import numpy as np
import lightgbm as lgb
import pytest

from carbon_forecast.training.registry import FeatureMismatchError, ModelRegistry

FEATURES = ["a", "b"]


def tiny_booster():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 2)), rng.normal(size=50)
    return lgb.train({"verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), num_boost_round=5)


def test_empty_registry_has_no_champion(tmp_path):
    reg = ModelRegistry(tmp_path)
    assert reg.champion_version() is None
    assert reg.load_champion(FEATURES) is None


def test_save_promote_and_reload(tmp_path):
    reg = ModelRegistry(tmp_path)
    booster = tiny_booster()
    reg.save_and_promote(booster, {"feature_columns": FEATURES, "decision_reason": "first"}, version="v1")

    loaded, meta = reg.load_champion(FEATURES)
    assert reg.champion_version() == "v1"
    assert meta["version"] == "v1"
    X = np.array([[0.1, 0.2], [1.0, -1.0]])
    assert np.allclose(loaded.predict(X), booster.predict(X))  # identical predictions after reload


def test_history_records_promotions_and_rejections(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.save_and_promote(tiny_booster(), {"feature_columns": FEATURES, "decision_reason": "first"}, version="v1")
    reg.record_rejection("regression: MAE worse", {"mae": 99.0})
    reg.save_and_promote(tiny_booster(), {"feature_columns": FEATURES, "decision_reason": "better"}, version="v2")

    history = reg.history()
    assert [h["promoted"] for h in history] == [True, False, True]
    assert reg.champion_version() == "v2"


def test_versions_are_immutable(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.save_and_promote(tiny_booster(), {"feature_columns": FEATURES}, version="v1")
    with pytest.raises(FileExistsError):
        reg.save_and_promote(tiny_booster(), {"feature_columns": FEATURES}, version="v1")


def test_feature_mismatch_refuses_to_load(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.save_and_promote(tiny_booster(), {"feature_columns": FEATURES}, version="v1")
    with pytest.raises(FeatureMismatchError):
        reg.load_champion(["a", "b", "c"])