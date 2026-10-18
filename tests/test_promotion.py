from carbon_forecast.training.promotion import decide

NAIVE = {"mae": 40.0, "regret_mean": 20.0}
TOL = {"mae_tolerance": 0.5, "regret_tolerance": 0.5}


def test_first_model_promoted_if_it_beats_naive():
    d = decide({"mae": 18.0, "regret_mean": 4.0}, NAIVE, None, **TOL)
    assert d.promote
    assert "first model" in d.reason


def test_rejected_if_it_does_not_beat_naive_even_without_champion():
    d = decide({"mae": 41.0, "regret_mean": 4.0}, NAIVE, None, **TOL)
    assert not d.promote
    assert "guardrail" in d.reason


def test_better_challenger_promoted():
    d = decide({"mae": 17.0, "regret_mean": 3.0}, NAIVE, {"mae": 18.0, "regret_mean": 4.0}, **TOL)
    assert d.promote


def test_tied_challenger_promoted_within_tolerance():
    d = decide({"mae": 18.3, "regret_mean": 4.2}, NAIVE, {"mae": 18.0, "regret_mean": 4.0}, **TOL)
    assert d.promote
    assert "no regression" in d.reason


def test_mae_regression_beyond_tolerance_rejected():
    d = decide({"mae": 19.0, "regret_mean": 3.0}, NAIVE, {"mae": 18.0, "regret_mean": 4.0}, **TOL)
    assert not d.promote
    assert "MAE" in d.reason


def test_regret_regression_beyond_tolerance_rejected():
    d = decide({"mae": 17.0, "regret_mean": 5.0}, NAIVE, {"mae": 18.0, "regret_mean": 4.0}, **TOL)
    assert not d.promote
    assert "regret" in d.reason