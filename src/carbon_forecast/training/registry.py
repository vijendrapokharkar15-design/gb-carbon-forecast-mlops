"""A small file-based model registry.

models/
  registry.json          champion version + history of every decision
  <version>/model.txt    LightGBM model in its native text format
  <version>/metadata.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb


class FeatureMismatchError(Exception):
    """The registered model expects different features from the current code."""


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "registry.json"

    # ---------- index ----------

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"champion": None, "history": []}

    def _save_index(self, index: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def champion_version(self) -> str | None:
        return self._load_index()["champion"]

    def history(self) -> list[dict]:
        return self._load_index()["history"]

    # ---------- read ----------

    def load_champion(self, expected_features: list[str]) -> tuple[lgb.Booster, dict] | None:
        """Return (booster, metadata) for the champion, or None if there isn't one.
        Refuses to load a model whose feature list doesn't match the current code."""
        version = self.champion_version()
        if version is None:
            return None
        folder = self.root / version
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        if metadata["feature_columns"] != list(expected_features):
            raise FeatureMismatchError(
                f"Champion {version} was trained on different features than the current code"
            )
        return lgb.Booster(model_file=str(folder / "model.txt")), metadata

    # ---------- write ----------

    def save_and_promote(self, booster: lgb.Booster, metadata: dict, version: str | None = None) -> str:
        """Save a new model version and make it the champion."""
        version = version or "v" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = self.root / version
        folder.mkdir(parents=True, exist_ok=False)
        booster.save_model(str(folder / "model.txt"))
        (folder / "metadata.json").write_text(json.dumps({**metadata, "version": version}, indent=2), encoding="utf-8")

        index = self._load_index()
        index["champion"] = version
        index["history"].append({"version": version, "promoted": True, "at": _now(), "reason": metadata.get("decision_reason", "")})
        self._save_index(index)
        return version

    def record_rejection(self, reason: str, metrics: dict) -> None:
        """Log a rejected challenger. Nothing is saved except the reason."""
        index = self._load_index()
        index["history"].append({"version": None, "promoted": False, "at": _now(), "reason": reason, "metrics": metrics})
        self._save_index(index)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")