"""Baseline forecasters. Every model must clearly beat these to be worth deploying."""
import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/London"


class NaiveForecaster:
    """Each target equals the actual at the same time 24h earlier.

    At a 12:00 issue time, the last target (11:30 tomorrow) would need 11:30 today,
    which isn't published yet. The harness hides it, so we fall back to 48h, 72h,
    then 7 days earlier. The same fallback covers outages and invalid values.
    """

    name = "naive_24h"
    fallback_days = (1, 2, 3, 7)

    def fit(self, history: pd.DataFrame, weather: pd.DataFrame) -> None:
        pass  # nothing to learn

    def predict(self, history, weather, issue_time, target_times) -> np.ndarray:
        known = history.set_index("period_start")["actual"]
        pred = pd.Series(np.nan, index=target_times)
        for days in self.fallback_days:
            candidate = known.reindex(target_times - pd.Timedelta(days=days)).to_numpy()
            pred = pred.fillna(pd.Series(candidate, index=target_times))
        last_valid = history["actual"].dropna().iloc[-1]
        return pred.fillna(last_valid).to_numpy(dtype="float64")


class ProfileForecaster:
    """Average of the last N weeks at the same UK local half-hour,
    split by weekday vs weekend. Knows the daily shape, not the weather."""

    def __init__(self, weeks: int = 4):
        self.weeks = weeks
        self.name = f"profile_{weeks}w"

    def fit(self, history: pd.DataFrame, weather: pd.DataFrame) -> None:
        pass  # profile is recomputed from recent history at each issue time

    @staticmethod
    def _slot_and_daytype(times: pd.DatetimeIndex | pd.Series):
        local = times.tz_convert(LOCAL_TZ) if isinstance(times, pd.DatetimeIndex) else times.dt.tz_convert(LOCAL_TZ)
        acc = local if isinstance(local, pd.DatetimeIndex) else local.dt
        slot = np.asarray(acc.hour * 2 + acc.minute // 30)
        weekend = np.asarray(acc.dayofweek >= 5)
        return slot, weekend

    def predict(self, history, weather, issue_time, target_times) -> np.ndarray:
        start = issue_time - pd.Timedelta(weeks=self.weeks)
        recent = history[(history["period_start"] >= start) & history["actual"].notna()]

        r_slot, r_weekend = self._slot_and_daytype(recent["period_start"])
        t_slot, t_weekend = self._slot_and_daytype(target_times)
        values = recent["actual"].to_numpy()

        by_slot_daytype = pd.Series(values).groupby([r_slot, r_weekend]).mean()
        pred = by_slot_daytype.reindex(pd.MultiIndex.from_arrays([t_slot, t_weekend])).to_numpy()

        # Fallbacks: same slot ignoring day type, then overall mean
        by_slot = pd.Series(values).groupby(r_slot).mean()
        pred = np.where(np.isnan(pred), by_slot.reindex(t_slot).to_numpy(), pred)
        pred = np.where(np.isnan(pred), values.mean(), pred)
        return pred.astype("float64")