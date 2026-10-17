"""Feature engineering.

One function, build_features, creates the 48 feature rows for a single issue time
from the history slice known at that moment. Training calls the same function for
every historical issue time, so training and serving cannot drift apart.
"""
import holidays
import numpy as np
import pandas as pd

from carbon_forecast.training.baselines import ProfileForecaster

LOCAL_TZ = "Europe/London"
WIND_INDEX_LOCATIONS = ["north_sea", "east_anglia", "irish_sea"]
WIND_CAP = 13.0                 # turbines reach rated output around here
GB_LAT, GB_LON = 53.5, -1.5     # rough centre of GB for solar geometry
_ENG_HOLIDAYS = holidays.country_holidays("GB", subdiv="ENG")

FEATURE_COLUMNS = [
    "horizon",
    # calendar (UK local time at the target)
    "slot_sin", "slot_cos", "day_of_week", "is_weekend", "is_bank_holiday",
    "is_christmas_period", "doy_sin", "doy_cos",
    # sun
    "solar_elevation",
    # weather at the target
    "wind_index", "wind_index_capped", "wind_north_sea", "wind_irish_sea", "wind_moray",
    "temp_london", "solar_radiation_south",
    # history known at issue time
    "last_known", "mean_3h", "mean_24h", "lag_24h", "lag_48h", "lag_168h", "profile_4w",
]


def solar_elevation(times_utc: pd.DatetimeIndex, lat: float = GB_LAT, lon: float = GB_LON) -> np.ndarray:
    """Sun elevation in degrees (NOAA approximation) at the middle of each half-hour."""
    t = times_utc + pd.Timedelta("15min")
    doy = np.asarray(t.dayofyear)
    hours = np.asarray(t.hour) + np.asarray(t.minute) / 60
    gamma = 2 * np.pi / 365 * (doy - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    hour_angle = np.radians((hours * 60 + eqtime + 4 * lon) / 4 - 180)
    phi = np.radians(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    return 90 - np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))


def calendar_features(target_times: pd.DatetimeIndex) -> dict:
    local = target_times.tz_convert(LOCAL_TZ)
    slot = np.asarray(local.hour) * 2 + np.asarray(local.minute) // 30
    doy = np.asarray(local.dayofyear)
    month, day = np.asarray(local.month), np.asarray(local.day)
    dow = np.asarray(local.dayofweek)
    return {
        "slot_sin": np.sin(2 * np.pi * slot / 48),
        "slot_cos": np.cos(2 * np.pi * slot / 48),
        "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(int),
        "is_bank_holiday": np.array([d in _ENG_HOLIDAYS for d in local.date], dtype=int),
        "is_christmas_period": (((month == 12) & (day >= 24)) | ((month == 1) & (day == 1))).astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }


def _weather_at(weather: pd.DataFrame, target_times: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather aligned to half-hours: :00 uses that hour, :30 averages the two hours around it."""
    w = weather.set_index("time")
    w.index = w.index.as_unit("ns")
    tt = target_times.as_unit("ns")
    floor = w.reindex(tt.floor("h")).to_numpy(dtype="float64")
    ceil = w.reindex(tt.ceil("h")).to_numpy(dtype="float64")
    values = np.where(np.isnan(ceil), floor, (floor + ceil) / 2)
    return pd.DataFrame(values, columns=w.columns, index=target_times)


def build_features(
    history: pd.DataFrame,
    weather: pd.DataFrame,
    issue_time: pd.Timestamp,
    target_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Feature rows for one issue time. `history` must contain only periods known at issue time."""
    feats = pd.DataFrame({"target_time": target_times, "horizon": np.arange(1, len(target_times) + 1)})

    for name, values in calendar_features(target_times).items():
        feats[name] = values
    feats["solar_elevation"] = solar_elevation(target_times)

    wx = _weather_at(weather, target_times)
    wind_cols = [f"wind_speed_100m__{loc}" for loc in WIND_INDEX_LOCATIONS]
    wind_index = wx[wind_cols].mean(axis=1).to_numpy()
    feats["wind_index"] = wind_index
    feats["wind_index_capped"] = np.minimum(wind_index, WIND_CAP)
    feats["wind_north_sea"] = wx["wind_speed_100m__north_sea"].to_numpy()
    feats["wind_irish_sea"] = wx["wind_speed_100m__irish_sea"].to_numpy()
    feats["wind_moray"] = wx["wind_speed_100m__moray_firth"].to_numpy()
    feats["temp_london"] = wx["temperature_2m__london"].to_numpy()
    feats["solar_radiation_south"] = wx["shortwave_radiation__south_england_solar"].to_numpy()

    actual = history["actual"]
    known = history.set_index("period_start")["actual"]
    known.index = known.index.as_unit("ns")
    tt = target_times.as_unit("ns")
    valid = actual.dropna()

    feats["last_known"] = valid.iloc[-1] if len(valid) else np.nan
    feats["mean_3h"] = actual.tail(6).mean()
    feats["mean_24h"] = actual.tail(48).mean()
    # Lags are only filled if that period was published by the cutoff; otherwise NaN
    for days, col in [(1, "lag_24h"), (2, "lag_48h"), (7, "lag_168h")]:
        feats[col] = known.reindex(tt - pd.Timedelta(days=days)).to_numpy()
    feats["profile_4w"] = ProfileForecaster(weeks=4).predict(history, None, issue_time, target_times)

    return feats[["target_time", *FEATURE_COLUMNS]]


def build_training_set(
    neso: pd.DataFrame,
    weather: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    horizon_steps: int = 48,
    publication_lag_minutes: int = 30,
) -> pd.DataFrame:
    """Features plus target for many issue times, sliced exactly like the backtest harness."""
    neso = neso.sort_values("period_start").reset_index(drop=True)
    weather = weather.sort_values("time").reset_index(drop=True)
    truth = neso.set_index("period_start")[["actual", "is_valid_target"]]
    lag = pd.Timedelta(minutes=publication_lag_minutes)

    frames = []
    for issue_time in issue_times:
        cutoff = issue_time - lag
        history = neso.iloc[: neso["period_start"].searchsorted(cutoff, side="left")]
        target_times = pd.date_range(issue_time, periods=horizon_steps, freq="30min").as_unit(truth.index.unit)

        feats = build_features(history, weather, issue_time, target_times)
        block = truth.reindex(target_times)
        feats.insert(0, "issue_time", issue_time)
        feats["y"] = block["actual"].to_numpy(dtype="float64")
        feats["is_valid_target"] = block["is_valid_target"].fillna(False).astype(bool).to_numpy()
        frames.append(feats)

    return pd.concat(frames, ignore_index=True)