"""Public dashboard: today's forecast, live track record, backtest results.

Run locally:
    streamlit run dashboard/app.py
"""
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = "https://carbon-forecast-api.delightfulfield-1ac7d123.uksouth.azurecontainerapps.io"
REPO = "vijendrapokharkar15-design/gb-carbon-forecast-mlops"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
LOCAL_TZ = "Europe/London"
BACKTEST_MAE = 19.5   # honest test-year MAE, using day-ahead weather forecasts

MODEL_LABELS = {
    "naive_24h": "Naive (same time yesterday)",
    "profile_4w": "Seasonal profile (last 4 weeks)",
    "lgbm_l1_all": "LightGBM, observed weather (optimistic ceiling)",
    "lgbm_l1_all_fcwx": "LightGBM, day-ahead weather forecasts (honest)",
    "neso_latest_reference": "NESO stored forecast (near-real-time, reference only)",
}

st.set_page_config(page_title="GB Carbon Intensity Forecast", page_icon="🌱", layout="wide")


# ---------- data access (cached so visitors don't hammer the API) ----------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_forecast() -> dict:
    """Latest 12:00 UTC forecast. Retries cover a cold start when the API was scaled to zero."""
    last_error = None
    for attempt in range(3):
        try:
            r = requests.get(
                f"{API_URL}/forecast", params={"window_hours": 4, "upcoming_only": "true"}, timeout=120
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(10 * (attempt + 1))
    raise last_error


@st.cache_data(ttl=1800, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Read a tracked CSV straight from the public repo, so new data appears without redeploying."""
    try:
        return pd.read_csv(f"{RAW}/{path}")
    except Exception:
        return pd.DataFrame()


def to_local_naive(values):
    """UK local wall-clock time without timezone info, so charts show UK time for every visitor.
    Works for a Series of timestamps or a single Timestamp."""
    if isinstance(values, pd.Series):
        return values.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return values.tz_convert(LOCAL_TZ).tz_localize(None)


# ---------- header ----------

st.title("🌱 GB electricity carbon intensity: day-ahead forecast")
st.markdown(
    "When should flexible electricity use (EV charging, batch computing, batteries) run to emit the "
    "least CO₂? A LightGBM model forecasts Great Britain's grid carbon intensity for the next 24 hours, "
    "**issued every day at 12:00 UTC**, and recommends the greenest window. Times below are **UK time**."
)

# ---------- 1. today's forecast ----------

st.header("Today's forecast")
try:
    with st.spinner("Fetching the forecast. If the API was idle it can take up to a minute to wake up..."):
        fc = fetch_forecast()
except Exception as exc:
    fc = None
    st.error(
        f"The forecast API is unavailable right now ({type(exc).__name__}). "
        "The track record below is still up to date."
    )

if fc:
    df = pd.DataFrame(fc["forecast"])
    df["start"] = to_local_naive(pd.to_datetime(df["period_start"], utc=True))
    issue = pd.Timestamp(fc["issue_time"]).tz_convert(LOCAL_TZ)
    win = fc.get("greenest_window")

    c1, c2, c3, c4 = st.columns(4)
    if win:
        ws = pd.Timestamp(win["start"]).tz_convert(LOCAL_TZ)
        we = pd.Timestamp(win["end"]).tz_convert(LOCAL_TZ)
        c1.metric("Greenest 4-hour window", f"{ws:%a %H:%M} to {we:%H:%M}")
        c2.metric("Average in that window", f"{win['mean_intensity']:.0f} gCO₂/kWh")
    else:
        c1.metric("Greenest 4-hour window", "None left in this forecast")
        c2.metric("Average in that window", "n/a")
    c3.metric("Forecast issued", f"{issue:%a %d %b, %H:%M}")
    c4.metric("Model version", fc["model_version"])

    line = alt.Chart(df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("start:T", title="UK time", axis=alt.Axis(format="%a %H:%M")),
        y=alt.Y("intensity:Q", title="Forecast carbon intensity (gCO₂/kWh)", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("start:T", title="Half-hour starting", format="%a %d %b %H:%M"),
            alt.Tooltip("intensity:Q", title="gCO₂/kWh", format=".0f"),
        ],
    )
    chart = line
    if win:
        band = pd.DataFrame({"start": [ws.tz_localize(None)], "end": [we.tz_localize(None)]})
        chart = alt.Chart(band).mark_rect(opacity=0.18, color="green").encode(x="start:T", x2="end:T") + line
    st.altair_chart(chart.properties(height=340))
    st.caption("Shaded: the recommended greenest 4-hour window among the half-hours that haven't started yet.")

# ---------- 2. live track record ----------

st.header("Live track record")
scores = load_csv("monitoring/daily_scores.csv")
if scores.empty:
    st.info("No scored days yet. Each forecast is scored automatically once its 24 hours have passed.")
else:
    scores["record_late"] = scores["record_late"].astype(str).str.lower().eq("true")
    scores["status"] = scores["record_late"].map({True: "late record (excluded)", False: "on time"})
    scores["date"] = pd.to_datetime(scores["issue_date"])
    on_time = scores[~scores["record_late"]]

    a, b, c = st.columns(3)
    a.metric("Days scored live", len(scores))
    a_mae = f"{on_time['mae'].mean():.1f}" if len(on_time) else "n/a"
    b.metric("Live MAE, on-time days (gCO₂/kWh)", a_mae)
    c.metric("Backtest MAE promised", f"{BACKTEST_MAE}")

    points = alt.Chart(scores).mark_line(point=True).encode(
        x=alt.X("issue_date:O", title="Forecast issue date"),
        y=alt.Y("mae:Q", title="Daily MAE (gCO₂/kWh)", scale=alt.Scale(zero=True)),
        color=alt.Color("status:N", title=None),
        tooltip=["issue_date", "mae", "bias", "regret", "status"],
    )
    reference = alt.Chart(pd.DataFrame({"y": [BACKTEST_MAE]})).mark_rule(strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart((points + reference).properties(height=300))
    st.caption("Dashed line: the backtest MAE. Live performance close to it means the evaluation was honest.")

    st.subheader("Head-to-head with NESO's day-ahead forecast")
    if "neso_mae_common" in scores.columns:
        h2h = on_time.dropna(subset=["neso_mae_common"])
    else:
        h2h = pd.DataFrame()
    if h2h.empty:
        st.caption(
            "Fair comparisons need both forecasts captured before 12:00 UTC on the same day. "
            "They accumulate daily; scored on the half-hours both forecasts cover."
        )
    else:
        st.dataframe(
            h2h[["issue_date", "common_periods", "ours_mae_common", "neso_mae_common",
                 "ours_regret_common", "neso_regret_common"]].rename(columns={
                "issue_date": "Issue date", "common_periods": "Shared half-hours",
                "ours_mae_common": "Our MAE", "neso_mae_common": "NESO MAE",
                "ours_regret_common": "Our regret", "neso_regret_common": "NESO regret",
            }),
            hide_index=True,
        )

    with st.expander("All scored days (raw)"):
        st.dataframe(scores.drop(columns=["date", "status"]), hide_index=True)

# ---------- 3. backtest results ----------

st.header("Backtest: test year (Oct 2025 to Sep 2026)")
results = load_csv("docs/results.csv")
if not results.empty:
    test = results[(results["period"] == "test") & results["model"].isin(MODEL_LABELS)].copy()
    order = ["lgbm_l1_all_fcwx", "lgbm_l1_all", "profile_4w", "naive_24h", "neso_latest_reference"]
    test["rank"] = test["model"].map({m: i for i, m in enumerate(order)})
    test["Model"] = test["model"].map(MODEL_LABELS)
    test["good_window_rate"] = (test["good_window_rate"] * 100).round(1).astype(str) + "%"
    test = test.sort_values("rank")[["Model", "mae", "regret_mean", "good_window_rate"]].rename(columns={
        "mae": "MAE (gCO₂/kWh)", "regret_mean": "Carbon regret", "good_window_rate": "Good-window rate",
    })
    st.dataframe(test, hide_index=True)
    st.caption(
        "Walk-forward backtest with monthly retraining. Carbon regret: how much dirtier the recommended "
        "4-hour window was than the truly greenest one. NESO's stored history is revised close to real time, "
        "so it is a reference, not a fair day-ahead benchmark."
    )

# ---------- 4. how it works ----------

st.header("How it works")
st.markdown(
    f"""
- **Data:** NESO Carbon Intensity API (national actuals) and Open-Meteo weather (offshore wind, solar, temperature), validated with Pandera
- **Model:** LightGBM (L1 loss) with 24 features built by one shared function for training and serving, so no training/serving skew
- **Evaluation:** walk-forward backtest with a leakage-proof harness; scored on MAE and carbon regret, against naive and seasonal baselines
- **MLOps:** champion/challenger promotion, Docker image on Azure Container Apps, daily forecast records and NESO snapshots via GitHub Actions, automatic monitoring with alerts, CI on every push

[GitHub repository](https://github.com/{REPO}) · [Live API docs]({API_URL}/docs)
"""
)
st.caption("Weather data by Open-Meteo.com (CC BY 4.0). Carbon intensity data from the NESO Carbon Intensity API.")