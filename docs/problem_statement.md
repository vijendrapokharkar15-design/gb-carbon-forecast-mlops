# Problem Statement: GB Carbon Intensity Day-Ahead Forecast

## 1. Business problem

Great Britain's electricity can be more than five times dirtier at some times than others,
depending mainly on how windy and sunny it is and on demand. Many electricity users have
**flexible loads** they can shift in time without harming operations:

- EV fleet depots charging vehicles overnight
- Data centres running batch jobs (backups, model training, reporting)
- Battery storage and heat-pump operators

These users need a reliable answer to one question each day:
**"When in the next 24 hours should I run my flexible load to emit the least carbon?"**

This project builds and operates a forecasting system that answers that question,
and proves its reliability every day against live data.

## 2. Prediction task

- **Target:** national GB carbon intensity (`actual`, gCO2/kWh) from the NESO Carbon Intensity API
- **Granularity:** half-hourly
- **Horizon:** the next 48 half-hours (24 hours)
- **Issue time:** once per day at a fixed time (configurable; default 12:00 UTC)
- **Output:** 48 point forecasts, plus the recommended greenest window for a 4-hour load

Only information available at issue time may be used. Lag features are computed relative
to the forecast issue time, never relative to the target period.

## 3. Data sources

| Source | Content | Resolution | History used |
|---|---|---|---|
| NESO Carbon Intensity API | National actual and forecast intensity | 30 min | Jan 2023 onwards |
| Open-Meteo archive | Wind at 100m, solar radiation, cloud cover, temperature at 4 GB points | 1 hour | Jan 2023 onwards |
| NESO forecast snapshots (own collection) | NESO's 24h-ahead forecast as issued each day | 30 min | From Oct 2026 |

## 4. Known data limitations

- **No regional actuals.** Regional endpoints (including London) provide forecasts only, so the
  target is national. London intensity differs materially from national, so the business framing is GB-wide.
- **Two source outages:** 20 to 22 Oct 2023 (91 periods) and 11 to 12 Jun 2024 (31 periods),
  confirmed by re-fetch. Not interpolated; excluded from training.
- **Invalid values:** 5 null and 6 zero `actual` readings (zeros occur where forecast was 62 to 185).
  Treated as invalid.
- **Historical NESO forecasts are not day-ahead.** The stored forecast has an MAE of only 9.9 gCO2/kWh,
  far too accurate for a 24h-ahead forecast; it very likely reflects near-real-time updates.
  It is therefore not a fair benchmark for this task.
- **Weather in training vs production.** Models train on observed historical weather but run on
  weather forecasts in production, so offline accuracy will be somewhat optimistic.
- **Regime change.** Mean intensity fell from 152 (2023) to ~125 (2024 onwards); 2023 may not
  represent today's grid.

## 5. Metrics

**Technical**
- **MAE** (gCO2/kWh) across all 48 steps, also reported by horizon and by hour of day

**Business**
- **Carbon regret:** mean actual intensity of our recommended 4-hour window minus mean actual
  intensity of the truly greenest 4-hour window that day (gCO2/kWh). Zero means we picked the best window.
- **Good-window rate:** share of days where carbon regret is 10 gCO2/kWh or less

The business metrics matter most: a forecast can have a mediocre MAE yet still rank the
hours correctly, which is what the user actually needs.

## 6. Baselines

1. **Naive:** each half-hour equals the same half-hour 24 hours earlier
2. **Seasonal profile:** average intensity for that hour, day type and month over recent weeks
3. **NESO day-ahead:** NESO's forecast as issued 24h ahead, from our own daily snapshots
   (available once enough days are collected)

The stored historical NESO forecast (MAE 9.9) is reported only as a near-real-time ceiling,
not as a target.

## 7. Success criteria

Fixed before any model is trained. Baseline scores are computed first in Phase 2 on the
same test period, then the model is judged against them.

**Model quality** (time-based test set, most recent months)
- MAE at least **25% lower** than the naive baseline
- Carbon regret at least **25% lower** than the naive baseline
- Once 30+ days of NESO snapshots exist: model MAE and carbon regret reported side by side
  with NESO day-ahead. Matching NESO is a strong result; beating it on carbon regret is the stretch goal.

**Operational**
- Forecast published every day by the issue time, with at least 95% pipeline success over 30 days
- Every data batch validated before use; failed validation stops the pipeline and alerts
- A retrained model is promoted only if it beats the current model on recent data
- Live MAE and carbon regret tracked daily; sustained degradation triggers an alert

## 8. Out of scope

- Regional forecasts (no regional ground truth)
- Intraday re-forecasting more than once per day
- Forecasting generation mix or electricity prices
- Controlling real devices (the system recommends; it does not act)
