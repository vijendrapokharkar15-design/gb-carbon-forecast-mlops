"""Issue-time logic, kept dependency-free so lightweight jobs can use it."""
import pandas as pd


def latest_issue_time(now: pd.Timestamp, issue_hour_utc: int = 12) -> pd.Timestamp:
    """The most recent official issue time: today's if it has passed, otherwise yesterday's."""
    now = pd.Timestamp(now).tz_convert("UTC")
    today = now.normalize() + pd.Timedelta(hours=issue_hour_utc)
    return today if now >= today else today - pd.Timedelta(days=1)