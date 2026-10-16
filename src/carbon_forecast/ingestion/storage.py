"""Save data as monthly Parquet files. Re-running never creates duplicates."""
from pathlib import Path

import pandas as pd


def save_monthly_partitions(
    df: pd.DataFrame,
    time_col: str,
    out_dir: Path,
    prefix: str,
    key_cols: list[str],
) -> list[Path]:
    """Write one Parquet file per month. If a month's file already exists,
    merge with it and drop duplicates on key_cols (new values win)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    months = df[time_col].dt.strftime("%Y-%m")
    for month, part in df.groupby(months):
        path = out_dir / f"{prefix}_{month}.parquet"
        if path.exists():
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        part = (
            part.drop_duplicates(subset=key_cols, keep="last")
            .sort_values(key_cols)
            .reset_index(drop=True)
        )
        part.to_parquet(path, index=False)
        written.append(path)

    return written

def load_partitions(folder: Path) -> pd.DataFrame:
    """Read every monthly Parquet file in a folder into one DataFrame."""
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {folder}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)