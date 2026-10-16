from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from carbon_forecast.ingestion.neso import make_chunks
from carbon_forecast.ingestion.storage import save_monthly_partitions

UTC = timezone.utc


def test_chunks_cover_range_without_gaps_or_overlaps():
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2023, 4, 15, tzinfo=UTC)
    chunks = make_chunks(start, end, 30)

    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (_, a_end), (b_start, _) in zip(chunks, chunks[1:]):
        assert a_end == b_start
    assert all(e - s <= timedelta(days=30) for s, e in chunks)


def test_chunks_empty_when_start_equals_end():
    t = datetime(2023, 1, 1, tzinfo=UTC)
    assert make_chunks(t, t, 30) == []


def test_chunks_reject_non_positive_size():
    t = datetime(2023, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        make_chunks(t, t + timedelta(days=1), 0)


def test_saving_twice_does_not_duplicate(tmp_path):
    times = pd.date_range("2023-01-31 23:00", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({"period_start": times, "actual": [1.0, 2.0, 3.0, 4.0]})

    save_monthly_partitions(df, "period_start", tmp_path, "test", ["period_start"])
    save_monthly_partitions(df, "period_start", tmp_path, "test", ["period_start"])

    files = sorted(tmp_path.glob("*.parquet"))
    result = pd.concat(pd.read_parquet(f) for f in files)

    assert len(files) == 2          # January and February
    assert len(result) == 4         # no duplicates after second save