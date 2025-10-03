"""Download daily SPY price data without third-party dependencies."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional
from urllib.request import urlopen


STOOQ_URL = "https://stooq.com/q/d/l/?s=spy.us&i=d"


@dataclass
class PriceBar:
    """Single OHLCV record for SPY."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


def _parse_rows(rows: Iterable[dict]) -> List[PriceBar]:
    bars: List[PriceBar] = []
    for row in rows:
        if row["Date"] == "":
            continue
        bars.append(
            PriceBar(
                date=datetime.strptime(row["Date"], "%Y-%m-%d").date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
        )
    bars.sort(key=lambda b: b.date)
    return bars


def download_spy_data(start: str = "2010-01-01", end: Optional[str] = None) -> List[PriceBar]:
    """Download historical SPY data from Stooq.

    Parameters
    ----------
    start:
        Earliest date to include (ISO format).
    end:
        Latest date to include (ISO format). Defaults to today.
    """

    if end is None:
        end = date.today().isoformat()

    with urlopen(STOOQ_URL, timeout=30) as response:
        reader = csv.DictReader(line.decode("utf-8") for line in response)
        bars = _parse_rows(reader)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()

    return [bar for bar in bars if start_date <= bar.date <= end_date]
