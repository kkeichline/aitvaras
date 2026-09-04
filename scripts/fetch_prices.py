"""Fetch real daily bars into data/prices/real/ (gitignored, never committed).

Optional. Tests and CI run entirely on the committed synthetic fixture; this
exists for when you want transcripts over real price action.

Real vendor data is deliberately *not* redistributed with this repo -- the terms
are murky and this repository is public. Fetch it yourself, locally, and keep it
out of git.

    uv run --with yfinance python scripts/fetch_prices.py

The output satisfies the same ``MarketDataSource`` contract as the synthetic
fixture, so nothing downstream changes:

    ReplayDataSource("data/prices/real")
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_TICKERS = (
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "JNJ",
    "JPM",
    "PG",
    "XOM",
    "KO",
    "TSLA",
    "SPY",
    "XLY",
)
OUT = Path(__file__).resolve().parents[1] / "data" / "prices" / "real"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-01-01")
    ap.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    args = ap.parse_args()

    try:
        import yfinance
    except ImportError:
        print(
            "yfinance is not installed. It is an optional, local-only dependency:\n"
            "  uv run --with yfinance python scripts/fetch_prices.py",
            file=sys.stderr,
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        df = yfinance.download(
            ticker, start=args.start, end=args.end, progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            print(f"{ticker}: no data returned", file=sys.stderr)
            continue
        with (OUT / f"{ticker}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for idx, row in df.iterrows():
                w.writerow(
                    [
                        idx.date().isoformat(),
                        f"{float(row['Open']):.4f}",
                        f"{float(row['High']):.4f}",
                        f"{float(row['Low']):.4f}",
                        f"{float(row['Close']):.4f}",
                        int(row["Volume"]),
                    ]
                )
        print(f"{ticker}: {len(df)} bars -> {OUT / f'{ticker}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
