"""Clean raw OHLC data for index backtests.

Rules:
- Datetime index, sorted, de-duplicated
- Drop rows with missing OHLC (no artificial candles)
- Enforce OHLC integrity
- Compute returns with clipping, plus RL-ready features
- Optional trading-calendar validation (no filling)

Usage:
    python clean_data.py --input data/NIFTY_50/raw/NIFTY_50_raw.parquet --out-dir data/NIFTY_50/clean
    python clean_data.py --all --data-dir data
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


PRICE_COLS = ["open", "high", "low", "close"]


def load_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    out.columns = [c.lower() for c in out.columns]
    for col in PRICE_COLS:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[PRICE_COLS]
    out = out.dropna()

    # Enforce OHLC integrity
    out = out[
        (out["low"] <= out["high"])
        & (out["open"] >= out["low"])
        & (out["open"] <= out["high"])
        & (out["close"] >= out["low"])
        & (out["close"] <= out["high"])
    ]

    out["returns"] = out["close"].pct_change(fill_method=None)
    out["clipped"] = False
    clip_mask = out["returns"].abs() > 0.20
    if clip_mask.any():
        out.loc[clip_mask, "returns"] = np.sign(out.loc[clip_mask, "returns"]) * 0.20
        out.loc[clip_mask, "clipped"] = True

    out["volatility"] = out["returns"].rolling(10).std()
    if not out.empty:
        out["norm_close"] = out["close"] / out["close"].iloc[0]
    else:
        out["norm_close"] = np.nan

    out = out.dropna()
    out.index.name = "date"
    return out


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=True)


def load_holidays(path: Optional[str]) -> Optional[pd.DatetimeIndex]:
    if not path:
        return None
    with open(path, "r") as f:
        raw = f.read()

    holidays: List[str] = []
    if path.lower().endswith(".json"):
        holidays = json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            holidays.append(line)

    if not holidays:
        return None
    return pd.to_datetime(holidays).tz_localize(None)


def expected_trading_days(
    start: pd.Timestamp,
    end: pd.Timestamp,
    use_nse_calendar: bool,
    holidays: Optional[pd.DatetimeIndex],
) -> pd.DatetimeIndex:
    if use_nse_calendar:
        try:
            import pandas_market_calendars as mcal
        except ImportError as exc:
            raise RuntimeError(
                "pandas_market_calendars is required for NSE calendar validation. "
                "Install it or omit --nse-calendar."
            ) from exc

        nse = mcal.get_calendar("NSE")
        schedule = nse.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(schedule.index).tz_localize(None)

    days = pd.bdate_range(start, end)
    if holidays is not None and len(holidays) > 0:
        days = days.difference(holidays)
    return days


def calendar_diagnostics(
    index: pd.DatetimeIndex,
    use_nse_calendar: bool,
    holidays: Optional[pd.DatetimeIndex],
) -> Dict[str, object]:
    expected = expected_trading_days(index.min(), index.max(), use_nse_calendar, holidays)
    missing = expected.difference(index)
    extra = index.difference(expected)
    return {
        "expected_days": int(len(expected)),
        "missing_days": int(len(missing)),
        "extra_days": int(len(extra)),
        "missing_sample": [d.strftime("%Y-%m-%d") for d in missing[:5]],
        "extra_sample": [d.strftime("%Y-%m-%d") for d in extra[:5]],
    }


def process_file(
    path: str,
    out_dir: str,
    use_nse_calendar: bool,
    holidays: Optional[pd.DatetimeIndex],
    validate_calendar: bool,
) -> Dict[str, object]:
    df = load_parquet(path)
    diagnostics: Dict[str, object] = {}
    if validate_calendar and not df.empty:
        diagnostics = calendar_diagnostics(df.index, use_nse_calendar, holidays)
    cleaned = clean_ohlc(df)
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{base.replace('_raw', '')}_clean.parquet")
    save_parquet(cleaned, out_path)
    payload: Dict[str, object] = {
        "input": path,
        "output": out_path,
        "rows": int(cleaned.shape[0]),
        "cols": list(cleaned.columns),
    }
    if diagnostics:
        payload["calendar"] = diagnostics
    return payload


def main(args: argparse.Namespace) -> None:
    outputs: List[Dict[str, object]] = []
    holidays = load_holidays(args.holidays_file)
    if args.all:
        pattern = os.path.join(args.data_dir, "*", "raw", "*_raw.parquet")
        paths = glob(pattern)
        if not paths:
            raise RuntimeError(f"No raw parquet files found at: {pattern}")
        for path in paths:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "clean")
            outputs.append(
                process_file(
                    path,
                    out_dir,
                    args.nse_calendar,
                    holidays,
                    args.validate_calendar,
                )
            )
    else:
        if not args.input or not args.out_dir:
            raise ValueError("--input and --out-dir are required unless --all is used")
        outputs.append(
            process_file(
                args.input,
                args.out_dir,
                args.nse_calendar,
                holidays,
                args.validate_calendar,
            )
        )

    manifest_path = os.path.join(args.data_dir if args.all else args.out_dir, "clean_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"cleaned": outputs}, f, indent=2)
    print("Saved clean manifest:", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data-dir", required=False, default="data")
    parser.add_argument("--validate-calendar", action="store_true")
    parser.add_argument("--nse-calendar", action="store_true")
    parser.add_argument("--holidays-file", required=False)
    main(parser.parse_args())
