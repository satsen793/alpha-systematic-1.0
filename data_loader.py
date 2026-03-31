"""Download and store India index data via NSEPython (local edition).

Usage:
    python data_loader.py --start 2007-01-01 --end 2025-12-31 --out-dir data
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict

import pandas as pd
from nsepython import index_history


INDEX_MAP = {
    "NIFTY 50": ["NIFTY 50"],
    "NIFTY BANK": ["NIFTY BANK"],
    "SENSEX": ["SENSEX", "S&P BSE SENSEX", "BSE SENSEX"],
}


def to_nse_date(d: str) -> str:
    return pd.to_datetime(d).strftime("%d-%b-%Y")


def normalize_index_history(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out.columns = [c.lower().strip().replace(" ", "_") for c in out.columns]
    if "historicaldate" in out.columns:
        out["date"] = pd.to_datetime(out["historicaldate"], errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "indexdate" in out.columns:
        out["date"] = pd.to_datetime(out["indexdate"], errors="coerce")
    else:
        raise RuntimeError("Missing date column in index history output.")
    out = out.sort_values("date")
    out = out.set_index("date")
    out = out.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        }
    )
    out["adj_close"] = out["close"]
    keep = ["open", "high", "low", "close", "adj_close"]
    out = out[keep]
    return out


def fetch_index_history(symbol: str, start: str, end: str, retries: int, sleep_seconds: int) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return index_history(symbol, start, end)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                jitter = random.uniform(0.0, 1.0)
                time.sleep(sleep_seconds + jitter)
    raise RuntimeError(f"Failed to fetch {symbol} after {retries} attempts: {last_error}")


def save_frame(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(out_path, index=True)
    elif fmt == "csv":
        df.to_csv(out_path, index=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def main(args: argparse.Namespace) -> None:
    start = to_nse_date(args.start)
    end = to_nse_date(args.end) if args.end else to_nse_date(datetime.utcnow().strftime("%Y-%m-%d"))

    manifest: Dict[str, object] = {
        "source": "nsepython_local",
        "start": args.start,
        "end": args.end or datetime.utcnow().strftime("%Y-%m-%d"),
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "indices": {},
    }

    for label, symbols in INDEX_MAP.items():
        df = pd.DataFrame()
        last_error: Exception | None = None
        for symbol in symbols:
            try:
                raw = fetch_index_history(symbol, start, end, args.retries, args.sleep)
                df = normalize_index_history(raw)
                if not df.empty:
                    break
                last_error = RuntimeError(f"Empty data for {symbol}")
            except Exception as exc:
                last_error = exc
                df = pd.DataFrame()
        if df.empty:
            if args.allow_missing:
                manifest["indices"][label] = {
                    "rows": 0,
                    "cols": [],
                    "path": None,
                    "error": str(last_error),
                }
                print(f"Skipped {label}: {last_error}")
                continue
            raise RuntimeError(f"No data returned for {label}. Last error: {last_error}")

        out_dir = os.path.join(args.out_dir, label.replace(" ", "_").upper(), "raw")
        ext = "parquet" if args.format == "parquet" else "csv"
        out_path = os.path.join(out_dir, f"{label.replace(' ', '_').upper()}_raw.{ext}")
        save_frame(df, out_path, args.format)

        manifest["indices"][label] = {
            "rows": int(df.shape[0]),
            "cols": list(df.columns),
            "path": out_path,
        }
        print(f"Saved {label} -> {out_path} ({df.shape[0]} rows)")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved manifest:", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=False, default="2007-01-01")
    parser.add_argument("--end", required=False, default=None)
    parser.add_argument("--out-dir", required=False, default="data")
    parser.add_argument("--format", required=False, default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--retries", required=False, type=int, default=5)
    parser.add_argument("--sleep", required=False, type=int, default=5)
    parser.add_argument("--allow-missing", action="store_true")
    main(parser.parse_args())
