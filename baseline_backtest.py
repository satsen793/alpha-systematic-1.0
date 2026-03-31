"""
Baseline backtest for Phase 1 (buy-and-hold + ret_5/ret_21 momentum).

Usage:
    python baseline_backtest.py --raw-path data/SPY/raw/spy_raw.parquet --out-dir data/SPY/baseline --snapshot-id s001
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import pandas as pd

import pipeline_spec as ps

TRADING_DAYS = 252
COST = ps.SPREAD + ps.SLIPPAGE + ps.COMMISSION


def compute_trade_returns(df: pd.DataFrame, signal: pd.Series) -> pd.Series:
    entry = df.get("open", df["adj_close"]).shift(-1)
    exit_price = df["adj_close"].shift(-1)
    gross = (exit_price / entry) - 1.0
    net = gross - COST
    return signal.fillna(0).astype(float) * net


def compute_buy_hold_returns(df: pd.DataFrame) -> pd.Series:
    return df["adj_close"].shift(-1) / df["adj_close"] - 1.0


def ann_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    mean = r.mean()
    std = r.std(ddof=0)
    if std == 0:
        return 0.0
    return float(mean / std * np.sqrt(TRADING_DAYS))


def cagr(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    return float(equity.iloc[-1] ** (TRADING_DAYS / len(r)) - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = 1.0 - (equity / peak)
    return float(drawdown.max())


def trades_per_month(signal: pd.Series) -> float:
    s = signal.dropna().astype(int)
    if s.empty:
        return float("nan")
    months = s.index.to_period("M")
    per_month = s.groupby(months).sum()
    return float(per_month.mean())


def summarize(name: str, returns: pd.Series, signal: pd.Series | None = None) -> Dict[str, float]:
    out = {
        "name": name,
        "ann_sharpe": ann_sharpe(returns),
        "cagr": cagr(returns),
        "max_dd": max_drawdown(returns),
        "mean_daily": float(returns.dropna().mean() if not returns.dropna().empty else np.nan),
        "std_daily": float(returns.dropna().std(ddof=0) if not returns.dropna().empty else np.nan),
        "days": int(returns.dropna().shape[0]),
    }
    if signal is not None:
        out["avg_trades_per_month"] = trades_per_month(signal)
        out["trade_days"] = int(signal.fillna(0).sum())
    return out


def eval_period(df: pd.DataFrame, start: str, end: str) -> Dict[str, Dict[str, float]]:
    d = df.loc[pd.to_datetime(start):pd.to_datetime(end)].copy()
    results = {}

    buy_hold = compute_buy_hold_returns(d)
    results["buy_hold"] = summarize("buy_hold", buy_hold)

    sig_5 = (d["ret_5"] > 0).astype(int)
    ret_5 = compute_trade_returns(d, sig_5)
    results["ret_5"] = summarize("ret_5", ret_5, sig_5)

    sig_21 = (d["ret_21"] > 0).astype(int)
    ret_21 = compute_trade_returns(d, sig_21)
    results["ret_21"] = summarize("ret_21", ret_21, sig_21)

    return results


def success_check(metrics: Dict[str, float]) -> Dict[str, bool]:
    return {
        "sharpe_gt_1_2": metrics.get("ann_sharpe", 0) > 1.2,
        "max_dd_lt_0_15": metrics.get("max_dd", 1) < 0.15,
    }


def main(args: argparse.Namespace) -> None:
    df_raw = ps.load_parquet(args.raw_path)
    df_clean = ps.preprocess(df_raw, start_date=args.start_date, end_date=args.end_date)
    df_feat = ps.compute_features(df_clean)

    # align features with clean data
    df = df_clean.join(df_feat[["ret_5", "ret_21"]], how="left")
    df = df.dropna(subset=["ret_5", "ret_21"]).copy()

    splits = ps.generate_static_splits()
    split = splits[0]
    results = {
        "split": split,
        "train": eval_period(df, split["train_start"], split["train_end"]),
        "val": eval_period(df, split["val_start"], split["val_end"]),
        "test": eval_period(df, split["test_start"], split["test_end"]),
    }

    # success checks on test for momentum baselines
    results["test_success"] = {
        "ret_5": success_check(results["test"]["ret_5"]),
        "ret_21": success_check(results["test"]["ret_21"]),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"baseline_report_{args.snapshot_id}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Saved baseline report:", out_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--snapshot-id", required=False, default="s001")
    parser.add_argument("--start-date", required=False)
    parser.add_argument("--end-date", required=False)
    main(parser.parse_args())
