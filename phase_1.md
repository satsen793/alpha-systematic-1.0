# Phase 1 

## Objective
- Primary: Maximize walk-forward annualized Sharpe ratio on `^NSEI` daily signals while controlling max drawdown.
- Secondary: Improve OOS Sharpe ≥10–20% vs static baseline, reduce drawdown, keep turnover realistic.

## Alpha Hypotheses (core, testable)
H1 (Momentum): Short-term momentum persists in low-volatility regimes; trend-following has positive expectancy.
H2 (Mean Reversion): Large negative deviations in high-volatility regimes partially revert within 1–5 days.
H3 (Volatility Clustering): High realized volatility reduces signal quality; exposure should be reduced when volatility spikes.

## Alpha Definition & Success Criteria
- Minimum OOS Sharpe: > 1.2
- Mean OOS Sharpe uplift ≥10% vs baseline OR bootstrap p-value < 0.05
- Max drawdown: < 15%
- Turnover constraint: < 2.0x annualized (configurable threshold)
- Cost stress test: +20% transaction cost sensitivity must retain positive Sharpe
- Reproducibility: given manifest + seed, run reproduces artifacts bitwise

## Baseline Benchmarks (must report)
- Buy & hold `^NSEI` (same period, same cost assumptions if applicable)
- Simple momentum rule (ret_5 or ret_21 threshold-based)

## Failure Conditions (kill criteria)
- Information coefficient (IC) ≈ 0 on OOS folds for 2 consecutive folds.
- Sharpe unstable across folds (e.g., fold Sharpe std dev > 0.7).
- Strategy loses profitability under +20% cost stress.
- Regimes not statistically distinct (no significant difference in return distributions).

## Decision-First Framing
- Core objective: decide when to take risk, not just predict returns.
- Pipeline output: signal strength → confidence → cost-adjusted edge → position size.
- Reality check: if a simple rule-based system with the same hypotheses cannot make money, the ML adds no real edge.

## One-Page Hypothesis Spec (required before coding)
1) Alpha hypotheses (max 3, above)
2) Feature support for each hypothesis
3) Regime conditions where each should work
4) Regime conditions where each should not work
5) Success/failure metrics and thresholds

## Architecture → Hypothesis Mapping
- Features: encode momentum (ret_1/ret_5/ema_slope), mean reversion (zscore_21), and volatility regime (vol_21/atr_14).
- HMM: detect when each hypothesis is expected to hold (low-vol trend vs. high-vol mean reversion).
- Model: estimate strength of the hypothesis conditional on regime and feature state.
- Meta-label: filter weak or cost-negative hypothesis instances.
- Backtest: validate hypothesis profitability after costs and constraints.

## Highest-Impact Improvements (prioritized for OOS Sharpe)
1) Trade only when edge is regime-consistent and cost-positive
  - Add strict trade filters: require regime posterior > 0.7 and expected return > cost buffer.
  - Rationale: improves signal quality by avoiding marginal trades that are noise after costs.
2) Stability-focused regime construction
  - Replace raw HMM states with HMM + minimum dwell + transition penalty; merge statistically indistinguishable regimes.
  - Rationale: removes regime flip-flop that destroys Sharpe.
3) Risk-aware position sizing
  - Use volatility targeting with drawdown caps and regime-specific risk budgets.
  - Rationale: reduces tail risk and improves risk-adjusted returns.
4) Economic feature triage
  - Drop any feature that does not contribute to Sharpe in walk-forward ablations.
  - Rationale: smaller, more robust feature set generalizes better.
5) Cost-aware labeling and meta-labeling
  - Meta-label based on net trade profit after costs; suppress low-edge trades.
  - Rationale: aligns model output with trading objective.

## Critical Evaluation Checklist (must pass)
- Economic meaning: every retained feature has a plausible economic mechanism (momentum, mean reversion, volatility risk premium, liquidity).
- Regime distinctness: HMM states must show statistically different return distributions and turnover profiles.
- Tradeability: post-cost P&L distribution must remain positive; otherwise discard signal.

## Signal Diagnostics (required)
- IC: feature vs forward return correlation (report mean and t-stat)
- IC stability: fold-by-fold IC variance and sign consistency

## Testable Hypotheses (3–5)
H1) Regime-confidence gating (posterior > 0.7) improves OOS Sharpe by reducing low-conviction trades.
H2) Regime dwell constraints (min 5–10 days) reduce turnover and improve Sharpe without reducing CAGR.
H3) Cost-aware meta-labeling improves P&L by filtering trades with negative expected net profit.
H4) Volatility-targeted sizing with drawdown caps reduces tail losses and improves Calmar.
H5) Feature pruning to 6–8 core features improves generalization vs. full feature set.

## Minimal-Change Ablation Experiments (exact design)
E1: Regime gating
- Change: add posterior threshold filter at trade time.
- Baseline: same model without gating.
- Measure: OOS Sharpe, turnover, max drawdown; use paired bootstrap on daily returns.
- Success: Sharpe improves and turnover drops without large CAGR loss.

E2: Regime stability constraint
- Change: enforce min dwell (5/10/15 days) and transition penalty in HMM posteriors.
- Baseline: raw HMM state sequence.
- Measure: Sharpe, turnover, regime persistence stats.
- Success: fewer regime flips and higher Sharpe.

E3: Meta-label filter
- Change: train meta-label classifier, require P(y_meta=1) > threshold.
- Baseline: no meta-label filtering.
- Measure: Sharpe, trade count, cost-adjusted hit rate.
- Success: Sharpe improves with fewer trades, lower drawdown.

E4: Feature pruning
- Change: restrict to top-k features by walk-forward Sharpe contribution (k=6,8,10).
- Baseline: full feature set.
- Measure: Sharpe, stability across folds.
- Success: similar or better Sharpe with tighter variance across folds.

E5: Risk sizing
- Change: volatility targeting + drawdown cap (reduce exposure when trailing drawdown > X%).
- Baseline: fixed position sizing.
- Measure: Sharpe, max drawdown, Calmar.
- Success: improved Calmar with minimal Sharpe loss.

## Expected Failure Modes & Detection
- False regime separation: regimes not statistically distinct → detect via KS tests on returns per regime.
- Over-filtering: too few trades → detect by low trade count and unstable Sharpe.
- Cost sensitivity: Sharpe collapses under realistic slippage → detect via cost sweep (±50%).
- Temporal decay: strong in-sample, weak out-of-sample → detect via fold-by-fold Sharpe instability.
- Feature drift: distribution shifts in recent years → detect via KS drift vs. training baseline.

## Data Spec
- Ticker: `^NSEI` (daily). Use `adj_close` for return computations.
- Raw fields: `date, open, high, low, close, adj_close, volume`.
- Date range: 2007-01-01 → latest. Static splits (baseline):
  - Train: 2007-01-01 → 2018-12-31
  - Val:   2019-01-01 → 2021-12-31
  - Test:  2022-01-01 → latest
- Storage: Parquet partitioned by year. Keep immutable snapshot_id.
- Data source: NSEPython (local edition). Use CSV/Parquet snapshots.
- Preprocessing:
  - Reindex to NY business days, forward-fill `adj_close` only for short holiday gaps (limit=1).
  - Compute `raw_ret = adj_close.pct_change()`, clip at ±20% (log clipped rows).
  - Timezone normalized; save snapshot manifest.

## Exact Feature Spec (shift-safe)
- Rule: Decision date = T. All features must use data ≤ T-1. Implement by creating base series shifted by 1: `close_s1 = adj_close.shift(1)` and compute features from shifted series.

Exact formulas (decision date T):
- ret_1 = close[t-1] / close[t-2] - 1
- ret_5 = close[t-1] / close[t-6] - 1
- ret_21 = close[t-1] / close[t-22] - 1
- vol_21 = std(ret_1 over t-22..t-2)
- mean_21 = mean(ret_1 over t-22..t-2)
- zscore_21 = (ret_1 - mean_21) / vol_21
- ema_slope_21 = (EMA_21[t-1] - EMA_21[t-6]) / EMA_21[t-6]  (compute EMA on close_s1)
- atr_14 = ATR computed on high_s1/low_s1/close_s1 with window=14
- autocorr_1 = lag-1 autocorrelation of ret_1 over t-22..t-2
- volume_adv21 = mean(volume over t-22..t-2)
- volume_z_21 = (volume[t-1] - volume_adv21) / std(volume over t-22..t-2)
- momentum_ratio = count(ret_1 > 0 over t-22..t-2) / 21

Implementation notes:
- Compute `close_s1 = adj_close.shift(1)` then derive rolling stats with `.rolling(window).shift(1)` or compute on already shifted series.
- Set `min_periods` (e.g., 15) to avoid noisy early samples; exclude early rows from training.

## Label Spec & Execution Assumptions
- Signal computed for date T (features use ≤ T-1).
- Execution: signal at T, enter at next open on T+1 (NSE), exit at next rebalance (daily/weekly/monthly).
- Regression target (one-day forward close-to-close):
  - y_reg = adj_close.shift(-1)[T] / adj_close[T] - 1
- Classification target:
  - y_cls = 1 if y_reg > 0 else 0
- Meta-label (profit after costs):
  - entry = open.shift(-1)
  - exit = adj_close.shift(-1)
  - gross = exit / entry - 1
  - cost = spread + slippage + commission (India baseline: 0.0002 + 0.0005 + 0.0001 = 0.0008)
  - y_meta = 1 if gross - cost > 0 else 0

All label computations use `.shift(-1)` consistently and are logged with snapshot IDs.

## CV Spec (leakage-proof)
- Static split (baseline): see Data Spec.
- Rolling walk-forward generator (strict):
  - train_years = 10, val_years = 2, test_years = 1, step = 1 year.
  - Fold i example: Train 2005–2014 | Val 2015–2016 | Test 2017
  - Advance start by 1 year; stop when test_end > last_date.
- For each fold: fit all training artifacts on TRAIN only (HMM, scalers, etc.), tune on VAL, evaluate on TEST.

## Leakage-Proof Rules (explicit)
- ALWAYS shift raw price series by 1 before computing features.
- Fit scalers/norms on TRAIN only; save and apply to VAL/TEST.
- HMM: train on TRAIN only; infer on VAL/TEST using saved params.
- Hyperparameter tuning uses TRAIN→VAL only. TEST is untouched until final evaluation.
- When computing meta_label, compute costs and execution assumptions using future prices consistently but separate from feature computation (features are shift-safe).

## Regime Detector (HMM baseline)
- Input vector: [vol_21, ema_slope_21, autocorr_1] normalized by TRAIN mean/std.
- HMM config: GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=SEED).
- Inference: posterior = predict_proba(input_at_T); smooth with EWMA(alpha=0.2); require min_dwell=5 days before accepting state change.
- Persist HMM params per-fold.

## Modeling (per-regime XGBoost)
- Per-regime models: XGBoost regressor for y_reg and classifier for y_cls/meta.
- Train per-regime using rows in TRAIN assigned to that regime.
- Global baseline: XGBoost trained on all TRAIN rows.
- Hyperparam search (Bayesian):
  - n_estimators: [100,1000]
  - learning_rate: [0.01,0.2]
  - max_depth: [3,10]
  - subsample: [0.5,1.0]
  - colsample_bytree: [0.5,1.0]
  - min_child_weight: [1,10]
- Tuning objective: VAL Sharpe from backtest (not raw prediction metric).
- Final model per-fold: retrain on TRAIN+VAL with chosen hyperparams (explicit option logged).

## Meta-Learner & Sizing
- Router: weight per-regime model outputs by smoothed HMM posterior × model_confidence; baseline hard-selection allowed for ablation.
- Position sizing: volatility targeting using vol_21 as realized vol proxy: pos = target_vol / vol_21, capped at max_leverage.
- Kelly cap: cap fraction at 0.2 of Kelly estimate; apply max position limits.
- Meta-label filter: suppress trades where y_meta probability < threshold (tuned on VAL).

## Backtester & Metrics
- Engine assumptions: vectorized daily backtester.
- Execution: entry at next open, exit next close (configurable).
- Costs: spread + slippage + commission (configurable constants). Market-impact sensitivity tests recommended.
- Metrics: annualized Sharpe, CAGR, max drawdown, Calmar, turnover, hit rate, profit factor.
- Statistical tests: bootstrap daily returns on TEST to compare to baseline (paired bootstrap), report p-values and 95% CI.

## Capacity Check (required)
- Turnover vs ADV: require turnover to stay below a fixed fraction of ADV (e.g., < 5%)
- Report implied capacity and slippage sensitivity under higher notional sizes

## Reproducibility & Artifacts
- Track with MLflow/W&B or manifest files.
- Persist: dataset snapshot_id, HMM params, scalers, model binaries, backtest logs, cv folds file.
- Requirements pinned in `requirements.txt`; include `Dockerfile`.
- Save an experiment manifest: {snapshot_id, fold_id, seed, hyperparams, model_ids}.

## Deliverables (Phase 1)
- Notebook + scripts demonstrating pipeline and top-line OOS results.
- `cv_folds.json`, `manifest.json`, models and backtest artifacts.
- OOS report: equity curve, per-regime P&L, transition heatmap, bootstrap p-values.

## Sanity & Leakage Checks
- For random sample dates in TEST, assert every feature value uses timestamps ≤ T-1.
- Re-run fold with same manifest/seed should reproduce hashes of saved artifacts.

---




##checklist for tomorrow
