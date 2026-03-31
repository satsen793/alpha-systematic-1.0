# Phase 2
## Objective
- Extend Phase 1 to a multi-asset universe with cross-sectional signals and controlled complexity that improves OOS Sharpe and reduces drawdown.

## Baseline Benchmarks (must report)
- Buy & hold `^NSEI`
- Simple momentum rule (ret_5 or ret_21 threshold-based) on each asset and on an equal-weight basket

## Phase 2 Gate (non-negotiable)
- Proceed only if Phase 1 passes:
  - OOS Sharpe > 1.0
  - Stable across folds (no collapse in any test fold)
  - Survives +20% transaction cost stress
- If any condition fails: stop and fix Phase 1.

## Execution Plan (strict order)
Step 1 — Multi-asset without ML changes
- Universe: ^NSEI, ^NSEBANK (SENSEX excluded for now).
- Add cross-sectional rank features: cs_rank_ret_21, cs_rank_vol_21.
- Keep SAME XGBoost, SAME HMM, SAME backtester.
- Goal: Sharpe up, drawdown down. If not, stop and fix signals.

Step 2 — Portfolio construction (expected alpha jump)
- Add risk-parity sizing: weight_i ∝ 1 / vol_i.
- Add drawdown cap: reduce exposure when trailing drawdown > threshold.
- Goal: improve Calmar without harming Sharpe.

Step 3 — Signal validation (multi-asset)
- Evaluate per-asset Sharpe, contribution, and signal correlation.
- Remove assets that dilute Sharpe or increase turnover.

Step 4 — Controlled NN test (only if Steps 1–3 succeed)
- Replace XGBoost with small LSTM (hidden ≤ 64, seq_len = 63).
- Early stop on VAL Sharpe (not loss).
- Decision rule: keep NN only if it consistently beats XGBoost across folds.

Step 5 — RL for sizing only (last step)
- Use RL only for position sizing; do NOT use for signal generation.
- Baseline: risk parity + meta-label filter.
- RL must beat baseline on OOS Sharpe and cost stress tests.

## Testable Hypotheses
H1) Cross-sectional rank features improve OOS Sharpe versus pure time-series features.
H2) Risk-parity + drawdown caps improve Calmar without reducing Sharpe.
H3) Small LSTM improves Sharpe only if it generalizes across folds; otherwise reject.
H4) RL sizing adds value only if it beats rule-based sizing under cost stress.

## Signal Diagnostics (required)
- IC: feature vs forward return correlation (per-asset and cross-sectional)
- IC stability: fold-by-fold IC variance and sign consistency

## Minimal-Change Ablation Experiments (exact design)
E1: Cross-sectional ranking
- Change: add cs_rank_ret_21 and cs_rank_vol_21, keep models fixed.
- Baseline: time-series-only features.
- Measure: OOS Sharpe, turnover, max drawdown across folds.
- Success: Sharpe improves and fold variance decreases.

E2: Risk-parity sizing
- Change: apply volatility-targeted risk-parity weights per asset + drawdown cap.
- Baseline: equal-weight positions.
- Measure: Sharpe, Calmar, drawdown.
- Success: lower drawdown and improved Calmar without reducing Sharpe.

E3: Small NN vs XGBoost
- Change: replace XGBoost per-asset with compact LSTM (hidden <= 64).
- Baseline: XGBoost per-asset.
- Measure: OOS Sharpe and stability across folds.
- Success: consistent Sharpe uplift and reduced fold variance.

E4: RL vs rule-based
- Change: RL allocator controlling position sizing only.
- Baseline: risk parity + meta-label filter.
- Measure: OOS Sharpe and drawdown; run cost stress tests.
- Success: RL beats baseline in multiple folds and under cost perturbations.

## Failure Modes & Detection
- Cross-sectional overfit: strong in-sample but unstable OOS → detect via fold dispersion and time-sliced Sharpe.
- Liquidity mismatch: signals require more turnover than feasible → detect by cost sweeps and capacity analysis.
- Model complexity trap: NN/RL underperform but add variance → detect via ablation and early stopping.
- Regime confusion across assets: non-stationary regime labels → detect via regime stability metrics and state persistence.

## Universe & Data
- Start with a small index universe (^NSEI, ^NSEBANK). SENSEX is excluded for now due to data availability.
- Frequency: daily initially; add intraday later if justified.
- Additional fields: cross-asset correlations, sector exposures, macro overlays (VIX, rates).
- Storage: columnar Parquet per ticker + unified panel index; maintain snapshot IDs per universe.

## Cross-Sectional Feature Design
- Standardize per-asset features (zscore computed on asset history in TRAIN fold) and compute cross-sectional ranks.
- Example features:
  - cs_rank_ret_21 = rank(ret_21) / N
  - cs_rank_vol_21 = rank(vol_21) / N
- Ensure shift-safety across cross-section: for date T, use data ≤ T-1 and train-only normalization.

## NN / LSTM Prototypes (only after Step 3)
- Input shapes:
  - Univariate sequence per asset: length L (e.g., 63) × feature_dim
  - Multivariate cross-asset tensor for portfolio model: assets × L × feature_dim
- Candidate architecture: compact LSTM (hidden ≤ 64, dropout ≤ 0.1) with dense head.
- Training rules:
  - Train on TRAIN sequences; early stop on VAL Sharpe.
  - Standardization: per-asset scaler fit on TRAIN; apply to VAL/TEST.

## RL Agent (sizing only, last step)
- State: recent signals per asset, regime posteriors, and portfolio stats.
- Action: allocation weights or exposure scaling per asset.
- Reward: daily P&L minus drawdown and turnover penalties.
- Only keep RL if it beats rule-based sizing under cost stress.

## Backtesting & Robustness
- Portfolio-level backtester with per-asset costs, constraints, and margin rules.
- Stress tests: cost sweeps (±50%), liquidity shocks, crisis subperiods.

## Capacity Check (required)
- Turnover vs ADV: require turnover to stay below a fixed fraction of ADV (e.g., < 5%) per asset
- Report implied capacity and slippage sensitivity under higher notional sizes

## Experimentation & Artifacts
- Track all experiments with manifests; save dataset snapshots and model artifacts.
- Save fold-by-fold metrics and stress test results.

## Deliverables (Phase 2)
- Multi-asset dataset snapshots and cross-sectional feature pipeline.
- Ablation report: cross-section, sizing, NN vs XGBoost, RL vs rule-based.
- Stress-test report and production checklist.

---