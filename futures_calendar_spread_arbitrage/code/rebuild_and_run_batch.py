# rebuild_and_run_batch.py
# ============================================================
# Self-contained batch runner for the unified "结课项目_en.ipynb" pipeline.
# - Reuses helpers from shfe_spread_batch_v2.py (FEATURE_COLS 20, volume loaders)
# - Adds: volume feature merge, Logistic Regression, max_drawdown/win_rate/calmar,
#         min_overlap=400 + near_leg avg daily volume >= 5000 filter
# - Writes batch_outputs_v2/RB_pair_analysis_summary_v2.csv
#
# Usage:
#   1) Sanity check (fast, ~2 min):
#      python rebuild_and_run_batch.py --mode pair --near-month 2 --far-month 4 --verbose-progress
#   2) Full 66-pair run (sleep-and-forget, a few hours depending on CPU):
#      python rebuild_and_run_batch.py --mode all --output-dir .\batch_outputs_v2 --verbose-progress
# ============================================================
import argparse
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 0) Reuse existing helpers (FEATURE_COLS, cost, scan, build_spread basics, etc.)
# ------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Re-import the "common denominator" functions from the batch scaffold
from shfe_spread_batch_v2 import (  # noqa: E402
    BASE_FEATURE_COLS,
    VOLUME_FEATURE_COLS,
    FEATURE_COLS,
    DEFAULT_EXCHANGE_ROOT,
    get_cost_parameters,
    read_futures_csv,
    read_futures_csv_with_volume,
    load_daily_volume,
    to_daily,
    make_stationary,
    check_stationarity,
    calc_trades_and_cost,
    calc_daily_cost_series,
    compute_rsi,
    calc_percentile,
    parse_contract_info,
    parse_month_from_name,
    parse_month_from_dir,
    canonicalize_pair,
    forward_month_gap,
    circular_month_gap,
    scan_month_files,
    build_daily_cache,
    describe_file_input,
    normalize_file_input,
    split_data,  # uses FEATURE_COLS (20 cols after our fix ✅)
    train_xgb_and_predict,
    train_rf_and_predict,
)

# ------------------------------------------------------------
# 1) Override: reasonable pairs filter
#    - min_overlap_days default 400 (交接文档口径)
#    - additionally require: near leg average daily volume >= 5000 contracts
# ------------------------------------------------------------
def list_reasonable_pairs_v2(
    month_files,
    min_gap=1,
    max_gap=6,
    min_overlap_days=400,
    min_near_avg_daily_volume=5000,
    verbose=False,
):
    daily_cache = build_daily_cache(month_files, verbose=verbose)

    # Pre-compute near-month average daily volume (for robustness filter)
    near_vol_cache = {}
    if min_near_avg_daily_volume and min_near_avg_daily_volume > 0:
        for month, path in month_files.items():
            try:
                dv = load_daily_volume(path, "near")
                if not dv.empty and "volume_near" in dv.columns:
                    near_vol_cache[month] = float(dv["volume_near"].mean())
                else:
                    near_vol_cache[month] = np.nan
            except Exception:
                near_vol_cache[month] = np.nan

    seen = set()
    pairs = []
    months = sorted(month_files.keys())
    for i, month_a in enumerate(months):
        for month_b in months[i + 1 :]:
            near_month, far_month, gap = canonicalize_pair(month_a, month_b)
            pair_key = (near_month, far_month)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            if not (min_gap <= gap <= max_gap):
                continue
            overlap_df = pd.merge(
                daily_cache[near_month],
                daily_cache[far_month],
                on="date",
                how="inner",
            )
            overlap_days = int(len(overlap_df))
            if overlap_days < min_overlap_days:
                continue
            near_avg_vol = near_vol_cache.get(near_month, np.nan)
            if min_near_avg_daily_volume and not pd.isna(near_avg_vol):
                if near_avg_vol < float(min_near_avg_daily_volume):
                    if verbose:
                        print(
                            f"  skip {near_month:02d}-{far_month:02d}: "
                            f"near vol mean={near_avg_vol:,.0f} < {min_near_avg_daily_volume}"
                        )
                    continue
            pairs.append(
                {
                    "near_month": near_month,
                    "far_month": far_month,
                    "forward_gap": gap,
                    "overlap_days": overlap_days,
                    "near_avg_daily_volume": near_avg_vol if not pd.isna(near_avg_vol) else None,
                    "near_path": str(month_files[near_month]),
                    "far_path": str(month_files[far_month]),
                }
            )
    if verbose:
        print(f"合理月份对筛选完成(min_overlap_days={min_overlap_days}, min_near_vol={min_near_avg_daily_volume})，共 {len(pairs)} 组。")
    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    return df.sort_values(["forward_gap", "near_month", "far_month"]).reset_index(drop=True)


# ------------------------------------------------------------
# 2) build_spread + volume attach + 20-feature engineering
#    Matches 结课项目_en.ipynb CELL 2 + CELL 3
# ------------------------------------------------------------
def build_spread_v2(near_path, far_path):
    near_paths = normalize_file_input(near_path)
    far_paths = normalize_file_input(far_path)
    variety, near_month_raw = parse_contract_info(near_paths[0])
    far_variety, far_month_raw = parse_contract_info(far_paths[0])
    if variety.upper() != far_variety.upper():
        raise ValueError(f"近远月品种不一致: {variety} vs {far_variety}")

    def _last2_month(name):
        s = str(name)
        try:
            return parse_month_from_name(Path(near_paths[0] if "月" in name else near_paths[0]).name)
        except Exception:
            return int(s[-2:]) if len(s) >= 2 else int(s)

    # Prefer month from filename for the new naming convention
    try:
        near_month = parse_month_from_name(near_paths[0].name)
    except Exception:
        near_month = int(str(near_month_raw)[-2:])
    try:
        far_month = parse_month_from_name(far_paths[0].name)
    except Exception:
        far_month = int(str(far_month_raw)[-2:])

    df_near = to_daily(read_futures_csv(near_path))
    df_far = to_daily(read_futures_csv(far_path))
    df = pd.merge(df_near, df_far, on="date", suffixes=("_near", "_far"))
    df["spread"] = df["close_near"] - df["close_far"]

    is_stat, adf_result = check_stationarity(df["spread"])
    if is_stat:
        df["spread_stationary"] = df["spread"]
        stationarity_method = "none"
    else:
        df["spread_diff"] = make_stationary(df["spread"], method="diff")
        is_stat_diff, result_diff = check_stationarity(df["spread_diff"])
        df["spread_stationary"] = df["spread_diff"]
        stationarity_method = "diff" if is_stat_diff else "diff_fallback"
        adf_result = result_diff

    # --- Volume loading (结课项目_en.ipynb CELL 3 load_daily_volume block) ---
    daily_volume_near = load_daily_volume(near_path, "near")
    daily_volume_far = load_daily_volume(far_path, "far")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if not daily_volume_near.empty:
        daily_volume_near["date"] = pd.to_datetime(daily_volume_near["date"]).dt.normalize()
        df = df.merge(daily_volume_near, on="date", how="left")
    else:
        df["volume_near"] = 0.0
    if not daily_volume_far.empty:
        daily_volume_far["date"] = pd.to_datetime(daily_volume_far["date"]).dt.normalize()
        df = df.merge(daily_volume_far, on="date", how="left")
    else:
        df["volume_far"] = 0.0
    df[["volume_near", "volume_far"]] = df[["volume_near", "volume_far"]].fillna(0.0)

    return {
        "df": df,
        "variety": variety.upper(),
        "near_month": near_month,
        "far_month": far_month,
        "stationarity_method": stationarity_method,
        "adf_pvalue": None if adf_result is None else float(adf_result[1]),
    }


def engineer_features_v2(df_input):
    """20 features: 16 base + 4 volume. Matches 结课项目_en.ipynb CELL 3/4 exactly."""
    df = df_input.copy()
    if "spread_stationary" in df.columns and not df["spread_stationary"].equals(df["spread"]):
        spread_col = "spread_stationary"
    else:
        spread_col = "spread"

    s = df[spread_col]
    window = 20
    df["spread_ma"] = s.rolling(window).mean()
    df["spread_std"] = s.rolling(window).std()
    df["zscore"] = (s - df["spread_ma"]) / df["spread_std"]

    for lag in [1, 2, 3, 5, 10]:
        df[f"spread_lag_{lag}"] = s.shift(lag)

    df["spread_ret_1"] = s.pct_change(1)
    df["spread_ret_3"] = s.pct_change(3)
    df["spread_ret_5"] = s.pct_change(5)
    df["volatility_5"] = s.rolling(5).std()
    df["volatility_10"] = s.rolling(10).std()
    df["volatility_20"] = s.rolling(20).std()
    df["rsi"] = compute_rsi(s, 14)
    # Bollinger width: (ma+2σ) − (ma−2σ) = 4σ  (结课项目_en.ipynb CELL 3)
    df["bb_width"] = (df["spread_ma"] + 2 * df["spread_std"]) - (df["spread_ma"] - 2 * df["spread_std"])
    df["high_low"] = s.diff().abs()
    df["atr"] = df["high_low"].rolling(14).mean()
    df["percentile_20"] = s.rolling(20).apply(calc_percentile, raw=False)

    # Volume derived (结课项目_en.ipynb CELL 3)
    if "volume_near" not in df.columns:
        df["volume_near"] = 0.0
    if "volume_far" not in df.columns:
        df["volume_far"] = 0.0
    df["volume_near"] = df["volume_near"].fillna(0.0)
    df["volume_far"] = df["volume_far"].fillna(0.0)
    df["volume_total"] = df["volume_near"] + df["volume_far"]
    df["volume_ratio_nf"] = df["volume_near"] / (df["volume_far"] + 1.0)
    df["near_volume_share"] = df["volume_near"] / (df["volume_total"] + 1.0)
    df["volume_total_change_1"] = df["volume_total"].pct_change(1)

    # Target: y_t = 1(s_{t+1} > s_t)
    df["target"] = (s.shift(-1) > s).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean, spread_col


# ------------------------------------------------------------
# 3) Logistic regression training helper
# ------------------------------------------------------------
def train_lr_and_predict(X_train, y_train, X_val):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


# ------------------------------------------------------------
# 4) Fusion parameter grid search + CV (adds Logistic Regression)
# ------------------------------------------------------------
def optimize_fusion_params_v2(
    df_clean,
    X_dev_raw,
    y_dev_raw,
    best_a,
    best_b,
    cost_per_op,
    n_splits=5,
    w_min=0.0,
    w_max=1.0,
    step_w=0.04,
    threshold_min=0.0,
    threshold_max=1.0,
    step_t=0.02,
    verbose=False,
    pair_label="",
):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    from shfe_spread_batch_v2 import grid_search_fusion_single

    w_values = np.arange(w_min, w_max + 1e-9, step_w).round(2).tolist()
    threshold_values = np.arange(threshold_min, threshold_max + 1e-9, step_t).round(2).tolist()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model_funcs = {
        "LogisticRegression": train_lr_and_predict,
        "RandomForest": train_rf_and_predict,
        "XGBoost": train_xgb_and_predict,
    }
    outputs = {}
    grid_count = len(w_values) * len(threshold_values)
    prefix = f"[{pair_label}] " if pair_label else ""

    if verbose:
        print(
            f"{prefix}步骤 2/4: 开始融合参数优化，共 {len(model_funcs)} 个模型，"
            f"每个模型 {n_splits} 折，w/threshold 组合数 {len(w_values)} x {len(threshold_values)} = {grid_count}"
        )
        print(f"{prefix}  当前融合搜索区间: w in [{w_min:.2f}, {w_max:.2f}], threshold in [{threshold_min:.2f}, {threshold_max:.2f}]")

    for model_name, model_func in model_funcs.items():
        if verbose:
            print(f"{prefix}  模型 {model_name}: 开始 CV 优化")
        all_grids = []
        for fold_no, (train_idx, val_idx) in enumerate(tscv.split(X_dev_raw), start=1):
            if verbose:
                print(f"{prefix}    {model_name}: 第 {fold_no}/{n_splits} 折")
            X_tr = X_dev_raw[train_idx]
            y_tr = y_dev_raw[train_idx]
            X_val = X_dev_raw[val_idx]
            val_df = df_clean.iloc[val_idx].copy()

            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            y_prob_val = model_func(X_tr_scaled, y_tr, X_val_scaled)

            _, _, df_grid = grid_search_fusion_single(
                val_df=val_df,
                y_prob_val=y_prob_val,
                upper_z=best_a,
                lower_z=best_b,
                cost_per_op=cost_per_op,
                w_values=w_values,
                threshold_values=threshold_values,
            )
            all_grids.append(df_grid)

        df_avg = (
            pd.concat(all_grids, keys=range(len(all_grids)), names=["fold", "idx"])
            .groupby(["w", "threshold"], as_index=False)[["net_return", "sharpe"]]
            .mean()
        )
        best_row = df_avg.loc[df_avg["net_return"].idxmax()]
        outputs[model_name] = {
            "best_w": float(best_row["w"]),
            "best_threshold": float(best_row["threshold"]),
            "best_net_return_cv": float(best_row["net_return"]),
            "grid": df_avg,
        }
        if verbose:
            print(
                f"{prefix}  模型 {model_name}: CV 完成，best_w={float(best_row['w']):.2f}, "
                f"best_threshold={float(best_row['threshold']):.2f}, "
                f"cv_net={float(best_row['net_return']):.2f}"
            )
    return outputs


# ------------------------------------------------------------
# 5) Backtest returns: added max_drawdown, win_rate, avg_trade, calmar
#    We reuse the signal logic but now we accumulate daily net series and derive:
#      - MaxDD (peak-to-trough) on net equity
#      - Win rate by completed round-turns (pair PnL > 0)
# ------------------------------------------------------------
def _derive_extra_metrics(signal_series, spread_change_series, gross_ret_series, net_ret_series, cost_per_op):
    """Return a dict with max_drawdown, win_rate, avg_trade, calmar."""
    from math import isnan

    net_ret = pd.Series(list(net_ret_series)).reset_index(drop=True).astype(float).fillna(0.0)
    spread_chg = pd.Series(list(spread_change_series)).reset_index(drop=True).astype(float).fillna(0.0)
    signal = pd.Series(list(signal_series)).reset_index(drop=True).astype(float).fillna(0.0)
    gross_ret = pd.Series(list(gross_ret_series)).reset_index(drop=True).astype(float).fillna(0.0)

    # --- Max drawdown (points, based on net equity cumulative) ---
    equity = net_ret.cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    net_profit = float(equity.iloc[-1]) if len(equity) else 0.0

    # --- Calmar ratio ---
    if max_dd > 1e-9:
        calmar = net_profit / max_dd
    else:
        calmar = np.nan

    # --- Completed round-trip win rate ---
    # Build completed trade-level PnLs. Rules:
    #   * Open  when position transitions from 0 -> +-1
    #   * Close when position transitions from +-1 -> 0   (back to flat)
    #   * Close+Reopen when position reverses sign +1 <-> -1 directly (no flat intermediate)
    #   * At end of sample, any still-open position is force-closed (final day price)
    completed_rounds = 0
    wins = 0
    opened = None  # (open_next_idx, side_int)
    pos = signal.copy().values.astype(float)
    for i in range(len(pos)):
        cur = float(pos[i])
        if opened is None:
            # No open position: only care about entering a non-zero side
            if cur != 0.0:
                opened = (i + 1, cur)
        else:
            open_idx, side = opened
            # Case A — Back to flat: close the open trade
            if cur == 0.0:
                if open_idx <= i:
                    realized_gross = float(spread_chg.iloc[open_idx : i + 1].sum()) * side
                    cost_round = 2.0 * float(cost_per_op)   # open + close
                    realized_net = realized_gross - cost_round
                    completed_rounds += 1
                    if realized_net > 0:
                        wins += 1
                opened = None
            # Case B — Sign flipped while still non-zero: CLOSE old position + REOPEN new side
            elif (cur > 0.0 and side < 0.0) or (cur < 0.0 and side > 0.0):
                # Close old side: realized over [open_idx, i] (we exit at end of i, price-change up to i included)
                if open_idx <= i:
                    realized_gross = float(spread_chg.iloc[open_idx : i + 1].sum()) * side
                    cost_round = 2.0 * float(cost_per_op)
                    realized_net = realized_gross - cost_round
                    completed_rounds += 1
                    if realized_net > 0:
                        wins += 1
                # Reopen new side (entry end of i -> next-day spread chg i+1 starts PnL)
                opened = (i + 1, cur)
            # else: cur same sign / same magnitude as open -> do nothing, keep position open

    # Still open at end-of-sample → force-close (exit at today's price, no future PnL)
    if opened is not None:
        open_idx, side = opened
        if open_idx < len(pos):
            realized_gross = float(spread_chg.iloc[open_idx:].sum()) * side
            cost_round = 2.0 * float(cost_per_op)
            realized_net = realized_gross - cost_round
            completed_rounds += 1
            if realized_net > 0:
                wins += 1

    win_rate = (float(wins) / float(completed_rounds)) if completed_rounds else np.nan

    # --- Avg trade = net profit / completed round-turns ---
    # We still report calc_trades_and_cost-compatible "round_trips" counts; they use ops_cnt/2 which
    # counts forced-close as a half-op so results can differ slightly. We derive "avg_trade" using the
    # same completed round-trip wins denominator for consistency.
    avg_trade = (net_profit / float(completed_rounds)) if completed_rounds else np.nan

    return {
        "max_drawdown": max_dd,
        "calmar_ratio": calmar if not (isinstance(calmar, float) and isnan(calmar)) else np.nan,
        "win_rate": win_rate if not (isinstance(win_rate, float) and isnan(win_rate)) else np.nan,
        "completed_rounds": completed_rounds,
        "winning_rounds": wins,
        "avg_trade": avg_trade if not (isinstance(avg_trade, float) and isnan(avg_trade)) else np.nan,
    }


def backtest_stat_only_v2(test_df, z_upper, z_lower, cost_per_op):
    from shfe_spread_batch_v2 import backtest_stat_only

    base = backtest_stat_only(test_df, z_upper, z_lower, cost_per_op)

    # Recompute signal + daily net series locally (mirrors original function)
    df_tmp = test_df.copy()
    df_tmp["signal_stat"] = 0
    df_tmp.loc[df_tmp["zscore"] > z_upper, "signal_stat"] = -1
    df_tmp.loc[df_tmp["zscore"] < z_lower, "signal_stat"] = 1
    df_tmp["spread_change"] = df_tmp["spread"].diff()
    df_tmp["return_stat_gross"] = (df_tmp["signal_stat"].shift(1) * df_tmp["spread_change"]).fillna(0.0)
    _, daily_cost_stat = calc_daily_cost_series(df_tmp["signal_stat"], cost_per_op=cost_per_op)
    df_tmp["return_stat_net"] = df_tmp["return_stat_gross"] - daily_cost_stat.values

    extra = _derive_extra_metrics(
        signal_series=df_tmp["signal_stat"].values,
        spread_change_series=df_tmp["spread_change"].fillna(0.0).values,
        gross_ret_series=df_tmp["return_stat_gross"].values,
        net_ret_series=df_tmp["return_stat_net"].values,
        cost_per_op=cost_per_op,
    )
    out = dict(base)
    out.update(extra)
    return out


def backtest_ml_v2(test_df, y_prob, z_upper, z_lower, w, fusion_threshold, cost_per_op):
    from shfe_spread_batch_v2 import backtest_ml

    base = backtest_ml(test_df, y_prob, z_upper, z_lower, w, fusion_threshold, cost_per_op)

    # Recompute signal + daily net series locally
    df_tmp = test_df.copy()
    if len(y_prob) != len(df_tmp):
        raise ValueError("预测概率长度与测试样本长度不一致")

    sqrt2 = math.sqrt(2)
    df_tmp["stat_prob_long"] = 0.5 * (1 + df_tmp["zscore"].apply(lambda z: math.erf((-z) / sqrt2)))
    df_tmp["stat_prob_short"] = 0.5 * (1 + df_tmp["zscore"].apply(lambda z: math.erf(z / sqrt2)))
    df_tmp["ml_score_long"] = y_prob
    df_tmp["ml_score_short"] = 1 - y_prob
    df_tmp["combined_long"] = w * df_tmp["stat_prob_long"] + (1 - w) * df_tmp["ml_score_long"]
    df_tmp["combined_short"] = w * df_tmp["stat_prob_short"] + (1 - w) * df_tmp["ml_score_short"]
    df_tmp["signal_ml"] = 0
    df_tmp.loc[df_tmp["combined_long"] > fusion_threshold, "signal_ml"] = 1
    df_tmp.loc[df_tmp["combined_short"] > fusion_threshold, "signal_ml"] = -1
    df_tmp["spread_change"] = df_tmp["spread"].diff()
    df_tmp["return_ml_gross"] = (df_tmp["signal_ml"].shift(1) * df_tmp["spread_change"]).fillna(0.0)
    _, daily_cost_ml = calc_daily_cost_series(df_tmp["signal_ml"], cost_per_op=cost_per_op)
    df_tmp["return_ml_net"] = df_tmp["return_ml_gross"] - daily_cost_ml.values

    extra = _derive_extra_metrics(
        signal_series=df_tmp["signal_ml"].values,
        spread_change_series=df_tmp["spread_change"].fillna(0.0).values,
        gross_ret_series=df_tmp["return_ml_gross"].values,
        net_ret_series=df_tmp["return_ml_net"].values,
        cost_per_op=cost_per_op,
    )
    out = dict(base)
    out.update(extra)
    return out


# ------------------------------------------------------------
# 6) analyze_pair_v2: runs entire pipeline for one contract pair
# ------------------------------------------------------------
def analyze_pair_v2(
    near_path,
    far_path,
    n_splits=5,
    boundary_min=0.0,
    boundary_max=1.5,
    boundary_step=0.05,
    w_min=0.0,
    w_max=1.0,
    step_w=0.04,
    threshold_min=0.0,
    threshold_max=1.0,
    step_t=0.02,
    verbose=False,
):
    import xgboost as xgb  # noqa: F401  (ensure importable; used indirectly via train_xgb_and_predict)
    from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401

    from shfe_spread_batch_v2 import optimize_stat_boundaries

    _, near_month_raw = parse_contract_info(normalize_file_input(near_path)[0])
    _, far_month_raw = parse_contract_info(normalize_file_input(far_path)[0])

    def _month(raw, path0):
        try:
            return parse_month_from_name(Path(path0).name)
        except Exception:
            return int(str(raw)[-2:])

    near_p, far_p = normalize_file_input(near_path)[0], normalize_file_input(far_path)[0]
    pair_label = f"{_month(near_month_raw, near_p):02d}-{_month(far_month_raw, far_p):02d}"
    if verbose:
        print(f"[{pair_label}] 步骤 0/4: 开始构建价差与特征(含20特征、4个volume派生列)")

    spread_info = build_spread_v2(near_path, far_path)
    df_clean, spread_col = engineer_features_v2(spread_info["df"])
    split_info = split_data(df_clean)

    variety = spread_info["variety"]
    rep_near_price = float(spread_info["df"]["close_near"].mean())
    rep_far_price = float(spread_info["df"]["close_far"].mean())
    commission, slippage, cost_per_op, cost_per_trade = get_cost_parameters(
        variety, near_price=rep_near_price, far_price=rep_far_price
    )
    if verbose:
        print(
            f"[{pair_label}] 成本口径: near={rep_near_price:.2f}, far={rep_far_price:.2f}, "
            f"commission={commission:.4f}, slippage={slippage:.4f}, cost_per_op={cost_per_op:.4f}"
        )
        print(f"[{pair_label}] 样本数 total={len(df_clean)}, dev={split_info['test_start_idx']}, test={len(df_clean) - split_info['test_start_idx']}")

    best_a, best_b, best_stat_net_cv, _ = optimize_stat_boundaries(
        df_clean=df_clean,
        X_dev_raw=split_info["X_dev_raw"],
        cost_per_op=cost_per_op,
        n_splits=n_splits,
        boundary_min=boundary_min,
        boundary_max=boundary_max,
        boundary_step=boundary_step,
        verbose=verbose,
        pair_label=pair_label,
    )
    fusion_results = optimize_fusion_params_v2(
        df_clean=df_clean,
        X_dev_raw=split_info["X_dev_raw"],
        y_dev_raw=split_info["y_dev_raw"],
        best_a=best_a,
        best_b=best_b,
        cost_per_op=cost_per_op,
        n_splits=n_splits,
        w_min=w_min,
        w_max=w_max,
        step_w=step_w,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        step_t=step_t,
        verbose=verbose,
        pair_label=pair_label,
    )

    # --- Step 3/4: Refit models on the FULL development set & generate test probabilities ---
    if verbose:
        print(f"[{pair_label}] 步骤 3/4: 用完整开发区重训 LR/XGBoost/RandomForest 并生成测试概率")
    from sklearn.ensemble import RandomForestClassifier as RFCls
    from sklearn.linear_model import LogisticRegression as LRCls
    import xgboost as xgb_local

    xgb_model = xgb_local.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(split_info["X_dev"], split_info["y_dev_raw"], verbose=False)
    y_prob_xgb_test = xgb_model.predict_proba(split_info["X_test"])[:, 1]

    rf_model = RFCls(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(split_info["X_dev"], split_info["y_dev_raw"])
    y_prob_rf_test = rf_model.predict_proba(split_info["X_test"])[:, 1]

    lr_model = LRCls(max_iter=1000, solver="lbfgs")
    lr_model.fit(split_info["X_dev"], split_info["y_dev_raw"])
    y_prob_lr_test = lr_model.predict_proba(split_info["X_test"])[:, 1]

    # --- Step 4/4: Backtest on held-out test set ---
    if verbose:
        print(f"[{pair_label}] 步骤 4/4: 在测试区执行统计/LR融合/XGBoost融合/RF融合回测(含MaxDD/WinRate/Calmar)")
    test_df = df_clean.iloc[split_info["test_start_idx"] :].copy()
    stat_result = backtest_stat_only_v2(test_df, best_a, best_b, cost_per_op)
    lr_result = backtest_ml_v2(
        test_df=test_df,
        y_prob=y_prob_lr_test,
        z_upper=best_a,
        z_lower=best_b,
        w=fusion_results["LogisticRegression"]["best_w"],
        fusion_threshold=fusion_results["LogisticRegression"]["best_threshold"],
        cost_per_op=cost_per_op,
    )
    xgb_result = backtest_ml_v2(
        test_df=test_df,
        y_prob=y_prob_xgb_test,
        z_upper=best_a,
        z_lower=best_b,
        w=fusion_results["XGBoost"]["best_w"],
        fusion_threshold=fusion_results["XGBoost"]["best_threshold"],
        cost_per_op=cost_per_op,
    )
    rf_result = backtest_ml_v2(
        test_df=test_df,
        y_prob=y_prob_rf_test,
        z_upper=best_a,
        z_lower=best_b,
        w=fusion_results["RandomForest"]["best_w"],
        fusion_threshold=fusion_results["RandomForest"]["best_threshold"],
        cost_per_op=cost_per_op,
    )

    net_values = {
        "Statistical": stat_result["net_return"],
        "LR": lr_result["net_return"],
        "XGBoost": xgb_result["net_return"],
        "RandomForest": rf_result["net_return"],
    }
    best_strategy = max(net_values, key=net_values.get)
    if verbose:
        print(
            f"[{pair_label}] 测试完成: best={best_strategy} | "
            f"stat={stat_result['net_return']:.2f}, LR={lr_result['net_return']:.2f}, "
            f"XGB={xgb_result['net_return']:.2f}, RF={rf_result['net_return']:.2f}"
        )

    out = {
        "variety": variety,
        "near_month": spread_info["near_month"],
        "far_month": spread_info["far_month"],
        "forward_gap": circular_month_gap(spread_info["near_month"], spread_info["far_month"]),
        "stationarity_method": spread_info["stationarity_method"],
        "adf_pvalue": spread_info["adf_pvalue"],
        "spread_col_used": spread_col,
        "sample_count": int(len(df_clean)),
        "dev_count": int(len(split_info["X_dev_raw"])),
        "test_count": int(len(split_info["X_test_raw"])),
        "rep_near_price": rep_near_price,
        "rep_far_price": rep_far_price,
        "commission": commission,
        "slippage": slippage,
        "cost_per_op": cost_per_op,
        "cost_per_trade": cost_per_trade,
        "best_a": best_a,
        "best_b": best_b,
        "best_stat_net_cv": best_stat_net_cv,
    }
    # per-model params
    for mdl, key_short in (
        ("LogisticRegression", "lr"),
        ("XGBoost", "xgb"),
        ("RandomForest", "rf"),
    ):
        out[f"best_w_{key_short}"] = fusion_results[mdl]["best_w"]
        out[f"best_threshold_{key_short}"] = fusion_results[mdl]["best_threshold"]
        out[f"cv_net_{key_short}"] = fusion_results[mdl]["best_net_return_cv"]
    # search ranges
    out.update(
        {
            "boundary_min": boundary_min,
            "boundary_max": boundary_max,
            "boundary_step": boundary_step,
            "w_min": w_min,
            "w_max": w_max,
            "step_w": step_w,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "step_t": step_t,
        }
    )
    # backtest metrics: stat / lr / xgb / rf -> _stat, _lr, _xgb, _rf
    def _flatten(src, suffix):
        out_local = {}
        mapping = {
            "gross_return": f"gross_{suffix}",
            "net_return": f"net_{suffix}",
            "sharpe": f"sharpe_{suffix}",
            "round_trips": f"trades_{suffix}",
            "cost": f"cost_{suffix}",
            "max_drawdown": f"maxdd_{suffix}",
            "calmar_ratio": f"calmar_{suffix}",
            "win_rate": f"winrate_{suffix}",
            "winning_rounds": f"wins_{suffix}",
            "completed_rounds": f"rounds_completed_{suffix}",
            "avg_trade": f"avgtrade_{suffix}",
        }
        for k, v in mapping.items():
            out_local[v] = src.get(k, np.nan)
        return out_local

    out.update(_flatten(stat_result, "stat"))
    out.update(_flatten(lr_result, "lr"))
    out.update(_flatten(xgb_result, "xgb"))
    out.update(_flatten(rf_result, "rf"))
    out["best_strategy"] = best_strategy
    out["near_path"] = describe_file_input(near_path)
    out["far_path"] = describe_file_input(far_path)
    return out


# ------------------------------------------------------------
# 7) Analyze-all-pairs orchestrator
# ------------------------------------------------------------
def analyze_all_pairs_v2(
    exchange_root,
    commodity_code,
    output_dir,
    min_gap,
    max_gap,
    min_overlap_days,
    min_near_avg_daily_volume,
    n_splits,
    boundary_min=0.0,
    boundary_max=1.5,
    boundary_step=0.05,
    w_min=0.0,
    w_max=1.0,
    step_w=0.04,
    threshold_min=0.0,
    threshold_max=1.0,
    step_t=0.02,
    verbose=False,
):
    _, month_files = scan_month_files(exchange_root, commodity_code)
    pairs_df = list_reasonable_pairs_v2(
        month_files=month_files,
        min_gap=min_gap,
        max_gap=max_gap,
        min_overlap_days=min_overlap_days,
        min_near_avg_daily_volume=min_near_avg_daily_volume,
        verbose=True,
    )
    if pairs_df.empty:
        raise ValueError("没有符合条件的合理月份对")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{commodity_code.upper()}_reasonable_pairs_v2.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    print(f"[v2] 合理月份对已保存: {pairs_path} (共 {len(pairs_df)} 组)", flush=True)

    result_path = output_dir / f"{commodity_code.upper()}_pair_analysis_summary_v2.csv"
    if result_path.exists():
        result_path.unlink()
    print(f"[v2] 批量分析结果将实时写入: {result_path}", flush=True)

    results = []
    total_pairs = len(pairs_df)
    for idx, row in enumerate(pairs_df.itertuples(index=False), start=1):
        print(
            f"\n=== [{idx}/{total_pairs}] 分析 {commodity_code.upper()} {row.near_month:02d}-{row.far_month:02d} "
            f"| gap={row.forward_gap} | overlap={row.overlap_days} ===",
            flush=True,
        )
        try:
            result = analyze_pair_v2(
                near_path=row.near_path,
                far_path=row.far_path,
                n_splits=n_splits,
                boundary_min=boundary_min,
                boundary_max=boundary_max,
                boundary_step=boundary_step,
                w_min=w_min,
                w_max=w_max,
                step_w=step_w,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                step_t=step_t,
                verbose=verbose,
            )
            results.append(result)
            # simple summary
            summary = (
                f"完成 {int(result['near_month']):02d}-{int(result['far_month']):02d} | "
                f"best={result['best_strategy']} | "
                f"net_stat={result['net_stat']:.2f}, net_lr={result['net_lr']:.2f}, "
                f"net_xgb={result['net_xgb']:.2f}, net_rf={result['net_rf']:.2f} | "
                f"best_a={result['best_a']:.2f}, best_b={result['best_b']:.2f} | "
                f"XGB (w={result['best_w_xgb']:.2f}, θ={result['best_threshold_xgb']:.2f}), "
                f"RF (w={result['best_w_rf']:.2f}, θ={result['best_threshold_rf']:.2f})"
            )
            print(summary, flush=True)
        except Exception as exc:
            print(f"[SKIP] {row.near_month:02d}-{row.far_month:02d} FAILED: {exc}", flush=True)
            continue

        current_df = pd.DataFrame(results).sort_values(
            ["best_strategy", "net_rf", "net_xgb", "net_stat"],
            ascending=[True, False, False, False],
        )
        current_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(f"已写入 {len(current_df)}/{total_pairs} 组结果到 CSV", flush=True)

    if not results:
        raise RuntimeError("所有组合均回测失败，请检查数据路径或Python依赖")
    result_df = pd.DataFrame(results).sort_values(
        ["best_strategy", "net_rf", "net_xgb", "net_stat"], ascending=[True, False, False, False]
    )
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return pairs_path, result_path, result_df


# ------------------------------------------------------------
# 8) CLI
# ------------------------------------------------------------
def summarize(result_df, commodity):
    def group_best_strategy(s):
        s = str(s).lower()
        if s.startswith("stat"):
            return "Z-score (Stat.)"
        if "lr" in s or "logistic" in s:
            return "Logistic Regression Fusion"
        if "xgb" in s:
            return "XGBoost Fusion"
        if "forest" in s or s.startswith("rf"):
            return "Random Forest Fusion"
        return "Other"

    df = result_df.copy()
    df["grp"] = df["best_strategy"].apply(group_best_strategy)

    print("\n" + "=" * 80)
    print(f"[v2] {commodity.upper()} — 66-pair Generalization Summary (net profit, net of cost)")
    print("=" * 80)
    rows = []
    for key, suffix in [
        ("Z-score (Stat.)", "stat"),
        ("Logistic Regression Fusion", "lr"),
        ("XGBoost Fusion", "xgb"),
        ("Random Forest Fusion", "rf"),
    ]:
        net_col = f"net_{suffix}"
        if net_col not in df.columns:
            continue
        positive = int((df[net_col] > 0).sum())
        total = int(df.shape[0])
        best_cnt = int((df["grp"] == key).sum())
        mean_np = float(df[net_col].mean())
        median_np = float(df[net_col].median())
        ratio = (positive / total * 100.0) if total else 0.0
        rows.append(
            {
                "Strategy": key,
                "Positive Pairs / N": f"{positive} / {total}",
                "Best Count": best_cnt,
                "Mean Net Profit (pts)": round(mean_np, 2),
                "Median Net Profit (pts)": round(median_np, 2),
                "Positive Ratio %": round(ratio, 2),
            }
        )
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    # ML (any of LR/XGB/RF) beats statistical baseline count
    if "net_stat" in df.columns and "net_lr" in df.columns and "net_xgb" in df.columns and "net_rf" in df.columns:
        best_ml_net = df[["net_lr", "net_xgb", "net_rf"]].max(axis=1)
        ml_beats_stat = int((best_ml_net > df["net_stat"]).sum())
        print(f"\nML fusion beats Statistical baseline (any of LR/XGB/RF net > Stat net): {ml_beats_stat} / {len(df)} pairs")
    return summary_df


def print_pairs_v2(pairs_df):
    if pairs_df.empty:
        print("未找到符合条件的月份对。")
        return
    show_df = pairs_df.copy()
    show_df["pair"] = show_df["near_month"].map(lambda x: f"{int(x):02d}") + "-" + show_df["far_month"].map(lambda x: f"{int(x):02d}")
    cols = ["pair", "forward_gap", "overlap_days"]
    if "near_avg_daily_volume" in show_df.columns:
        cols.append("near_avg_daily_volume")
    print(show_df[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="[v2] SHFE 月份连续跨期价差批量分析 (20特征, 含Logistic回归, MaxDD/WinRate/Calmar, min_overlap=400)")
    parser.add_argument("--exchange-root", default=str(DEFAULT_EXCHANGE_ROOT), help="上海期货交易所根目录")
    parser.add_argument("--commodity", default="RB", help="品种代码，默认 RB")
    parser.add_argument("--mode", choices=["pairs", "pair", "all"], default="pairs", help="pairs=列出合理月份对；pair=单组；all=批量")
    parser.add_argument("--near-month", type=int, help="单组近月，如 2")
    parser.add_argument("--far-month", type=int, help="单组远月，如 4")
    parser.add_argument("--min-gap", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=6)
    parser.add_argument("--min-overlap-days", type=int, default=400, help="交接文档口径：≥400 重叠交易日")
    parser.add_argument("--min-near-avg-daily-volume", type=int, default=5000, help="近月合约日均成交量下限（默认 5000 手）")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--boundary-min", type=float, default=0.00)
    parser.add_argument("--boundary-max", type=float, default=1.50)
    parser.add_argument("--boundary-step", type=float, default=0.05)
    parser.add_argument("--w-min", type=float, default=0.00)
    parser.add_argument("--w-max", type=float, default=1.00)
    parser.add_argument("--step-w", type=float, default=0.04)
    parser.add_argument("--threshold-min", type=float, default=0.00)
    parser.add_argument("--threshold-max", type=float, default=1.00)
    parser.add_argument("--step-t", type=float, default=0.02)
    parser.add_argument("--verbose-progress", action="store_true")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "batch_outputs_v2"))
    args = parser.parse_args()

    commodity_dir, month_files = scan_month_files(args.exchange_root, args.commodity)
    print(f"品种目录: {commodity_dir}")
    print(f"已发现月份文件: {', '.join(f'{int(m):02d}' for m in sorted(month_files))}")

    if args.mode == "pairs":
        pairs_df = list_reasonable_pairs_v2(
            month_files=month_files,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            min_overlap_days=args.min_overlap_days,
            min_near_avg_daily_volume=args.min_near_avg_daily_volume,
        )
        print_pairs_v2(pairs_df)
        return

    try:
        if args.mode == "pair":
            if args.near_month is None or args.far_month is None:
                raise ValueError("单组分析必须同时提供 --near-month 和 --far-month")
            if args.near_month == args.far_month:
                raise ValueError("近远月不能相同")
            if args.near_month not in month_files:
                avail = ", ".join(f"{int(m):02d}" for m in sorted(month_files))
                raise ValueError(f"近月 {args.near_month:02d} 没有对应文件，可用月份: {avail}")
            if args.far_month not in month_files:
                avail = ", ".join(f"{int(m):02d}" for m in sorted(month_files))
                raise ValueError(f"远月 {args.far_month:02d} 没有对应文件，可用月份: {avail}")

            near_month, far_month, _ = canonicalize_pair(args.near_month, args.far_month)
            if (near_month, far_month) != (args.near_month, args.far_month):
                print(f"已按前向月差规范化月份对: {args.near_month:02d}-{args.far_month:02d} -> {near_month:02d}-{far_month:02d}")

            pairs_df = list_reasonable_pairs_v2(
                month_files=month_files,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                min_overlap_days=args.min_overlap_days,
                min_near_avg_daily_volume=args.min_near_avg_daily_volume,
            )
            pair_keys = {(int(row.near_month), int(row.far_month)) for row in pairs_df.itertuples(index=False)}
            if (near_month, far_month) not in pair_keys:
                raise ValueError(
                    f"月份对 {near_month:02d}-{far_month:02d} 不在当前合理范围内，"
                    f"请调整 --min-gap/--max-gap/--min-overlap-days/--min-near-avg-daily-volume。"
                )

            near_path = month_files[near_month]
            far_path = month_files[far_month]
            result = analyze_pair_v2(
                near_path=near_path,
                far_path=far_path,
                n_splits=args.n_splits,
                boundary_min=args.boundary_min,
                boundary_max=args.boundary_max,
                boundary_step=args.boundary_step,
                w_min=args.w_min,
                w_max=args.w_max,
                step_w=args.step_w,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                step_t=args.step_t,
                verbose=args.verbose_progress,
            )
            # Print one-row human summary
            show = {
                "near_month": result["near_month"],
                "far_month": result["far_month"],
                "best_strategy": result["best_strategy"],
                "best_a": round(result["best_a"], 2),
                "best_b": round(result["best_b"], 2),
                "w_xgb / θ": f"{result['best_w_xgb']:.2f} / {result['best_threshold_xgb']:.2f}",
                "w_rf / θ": f"{result['best_w_rf']:.2f} / {result['best_threshold_rf']:.2f}",
                "net_stat": round(result["net_stat"], 2),
                "net_xgb": round(result["net_xgb"], 2),
                "net_rf": round(result["net_rf"], 2),
                "SR_stat": round(result["sharpe_stat"], 2),
                "SR_xgb": round(result["sharpe_xgb"], 2),
                "SR_rf": round(result["sharpe_rf"], 2),
                "trades_stat / xgb / rf": f"{result['trades_stat']:.0f} / {result['trades_xgb']:.0f} / {result['trades_rf']:.0f}",
                "MaxDD stat / xgb / rf": f"{result['maxdd_stat']:.2f} / {result['maxdd_xgb']:.2f} / {result['maxdd_rf']:.2f}",
                "WinRate stat / xgb / rf": (
                    f"{result['winrate_stat'] * 100:.1f}% / {result['winrate_xgb'] * 100:.1f}% / {result['winrate_rf'] * 100:.1f}%"
                    if pd.notna(result["winrate_stat"])
                    else "N/A"
                ),
                "Calmar stat / xgb / rf": (
                    f"{result['calmar_stat']:.2f} / {result['calmar_xgb']:.2f} / {result['calmar_rf']:.2f}"
                    if pd.notna(result["calmar_stat"])
                    else "N/A"
                ),
            }
            print(pd.DataFrame([show]).T.to_string(header=False))
            return

        # --- all mode ---
        pairs_path, result_path, result_df = analyze_all_pairs_v2(
            exchange_root=args.exchange_root,
            commodity_code=args.commodity,
            output_dir=args.output_dir,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            min_overlap_days=args.min_overlap_days,
            min_near_avg_daily_volume=args.min_near_avg_daily_volume,
            n_splits=args.n_splits,
            boundary_min=args.boundary_min,
            boundary_max=args.boundary_max,
            boundary_step=args.boundary_step,
            w_min=args.w_min,
            w_max=args.w_max,
            step_w=args.step_w,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            step_t=args.step_t,
            verbose=args.verbose_progress,
        )
        print(f"\n[v2] 合理月份对已保存: {pairs_path}")
        print(f"[v2] 批量分析汇总已保存: {result_path}")
        summarize(result_df, args.commodity)
        print("\n净利润前 5 名（按 best_net = max(stat,lr,xgb,rf)）：")
        top5 = (
            result_df.assign(
                best_net=result_df[["net_stat", "net_lr", "net_xgb", "net_rf"]].max(axis=1)
            )
            .sort_values("best_net", ascending=False)
            .head(5)
        )
        print(top5[["near_month", "far_month", "best_strategy", "best_net", "best_a", "best_b", "best_w_xgb", "best_threshold_xgb"]].to_string(index=False))
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "未知模块")
        raise SystemExit(
            f"当前 Python 环境缺少依赖: {missing}。\n"
            "请在与 Notebook 相同的量化环境中运行（通常需: numpy/pandas/scikit-learn/xgboost/statsmodels/tqdm）。"
        ) from exc
    except (ValueError, FileNotFoundError, KeyError, TypeError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
