import argparse
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_EXCHANGE_ROOT = Path(r"C:\Users\Asus\Desktop\SHFE_上海期货交易所")
BASE_FEATURE_COLS = [
    "zscore",
    "rsi",
    "atr",
    "bb_width",
    "spread_lag_1",
    "spread_lag_2",
    "spread_lag_3",
    "spread_lag_5",
    "spread_lag_10",
    "spread_ret_1",
    "spread_ret_3",
    "spread_ret_5",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "percentile_20",
]
VOLUME_FEATURE_COLS = [
    "volume_total",
    "volume_ratio_nf",
    "near_volume_share",
    "volume_total_change_1",
]
FEATURE_COLS = BASE_FEATURE_COLS + VOLUME_FEATURE_COLS


def get_cost_parameters(variety, near_price=None, far_price=None):
    """
    Return cost parameters needed for calendar spread backtesting, all expressed in "price ticks".

    1. COMMISSION:
       - Based on SHFE 2025 futures commission rules
       - Fixed-fee contracts (e.g. AU) charged directly in yuan per lot
       - Proportional-fee contracts (e.g. RB / HC / AG) computed as trade_price * contract_multiplier * fee_rate
       - One single-leg operation on a spread = 1 near leg + 1 far leg, so commissions of both legs are summed
       - Then divide by the contract multiplier to convert everything into "price ticks"
    2. SLIPPAGE:
       - Not an official fee, but an optimistic fill assumption in backtesting
       - Default slippage per leg is 0.5 tick, so single-side spread operation slippage = 1 * tick_size
    3. default_price in specs:
       - Used only as a fallback representative price when near_price / far_price are missing
       - Normally the average near/far price over the sample window is preferred for proportional-fee conversion
    """
    specs = {
        "RB": {
            "multiplier": 10.0,
            "tick_size": 1.0,
            "fee_type": "rate",
            "fee_rate": 0.0001,   # 0.1 per mille
            "fee_fixed": None,
            "default_price": 3200.0,  # fallback representative price only
        },
        "HC": {
            "multiplier": 10.0,
            "tick_size": 1.0,
            "fee_type": "rate",
            "fee_rate": 0.0001,   # 0.1 per mille
            "fee_fixed": None,
            "default_price": 3400.0,  # fallback representative price only
        },
        "AG": {
            "multiplier": 15.0,
            "tick_size": 1.0,
            "fee_type": "rate",
            "fee_rate": 0.00005,  # 0.05 per mille
            "fee_fixed": None,
            "default_price": 6000.0,  # fallback representative price only
        },
        "AU": {
            "multiplier": 1000.0,
            "tick_size": 0.02,
            "fee_type": "fixed",
            "fee_rate": None,
            "fee_fixed": 10.0,    # 10 yuan per lot per side
            "default_price": 600.0,  # fallback representative price only
        },
    }
    default_spec = {
        "multiplier": 10.0,
        "tick_size": 1.0,
        "fee_type": "rate",
        "fee_rate": 0.0001,
        "fee_fixed": None,
        "default_price": 3000.0,
    }

    spec = specs.get(variety.upper(), default_spec)
    multiplier = float(spec["multiplier"])
    tick_size = float(spec["tick_size"])

    if near_price is None:
        near_price = spec["default_price"]
    if far_price is None:
        far_price = spec["default_price"]

    near_price = float(near_price)
    far_price = float(far_price)

    if spec["fee_type"] == "fixed":
        fee_yuan_near = float(spec["fee_fixed"])
        fee_yuan_far = float(spec["fee_fixed"])
    else:
        fee_rate = float(spec["fee_rate"])
        fee_yuan_near = near_price * multiplier * fee_rate
        fee_yuan_far = far_price * multiplier * fee_rate

    commission_points = (fee_yuan_near + fee_yuan_far) / multiplier
    slippage_points = 1.0 * tick_size
    cost_per_op = commission_points + slippage_points
    cost_per_trade = 2.0 * cost_per_op
    return commission_points, slippage_points, cost_per_op, cost_per_trade


def normalize_file_input(file_input):
    if isinstance(file_input, (str, Path)):
        return [Path(file_input)]
    if isinstance(file_input, (list, tuple, set)):
        return [Path(path) for path in file_input]
    raise TypeError(f"Unsupported file input type: {type(file_input)}")


def describe_file_input(file_input):
    return " | ".join(str(path) for path in normalize_file_input(file_input))


def log_progress(message, verbose=True):
    if verbose:
        print(message, flush=True)


def detect_datetime_close_columns(filepath, encoding):
    header_df = pd.read_csv(filepath, encoding=encoding, nrows=0)
    date_col_candidates = ["datetime", "date", "Date", "time", "Time", "TradingDay"]
    close_col_candidates = ["close", "Close", "closing", "settlement", "Settlement", "last", "Last"]

    date_col = next((col for col in date_col_candidates if col in header_df.columns), None)
    close_col = next((col for col in close_col_candidates if col in header_df.columns), None)

    if date_col is None or close_col is None:
        if header_df.shape[1] < 10:
            raise ValueError(f"Insufficient columns, cannot identify date and close price: {filepath}")
        date_col = header_df.columns[0]
        close_col = header_df.columns[9] if "close" in str(header_df.columns[9]).lower() else header_df.columns[8]
    return date_col, close_col


def detect_datetime_volume_columns(filepath, encoding):
    header_df = pd.read_csv(filepath, encoding=encoding, nrows=0)
    date_col_candidates = ["datetime", "date", "Date", "time", "Time", "TradingDay"]
    volume_col_candidates = ["volume", "Volume", "vol", "Vol"]

    date_col = next((col for col in date_col_candidates if col in header_df.columns), None)
    volume_col = next((col for col in volume_col_candidates if col in header_df.columns), None)

    if date_col is None or volume_col is None:
        # fall back heuristically: 1st col date, 6th col volume
        if header_df.shape[1] >= 6:
            date_col = header_df.columns[0] if date_col is None else date_col
            volume_col = header_df.columns[5] if volume_col is None else volume_col
    return date_col, volume_col


def read_futures_csv(filepath):
    filepaths = normalize_file_input(filepath)
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1", "cp936"]
    results = []

    for single_path in filepaths:
        df = None
        last_error = None
        for enc in encodings:
            try:
                date_col, close_col = detect_datetime_close_columns(single_path, enc)
                df = pd.read_csv(
                    single_path,
                    encoding=enc,
                    usecols=[date_col, close_col],
                    on_bad_lines="skip",
                )
                break
            except Exception as exc:  # pragma: no cover - diagnostic path
                last_error = exc
        if df is None:
            raise ValueError(f"Read failed: {single_path} | {last_error}")

        result = pd.DataFrame(
            {
                "datetime": pd.to_datetime(df[date_col], errors="coerce"),
                "close": pd.to_numeric(df[close_col], errors="coerce"),
            }
        ).dropna()
        results.append(result)

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
    return combined


def read_futures_csv_with_volume(filepath):
    filepaths = normalize_file_input(filepath)
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1", "cp936"]
    results_close = []
    results_volume = []

    for single_path in filepaths:
        df_close = None
        df_volume = None
        last_error = None
        for enc in encodings:
            try:
                # read header once to get columns
                header_df = pd.read_csv(single_path, encoding=enc, nrows=0)
                date_col, close_col = detect_datetime_close_columns(single_path, enc)
                _, volume_col = detect_datetime_volume_columns(single_path, enc)
                usecols = []
                if date_col in header_df.columns:
                    usecols.append(date_col)
                if close_col in header_df.columns:
                    usecols.append(close_col)
                if volume_col in header_df.columns:
                    usecols.append(volume_col)
                raw = pd.read_csv(single_path, encoding=enc, usecols=usecols, on_bad_lines="skip")
                datetime_series = pd.to_datetime(raw[date_col], errors="coerce")
                df_close = pd.DataFrame({
                    "datetime": datetime_series,
                    "close": pd.to_numeric(raw[close_col], errors="coerce"),
                }).dropna()
                if volume_col in raw.columns:
                    df_volume = pd.DataFrame({
                        "datetime": datetime_series,
                        "volume": pd.to_numeric(raw[volume_col], errors="coerce"),
                    }).dropna()
                else:
                    df_volume = pd.DataFrame(columns=["datetime", "volume"])
                break
            except Exception as exc:
                last_error = exc
        if df_close is None:
            raise ValueError(f"Read failed (incl. volume): {single_path} | {last_error}")
        results_close.append(df_close)
        results_volume.append(df_volume if df_volume is not None else pd.DataFrame(columns=["datetime", "volume"]))

    combined_close = pd.concat(results_close, ignore_index=True).sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    combined_volume = pd.concat(results_volume, ignore_index=True).sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)
    return combined_close, combined_volume


def load_daily_volume(filepath, prefix):
    daily = pd.DataFrame(columns=["date", f"volume_{prefix}"])
    try:
        _, volume_1m = read_futures_csv_with_volume(filepath)
        if volume_1m.empty:
            return daily
        volume_1m["date"] = volume_1m["datetime"].dt.normalize()
        grouped = volume_1m.groupby("date", as_index=False)["volume"].sum()
        daily = grouped.rename(columns={"volume": f"volume_{prefix}"})
    except Exception:
        pass
    return daily


def to_daily(df):
    if df.empty:
        empty = df.copy()
        if "datetime" in empty.columns:
            empty = empty.rename(columns={"datetime": "date"})
        return empty
    daily = df.set_index("datetime").resample("D").last().dropna().reset_index()
    return daily.rename(columns={"datetime": "date"})


def make_stationary(series, method="diff"):
    if method == "diff":
        return series.diff()
    if method == "none":
        return series
    raise ValueError(f"Unsupported stationarization method: {method}")


def check_stationarity(series, alpha=0.05):
    from statsmodels.tsa.stattools import adfuller

    series_clean = series.dropna()
    if len(series_clean) < 10:
        return False, None
    result = adfuller(series_clean, autolag="AIC")
    return result[1] < alpha, result


def calc_trades_and_cost(signal_series, cost_per_op):
    pos = pd.Series(list(signal_series)).reset_index(drop=True).astype(float).fillna(0.0)
    if len(pos) == 0:
        return 0.0, 0.0, 0.0
    delta = pos.diff().fillna(pos.iloc[0])
    ops_cnt = float(delta.abs().sum())
    if pos.iloc[-1] != 0.0:
        ops_cnt += float(abs(pos.iloc[-1]))
    total_cost = ops_cnt * float(cost_per_op)
    return ops_cnt, ops_cnt / 2.0, total_cost


def calc_daily_cost_series(signal_series, cost_per_op):
    pos = pd.Series(list(signal_series)).reset_index(drop=True).astype(float).fillna(0.0)
    if len(pos) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    daily_ops = pos.diff().fillna(pos.iloc[0]).abs()
    if pos.iloc[-1] != 0.0:
        daily_ops.iloc[-1] += float(abs(pos.iloc[-1]))
    return daily_ops, daily_ops * float(cost_per_op)


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_percentile(series):
    if series.max() - series.min() < 1e-10:
        return 0.5
    return (series.iloc[-1] - series.min()) / (series.max() - series.min() + 1e-10)


def parse_contract_info(filepath):
    """
    Supports two naming conventions:
    1. SHFE.au2602.csv -> ('au', '2602')
    2. RB_02月_1m.csv   -> ('RB', '02')
    """
    filename = str(filepath).replace("/", "\\").split("\\")[-1]
    match_new = re.match(r"([A-Za-z]+)_(\d{2})月_.*\.csv$", filename, flags=re.IGNORECASE)
    if match_new:
        return match_new.group(1), match_new.group(2)

    parts = filename.split(".")
    if len(parts) >= 2:
        contract_str = parts[1]
        match_old = re.match(r"([a-zA-Z]+)(\d+)", contract_str)
        if match_old:
            return match_old.group(1), match_old.group(2)

    raise ValueError(f"Cannot parse filename: {filename}")


def parse_month_from_dir(dirname):
    match = re.search(r"^(\d{2})月$", dirname)
    if not match:
        raise ValueError(f"Cannot parse month from directory name: {dirname}")
    return int(match.group(1))


def parse_month_from_name(filename):
    match = re.search(r"_(\d{2})月(?:_[^.]+)?\.csv$", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse month from filename: {filename}")
    return int(match.group(1))


def find_commodity_dir(exchange_root, commodity_code):
    exchange_root = Path(exchange_root)
    commodity_code = commodity_code.upper()
    matches = [path for path in exchange_root.iterdir() if path.is_dir() and path.name.upper().startswith(f"{commodity_code}_")]
    if not matches:
        raise FileNotFoundError(f"Commodity directory not found: {commodity_code} @ {exchange_root}")
    return matches[0]


def scan_month_files(exchange_root, commodity_code):
    commodity_dir = find_commodity_dir(exchange_root, commodity_code)
    month_root = commodity_dir / "月份连续"
    if not month_root.exists():
        raise FileNotFoundError(f"Consecutive-month directory not found: {month_root}")

    month_files = {}
    for month_dir in sorted(month_root.iterdir(), key=lambda p: p.name):
        if not month_dir.is_dir():
            continue
        try:
            month = parse_month_from_dir(month_dir.name)
        except ValueError:
            continue
        csv_files = sorted(month_dir.glob("*.csv"), key=lambda p: p.name)
        if not csv_files:
            continue

        matched_files = []
        prefix = f"{commodity_code.upper()}_"
        for csv_path in csv_files:
            if not csv_path.name.upper().startswith(prefix):
                continue
            try:
                file_month = parse_month_from_name(csv_path.name)
            except ValueError:
                continue
            if file_month == month:
                matched_files.append(csv_path)

        if not matched_files:
            names = ", ".join(path.name for path in csv_files)
            raise ValueError(f"{month_dir} has no standard CSV files matching {commodity_code.upper()} month {month:02d}. Existing files: {names}")

        preferred_1m = [
            path for path in matched_files
            if re.search(rf"^{re.escape(commodity_code.upper())}_{month:02d}月_1m\.csv$", path.name, flags=re.IGNORECASE)
        ]
        if len(preferred_1m) == 1:
            selected_file = preferred_1m[0]
        else:
            selected_file = matched_files[0] if len(matched_files) == 1 else tuple(matched_files)

        if month in month_files:
            raise ValueError(f"Duplicate month file detected: month {month:02d} -> {month_files[month]} | {selected_file}")
        month_files[month] = selected_file
    return commodity_dir, month_files


def forward_month_gap(near_month, far_month):
    gap = (far_month - near_month) % 12
    return 12 if gap == 0 else gap


def circular_month_gap(month_a, month_b):
    if month_a == month_b:
        return 0
    return min(forward_month_gap(month_a, month_b), forward_month_gap(month_b, month_a))


def canonicalize_pair(month_a, month_b):
    if month_a == month_b:
        raise ValueError("Months cannot be identical")
    gap_ab = forward_month_gap(month_a, month_b)
    gap_ba = forward_month_gap(month_b, month_a)
    if gap_ab < gap_ba:
        return month_a, month_b, gap_ab
    if gap_ba < gap_ab:
        return month_b, month_a, gap_ba
    return min(month_a, month_b), max(month_a, month_b), gap_ab


def build_daily_cache(month_files, verbose=False):
    cache = {}
    total = len(month_files)
    for idx, (month, path) in enumerate(month_files.items(), start=1):
        if verbose:
            print(f"[{idx}/{total}] Reading month {month:02d} file and converting to daily frequency...")
        cache[month] = to_daily(read_futures_csv(path))[["date", "close"]].rename(columns={"close": f"close_{month:02d}"})
    return cache


def list_reasonable_pairs(month_files, min_gap=1, max_gap=6, min_overlap_days=200, verbose=False):
    daily_cache = build_daily_cache(month_files, verbose=verbose)
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

            pairs.append(
                {
                    "near_month": near_month,
                    "far_month": far_month,
                    "forward_gap": gap,
                    "overlap_days": overlap_days,
                    "near_path": str(month_files[near_month]),
                    "far_path": str(month_files[far_month]),
                }
            )

    if verbose:
        print(f"Reasonable month-pair screening complete, total {len(pairs)} pairs.")
    return pd.DataFrame(pairs).sort_values(["forward_gap", "near_month", "far_month"]).reset_index(drop=True)


def build_spread(near_path, far_path):
    near_paths = normalize_file_input(near_path)
    far_paths = normalize_file_input(far_path)
    variety, near_month_raw = parse_contract_info(near_paths[0])
    far_variety, far_month_raw = parse_contract_info(far_paths[0])
    if variety.upper() != far_variety.upper():
        raise ValueError(f"Near/far contract varieties do not match: {variety} vs {far_variety}")
    near_month = parse_month_from_name(near_paths[0].name) if "月" in near_paths[0].name else int(str(near_month_raw)[-2:])
    far_month = parse_month_from_name(far_paths[0].name) if "月" in far_paths[0].name else int(str(far_month_raw)[-2:])

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

    return {
        "df": df,
        "variety": variety.upper(),
        "near_month": near_month,
        "far_month": far_month,
        "stationarity_method": stationarity_method,
        "adf_pvalue": None if adf_result is None else float(adf_result[1]),
    }


def engineer_features(df):
    df = df.copy()
    if "spread_stationary" in df.columns and not df["spread_stationary"].equals(df["spread"]):
        spread_col = "spread_stationary"
    else:
        spread_col = "spread"

    s = df[spread_col]
    df["spread_ma"] = s.rolling(20).mean()
    df["spread_std"] = s.rolling(20).std()
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
    df["bb_width"] = 4 * df["spread_std"]
    df["high_low"] = s.diff().abs()
    df["atr"] = df["high_low"].rolling(14).mean()
    df["percentile_20"] = s.rolling(20).apply(calc_percentile, raw=False)
    df["target"] = (s.shift(-1) > s).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean, spread_col


def split_data(df_clean):
    from sklearn.preprocessing import StandardScaler

    X = df_clean[FEATURE_COLS].values
    y = df_clean["target"].values
    total_len = len(X)
    test_len = int(total_len * 0.3)
    dev_len = total_len - test_len

    X_dev_raw = X[:dev_len]
    y_dev_raw = y[:dev_len]
    X_test_raw = X[dev_len:]
    y_test_raw = y[dev_len:]

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev_raw)
    X_test = scaler.transform(X_test_raw)

    return {
        "X_dev_raw": X_dev_raw,
        "y_dev_raw": y_dev_raw,
        "X_test_raw": X_test_raw,
        "y_test_raw": y_test_raw,
        "X_dev": X_dev,
        "X_test": X_test,
        "test_start_idx": dev_len,
    }


def train_xgb_and_predict(X_train, y_train, X_val):
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, verbose=False)
    return model.predict_proba(X_val)[:, 1]


def train_rf_and_predict(X_train, y_train, X_val):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val)[:, 1]


def optimize_stat_boundaries(
    df_clean,
    X_dev_raw,
    cost_per_op,
    n_splits=5,
    boundary_min=0.0,
    boundary_max=1.5,
    boundary_step=0.05,
    verbose=False,
    pair_label="",
):
    from sklearn.model_selection import TimeSeriesSplit

    upper_bounds = np.arange(boundary_min, boundary_max + 1e-9, boundary_step).round(2).tolist()
    lower_bounds = (-np.arange(boundary_min, boundary_max + 1e-9, boundary_step)).round(2).tolist()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_grids = []
    grid_count = len(upper_bounds) * len(lower_bounds)
    prefix = f"[{pair_label}] " if pair_label else ""

    log_progress(
        f"{prefix}Step 1/4: Starting statistical boundary optimization, total {n_splits} folds, "
        f"a/b combinations {len(upper_bounds)} x {len(lower_bounds)} = {grid_count}",
        verbose=verbose,
    )

    for fold_no, (_, val_idx) in enumerate(tscv.split(X_dev_raw), start=1):
        log_progress(
            f"{prefix}  Statistical boundary optimization: fold {fold_no}/{n_splits}",
            verbose=verbose,
        )
        val_df = df_clean.iloc[val_idx].copy()
        results = []
        for a in upper_bounds:
            for b in lower_bounds:
                df_tmp = val_df.copy()
                df_tmp["signal_stat"] = 0
                df_tmp.loc[df_tmp["zscore"] > a, "signal_stat"] = -1
                df_tmp.loc[df_tmp["zscore"] < b, "signal_stat"] = 1
                df_tmp["spread_change"] = df_tmp["spread"].diff()
                df_tmp["return_stat"] = (df_tmp["signal_stat"].shift(1) * df_tmp["spread_change"]).fillna(0.0)
                total_ret = df_tmp["return_stat"].cumsum().iloc[-1]
                _, _, total_cost = calc_trades_and_cost(df_tmp["signal_stat"], cost_per_op=cost_per_op)
                results.append({"upper_z": a, "lower_z": b, "net_return": total_ret - total_cost})
        all_grids.append(pd.DataFrame(results))

    df_stat_grid = (
        pd.concat(all_grids, keys=range(len(all_grids)), names=["fold", "idx"])
        .groupby(["upper_z", "lower_z"], as_index=False)["net_return"]
        .mean()
    )
    best_row = df_stat_grid.loc[df_stat_grid["net_return"].idxmax()]
    log_progress(
        f"{prefix}Statistical boundary optimization complete: best_a={float(best_row['upper_z']):.2f}, "
        f"best_b={float(best_row['lower_z']):.2f}, cv_net={float(best_row['net_return']):.2f}",
        verbose=verbose,
    )
    return float(best_row["upper_z"]), float(best_row["lower_z"]), float(best_row["net_return"]), df_stat_grid


def grid_search_fusion_single(val_df, y_prob_val, upper_z, lower_z, cost_per_op, w_values, threshold_values):
    # upper_z / lower_z retained only for backward-compatible calls; current fusion signal no longer additionally stacks statistical hard boundaries
    results = []
    sqrt2 = math.sqrt(2)
    for w in w_values:
        for threshold in threshold_values:
            df_tmp = val_df.copy()
            df_tmp["stat_prob_long"] = 0.5 * (1 + df_tmp["zscore"].apply(lambda z: math.erf((-z) / sqrt2)))
            df_tmp["stat_prob_short"] = 0.5 * (1 + df_tmp["zscore"].apply(lambda z: math.erf(z / sqrt2)))
            df_tmp["ml_score_long"] = y_prob_val
            df_tmp["ml_score_short"] = 1 - y_prob_val
            df_tmp["combined_long"] = w * df_tmp["stat_prob_long"] + (1 - w) * df_tmp["ml_score_long"]
            df_tmp["combined_short"] = w * df_tmp["stat_prob_short"] + (1 - w) * df_tmp["ml_score_short"]
            df_tmp["signal_ml"] = 0
            df_tmp.loc[df_tmp["combined_long"] > threshold, "signal_ml"] = 1
            df_tmp.loc[df_tmp["combined_short"] > threshold, "signal_ml"] = -1
            df_tmp["spread_change"] = df_tmp["spread"].diff()
            df_tmp["return_ml_gross"] = (df_tmp["signal_ml"].shift(1) * df_tmp["spread_change"]).fillna(0.0)
            _, daily_cost_ml = calc_daily_cost_series(df_tmp["signal_ml"], cost_per_op=cost_per_op)
            df_tmp["return_ml_net"] = df_tmp["return_ml_gross"] - daily_cost_ml.values
            results.append(
                {
                    "w": round(float(w), 2),
                    "threshold": round(float(threshold), 2),
                    "net_return": float(df_tmp["return_ml_net"].cumsum().iloc[-1]),
                    "sharpe": float((df_tmp["return_ml_net"].mean() / df_tmp["return_ml_net"].std()) * np.sqrt(252))
                    if df_tmp["return_ml_net"].std() != 0
                    else np.nan,
                }
            )
    df_grid = pd.DataFrame(results)
    best = df_grid.loc[df_grid["net_return"].idxmax()]
    return float(best["w"]), float(best["threshold"]), df_grid


def optimize_fusion_params(
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

    w_values = np.arange(w_min, w_max + 1e-9, step_w).round(2).tolist()
    threshold_values = np.arange(threshold_min, threshold_max + 1e-9, step_t).round(2).tolist()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model_funcs = {
        "RandomForest": train_rf_and_predict,
        "XGBoost": train_xgb_and_predict,
    }
    outputs = {}
    grid_count = len(w_values) * len(threshold_values)
    prefix = f"[{pair_label}] " if pair_label else ""

    log_progress(
        f"{prefix}Step 2/4: Starting fusion parameter optimization, total {len(model_funcs)} models, "
        f"each model {n_splits} folds, w/threshold combinations {len(w_values)} x {len(threshold_values)} = {grid_count}",
        verbose=verbose,
    )
    log_progress(
        f"{prefix}  Current fusion search range: w in [{w_min:.2f}, {w_max:.2f}], "
        f"threshold in [{threshold_min:.2f}, {threshold_max:.2f}]",
        verbose=verbose,
    )

    for model_name, model_func in model_funcs.items():
        log_progress(f"{prefix}  Model {model_name}: starting CV optimization", verbose=verbose)
        all_grids = []
        for fold_no, (train_idx, val_idx) in enumerate(tscv.split(X_dev_raw), start=1):
            log_progress(
                f"{prefix}    {model_name}: fold {fold_no}/{n_splits}",
                verbose=verbose,
            )
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
        log_progress(
            f"{prefix}  Model {model_name}: CV complete, best_w={float(best_row['w']):.2f}, "
            f"best_threshold={float(best_row['threshold']):.2f}, "
            f"cv_net={float(best_row['net_return']):.2f}",
            verbose=verbose,
        )

    return outputs


def backtest_stat_only(test_df, z_upper, z_lower, cost_per_op):
    df_tmp = test_df.copy()
    df_tmp["signal_stat"] = 0
    df_tmp.loc[df_tmp["zscore"] > z_upper, "signal_stat"] = -1
    df_tmp.loc[df_tmp["zscore"] < z_lower, "signal_stat"] = 1
    df_tmp["spread_change"] = df_tmp["spread"].diff()
    df_tmp["return_stat_gross"] = (df_tmp["signal_stat"].shift(1) * df_tmp["spread_change"]).fillna(0.0)
    ops_stat, bilat_stat, cost_stat = calc_trades_and_cost(df_tmp["signal_stat"], cost_per_op=cost_per_op)
    _, daily_cost_stat = calc_daily_cost_series(df_tmp["signal_stat"], cost_per_op=cost_per_op)
    df_tmp["return_stat_net"] = df_tmp["return_stat_gross"] - daily_cost_stat.values
    sharpe_stat = (
        (df_tmp["return_stat_net"].mean() / df_tmp["return_stat_net"].std()) * np.sqrt(252)
        if df_tmp["return_stat_net"].std() != 0
        else np.nan
    )
    return {
        "gross_return": float(df_tmp["return_stat_gross"].cumsum().iloc[-1]),
        "net_return": float(df_tmp["return_stat_net"].cumsum().iloc[-1]),
        "sharpe": float(sharpe_stat) if not pd.isna(sharpe_stat) else np.nan,
        "ops": float(ops_stat),
        "round_trips": float(bilat_stat),
        "cost": float(cost_stat),
    }


def backtest_ml(test_df, y_prob, z_upper, z_lower, w, fusion_threshold, cost_per_op):
    df_tmp = test_df.copy()
    if len(y_prob) != len(df_tmp):
        raise ValueError("Predicted probability length does not match test sample length")

    # z_upper / z_lower retained only for backward-compatible calls; current fusion signal no longer additionally stacks statistical hard boundaries
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
    ops_ml, bilat_ml, cost_ml = calc_trades_and_cost(df_tmp["signal_ml"], cost_per_op=cost_per_op)
    _, daily_cost_ml = calc_daily_cost_series(df_tmp["signal_ml"], cost_per_op=cost_per_op)
    df_tmp["return_ml_net"] = df_tmp["return_ml_gross"] - daily_cost_ml.values
    sharpe_ml = (
        (df_tmp["return_ml_net"].mean() / df_tmp["return_ml_net"].std()) * np.sqrt(252)
        if df_tmp["return_ml_net"].std() != 0
        else np.nan
    )
    return {
        "gross_return": float(df_tmp["return_ml_gross"].cumsum().iloc[-1]),
        "net_return": float(df_tmp["return_ml_net"].cumsum().iloc[-1]),
        "sharpe": float(sharpe_ml) if not pd.isna(sharpe_ml) else np.nan,
        "ops": float(ops_ml),
        "round_trips": float(bilat_ml),
        "cost": float(cost_ml),
    }


def analyze_pair(
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
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    _, near_month_raw = parse_contract_info(normalize_file_input(near_path)[0])
    _, far_month_raw = parse_contract_info(normalize_file_input(far_path)[0])
    pair_label = f"{int(str(near_month_raw)[-2:]):02d}-{int(str(far_month_raw)[-2:]):02d}"
    log_progress(f"[{pair_label}] Step 0/4: Starting spread construction and feature engineering", verbose=verbose)

    spread_info = build_spread(near_path, far_path)
    df_clean, spread_col = engineer_features(spread_info["df"])
    split_info = split_data(df_clean)

    variety = spread_info["variety"]
    rep_near_price = float(spread_info["df"]["close_near"].mean())
    rep_far_price = float(spread_info["df"]["close_far"].mean())
    commission, slippage, cost_per_op, cost_per_trade = get_cost_parameters(
        variety,
        near_price=rep_near_price,
        far_price=rep_far_price,
    )
    log_progress(
        f"[{pair_label}] Cost basis: near={rep_near_price:.2f}, far={rep_far_price:.2f}, "
        f"commission={commission:.4f}, slippage={slippage:.4f}, cost_per_op={cost_per_op:.4f}",
        verbose=verbose,
    )

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
    fusion_results = optimize_fusion_params(
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

    log_progress(f"[{pair_label}] Step 3/4: Re-training XGBoost and RandomForest on the full dev set", verbose=verbose)
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_model.fit(split_info["X_dev"], split_info["y_dev_raw"], verbose=False)
    y_prob_xgb_test = xgb_model.predict_proba(split_info["X_test"])[:, 1]

    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(split_info["X_dev"], split_info["y_dev_raw"])
    y_prob_rf_test = rf_model.predict_proba(split_info["X_test"])[:, 1]

    log_progress(f"[{pair_label}] Step 4/4: Running statistical / fusion backtests on the test set", verbose=verbose)
    test_df = df_clean.iloc[split_info["test_start_idx"] :].copy()
    stat_result = backtest_stat_only(test_df, best_a, best_b, cost_per_op)
    xgb_result = backtest_ml(
        test_df=test_df,
        y_prob=y_prob_xgb_test,
        z_upper=best_a,
        z_lower=best_b,
        w=fusion_results["XGBoost"]["best_w"],
        fusion_threshold=fusion_results["XGBoost"]["best_threshold"],
        cost_per_op=cost_per_op,
    )
    rf_result = backtest_ml(
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
        "XGBoost": xgb_result["net_return"],
        "RandomForest": rf_result["net_return"],
    }
    best_strategy = max(net_values, key=net_values.get)
    log_progress(
        f"[{pair_label}] Test complete: best={best_strategy}, "
        f"net_stat={stat_result['net_return']:.2f}, "
        f"net_xgb={xgb_result['net_return']:.2f}, "
        f"net_rf={rf_result['net_return']:.2f}",
        verbose=verbose,
    )

    return {
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
        "best_w_xgb": fusion_results["XGBoost"]["best_w"],
        "best_threshold_xgb": fusion_results["XGBoost"]["best_threshold"],
        "best_w_rf": fusion_results["RandomForest"]["best_w"],
        "best_threshold_rf": fusion_results["RandomForest"]["best_threshold"],
        "cv_net_xgb": fusion_results["XGBoost"]["best_net_return_cv"],
        "cv_net_rf": fusion_results["RandomForest"]["best_net_return_cv"],
        "boundary_min": boundary_min,
        "boundary_max": boundary_max,
        "boundary_step": boundary_step,
        "w_min": w_min,
        "w_max": w_max,
        "step_w": step_w,
        "threshold_min": threshold_min,
        "threshold_max": threshold_max,
        "step_t": step_t,
        "gross_stat": stat_result["gross_return"],
        "net_stat": stat_result["net_return"],
        "sharpe_stat": stat_result["sharpe"],
        "trades_stat": stat_result["round_trips"],
        "cost_stat": stat_result["cost"],
        "gross_xgb": xgb_result["gross_return"],
        "net_xgb": xgb_result["net_return"],
        "sharpe_xgb": xgb_result["sharpe"],
        "trades_xgb": xgb_result["round_trips"],
        "cost_xgb": xgb_result["cost"],
        "gross_rf": rf_result["gross_return"],
        "net_rf": rf_result["net_return"],
        "sharpe_rf": rf_result["sharpe"],
        "trades_rf": rf_result["round_trips"],
        "cost_rf": rf_result["cost"],
        "best_strategy": best_strategy,
        "near_path": describe_file_input(near_path),
        "far_path": describe_file_input(far_path),
    }


def print_pairs(pairs_df):
    if pairs_df.empty:
        print("No qualifying month pairs found.")
        return
    show_df = pairs_df.copy()
    show_df["pair"] = show_df["near_month"].map(lambda x: f"{x:02d}") + "-" + show_df["far_month"].map(lambda x: f"{x:02d}")
    print(show_df[["pair", "forward_gap", "overlap_days"]].to_string(index=False))


def summarize_pair_result(result):
    pair = f"{int(result['near_month']):02d}-{int(result['far_month']):02d}"
    return (
        f"Done {pair} | best={result['best_strategy']} | "
        f"net_stat={result['net_stat']:.2f} | "
        f"net_xgb={result['net_xgb']:.2f} | "
        f"net_rf={result['net_rf']:.2f}"
    )


def analyze_all_pairs(
    exchange_root,
    commodity_code,
    output_dir,
    min_gap,
    max_gap,
    min_overlap_days,
    n_splits,
    boundary_min=0.0,
    boundary_max=1.5,
    boundary_step=0.02,
    w_min=0.0,
    w_max=1.0,
    step_w=0.02,
    threshold_min=0.0,
    threshold_max=1.0,
    step_t=0.01,
    verbose=False,
):
    _, month_files = scan_month_files(exchange_root, commodity_code)
    pairs_df = list_reasonable_pairs(
        month_files=month_files,
        min_gap=min_gap,
        max_gap=max_gap,
        min_overlap_days=min_overlap_days,
        verbose=True,
    )
    if pairs_df.empty:
        raise ValueError("No qualifying reasonable month pairs")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = output_dir / f"{commodity_code.upper()}_reasonable_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    print(f"Reasonable month pairs saved to: {pairs_path}", flush=True)

    result_path = output_dir / f"{commodity_code.upper()}_pair_analysis_summary.csv"
    if result_path.exists():
        result_path.unlink()
    print(f"Batch analysis results will be written live to: {result_path}", flush=True)

    results = []
    total_pairs = len(pairs_df)
    for idx, row in enumerate(pairs_df.itertuples(index=False), start=1):
        print(
            f"\n=== [{idx}/{total_pairs}] Analyzing {commodity_code.upper()} {row.near_month:02d}-{row.far_month:02d} "
            f"| gap={row.forward_gap} | overlap={row.overlap_days} ===",
            flush=True,
        )
        result = analyze_pair(
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
        print(summarize_pair_result(result), flush=True)

        current_df = pd.DataFrame(results).sort_values(
            ["best_strategy", "net_rf", "net_xgb", "net_stat"],
            ascending=[True, False, False, False],
        )
        current_df.to_csv(result_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(current_df)}/{total_pairs} results to CSV", flush=True)

    result_df = pd.DataFrame(results).sort_values(["best_strategy", "net_rf", "net_xgb", "net_stat"], ascending=[True, False, False, False])
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    return pairs_path, result_path, result_df


def main():
    parser = argparse.ArgumentParser(description="Batch analysis script for calendar spreads on SHFE consecutive-month data")
    parser.add_argument("--exchange-root", default=str(DEFAULT_EXCHANGE_ROOT), help="SHFE root data directory")
    parser.add_argument("--commodity", required=True, help="Commodity code, e.g. RB / AU / AG")
    parser.add_argument("--mode", choices=["pairs", "pair", "all"], default="pairs", help="pairs=list reasonable month pairs, pair=analyze single pair, all=batch analysis")
    parser.add_argument("--near-month", type=int, help="Near month for single-pair analysis, e.g. 2")
    parser.add_argument("--far-month", type=int, help="Far month for single-pair analysis, e.g. 4")
    parser.add_argument("--min-gap", type=int, default=1, help="Minimum circular month gap for reasonable pairs (12-month circular, take the smaller value; e.g. 05-08 and 11-02 both count as 3)")
    parser.add_argument("--max-gap", type=int, default=6, help="Maximum circular month gap for reasonable pairs (12-month circular, take the smaller value; e.g. 05-08 and 11-02 both count as 3)")
    parser.add_argument("--min-overlap-days", type=int, default=400, help="Minimum overlapping trading days between two consecutive-month series")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of TimeSeriesSplit folds")
    parser.add_argument("--boundary-min", type=float, default=0.00, help="Statistical boundary search minimum value")
    parser.add_argument("--boundary-max", type=float, default=1.50, help="Statistical boundary search maximum value")
    parser.add_argument("--boundary-step", type=float, default=0.05, help="Statistical boundary search step size")
    parser.add_argument("--w-min", type=float, default=0.00, help="Fusion weight minimum value")
    parser.add_argument("--w-max", type=float, default=1.00, help="Fusion weight maximum value")
    parser.add_argument("--step-w", type=float, default=0.04, help="Fusion weight search step size")
    parser.add_argument("--threshold-min", type=float, default=0.00, help="Fusion threshold minimum value")
    parser.add_argument("--threshold-max", type=float, default=1.00, help="Fusion threshold maximum value")
    parser.add_argument("--step-t", type=float, default=0.02, help="Fusion threshold search step size")
    parser.add_argument("--verbose-progress", action="store_true", help="Print more detailed stage progress")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "batch_outputs"), help="Batch output directory")
    args = parser.parse_args()

    commodity_dir, month_files = scan_month_files(args.exchange_root, args.commodity)
    print(f"Commodity directory: {commodity_dir}")
    print(f"Discovered month files: {', '.join(f'{m:02d}' for m in sorted(month_files))}")

    if args.mode == "pairs":
        pairs_df = list_reasonable_pairs(
            month_files=month_files,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            min_overlap_days=args.min_overlap_days,
        )
        print_pairs(pairs_df)
        return

    try:
        if args.mode == "pair":
            if args.near_month is None or args.far_month is None:
                raise ValueError("Single-pair analysis must provide both --near-month and --far-month")
            if args.near_month == args.far_month:
                raise ValueError("Near month and far month cannot be the same for single-pair analysis")
            if args.near_month not in month_files:
                raise ValueError(f"No file found for near month {args.near_month:02d}, available months: {', '.join(f'{m:02d}' for m in sorted(month_files))}")
            if args.far_month not in month_files:
                raise ValueError(f"No file found for far month {args.far_month:02d}, available months: {', '.join(f'{m:02d}' for m in sorted(month_files))}")

            near_month, far_month, _ = canonicalize_pair(args.near_month, args.far_month)
            if (near_month, far_month) != (args.near_month, args.far_month):
                print(f"Canonicalized month pair by forward gap: {args.near_month:02d}-{args.far_month:02d} -> {near_month:02d}-{far_month:02d}")

            pairs_df = list_reasonable_pairs(
                month_files=month_files,
                min_gap=args.min_gap,
                max_gap=args.max_gap,
                min_overlap_days=args.min_overlap_days,
            )
            pair_keys = {(int(row.near_month), int(row.far_month)) for row in pairs_df.itertuples(index=False)}
            if (near_month, far_month) not in pair_keys:
                raise ValueError(
                    f"Month pair {near_month:02d}-{far_month:02d} is not within the current reasonable range. "
                    f"Please check --min-gap/--max-gap/--min-overlap-days parameters."
                )

            near_path = month_files[near_month]
            far_path = month_files[far_month]
            result = analyze_pair(
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
            print(pd.DataFrame([result]).T.to_string(header=False))
            return

        pairs_path, result_path, result_df = analyze_all_pairs(
            exchange_root=args.exchange_root,
            commodity_code=args.commodity,
            output_dir=args.output_dir,
            min_gap=args.min_gap,
            max_gap=args.max_gap,
            min_overlap_days=args.min_overlap_days,
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
        print(f"\nReasonable month pairs saved to: {pairs_path}")
        print(f"Batch analysis summary saved to: {result_path}")
        print("\nTop 5 by net profit:")
        top5 = result_df.assign(best_net=result_df[["net_stat", "net_xgb", "net_rf"]].max(axis=1)).sort_values("best_net", ascending=False).head(5)
        print(top5[["near_month", "far_month", "best_strategy", "best_net"]].to_string(index=False))
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "unknown module")
        raise SystemExit(
            f"Current Python environment is missing dependency: {missing_name}. "
            "Please run this script in the same quantitative environment as your notebook, or install the required dependencies before running analysis mode."
        ) from exc
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc



# ---------------------------------------------------------------------------
# v2-compatible name aliases (so generalization.ipynb can import *v2() names)
# ---------------------------------------------------------------------------
analyze_pair_v2 = analyze_pair
list_reasonable_pairs_v2 = list_reasonable_pairs
print_pairs_v2 = print_pairs


def analyze_all_pairs_v2(
    exchange_root,
    commodity_code,
    output_dir,
    min_gap,
    max_gap,
    min_overlap_days,
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
    min_near_avg_daily_volume=0,  # kwarg accepted for API parity; filtering is done inside list_reasonable_pairs when >= v2; our analyze_all_pairs does not use it.
):
    """Wrapper so the notebook can call analyze_all_pairs_v2 with the same
    3-tuple return contract as rebuild_and_run_batch.analyze_all_pairs_v2.

    Returns
    -------
    (pairs_csv_path, summary_csv_path, summary_DataFrame)
    """
    pairs_path, summary_path, summary_df = analyze_all_pairs(
        exchange_root=exchange_root,
        commodity_code=commodity_code,
        output_dir=output_dir,
        min_gap=min_gap,
        max_gap=max_gap,
        min_overlap_days=min_overlap_days,
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
    return pairs_path, summary_path, summary_df


def canonicalize_pair(near_month, far_month):
    """Canonicalize (near, far) month pair to forward-month gap convention.

    Reimplemented locally to remove the rebuild_and_run_batch → shfe_spread_batch
    dep chain for notebook users. Matches shfe_spread_batch_v2.canonicalize_pair
    1:1.

    Returns
    -------
    (canonical_near, canonical_far, forward_gap)
    """
    gap_fwd = (int(far_month) - int(near_month)) % 12
    gap_rev = (int(near_month) - int(far_month)) % 12
    if gap_fwd <= gap_rev:
        return int(near_month), int(far_month), gap_fwd
    return int(far_month), int(near_month), gap_rev


# Make commonly-needed imports/globals available at module top-level for notebooks
# that do `from generalization import *` during porting.
__all__ = list(vars().keys())


if __name__ == "__main__":
    main()
