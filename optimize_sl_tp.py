"""
Grid-search optimizer for SL/TP ATR multipliers.

Replicates the Pine Script strategy:
  BB Dual Mode + 30m RSI/MACD Regime Weight (Directional)

Sweeps slMult, tpMult, and revTpBasis over ~6 months of SOL/USDT 15m data
from Binance, then reports top parameter combinations by profit factor,
Sharpe ratio, and total return.
"""

import os, sys, time, itertools

# Fix Windows console encoding for box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import ccxt
import talib

# ── ta library imports ──────────────────────────────────────────────────────
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import MACD

# ── paths ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CACHE_15M = os.path.join(CACHE_DIR, "sol_usdt_15m.csv")
CACHE_30M = os.path.join(CACHE_DIR, "sol_usdt_30m.csv")

# ── default strategy parameters (matching Pine Script) ──────────────────────
BB_LEN = 20
BB_MULT = 2.0
RSI_LEN = 14
RSI_BULL = 50
RSI_BEAR = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_LEN = 14
ATR_FILTER_LEN = 20
MIN_SCORE = 0.5
COMMISSION_PCT = 0.1  # 0.1% per trade (each side)

# Directional weights
WEIGHTS = {
    "rev_long":  {"bull": 1.0, "side": 1.0, "bear": 0.2},
    "rev_short": {"bull": 0.2, "side": 1.0, "bear": 1.0},
    "brk_long":  {"bull": 1.0, "side": 0.3, "bear": 0.0},
    "brk_short": {"bull": 0.0, "side": 0.3, "bear": 1.0},
}

# ── grid search ranges ──────────────────────────────────────────────────────
SL_RANGE = np.arange(0.5, 3.01, 0.25)
TP_RANGE = np.arange(0.5, 4.01, 0.25)
REV_TP_BASIS = [True, False]
USE_PATTERNS = [True, False]


# ═════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol: str, timeframe: str, cache_path: str,
                days: int = 180) -> pd.DataFrame:
    """Fetch OHLCV from Binance via ccxt, with CSV caching."""
    if os.path.exists(cache_path):
        age_hrs = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hrs < 12:
            print(f"  Using cached {cache_path} ({age_hrs:.1f}h old)")
            df = pd.read_csv(cache_path, parse_dates=["timestamp"])
            return df

    print(f"  Fetching {symbol} {timeframe} (~{days}d) ...")
    # Try multiple exchanges — Binance is geo-restricted in the US
    exchanges_to_try = [
        ("binanceus", ccxt.binanceus, {}),
        ("bybit",     ccxt.bybit,     {}),
        ("okx",       ccxt.okx,       {}),
    ]
    exchange = None
    for name, cls, opts in exchanges_to_try:
        try:
            ex = cls({"enableRateLimit": True, **opts})
            # Quick test fetch
            ex.fetch_ohlcv(symbol, timeframe, limit=1)
            exchange = ex
            print(f"  Using exchange: {name}")
            break
        except Exception as e:
            print(f"  {name} unavailable: {e!r:.80s}")
            continue
    if exchange is None:
        raise RuntimeError("No exchange available for " + symbol)

    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days * 86_400_000
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + tf_ms
        if len(candles) < 1000:
            break

    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} bars -> {cache_path}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_30m_regime(df30: pd.DataFrame) -> pd.DataFrame:
    """Compute regime (+1 bull, -1 bear, 0 sideways) on 30m data."""
    rsi = RSIIndicator(df30["close"], window=RSI_LEN).rsi()
    macd_obj = MACD(df30["close"], window_slow=MACD_SLOW,
                    window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    macd_line = macd_obj.macd()
    signal_line = macd_obj.macd_signal()

    bull = (rsi > RSI_BULL) & (macd_line > signal_line)
    bear = (rsi < RSI_BEAR) & (macd_line < signal_line)
    regime = np.where(bull, 1, np.where(bear, -1, 0))

    df30 = df30.copy()
    df30["regime"] = regime
    return df30


def map_regime_to_15m(df15: pd.DataFrame, df30: pd.DataFrame) -> pd.Series:
    """
    Map 30m regime onto 15m bars.
    Pine uses request.security with lookahead_off, so each 15m bar sees the
    regime from the most recently *closed* 30m bar.
    """
    # The 30m bar at time T covers [T, T+30m). It is "closed" at T+30m.
    # So a 15m bar at time T should see the regime from the 30m bar that
    # closed at or before T, i.e. the 30m bar whose timestamp <= T - 30m.
    df30_sorted = df30[["timestamp", "regime"]].copy()
    # Shift regime forward by one bar to represent "closed" bar value
    df30_sorted["regime"] = df30_sorted["regime"].shift(1)
    df30_sorted.dropna(inplace=True)

    # Use merge_asof: for each 15m timestamp, find the last 30m bar <= that time
    df15 = df15.copy()
    regime_mapped = pd.merge_asof(
        df15[["timestamp"]],
        df30_sorted,
        on="timestamp",
        direction="backward"
    )["regime"].fillna(0).astype(int)

    return regime_mapped


def compute_15m_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add BB, ATR, and signals to the 15m dataframe."""
    df = df.copy()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=BB_LEN, window_dev=BB_MULT)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_basis"] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_LEN)
    df["atr"] = atr.average_true_range()
    df["atr_sma"] = df["atr"].rolling(ATR_FILTER_LEN).mean()
    df["atr_ok"] = df["atr"] > df["atr_sma"]

    # Signals (vectorized)
    prev_close = df["close"].shift(1)
    prev_lower = df["bb_lower"].shift(1)
    prev_upper = df["bb_upper"].shift(1)

    # Mean Reversion
    df["rev_long"] = (prev_close < prev_lower) & (df["close"] > df["bb_lower"])
    df["rev_short"] = (prev_close > prev_upper) & (df["close"] < df["bb_upper"])

    # Breakout (crossover/crossunder)
    df["brk_long"] = (prev_close <= prev_upper) & (df["close"] > df["bb_upper"])
    df["brk_short"] = (prev_close >= prev_lower) & (df["close"] < df["bb_lower"])

    return df


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern boolean columns using TA-Lib."""
    df = df.copy()
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # TA-Lib pattern functions return 100/-100/0
    engulfing = talib.CDLENGULFING(o, h, l, c)
    df["bullish_engulfing"] = engulfing > 0
    df["bearish_engulfing"] = engulfing < 0

    df["hammer"] = talib.CDLHAMMER(o, h, l, c) > 0
    df["shooting_star"] = talib.CDLSHOOTINGSTAR(o, h, l, c) > 0

    # Strong candle: body > 0.7*ATR, max wick < 15% of candle range
    body = (c - o)
    candle_range = h - l
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    max_wick = np.maximum(upper_wick, lower_wick)

    # Avoid division by zero
    safe_range = np.where(candle_range > 0, candle_range, 1.0)
    wick_pct = max_wick / safe_range

    atr = df["atr"].values
    df["strong_bull_candle"] = (body > 0.7 * atr) & (wick_pct < 0.15) & (body > 0)
    df["moderate_bull_candle"] = (body > 0.4 * atr) & (body > 0)
    df["strong_bear_candle"] = ((-body) > 0.7 * atr) & (wick_pct < 0.15) & (body < 0)

    return df


def score_for_regime(regime: int, w: dict) -> float:
    if regime == 1:
        return w["bull"]
    elif regime == -1:
        return w["bear"]
    return w["side"]


def qualify_signals(df: pd.DataFrame, use_patterns: bool = False) -> pd.DataFrame:
    """Apply regime weights, minScore filter, and optional pattern confirmation."""
    df = df.copy()

    # Vectorized scoring
    regime = df["regime"].values
    for sig_name in ["rev_long", "rev_short", "brk_long", "brk_short"]:
        w = WEIGHTS[sig_name]
        scores = np.where(regime == 1, w["bull"],
                          np.where(regime == -1, w["bear"], w["side"]))
        df[f"{sig_name}_score"] = scores
        # Qualified = signal fired AND score >= min AND ATR filter ok
        df[f"{sig_name}_q"] = (df[sig_name] & (scores >= MIN_SCORE) & df["atr_ok"])

    # Apply candlestick pattern confirmation
    if use_patterns:
        rev_long_confirm  = df["bullish_engulfing"] | df["hammer"]
        rev_short_confirm = df["bearish_engulfing"] | df["shooting_star"]
        brk_long_confirm  = df["moderate_bull_candle"]
        brk_short_confirm = df["strong_bear_candle"]

        df["rev_long_q"]  = df["rev_long_q"]  & rev_long_confirm
        df["rev_short_q"] = df["rev_short_q"] & rev_short_confirm
        df["brk_long_q"]  = df["brk_long_q"]  & brk_long_confirm
        df["brk_short_q"] = df["brk_short_q"] & brk_short_confirm

    return df


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, sl_mult: float, tp_mult: float,
                 rev_tp_basis: bool, use_patterns: bool = False) -> dict:
    """
    Simulate the Pine strategy bar-by-bar.

    Position tracking: 0 = flat, 1 = long, -1 = short.
    Uses close prices for entries (matching Pine strategy default).
    SL/TP checked against high/low of subsequent bars.
    """
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values
    bb_basis = df["bb_basis"].values

    rev_long_q = df["rev_long_q"].values
    rev_short_q = df["rev_short_q"].values
    brk_long_q = df["brk_long_q"].values
    brk_short_q = df["brk_short_q"].values

    n = len(df)
    position = 0       # 0=flat, 1=long, -1=short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    is_rev = False      # whether current trade is mean-reversion

    equity = 10000.0
    peak_equity = 10000.0
    max_drawdown = 0.0
    trades = []         # list of pnl percentages per trade

    for i in range(1, n):
        # ── Check SL/TP on current bar if in a position ─────────────────
        if position != 0:
            hit_sl = False
            hit_tp = False

            if position == 1:  # long
                if lows[i] <= stop_loss:
                    hit_sl = True
                if highs[i] >= take_profit:
                    hit_tp = True
            else:  # short
                if highs[i] >= stop_loss:
                    hit_sl = True
                if lows[i] <= take_profit:
                    hit_tp = True

            if hit_sl and hit_tp:
                # Both hit in same bar — assume SL hit first (conservative)
                hit_tp = False

            if hit_sl:
                exit_price = stop_loss
                pnl_pct = ((exit_price - entry_price) / entry_price * position) * 100
                pnl_pct -= COMMISSION_PCT  # exit commission
                trades.append(pnl_pct)
                equity *= (1 + pnl_pct / 100)
                position = 0
            elif hit_tp:
                exit_price = take_profit
                pnl_pct = ((exit_price - entry_price) / entry_price * position) * 100
                pnl_pct -= COMMISSION_PCT
                trades.append(pnl_pct)
                equity *= (1 + pnl_pct / 100)
                position = 0

        # ── Check for new signals ───────────────────────────────────────
        # Priority: rev_long, brk_long, rev_short, brk_short
        # Opposite signals close existing position first
        new_pos = 0
        new_is_rev = False

        if rev_long_q[i]:
            new_pos = 1
            new_is_rev = True
        elif brk_long_q[i]:
            new_pos = 1
            new_is_rev = False
        elif rev_short_q[i]:
            new_pos = -1
            new_is_rev = True
        elif brk_short_q[i]:
            new_pos = -1
            new_is_rev = False

        if new_pos != 0:
            # Pine with pyramiding=0: same-direction, same signal type
            # is ignored. But different signal type (rev vs brk) replaces
            # the position. Opposite direction always closes + enters.
            skip = False
            if position == new_pos and is_rev == new_is_rev:
                skip = True  # exact same type already open

            if not skip:
                # Close existing position if any
                if position != 0:
                    exit_price = closes[i]
                    pnl_pct = ((exit_price - entry_price) / entry_price * position) * 100
                    pnl_pct -= COMMISSION_PCT
                    trades.append(pnl_pct)
                    equity *= (1 + pnl_pct / 100)
                    position = 0

                # Enter new position
                entry_price = closes[i]
                position = new_pos
                is_rev = new_is_rev
                atr_now = atrs[i]

                if position == 1:  # long
                    stop_loss = entry_price - atr_now * sl_mult
                    if is_rev and rev_tp_basis:
                        take_profit = bb_basis[i]
                        if take_profit <= entry_price:
                            take_profit = entry_price + atr_now * tp_mult
                    else:
                        take_profit = entry_price + atr_now * tp_mult
                else:  # short
                    stop_loss = entry_price + atr_now * sl_mult
                    if is_rev and rev_tp_basis:
                        take_profit = bb_basis[i]
                        if take_profit >= entry_price:
                            take_profit = entry_price - atr_now * tp_mult
                    else:
                        take_profit = entry_price - atr_now * tp_mult

                equity *= (1 - COMMISSION_PCT / 100)  # entry commission

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_drawdown:
            max_drawdown = dd

    # Close any remaining position at last bar
    if position != 0:
        exit_price = closes[-1]
        pnl_pct = ((exit_price - entry_price) / entry_price * position) * 100
        pnl_pct -= COMMISSION_PCT
        trades.append(pnl_pct)
        equity *= (1 + pnl_pct / 100)

    # ── Compute metrics ─────────────────────────────────────────────────
    trades_arr = np.array(trades) if trades else np.array([0.0])
    n_trades = len(trades)
    wins = trades_arr[trades_arr > 0]
    losses = trades_arr[trades_arr <= 0]

    total_return = (equity / 10000 - 1) * 100
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

    # Sharpe ratio (annualized, assuming 15m bars ≈ 35040 bars/year)
    if n_trades > 1 and trades_arr.std() > 0:
        sharpe = (trades_arr.mean() / trades_arr.std()) * np.sqrt(n_trades)
    else:
        sharpe = 0.0

    return {
        "sl_mult": sl_mult,
        "tp_mult": tp_mult,
        "rev_tp_basis": rev_tp_basis,
        "use_patterns": use_patterns,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 1),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe": round(sharpe, 3),
        "n_trades": n_trades,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  SL/TP Grid Search Optimizer")
    print("  Strategy: BB Dual Mode + 30m RSI/MACD Regime")
    print("  Asset: SOL/USDT  |  Timeframe: 15m  |  ~6 months")
    print("=" * 70)

    # 1) Fetch data
    print("\n[1/4] Fetching data ...")
    df15 = fetch_ohlcv("SOL/USDT", "15m", CACHE_15M, days=180)
    df30 = fetch_ohlcv("SOL/USDT", "30m", CACHE_30M, days=180)
    print(f"  15m bars: {len(df15):,}  |  30m bars: {len(df30):,}")

    # 2) Compute indicators
    print("\n[2/5] Computing indicators ...")
    df30 = compute_30m_regime(df30)
    df15 = compute_15m_indicators(df15)
    df15 = detect_candlestick_patterns(df15)
    df15["regime"] = map_regime_to_15m(df15, df30)

    # Drop warmup period (first 30 bars)
    df15 = df15.iloc[30:].reset_index(drop=True)
    print(f"  Working bars: {len(df15):,}")

    # Build qualified dataframes for both modes
    df15_no_pat = qualify_signals(df15, use_patterns=False)
    df15_pat    = qualify_signals(df15, use_patterns=True)

    # Count signals for both modes
    print("\n  Signal counts (no patterns / with patterns):")
    for sig in ["rev_long_q", "rev_short_q", "brk_long_q", "brk_short_q"]:
        n_no = df15_no_pat[sig].sum()
        n_yes = df15_pat[sig].sum()
        print(f"    {sig}: {n_no:>4} / {n_yes:>4}")

    # 3) Grid search
    combos = list(itertools.product(SL_RANGE, TP_RANGE, REV_TP_BASIS, USE_PATTERNS))
    print(f"\n[3/5] Running grid search ({len(combos)} combinations) ...")
    t0 = time.time()

    results = []
    for sl, tp, rev_basis, use_pat in combos:
        df_run = df15_pat if use_pat else df15_no_pat
        res = run_backtest(df_run, sl, tp, rev_basis, use_pat)
        results.append(res)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 4) Pattern impact comparison
    print("\n[4/5] Pattern Impact Comparison")
    rdf = pd.DataFrame(results)

    rdf_no_pat = rdf[~rdf["use_patterns"]].copy()
    rdf_pat    = rdf[rdf["use_patterns"]].copy()

    def summarize_group(group, label):
        valid = group[group["n_trades"] >= 5]
        if len(valid) == 0:
            print(f"  {label}: No combos with >= 5 trades")
            return
        print(f"  {label}:")
        print(f"    Combos with >= 5 trades: {len(valid)}")
        print(f"    Avg trades/combo:  {valid['n_trades'].mean():.0f}")
        print(f"    Avg total return:  {valid['total_return'].mean():.2f}%")
        print(f"    Avg win rate:      {valid['win_rate'].mean():.1f}%")
        print(f"    Avg profit factor: {valid['profit_factor'].mean():.3f}")
        pos = valid[valid["total_return"] > 0]
        print(f"    Positive-return combos: {len(pos)} / {len(valid)}")

    summarize_group(rdf_no_pat, "WITHOUT patterns")
    summarize_group(rdf_pat,    "WITH patterns")

    # 5) Results
    print("\n[5/5] Results")

    # Use lower min-trades threshold for pattern mode (fewer trades expected)
    rdf_valid = rdf[rdf["n_trades"] >= 5].copy()
    if len(rdf_valid) == 0:
        print("  WARNING: No combos with >= 5 trades. Showing all results.")
        rdf_valid = rdf.copy()

    # ── Top 20 by Profit Factor ─────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  TOP 20 by PROFIT FACTOR (min 5 trades)")
    print("─" * 80)
    top_pf = rdf_valid.sort_values("profit_factor", ascending=False).head(20)
    print(top_pf.to_string(index=False))

    # ── Top 10 by Sharpe Ratio ──────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  TOP 10 by SHARPE RATIO")
    print("─" * 80)
    top_sharpe = rdf_valid.sort_values("sharpe", ascending=False).head(10)
    print(top_sharpe.to_string(index=False))

    # ── Top 10 by Total Return ──────────────────────────────────────────
    print("\n" + "─" * 80)
    print("  TOP 10 by TOTAL RETURN")
    print("─" * 80)
    top_ret = rdf_valid.sort_values("total_return", ascending=False).head(10)
    print(top_ret.to_string(index=False))

    # ── Best balanced pick ──────────────────────────────────────────────
    # Rank by combined score: normalize PF + Sharpe + Return, penalize drawdown
    rdf_valid = rdf_valid.copy()
    for col in ["profit_factor", "sharpe", "total_return"]:
        mn, mx = rdf_valid[col].min(), rdf_valid[col].max()
        rng = mx - mn if mx != mn else 1
        rdf_valid[f"{col}_norm"] = (rdf_valid[col] - mn) / rng
    dd_mn, dd_mx = rdf_valid["max_drawdown"].min(), rdf_valid["max_drawdown"].max()
    dd_rng = dd_mx - dd_mn if dd_mx != dd_mn else 1
    rdf_valid["dd_norm"] = 1 - (rdf_valid["max_drawdown"] - dd_mn) / dd_rng

    rdf_valid["composite"] = (
        rdf_valid["profit_factor_norm"] * 0.30 +
        rdf_valid["sharpe_norm"] * 0.30 +
        rdf_valid["total_return_norm"] * 0.20 +
        rdf_valid["dd_norm"] * 0.20
    )

    best = rdf_valid.sort_values("composite", ascending=False).iloc[0]
    print("\n" + "=" * 80)
    print("  RECOMMENDED PARAMETERS (best composite score)")
    print("=" * 80)
    print(f"  slMult          = {best['sl_mult']:.2f}")
    print(f"  tpMult          = {best['tp_mult']:.2f}")
    print(f"  revTpBasis      = {'true' if best['rev_tp_basis'] else 'false'}")
    print(f"  usePatternConf  = {'true' if best['use_patterns'] else 'false'}")
    print(f"  ────────────────────────────────")
    print(f"  Total Return = {best['total_return']:.2f}%")
    print(f"  Win Rate     = {best['win_rate']:.1f}%")
    print(f"  Max Drawdown = {best['max_drawdown']:.2f}%")
    print(f"  Profit Factor= {best['profit_factor']:.3f}")
    print(f"  Sharpe Ratio = {best['sharpe']:.3f}")
    print(f"  Trades       = {int(best['n_trades'])}")
    print("=" * 80)

    # ── Parameter stability check ───────────────────────────────────────
    print("\n  Parameter stability (top 10 composite — look for clustering):")
    stability = rdf_valid.sort_values("composite", ascending=False).head(10)[
        ["sl_mult", "tp_mult", "rev_tp_basis", "use_patterns",
         "profit_factor", "sharpe", "total_return", "n_trades"]
    ]
    print(stability.to_string(index=False))

    # ── Suggested Pine Script update ────────────────────────────────────
    print(f"\n  To update Pine Script defaults:")
    print(f'    slMult          = input.float({best["sl_mult"]:.1f}, ...)')
    print(f'    tpMult          = input.float({best["tp_mult"]:.1f}, ...)')
    rev_str = "true" if best["rev_tp_basis"] else "false"
    pat_str = "true" if best["use_patterns"] else "false"
    print(f'    revTpBasis      = input.bool({rev_str}, ...)')
    print(f'    usePatternConf  = input.bool({pat_str}, ...)')


if __name__ == "__main__":
    main()
