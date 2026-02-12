#!/usr/bin/env python3
"""
Fractal Adaptive Trend System (FATS) — Multi-Timeframe Backtester
Backtests FATS on AVAX/USDT across 15m, 30m, 1H, 4H over 12 months.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — matches Pine Script defaults
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "AVAX/USDT"
TIMEFRAMES = ["15m", "30m", "1h", "4h"]
HTF_MAP = {"15m": "4h", "30m": "4h", "1h": "4h", "4h": "1d"}
LOOKBACK_DAYS = 365
WARMUP_DAYS = 90
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.075
SLIPPAGE_PCT = 0.05

# Fractal Efficiency — optimized for 4H AVAX
ER_LEN = 10
ER_SMOOTH = 3
ER_TREND_THR = 0.38
ER_CHOP_THR = 0.18

# KAMA
KAMA_LEN = 10
KAMA_FAST = 2
KAMA_SLOW = 30

# Squeeze
SQZ_BB_LEN = 20
SQZ_BB_MULT = 2.0
SQZ_KC_LEN = 20
SQZ_KC_MULT = 1.5

# Momentum
MOM_LEN = 12
MOM_SMOOTH = 3

# Volume
VOL_LEN = 20
VOL_THR = 2.5

# HTF
HTF_EMA_LEN = 50

# Entry modes
USE_BURST = True

# Risk — optimized
ATR_LEN = 14
SL_MULT = 2.0
TP_MULT = 5.0
TRAIL_MULT = 3.0
ADAPT_STOPS = True
COOLDOWN_BARS = 15


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def create_exchange():
    """Try exchanges until one works."""
    for name in ["binanceus", "bybit", "kraken", "binance"]:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            sym = SYMBOL
            if sym in ex.markets:
                print(f"  Connected: {name}")
                return ex, sym
            # Kraken uses AVAX/USD
            if name == "kraken" and "AVAX/USD" in ex.markets:
                print(f"  Connected: {name} (AVAX/USD)")
                return ex, "AVAX/USD"
        except Exception:
            continue
    raise RuntimeError("No exchange found with AVAX data")


def fetch_ohlcv(exchange, symbol, timeframe, since_ms, until_ms=None):
    """Fetch OHLCV data with pagination."""
    all_data = []
    current = since_ms
    batch = 1000

    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=batch)
        except Exception as e:
            print(f"    Error: {e}")
            break

        if not data:
            break
        all_data.extend(data)
        last_ts = data[-1][0]
        if until_ms and last_ts >= until_ms:
            break
        if len(data) < batch:
            break
        current = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if until_ms:
        df = df[df.index <= pd.to_datetime(until_ms, unit="ms")]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ema_calc(arr, period):
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def sma_calc(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).mean().values


def stdev_calc(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).std(ddof=0).values


def atr_calc(high, low, close, period):
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


def linreg_calc(arr, period):
    """Pine ta.linreg(source, length, 0) — endpoint of fitted line."""
    n = len(arr)
    out = np.zeros(n, dtype=float)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, n):
        y = arr[i - period + 1 : i + 1]
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / x_var if x_var > 0 else 0
        out[i] = slope * (period - 1) + (y_mean - slope * x_mean)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE ALL STRATEGY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(df, htf_ema_s, htf_close_s):
    """Vectorized indicator computation, returns DataFrame with signals."""
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)

    # ── Fractal Efficiency Ratio ──
    abs_chg = np.abs(np.diff(c, prepend=c[0]))
    path_len = pd.Series(abs_chg).rolling(ER_LEN, min_periods=ER_LEN).sum().values
    price_chg = np.zeros(n)
    for i in range(ER_LEN, n):
        price_chg[i] = abs(c[i] - c[i - ER_LEN])
    raw_er = np.where(path_len > 0, price_chg / path_len, 0.0)
    raw_er = np.nan_to_num(raw_er, nan=0.0)
    er = ema_calc(raw_er, ER_SMOOTH)

    is_trend = er > ER_TREND_THR
    is_chop = er < ER_CHOP_THR
    is_trans = ~is_trend & ~is_chop

    # ── Volume MA ──
    vol_ma = sma_calc(v, VOL_LEN)
    vol_ma = np.nan_to_num(vol_ma, nan=np.nanmean(v))

    # ── Volume-Weighted KAMA ──
    fast_sc = 2.0 / (KAMA_FAST + 1.0)
    slow_sc = 2.0 / (KAMA_SLOW + 1.0)

    # Pre-compute rolling path for KAMA efficiency
    kama_path = pd.Series(abs_chg).rolling(KAMA_LEN, min_periods=KAMA_LEN).sum().values

    kama = np.zeros(n)
    kama[0] = c[0]
    for i in range(1, n):
        if i < KAMA_LEN or np.isnan(kama_path[i]) or kama_path[i] == 0:
            kama[i] = c[i]
            continue
        k_er = abs(c[i] - c[i - KAMA_LEN]) / kama_path[i]
        sc = (k_er * (fast_sc - slow_sc) + slow_sc) ** 2
        vr = v[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        adj = min(sc * (0.5 + 0.5 * min(vr, 2.0)), 1.0)
        kama[i] = kama[i - 1] + adj * (c[i] - kama[i - 1])

    kama_slope = np.diff(kama, prepend=kama[0])
    kama_bull = kama_slope > 0
    kama_bear = kama_slope < 0

    # ── Squeeze ──
    sqz_basis = sma_calc(c, SQZ_BB_LEN)
    sqz_dev = SQZ_BB_MULT * stdev_calc(c, SQZ_BB_LEN)
    sqz_bb_up = np.nan_to_num(sqz_basis + sqz_dev, nan=np.inf)
    sqz_bb_lo = np.nan_to_num(sqz_basis - sqz_dev, nan=0)

    kc_basis = sma_calc(c, SQZ_KC_LEN)
    kc_range = SQZ_KC_MULT * atr_calc(h, l, c, SQZ_KC_LEN)
    kc_up = np.nan_to_num(kc_basis + kc_range, nan=np.inf)
    kc_lo = np.nan_to_num(kc_basis - kc_range, nan=0)

    sqz_on = (sqz_bb_lo > kc_lo) & (sqz_bb_up < kc_up)
    sqz_prev = np.roll(sqz_on, 1); sqz_prev[0] = False
    sqz_fire = sqz_prev & ~sqz_on

    sqz_basis_safe = np.nan_to_num(sqz_basis, nan=np.nanmean(c))
    sqz_mom = linreg_calc(c - sqz_basis_safe, SQZ_BB_LEN)

    # ── Momentum ──
    mom = np.zeros(n)
    for i in range(MOM_LEN, n):
        mom[i] = c[i] - c[i - MOM_LEN]
    mom_s = ema_calc(mom, MOM_SMOOTH)
    mom_acc = np.diff(mom_s, prepend=mom_s[0])

    # ── Volume Confirm ──
    vol_ok = v > (vol_ma * VOL_THR)

    # ── HTF Trend (forward-filled, shifted to prevent lookahead) ──
    htf_ema_ff = htf_ema_s.reindex(df.index, method="ffill").values
    htf_close_ff = htf_close_s.reindex(df.index, method="ffill").values
    htf_ema_ff = np.nan_to_num(htf_ema_ff, nan=np.nanmean(c))
    htf_close_ff = np.nan_to_num(htf_close_ff, nan=np.nanmean(c))
    htf_trend = np.where(htf_close_ff > htf_ema_ff, 1, np.where(htf_close_ff < htf_ema_ff, -1, 0))

    # ── ATR ──
    atr_v = atr_calc(h, l, c, ATR_LEN)

    # ── Adaptive multipliers ──
    if ADAPT_STOPS:
        sl_adj = np.where(is_trend, 1.15, np.where(is_trans, 0.85, 1.0))
        tp_adj = np.where(is_trend, 1.25, np.where(is_trans, 0.80, 1.0))
        tr_adj = np.where(is_trend, 1.20, np.where(is_trans, 0.85, 1.0))
    else:
        sl_adj = tp_adj = tr_adj = np.ones(n)

    # ── Crossovers ──
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    prev_k = np.roll(kama, 1); prev_k[0] = kama[0]
    cross_above = (c > kama) & (prev_c <= prev_k)
    cross_below = (c < kama) & (prev_c >= prev_k)

    # ── Entry Signals ──
    sqz_long = sqz_fire & (sqz_mom > 0) & (mom_acc > 0)
    trend_long = is_trend & kama_bull & cross_above & (mom_s > 0)
    if USE_BURST:
        burst_long = vol_ok & (c > kama) & (mom_s > 0) & ~is_chop
        burst_short_v = vol_ok & (c < kama) & (mom_s < 0) & ~is_chop
    else:
        burst_long = np.zeros(n, dtype=bool)
        burst_short_v = np.zeros(n, dtype=bool)
    long_sig = (sqz_long | trend_long | burst_long) & (htf_trend >= 0) & ~is_chop

    sqz_short = sqz_fire & (sqz_mom < 0) & (mom_acc < 0)
    trend_short = is_trend & kama_bear & cross_below & (mom_s < 0)
    short_sig = (sqz_short | trend_short | burst_short_v) & (htf_trend <= 0) & ~is_chop

    return pd.DataFrame({
        "open": df["open"].values, "high": h, "low": l, "close": c,
        "atr": atr_v, "er": er,
        "is_chop": is_chop, "is_trend": is_trend, "is_trans": is_trans,
        "long_sig": long_sig, "short_sig": short_sig,
        "sl_adj": sl_adj, "tp_adj": tp_adj, "tr_adj": tr_adj,
    }, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(sig):
    """Event-driven backtester matching Pine execution model."""
    n = len(sig)
    o = sig["open"].values
    h = sig["high"].values
    l = sig["low"].values
    c = sig["close"].values
    atr_v = sig["atr"].values
    er_v = sig["er"].values
    chop = sig["is_chop"].values
    lsig = sig["long_sig"].values
    ssig = sig["short_sig"].values
    sl_a = sig["sl_adj"].values
    tp_a = sig["tp_adj"].values
    tr_a = sig["tr_adj"].values

    equity = INITIAL_CAPITAL
    pos = 0          # 1=long, -1=short, 0=flat
    entry_px = 0.0
    stop_px = 0.0
    tp_px = 0.0
    trail_px = 0.0
    last_exit = -100
    trades = []
    eq_curve = np.full(n, INITIAL_CAPITAL)

    comm = COMMISSION_PCT / 100
    slip_buy = 1 + SLIPPAGE_PCT / 100
    slip_sell = 1 - SLIPPAGE_PCT / 100

    def close_long(exit_px, reason, bar):
        nonlocal equity, pos, last_exit
        pnl = (exit_px / entry_px - 1) * 100
        equity *= (exit_px / entry_px) * (1 - comm)
        trades.append({"dir": "L", "entry": entry_px, "exit": exit_px, "pnl": pnl, "reason": reason})
        pos = 0
        last_exit = bar

    def close_short(exit_px, reason, bar):
        nonlocal equity, pos, last_exit
        pnl = (1 - exit_px / entry_px) * 100
        equity *= (2 - exit_px / entry_px) * (1 - comm)
        trades.append({"dir": "S", "entry": entry_px, "exit": exit_px, "pnl": pnl, "reason": reason})
        pos = 0
        last_exit = bar

    def enter_long(px, bar):
        nonlocal equity, pos, entry_px, stop_px, tp_px, trail_px
        entry_px = px
        equity *= (1 - comm)
        a = atr_v[bar - 1] if bar > 0 else atr_v[bar]
        sa, ta2, tra = sl_a[bar - 1], tp_a[bar - 1], tr_a[bar - 1]
        stop_px = px - SL_MULT * sa * a
        tp_px = px + TP_MULT * ta2 * a
        trail_px = px - TRAIL_MULT * tra * a
        pos = 1

    def enter_short(px, bar):
        nonlocal equity, pos, entry_px, stop_px, tp_px, trail_px
        entry_px = px
        equity *= (1 - comm)
        a = atr_v[bar - 1] if bar > 0 else atr_v[bar]
        sa, ta2, tra = sl_a[bar - 1], tp_a[bar - 1], tr_a[bar - 1]
        stop_px = px + SL_MULT * sa * a
        tp_px = px - TP_MULT * ta2 * a
        trail_px = px + TRAIL_MULT * tra * a
        pos = -1

    for i in range(1, n):
        cooldown_ok = (i - last_exit) >= COOLDOWN_BARS

        # ── Phase 1: Fill pending entries at this bar's open ──
        if i > 0 and cooldown_ok:
            if lsig[i - 1] and pos <= 0:
                if pos == -1:
                    close_short(o[i] * slip_buy, "rev", i)
                enter_long(o[i] * slip_buy, i)
            elif ssig[i - 1] and pos >= 0:
                if pos == 1:
                    close_long(o[i] * slip_sell, "rev", i)
                enter_short(o[i] * slip_sell, i)

        # ── Phase 2: Check stop / TP during this bar ──
        if pos == 1:
            eff_stop = max(stop_px, trail_px) if True else stop_px
            both_hit = (l[i] <= eff_stop) and (h[i] >= tp_px)
            if both_hit:
                # Pessimistic: if open closer to stop, stop hit first
                if abs(o[i] - eff_stop) <= abs(o[i] - tp_px):
                    close_long(eff_stop * slip_sell, "stop", i)
                else:
                    close_long(tp_px * slip_sell, "tp", i)
            elif l[i] <= eff_stop:
                close_long(eff_stop * slip_sell, "stop", i)
            elif h[i] >= tp_px:
                close_long(tp_px * slip_sell, "tp", i)

        elif pos == -1:
            eff_stop = min(stop_px, trail_px) if True else stop_px
            both_hit = (h[i] >= eff_stop) and (l[i] <= tp_px)
            if both_hit:
                if abs(o[i] - eff_stop) <= abs(o[i] - tp_px):
                    close_short(eff_stop * slip_buy, "stop", i)
                else:
                    close_short(tp_px * slip_buy, "tp", i)
            elif h[i] >= eff_stop:
                close_short(eff_stop * slip_buy, "stop", i)
            elif l[i] <= tp_px:
                close_short(tp_px * slip_buy, "tp", i)

        # ── Phase 3: Update trailing stop ──
        if pos == 1:
            new_tr = c[i] - TRAIL_MULT * tr_a[i] * atr_v[i]
            trail_px = max(trail_px, new_tr)
        elif pos == -1:
            new_tr = c[i] + TRAIL_MULT * tr_a[i] * atr_v[i]
            trail_px = min(trail_px, new_tr)

        # ── Phase 4: Regime exit ──
        if pos != 0 and chop[i] and er_v[i] < ER_CHOP_THR * 0.65:
            if pos == 1:
                close_long(c[i] * slip_sell, "chop", i)
            elif pos == -1:
                close_short(c[i] * slip_buy, "chop", i)

        # ── Record equity ──
        if pos == 0:
            eq_curve[i] = equity
        elif pos == 1:
            eq_curve[i] = equity * (c[i] / entry_px)
        else:
            eq_curve[i] = equity * (2 - c[i] / entry_px)

    # Close open position at end
    if pos == 1:
        close_long(c[-1] * slip_sell, "end", n - 1)
        eq_curve[-1] = equity
    elif pos == -1:
        close_short(c[-1] * slip_buy, "end", n - 1)
        eq_curve[-1] = equity

    return trades, eq_curve


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(trades, eq_curve):
    if not trades:
        return {k: 0.0 for k in [
            "ret", "equity", "dd", "trades", "longs", "shorts",
            "winr", "pf", "sharpe", "avg", "avg_w", "avg_l", "best", "worst"
        ]}

    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    peak = np.maximum.accumulate(eq_curve)
    dd = ((eq_curve - peak) / peak * 100).min()

    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gp / gl if gl > 0 else 999.0

    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls))
    else:
        sharpe = 0.0

    return {
        "ret": (eq_curve[-1] / INITIAL_CAPITAL - 1) * 100,
        "equity": eq_curve[-1],
        "dd": dd,
        "trades": len(trades),
        "longs": sum(1 for t in trades if t["dir"] == "L"),
        "shorts": sum(1 for t in trades if t["dir"] == "S"),
        "winr": len(wins) / len(trades) * 100,
        "pf": min(pf, 99.99),
        "sharpe": sharpe,
        "avg": np.mean(pnls),
        "avg_w": np.mean(wins) if wins else 0,
        "avg_l": np.mean(losses) if losses else 0,
        "best": max(pnls),
        "worst": min(pnls),
    }


def print_table(results, buy_hold):
    W = 74
    tfs = list(results.keys())

    print()
    print("=" * W)
    print(f"  FATS BACKTEST — {SYMBOL} — {LOOKBACK_DAYS}-Day Lookback".center(W))
    print("=" * W)

    hdr = f"{'Metric':<20}"
    for tf in tfs:
        hdr += f" | {tf:>9}"
    print(hdr)
    print("-" * W)

    rows = [
        ("Total Return",   "ret",    "{:>+8.1f}%"),
        ("Final Equity",   "equity", "${:>8,.0f}"),
        ("Max Drawdown",   "dd",     "{:>+8.1f}%"),
        ("Total Trades",   "trades", "{:>9.0f}"),
        ("  Long / Short", None,     None),
        ("Win Rate",       "winr",   "{:>8.1f}%"),
        ("Profit Factor",  "pf",     "{:>8.2f}x"),
        ("Sharpe Ratio",   "sharpe", "{:>9.2f}"),
        ("Avg Trade",      "avg",    "{:>+8.2f}%"),
        ("Avg Winner",     "avg_w",  "{:>+8.2f}%"),
        ("Avg Loser",      "avg_l",  "{:>+8.2f}%"),
        ("Best Trade",     "best",   "{:>+8.2f}%"),
        ("Worst Trade",    "worst",  "{:>+8.2f}%"),
    ]

    for label, key, fmt in rows:
        line = f"{label:<20}"
        if key is None:
            # Special row for long/short counts
            for tf in tfs:
                r = results[tf]
                val = f"{int(r['longs'])} / {int(r['shorts'])}"
                line += f" | {val:>9}"
        else:
            for tf in tfs:
                val = results[tf][key]
                line += f" | {fmt.format(val)}"
        print(line)

    print("-" * W)
    bh_line = f"{'Buy & Hold':<20}"
    for _ in tfs:
        bh_line += f" | {buy_hold:>+8.1f}%"
    print(bh_line)
    print("=" * W)

    # Exit reason breakdown
    print(f"\n{'Exit Reasons':<20}", end="")
    for tf in tfs:
        print(f" | {tf:>9}", end="")
    print()
    print("-" * W)
    for reason, label in [("tp", "Take Profit"), ("stop", "Stop Loss"),
                          ("chop", "Regime Exit"), ("rev", "Reversal"), ("end", "End/Open")]:
        line = f"  {label:<18}"
        for tf in tfs:
            count = results[tf].get(f"_{reason}", 0)
            line += f" | {count:>9}"
        print(line)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 56)
    print("  FRACTAL ADAPTIVE TREND SYSTEM — Backtester")
    print("=" * 56)

    # Connect
    print("\n[1/3] Connecting to exchange...")
    exchange, symbol = create_exchange()

    now = datetime.utcnow()
    since = now - timedelta(days=LOOKBACK_DAYS + WARMUP_DAYS)
    since_ms = int(since.timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)
    test_start = now - timedelta(days=LOOKBACK_DAYS)

    # Fetch data
    print("\n[2/3] Fetching OHLCV data...")
    needed_tfs = sorted(set(TIMEFRAMES) | set(HTF_MAP.values()))
    all_data = {}
    for tf in needed_tfs:
        print(f"  {tf:>3}...", end=" ", flush=True)
        df = fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
        print(f"{len(df):,} candles ({'ok' if len(df) > 50 else 'insufficient'})")
        all_data[tf] = df

    # Buy & Hold baseline
    ref_df = all_data.get("1h", all_data.get("4h", list(all_data.values())[0]))
    ref_test = ref_df[ref_df.index >= test_start]
    if len(ref_test) > 0:
        buy_hold = (ref_test.iloc[-1]["close"] / ref_test.iloc[0]["close"] - 1) * 100
    else:
        buy_hold = 0.0

    # Run backtests
    print(f"\n[3/3] Running backtests (start: {test_start.strftime('%Y-%m-%d')})...")
    all_results = {}

    for tf in TIMEFRAMES:
        print(f"\n  {'=' * 50}")
        print(f"  {tf.upper()} Timeframe")
        print(f"  {'=' * 50}")

        chart_df = all_data.get(tf, pd.DataFrame())
        htf_tf = HTF_MAP[tf]
        htf_df = all_data.get(htf_tf, pd.DataFrame())

        if chart_df.empty or htf_df.empty or len(chart_df) < 100:
            print(f"  Skipped — insufficient data")
            all_results[tf] = {k: 0.0 for k in [
                "ret", "equity", "dd", "trades", "longs", "shorts",
                "winr", "pf", "sharpe", "avg", "avg_w", "avg_l", "best", "worst"
            ]}
            continue

        # HTF EMA — shift by 1 to use only confirmed (closed) bars (no lookahead)
        htf_ema_v = ema_calc(htf_df["close"].values.astype(float), HTF_EMA_LEN)
        htf_ema_s = pd.Series(htf_ema_v, index=htf_df.index).shift(1)
        htf_close_s = htf_df["close"].shift(1)

        print(f"  Computing indicators...", flush=True)
        signals = compute_signals(chart_df, htf_ema_s, htf_close_s)

        # Trim to test period
        signals = signals[signals.index >= test_start]
        if len(signals) < 50:
            print(f"  Skipped — only {len(signals)} bars in test window")
            all_results[tf] = {k: 0.0 for k in [
                "ret", "equity", "dd", "trades", "longs", "shorts",
                "winr", "pf", "sharpe", "avg", "avg_w", "avg_l", "best", "worst"
            ]}
            continue

        print(f"  Backtesting {len(signals):,} bars...", flush=True)
        trades, eq_curve = run_backtest(signals)
        res = analyze(trades, eq_curve)

        # Add exit reason counts
        for reason in ["tp", "stop", "chop", "rev", "end"]:
            res[f"_{reason}"] = sum(1 for t in trades if t["reason"] == reason)

        all_results[tf] = res

        print(f"  Trades: {res['trades']:.0f} | Return: {res['ret']:+.1f}% | "
              f"WR: {res['winr']:.1f}% | PF: {res['pf']:.2f}x | MaxDD: {res['dd']:.1f}%")

    # Final table
    print_table(all_results, buy_hold)


if __name__ == "__main__":
    main()
