#!/usr/bin/env python3
"""
FATS Parameter Optimizer — AVAX/USDT
Grid-searches optimal parameters per timeframe (15m, 30m, 1H, 4H).
Pre-computes expensive indicators once, then sweeps thresholds + risk params.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import product
import ccxt

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED PARAMETERS (not optimized)
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL = "AVAX/USDT"
TIMEFRAMES = ["15m", "30m", "1h", "4h"]
HTF_MAP = {"15m": "4h", "30m": "4h", "1h": "4h", "4h": "1d"}
LOOKBACK_DAYS = 365
WARMUP_DAYS = 90
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.075
SLIPPAGE_PCT = 0.05

ER_LEN = 10
ER_SMOOTH = 3
KAMA_LEN = 10
KAMA_FAST = 2
KAMA_SLOW = 30
SQZ_BB_LEN = 20
SQZ_BB_MULT = 2.0
SQZ_KC_LEN = 20
SQZ_KC_MULT = 1.5
MOM_LEN = 12
MOM_SMOOTH = 3
VOL_LEN = 20
HTF_EMA_LEN = 50
ATR_LEN = 14

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION GRID
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    "vol_thr":    [1.0, 1.5, 2.0, 2.5],
    "er_trend":   [0.38, 0.45, 0.55],
    "er_chop":    [0.18, 0.25, 0.32],
    "sl_mult":    [1.5, 2.0, 2.5],
    "tp_mult":    [2.5, 3.5, 5.0],
    "trail_mult": [2.0, 3.0],
    "cooldown":   [3, 8, 15],
    "use_burst":  [True, False],
}

PARAM_NAMES = list(GRID.keys())
COMBOS = list(product(*GRID.values()))
TOTAL_COMBOS = len(COMBOS)

# Minimum trades required per timeframe for statistical validity
MIN_TRADES = {"15m": 40, "30m": 25, "1h": 15, "4h": 8}

# Default params for comparison
DEFAULTS = {
    "vol_thr": 1.3, "er_trend": 0.42, "er_chop": 0.22,
    "sl_mult": 2.0, "tp_mult": 3.0, "trail_mult": 2.5,
    "cooldown": 5, "use_burst": True,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ema_calc(arr, period):
    a = 2.0 / (period + 1.0)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def sma_calc(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).mean().values

def stdev_calc(arr, period):
    return pd.Series(arr).rolling(period, min_periods=period).std(ddof=0).values

def atr_calc(high, low, close, period):
    pc = np.roll(close, 1); pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return pd.Series(tr).rolling(period, min_periods=1).mean().values

def linreg_calc(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=float)
    x = np.arange(period, dtype=float)
    xm = x.mean()
    xv = np.sum((x - xm) ** 2)
    for i in range(period - 1, n):
        y = arr[i - period + 1 : i + 1]
        ym = np.mean(y)
        sl = np.sum((x - xm) * (y - ym)) / xv if xv > 0 else 0
        out[i] = sl * (period - 1) + (ym - sl * xm)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def create_exchange():
    for name in ["binanceus", "bybit", "kraken", "binance"]:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            if SYMBOL in ex.markets:
                print(f"  Connected: {name}")
                return ex, SYMBOL
        except Exception:
            continue
    raise RuntimeError("No exchange found")

def fetch_ohlcv(exchange, symbol, timeframe, since_ms, until_ms=None):
    all_data = []
    current = since_ms
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=1000)
        except Exception as e:
            print(f"    Fetch error: {e}")
            break
        if not data:
            break
        all_data.extend(data)
        if until_ms and data[-1][0] >= until_ms:
            break
        if len(data) < 1000:
            break
        current = data[-1][0] + 1
        _time.sleep(exchange.rateLimit / 1000)

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
# PRE-COMPUTE RAW INDICATORS (once per timeframe)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute(df, htf_ema_s, htf_close_s):
    """Pre-compute all indicators that don't depend on optimized params."""
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    lo = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)

    # Efficiency Ratio
    abs_chg = np.abs(np.diff(c, prepend=c[0]))
    path_len = pd.Series(abs_chg).rolling(ER_LEN, min_periods=ER_LEN).sum().values
    price_chg = np.zeros(n)
    for i in range(ER_LEN, n):
        price_chg[i] = abs(c[i] - c[i - ER_LEN])
    raw_er = np.where(path_len > 0, price_chg / path_len, 0.0)
    er = ema_calc(np.nan_to_num(raw_er, nan=0.0), ER_SMOOTH)

    # Volume MA
    vol_ma = np.nan_to_num(sma_calc(v, VOL_LEN), nan=np.nanmean(v))

    # KAMA
    fsc = 2.0 / (KAMA_FAST + 1.0)
    ssc = 2.0 / (KAMA_SLOW + 1.0)
    kp = pd.Series(abs_chg).rolling(KAMA_LEN, min_periods=KAMA_LEN).sum().values
    kama = np.zeros(n)
    kama[0] = c[0]
    for i in range(1, n):
        if i < KAMA_LEN or np.isnan(kp[i]) or kp[i] == 0:
            kama[i] = c[i]
            continue
        ke = abs(c[i] - c[i - KAMA_LEN]) / kp[i]
        sc = (ke * (fsc - ssc) + ssc) ** 2
        vr = v[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        kama[i] = kama[i - 1] + min(sc * (0.5 + 0.5 * min(vr, 2.0)), 1.0) * (c[i] - kama[i - 1])

    ks = np.diff(kama, prepend=kama[0])

    # Squeeze
    sb = sma_calc(c, SQZ_BB_LEN)
    sd = SQZ_BB_MULT * stdev_calc(c, SQZ_BB_LEN)
    bbu = np.nan_to_num(sb + sd, nan=np.inf)
    bbl = np.nan_to_num(sb - sd, nan=0)
    kb = sma_calc(c, SQZ_KC_LEN)
    kr = SQZ_KC_MULT * atr_calc(h, lo, c, SQZ_KC_LEN)
    kcu = np.nan_to_num(kb + kr, nan=np.inf)
    kcl = np.nan_to_num(kb - kr, nan=0)
    sqz_on = (bbl > kcl) & (bbu < kcu)
    sqz_prev = np.roll(sqz_on, 1); sqz_prev[0] = False
    sqz_fire = sqz_prev & ~sqz_on
    sbs = np.nan_to_num(sb, nan=np.nanmean(c))
    sqz_mom = linreg_calc(c - sbs, SQZ_BB_LEN)

    # Momentum
    mom = np.zeros(n)
    for i in range(MOM_LEN, n):
        mom[i] = c[i] - c[i - MOM_LEN]
    mom_s = ema_calc(mom, MOM_SMOOTH)
    mom_a = np.diff(mom_s, prepend=mom_s[0])

    # HTF
    htf_e = np.nan_to_num(htf_ema_s.reindex(df.index, method="ffill").values, nan=np.nanmean(c))
    htf_c = np.nan_to_num(htf_close_s.reindex(df.index, method="ffill").values, nan=np.nanmean(c))
    htf_trend = np.where(htf_c > htf_e, 1, np.where(htf_c < htf_e, -1, 0))

    # ATR
    atr_v = atr_calc(h, lo, c, ATR_LEN)

    # Crossovers
    pc = np.roll(c, 1); pc[0] = c[0]
    pk = np.roll(kama, 1); pk[0] = kama[0]
    cx_above = (c > kama) & (pc <= pk)
    cx_below = (c < kama) & (pc >= pk)

    # Squeeze base signals (threshold-independent)
    sqz_l = sqz_fire & (sqz_mom > 0) & (mom_a > 0)
    sqz_s = sqz_fire & (sqz_mom < 0) & (mom_a < 0)

    return {
        "o": df["open"].values.astype(float), "h": h, "l": lo, "c": c, "v": v,
        "er": er, "vol_ma": vol_ma, "kama": kama,
        "kama_bull": ks > 0, "kama_bear": ks < 0,
        "sqz_l": sqz_l, "sqz_s": sqz_s,
        "mom_s": mom_s, "mom_a": mom_a,
        "htf_trend": htf_trend, "atr": atr_v,
        "cx_above": cx_above, "cx_below": cx_below,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FAST BACKTEST — inlined for speed
# ═══════════════════════════════════════════════════════════════════════════════

def run_opt_backtest(raw, params):
    """Generate signals + run backtest for one parameter combination.
    Returns (total_return_pct, max_dd_pct, profit_factor, num_trades, win_rate)."""

    er = raw["er"]
    c = raw["c"]
    v = raw["v"]
    n = len(c)

    # Thresholds
    vt = params["vol_thr"]
    ert = params["er_trend"]
    erc = params["er_chop"]

    is_trend = er > ert
    is_chop = er < erc
    is_trans = ~is_trend & ~is_chop

    vol_ok = v > raw["vol_ma"] * vt

    # Signals
    tl = is_trend & raw["kama_bull"] & raw["cx_above"] & (raw["mom_s"] > 0)
    ts = is_trend & raw["kama_bear"] & raw["cx_below"] & (raw["mom_s"] < 0)

    if params["use_burst"]:
        bl = vol_ok & (c > raw["kama"]) & (raw["mom_s"] > 0) & ~is_chop
        bs = vol_ok & (c < raw["kama"]) & (raw["mom_s"] < 0) & ~is_chop
        lsig = (raw["sqz_l"] | tl | bl) & (raw["htf_trend"] >= 0) & ~is_chop
        ssig = (raw["sqz_s"] | ts | bs) & (raw["htf_trend"] <= 0) & ~is_chop
    else:
        lsig = (raw["sqz_l"] | tl) & (raw["htf_trend"] >= 0) & ~is_chop
        ssig = (raw["sqz_s"] | ts) & (raw["htf_trend"] <= 0) & ~is_chop

    # Adaptive stop adjustments
    sl_a = np.where(is_trend, 1.15, np.where(is_trans, 0.85, 1.0))
    tp_a = np.where(is_trend, 1.25, np.where(is_trans, 0.80, 1.0))
    tr_a = np.where(is_trend, 1.20, np.where(is_trans, 0.85, 1.0))

    # Risk params
    SL = params["sl_mult"]
    TP = params["tp_mult"]
    TR = params["trail_mult"]
    CD = params["cooldown"]

    o = raw["o"]
    h = raw["h"]
    lo = raw["l"]
    atr_v = raw["atr"]
    chop_deep_thr = erc * 0.65

    comm = COMMISSION_PCT / 100
    slip_b = 1 + SLIPPAGE_PCT / 100
    slip_s = 1 - SLIPPAGE_PCT / 100

    equity = INITIAL_CAPITAL
    peak_eq = equity
    max_dd = 0.0
    pos = 0
    entry_px = 0.0
    stop_px = 0.0
    tp_px = 0.0
    trail_px = 0.0
    last_exit = -100
    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(1, n):
        cd_ok = (i - last_exit) >= CD

        # Phase 1: Entries at open
        if cd_ok:
            if lsig[i - 1] and pos <= 0:
                if pos == -1:
                    # Close short
                    ep = o[i] * slip_b
                    pnl = (1 - ep / entry_px)
                    equity *= (2 - ep / entry_px) * (1 - comm)
                    n_trades += 1
                    if pnl > 0:
                        n_wins += 1; gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    pos = 0; last_exit = i
                # Enter long
                entry_px = o[i] * slip_b
                equity *= (1 - comm)
                a = atr_v[i - 1]
                stop_px = entry_px - SL * sl_a[i - 1] * a
                tp_px = entry_px + TP * tp_a[i - 1] * a
                trail_px = entry_px - TR * tr_a[i - 1] * a
                pos = 1

            elif ssig[i - 1] and pos >= 0:
                if pos == 1:
                    ep = o[i] * slip_s
                    pnl = (ep / entry_px - 1)
                    equity *= (ep / entry_px) * (1 - comm)
                    n_trades += 1
                    if pnl > 0:
                        n_wins += 1; gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    pos = 0; last_exit = i
                entry_px = o[i] * slip_s
                equity *= (1 - comm)
                a = atr_v[i - 1]
                stop_px = entry_px + SL * sl_a[i - 1] * a
                tp_px = entry_px - TP * tp_a[i - 1] * a
                trail_px = entry_px + TR * tr_a[i - 1] * a
                pos = -1

        # Phase 2: Exits
        if pos == 1:
            es = max(stop_px, trail_px)
            hit_stop = lo[i] <= es
            hit_tp = h[i] >= tp_px
            if hit_stop and hit_tp:
                if abs(o[i] - es) <= abs(o[i] - tp_px):
                    ep = es * slip_s
                else:
                    ep = tp_px * slip_s
                pnl = (ep / entry_px - 1)
                equity *= (ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i
            elif hit_stop:
                ep = es * slip_s
                pnl = (ep / entry_px - 1)
                equity *= (ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i
            elif hit_tp:
                ep = tp_px * slip_s
                pnl = (ep / entry_px - 1)
                equity *= (ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i

        elif pos == -1:
            es = min(stop_px, trail_px)
            hit_stop = h[i] >= es
            hit_tp = lo[i] <= tp_px
            if hit_stop and hit_tp:
                if abs(o[i] - es) <= abs(o[i] - tp_px):
                    ep = es * slip_b
                else:
                    ep = tp_px * slip_b
                pnl = (1 - ep / entry_px)
                equity *= (2 - ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i
            elif hit_stop:
                ep = es * slip_b
                pnl = (1 - ep / entry_px)
                equity *= (2 - ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i
            elif hit_tp:
                ep = tp_px * slip_b
                pnl = (1 - ep / entry_px)
                equity *= (2 - ep / entry_px) * (1 - comm)
                n_trades += 1
                if pnl > 0: n_wins += 1; gross_profit += pnl
                else: gross_loss += abs(pnl)
                pos = 0; last_exit = i

        # Phase 3: Trail update
        if pos == 1:
            nt = c[i] - TR * tr_a[i] * atr_v[i]
            if nt > trail_px:
                trail_px = nt
        elif pos == -1:
            nt = c[i] + TR * tr_a[i] * atr_v[i]
            if nt < trail_px:
                trail_px = nt

        # Phase 4: Regime exit
        if pos != 0 and is_chop[i] and er[i] < chop_deep_thr:
            if pos == 1:
                ep = c[i] * slip_s
                pnl = (ep / entry_px - 1)
                equity *= (ep / entry_px) * (1 - comm)
            else:
                ep = c[i] * slip_b
                pnl = (1 - ep / entry_px)
                equity *= (2 - ep / entry_px) * (1 - comm)
            n_trades += 1
            if pnl > 0: n_wins += 1; gross_profit += pnl
            else: gross_loss += abs(pnl)
            pos = 0; last_exit = i

        # Track drawdown
        cur_eq = equity
        if pos == 1:
            cur_eq = equity * (c[i] / entry_px)
        elif pos == -1:
            cur_eq = equity * (2 - c[i] / entry_px)
        if cur_eq > peak_eq:
            peak_eq = cur_eq
        dd = (cur_eq - peak_eq) / peak_eq * 100
        if dd < max_dd:
            max_dd = dd

    # Close open position
    if pos == 1:
        equity *= (c[-1] * slip_s / entry_px) * (1 - comm)
        n_trades += 1
    elif pos == -1:
        equity *= (2 - c[-1] * slip_b / entry_px) * (1 - comm)
        n_trades += 1

    ret = (equity / INITIAL_CAPITAL - 1) * 100
    pf = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    wr = n_wins / n_trades * 100 if n_trades > 0 else 0.0

    return ret, max_dd, pf, n_trades, wr


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_result(ret, dd, pf, trades, min_t):
    """Score: Calmar ratio * profit factor bonus. Higher = better."""
    if trades < min_t:
        return -999.0
    if dd >= 0:
        dd = -0.1
    calmar = ret / abs(dd)
    pf_bonus = min(pf, 3.0) / 1.5
    return calmar * pf_bonus


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  FATS PARAMETER OPTIMIZER — AVAX/USDT")
    print("=" * 60)
    print(f"  Grid: {TOTAL_COMBOS:,} combinations per timeframe")
    print(f"  Total: {TOTAL_COMBOS * len(TIMEFRAMES):,} backtests")

    # Connect & fetch
    print("\n[1/3] Connecting...")
    exchange, symbol = create_exchange()

    now = datetime.utcnow()
    since_ms = int((now - timedelta(days=LOOKBACK_DAYS + WARMUP_DAYS)).timestamp() * 1000)
    until_ms = int(now.timestamp() * 1000)
    test_start = now - timedelta(days=LOOKBACK_DAYS)

    print("\n[2/3] Fetching data...")
    needed = sorted(set(TIMEFRAMES) | set(HTF_MAP.values()))
    all_data = {}
    for tf in needed:
        print(f"  {tf:>3}...", end=" ", flush=True)
        df = fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
        print(f"{len(df):,} candles")
        all_data[tf] = df

    # Buy & hold baseline
    ref = all_data.get("1h", all_data.get("4h"))
    ref_t = ref[ref.index >= test_start]
    buy_hold = (ref_t.iloc[-1]["close"] / ref_t.iloc[0]["close"] - 1) * 100 if len(ref_t) > 0 else 0

    # Optimize each timeframe
    print(f"\n[3/3] Optimizing (this will take several minutes)...")
    best_results = {}

    for tf in TIMEFRAMES:
        print(f"\n{'=' * 60}")
        print(f"  Optimizing {tf.upper()}")
        print(f"{'=' * 60}")

        chart_df = all_data.get(tf, pd.DataFrame())
        htf_tf = HTF_MAP[tf]
        htf_df = all_data.get(htf_tf, pd.DataFrame())

        if chart_df.empty or htf_df.empty or len(chart_df) < 100:
            print(f"  Skipped — insufficient data")
            best_results[tf] = None
            continue

        # HTF EMA (shifted for no lookahead)
        htf_ema_v = ema_calc(htf_df["close"].values.astype(float), HTF_EMA_LEN)
        htf_ema_s = pd.Series(htf_ema_v, index=htf_df.index).shift(1)
        htf_close_s = htf_df["close"].shift(1)

        # Pre-compute raw indicators
        print(f"  Pre-computing indicators...", flush=True)
        raw_full = precompute(chart_df, htf_ema_s, htf_close_s)

        # Trim to test period
        mask = chart_df.index >= test_start
        idx = np.where(mask)[0]
        if len(idx) < 50:
            print(f"  Skipped — only {len(idx)} bars")
            best_results[tf] = None
            continue

        start_idx = idx[0]
        raw = {}
        for k, val in raw_full.items():
            raw[k] = val[start_idx:]

        n_bars = len(raw["c"])
        min_t = MIN_TRADES[tf]
        print(f"  {n_bars:,} bars | {TOTAL_COMBOS:,} combos | min trades: {min_t}")

        # Default params result
        def_ret, def_dd, def_pf, def_trades, def_wr = run_opt_backtest(raw, DEFAULTS)
        def_score = score_result(def_ret, def_dd, def_pf, def_trades, min_t)
        print(f"  Default: ret={def_ret:+.1f}%  dd={def_dd:.1f}%  pf={def_pf:.2f}x  "
              f"trades={def_trades}  wr={def_wr:.1f}%  score={def_score:.3f}")

        # Grid search
        best_score = -9999.0
        best_params = None
        best_metrics = None
        t0 = _time.time()

        for ci, combo in enumerate(COMBOS):
            p = dict(zip(PARAM_NAMES, combo))
            ret, dd, pf, trades, wr = run_opt_backtest(raw, p)
            sc = score_result(ret, dd, pf, trades, min_t)

            if sc > best_score:
                best_score = sc
                best_params = p.copy()
                best_metrics = (ret, dd, pf, trades, wr)

            # Progress
            if (ci + 1) % 500 == 0 or ci == TOTAL_COMBOS - 1:
                elapsed = _time.time() - t0
                pct = (ci + 1) / TOTAL_COMBOS * 100
                eta = elapsed / (ci + 1) * (TOTAL_COMBOS - ci - 1)
                br, bdd, bpf, bt, bwr = best_metrics if best_metrics else (0, 0, 0, 0, 0)
                print(f"\r  [{pct:5.1f}%] {ci + 1:,}/{TOTAL_COMBOS:,} | "
                      f"best: ret={br:+.1f}% dd={bdd:.1f}% pf={bpf:.2f}x "
                      f"trades={bt} | ETA {eta:.0f}s   ", end="", flush=True)

        elapsed = _time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s")

        if best_params:
            br, bdd, bpf, bt, bwr = best_metrics
            print(f"\n  BEST: ret={br:+.1f}%  dd={bdd:.1f}%  pf={bpf:.2f}x  "
                  f"trades={bt}  wr={bwr:.1f}%  score={best_score:.3f}")
            print(f"  Params: {best_params}")

        best_results[tf] = {
            "params": best_params,
            "metrics": best_metrics,
            "score": best_score,
            "default_metrics": (def_ret, def_dd, def_pf, def_trades, def_wr),
            "default_score": def_score,
        }

    # ── FINAL SUMMARY ──
    print("\n\n" + "=" * 80)
    print("  OPTIMIZATION RESULTS — AVAX/USDT".center(80))
    print("=" * 80)

    for tf in TIMEFRAMES:
        r = best_results.get(tf)
        if not r or not r["params"]:
            print(f"\n  {tf.upper()}: No valid parameters found")
            continue

        p = r["params"]
        br, bdd, bpf, bt, bwr = r["metrics"]
        dr, ddd, dpf, dt, dwr = r["default_metrics"]

        print(f"\n  {'─' * 76}")
        print(f"  {tf.upper()} TIMEFRAME")
        print(f"  {'─' * 76}")

        # Comparison table
        print(f"  {'Metric':<20} {'Default':>12} {'Optimized':>12} {'Change':>12}")
        print(f"  {'─' * 56}")
        print(f"  {'Return':<20} {dr:>+11.1f}% {br:>+11.1f}% {br - dr:>+11.1f}%")
        print(f"  {'Max Drawdown':<20} {ddd:>+11.1f}% {bdd:>+11.1f}% {bdd - ddd:>+11.1f}%")
        print(f"  {'Profit Factor':<20} {dpf:>11.2f}x {bpf:>11.2f}x {bpf - dpf:>+11.2f}x")
        print(f"  {'Trades':<20} {dt:>12} {bt:>12} {bt - dt:>+12}")
        print(f"  {'Win Rate':<20} {dwr:>11.1f}% {bwr:>11.1f}% {bwr - dwr:>+11.1f}%")

        print(f"\n  Optimal Parameters:")
        print(f"    ER Trend Threshold  = {p['er_trend']}")
        print(f"    ER Chop Threshold   = {p['er_chop']}")
        print(f"    Volume Threshold    = {p['vol_thr']}x")
        print(f"    Stop Loss (ATR)     = {p['sl_mult']}x")
        print(f"    Take Profit (ATR)   = {p['tp_mult']}x")
        print(f"    Trail Stop (ATR)    = {p['trail_mult']}x")
        print(f"    Cooldown Bars       = {p['cooldown']}")
        print(f"    Use Burst Entries   = {p['use_burst']}")

    # Pine Script settings
    print(f"\n\n{'=' * 80}")
    print("  PINE SCRIPT INPUT VALUES (copy into TradingView)".center(80))
    print("=" * 80)

    for tf in TIMEFRAMES:
        r = best_results.get(tf)
        if not r or not r["params"]:
            continue
        p = r["params"]
        br, bdd, bpf, bt, bwr = r["metrics"]

        print(f"\n  // {tf.upper()} — Return: {br:+.1f}% | PF: {bpf:.2f}x | DD: {bdd:.1f}% | Trades: {bt}")
        print(f"  // Fractal Efficiency")
        print(f"  //   Trending Threshold = {p['er_trend']}")
        print(f"  //   Chop Threshold     = {p['er_chop']}")
        print(f"  // Volume")
        print(f"  //   Volume Threshold   = {p['vol_thr']}")
        print(f"  // Risk Management")
        print(f"  //   Stop Loss ATR Mult = {p['sl_mult']}")
        print(f"  //   Take Profit ATR    = {p['tp_mult']}")
        print(f"  //   Trail ATR Mult     = {p['trail_mult']}")
        print(f"  // Cooldown")
        print(f"  //   Cooldown Bars      = {p['cooldown']}")
        print(f"  //   Use Burst Entries  = {p['use_burst']}")

    print(f"\n  Buy & Hold (12mo): {buy_hold:+.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
