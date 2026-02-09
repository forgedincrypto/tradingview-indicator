# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-file TradingView Pine Script v5 indicator: `BB_Dual_Mode_Ichimoku_Regime.pine`. No build system, tests, or dependencies — the script is copied directly into the TradingView Pine Editor.

## Architecture

The indicator combines two timeframes into a scored signal system:

1. **1H Ichimoku Regime** — `request.security()` fetches hourly Ichimoku cloud data (Tenkan/Kijun/Senkou) and classifies the market as Bullish (+1), Sideways (0), or Bearish (-1). Uses `lookahead_off` to prevent repainting.

2. **15m Bollinger Band Signals** — Four signal types on the working timeframe:
   - **REV Long/Short** (mean reversion): price crosses back inside the band
   - **BRK Long/Short** (breakout): price crosses outside the band

3. **Directional Scoring** — Each of the 4 signals has 3 weight inputs (one per regime). The regime selects which weight applies. Signals only display if their score >= `minScoreToShow`. This creates directional bias: bullish regime favors longs, bearish favors shorts, sideways favors reversion over breakout.

## Key Constraints

- Pine Script v5 syntax — use `ta.*` and `math.*` namespaces, not legacy function calls.
- `request.security()` must use `barmerge.lookahead_off` to avoid future data leaks.
- Ichimoku cloud is evaluated at current position (no forward displacement) for regime decisions.
- `alertcondition()` messages are static strings — Pine does not support dynamic interpolation in alert messages.
- `max_labels_count=500` is set on the indicator declaration to support text labels.
