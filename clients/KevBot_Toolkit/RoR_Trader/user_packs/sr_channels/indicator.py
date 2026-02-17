import pandas as pd
import numpy as np


def calculate_sr_channels(df: pd.DataFrame, **params) -> pd.DataFrame:
    """
    Calculate Support/Resistance channels from pivot point clusters.
    
    Identifies pivot highs/lows, clusters them into S/R zones based on
    proximity, ranks zones by strength, and outputs the nearest S/R
    channel boundaries plus breakout signals for each bar.
    """
    prd = params.get("pivot_period", 10)
    ppsrc = params.get("pivot_source", "High/Low")
    channel_w = params.get("channel_width_pct", 5)
    min_strength = params.get("min_strength", 1)
    max_num_sr = params.get("max_num_sr", 6)
    loopback = params.get("loopback", 290)

    result = df.copy()
    n = len(result)

    # Pivot source selection
    if ppsrc == "High/Low":
        src_high = result["high"].values
        src_low = result["low"].values
    else:
        src_high = np.maximum(result["close"].values, result["open"].values)
        src_low = np.minimum(result["close"].values, result["open"].values)

    highs = result["high"].values
    lows = result["low"].values
    closes = result["close"].values

    # Detect pivot highs and pivot lows (confirmed prd bars later)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)

    for i in range(prd, n - prd):
        # Pivot high: src_high[i] is highest in window [i-prd, i+prd]
        window_h = src_high[max(0, i - prd):i + prd + 1]
        if src_high[i] == np.max(window_h) and np.sum(window_h == src_high[i]) == 1:
            pivot_highs[i] = src_high[i]
        # Pivot low: src_low[i] is lowest in window
        window_l = src_low[max(0, i - prd):i + prd + 1]
        if src_low[i] == np.min(window_l) and np.sum(window_l == src_low[i]) == 1:
            pivot_lows[i] = src_low[i]

    # For each bar, compute S/R channels using historical pivots
    nearest_top = np.full(n, np.nan)
    nearest_bot = np.full(n, np.nan)
    num_channels = np.zeros(n)
    in_channel = np.zeros(n)
    res_broken = np.zeros(n)
    sup_broken = np.zeros(n)

    for i in range(prd * 2, n):
        # Collect pivot values and locations within loopback
        # Pivots are confirmed at index j, but detected at j (the peak/trough bar)
        # We only know about pivot at j once we've seen j + prd bars
        confirm_limit = i - prd  # latest bar whose pivot status is confirmed
        pivot_vals = []
        pivot_idxs = []
        for j in range(max(0, confirm_limit - loopback), confirm_limit + 1):
            if not np.isnan(pivot_highs[j]):
                pivot_vals.append(pivot_highs[j])
                pivot_idxs.append(j)
            if not np.isnan(pivot_lows[j]):
                pivot_vals.append(pivot_lows[j])
                pivot_idxs.append(j)

        if len(pivot_vals) == 0:
            continue

        # Channel width based on 300-bar range ending at bar i
        range_start = max(0, i - 299)
        prdhighest = np.max(highs[range_start:i + 1])
        prdlowest = np.min(lows[range_start:i + 1])
        cwidth = (prdhighest - prdlowest) * channel_w / 100.0

        if cwidth <= 0:
            continue

        # Build S/R channels from pivot clusters
        pv = np.array(pivot_vals)
        num_pivots = len(pv)

        # For each pivot, find the channel (hi, lo) and base strength
        channels = []
        for k in range(num_pivots):
            lo = pv[k]
            hi = pv[k]
            pp_count = 0
            for m in range(num_pivots):
                cpp = pv[m]
                wdth = hi - cpp if cpp <= hi else cpp - lo
                if wdth <= cwidth:
                    lo = min(lo, cpp)
                    hi = max(hi, cpp)
                    pp_count += 20
            # Add bar-touch strength
            touch_count = 0
            lb = min(loopback, i)
            for y in range(lb + 1):
                idx = i - y
                if idx < 0:
                    break
                if (highs[idx] <= hi and highs[idx] >= lo) or (lows[idx] <= hi and lows[idx] >= lo):
                    touch_count += 1
            total_strength = pp_count + touch_count
            channels.append((total_strength, hi, lo))

        # Select strongest non-overlapping channels
        sr_levels = []
        used = [False] * num_pivots
        ch_strengths = [c[0] for c in channels]
        ch_his = [c[1] for c in channels]
        ch_los = [c[2] for c in channels]

        for _ in range(min(10, max_num_sr)):
            best_val = -1
            best_idx = -1
            for k in range(num_pivots):
                if not used[k] and ch_strengths[k] > best_val and ch_strengths[k] >= min_strength * 20:
                    best_val = ch_strengths[k]
                    best_idx = k
            if best_idx < 0:
                break
            hh = ch_his[best_idx]
            ll = ch_los[best_idx]
            sr_levels.append((best_val, hh, ll))
            # Zero out overlapping channels
            for k in range(num_pivots):
                if (ch_his[k] <= hh and ch_his[k] >= ll) or (ch_los[k] <= hh and ch_los[k] >= ll):
                    used[k] = True
                    ch_strengths[k] = -1

        # Sort by strength descending
        sr_levels.sort(key=lambda x: -x[0])
        sr_levels = sr_levels[:max_num_sr]

        num_channels[i] = len(sr_levels)

        if len(sr_levels) == 0:
            continue

        # Find nearest channel to current close
        c = closes[i]
        best_dist = float("inf")
        best_top = np.nan
        best_bot = np.nan
        is_in_channel = False

        for _, top, bot in sr_levels:
            mid = (top + bot) / 2.0
            dist = abs(c - mid)
            if c <= top and c >= bot:
                is_in_channel = True
            if dist < best_dist:
                best_dist = dist
                best_top = top
                best_bot = bot

        nearest_top[i] = best_top
        nearest_bot[i] = best_bot
        in_channel[i] = 1.0 if is_in_channel else 0.0

        # Breakout detection
        if not is_in_channel and i > 0:
            prev_c = closes[i - 1]
            for _, top, bot in sr_levels:
                if prev_c <= top and c > top:
                    res_broken[i] = 1.0
                if prev_c >= bot and c < bot:
                    sup_broken[i] = 1.0

    result["src_nearest_top"] = nearest_top
    result["src_nearest_bot"] = nearest_bot
    result["src_num_channels"] = num_channels
    result["src_in_channel"] = in_channel
    result["src_res_broken"] = res_broken
    result["src_sup_broken"] = sup_broken

    return result