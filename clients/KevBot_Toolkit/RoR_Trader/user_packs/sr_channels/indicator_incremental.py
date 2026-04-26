"""Support/Resistance Channels — incremental indicator class.

Identifies pivot highs/lows over a rolling buffer, clusters them into
channels by proximity, ranks by strength (pivot count + bar-touch
count), and outputs the nearest channel boundaries plus breakout
flags. Mirrors the original batch algorithm exactly — same O(loopback²)
inner loop per bar — but reuses a bounded rolling buffer so memory
stays constant and the per-bar cost matches the batch (the batch is
already O(n × loopback²) overall).

Key invariants:
  - Pivots at bar i are "confirmed" only at bar i + prd (we need prd
    forward bars to verify the local extremum). The buffer therefore
    needs to hold at least `prd + loopback + 1` bars so we can both
    confirm the latest pivot and walk back `loopback` bars of pivots.
  - The cwidth calc uses the last 300 bars; buffer covers that too.

Triggers:
  - resistance_broken / support_broken: bar-close breakout of the
    nearest channel. Level columns (`src_nearest_top_prev` /
    `src_nearest_bot_prev`) drive the L-type intra-bar variants.
  - enter_sr_zone / exit_sr_zone: pure pattern transitions; only C/CC.
"""


def _isnan(x):
    return x != x


class SrChannelsIncremental:
    def __init__(self, **params):
        self.prd = int(params.get("pivot_period", 10))
        self.pivot_source = params.get("pivot_source", "High/Low")
        self.channel_w = int(params.get("channel_width_pct", 5))
        self.min_strength = int(params.get("min_strength", 1))
        self.max_num_sr = int(params.get("max_num_sr", 6))
        self.loopback = int(params.get("loopback", 290))

        # Buffer size: cover both the cwidth window (300) and the
        # pivot-history window (loopback + prd + 1). Add a few bars
        # of slack so trim-then-confirm always works.
        self._max_buf = max(300, self.loopback + self.prd * 2 + 2)

        # Parallel buffers — same length, trimmed together.
        self._highs: list = []
        self._lows: list = []
        self._closes: list = []
        self._opens: list = []
        # Pivot arrays — NaN where not a pivot, value where confirmed.
        self._pivot_highs: list = []
        self._pivot_lows: list = []

        # Output state from previous bar (for `_prev` columns and
        # for breakout detection's prev-close lookup).
        self._prev_nearest_top = float("nan")
        self._prev_nearest_bot = float("nan")
        self._prev_in_channel = 0.0

    def warmup(self, df) -> None:
        for _, row in df.iterrows():
            self.update_bar({
                "open": row.get("open", 0.0),
                "high": row.get("high", 0.0),
                "low": row.get("low", 0.0),
                "close": row.get("close", 0.0),
                "volume": row.get("volume", 0.0),
            })

    def _src_high(self, idx: int) -> float:
        if self.pivot_source == "High/Low":
            return self._highs[idx]
        return max(self._closes[idx], self._opens[idx])

    def _src_low(self, idx: int) -> float:
        if self.pivot_source == "High/Low":
            return self._lows[idx]
        return min(self._closes[idx], self._opens[idx])

    def _confirm_pivot_at(self, p: int) -> None:
        """Check whether the bar at buffer index p is a pivot.

        Called when we have at least `prd` bars after p available, so
        the (-prd, +prd) window centered on p is fully populated.
        Mirrors the original batch's pivot detection: src is the
        unique max/min in [p-prd, p+prd].
        """
        n = len(self._highs)
        lo_idx = max(0, p - self.prd)
        hi_idx = min(n - 1, p + self.prd)
        sh_p = self._src_high(p)
        sl_p = self._src_low(p)

        # Pivot high: src_high[p] is the unique max in window
        is_pivot_high = True
        max_count = 0
        for j in range(lo_idx, hi_idx + 1):
            sh_j = self._src_high(j)
            if sh_j > sh_p:
                is_pivot_high = False
                break
            if sh_j == sh_p:
                max_count += 1
        if is_pivot_high and max_count == 1:
            self._pivot_highs[p] = sh_p

        # Pivot low: src_low[p] is the unique min in window
        is_pivot_low = True
        min_count = 0
        for j in range(lo_idx, hi_idx + 1):
            sl_j = self._src_low(j)
            if sl_j < sl_p:
                is_pivot_low = False
                break
            if sl_j == sl_p:
                min_count += 1
        if is_pivot_low and min_count == 1:
            self._pivot_lows[p] = sl_p

    def _build_channels(self, current_idx: int):
        """Return ranked S/R channel list for the bar at `current_idx`.

        Output: list of (strength, top, bot) tuples, sorted desc by
        strength, capped to max_num_sr. Empty list if no channels.
        """
        n = len(self._highs)
        confirm_limit = current_idx - self.prd
        if confirm_limit < 0:
            return []

        # Collect confirmed pivots within the loopback window
        pv: list = []
        lookback_start = max(0, confirm_limit - self.loopback)
        for j in range(lookback_start, confirm_limit + 1):
            v = self._pivot_highs[j]
            if not _isnan(v):
                pv.append(v)
            v = self._pivot_lows[j]
            if not _isnan(v):
                pv.append(v)
        if not pv:
            return []

        # cwidth from the 300-bar range ending at current bar
        range_start = max(0, current_idx - 299)
        prdhighest = max(self._highs[range_start:current_idx + 1])
        prdlowest = min(self._lows[range_start:current_idx + 1])
        cwidth = (prdhighest - prdlowest) * self.channel_w / 100.0
        if cwidth <= 0:
            return []

        num_pivots = len(pv)

        # For each pivot, build a channel and score it.
        ch_strengths: list = [0] * num_pivots
        ch_his: list = [0.0] * num_pivots
        ch_los: list = [0.0] * num_pivots

        # Pre-compute bar-touch arrays for the loopback window
        lb = min(self.loopback, current_idx)
        touch_start = current_idx - lb
        # We'll use highs/lows[touch_start..current_idx]

        for k in range(num_pivots):
            lo = pv[k]
            hi = pv[k]
            pp_count = 0
            for m in range(num_pivots):
                cpp = pv[m]
                wdth = hi - cpp if cpp <= hi else cpp - lo
                if wdth <= cwidth:
                    if cpp < lo:
                        lo = cpp
                    if cpp > hi:
                        hi = cpp
                    pp_count += 20
            # Bar-touch count over the last `lb+1` bars
            touch_count = 0
            for idx in range(touch_start, current_idx + 1):
                h_idx = self._highs[idx]
                l_idx = self._lows[idx]
                if (h_idx <= hi and h_idx >= lo) or (l_idx <= hi and l_idx >= lo):
                    touch_count += 1
            ch_strengths[k] = pp_count + touch_count
            ch_his[k] = hi
            ch_los[k] = lo

        # Pick strongest non-overlapping channels
        used = [False] * num_pivots
        sr_levels: list = []
        for _ in range(min(10, self.max_num_sr)):
            best_val = -1
            best_idx = -1
            for k in range(num_pivots):
                if (not used[k]) and ch_strengths[k] > best_val \
                        and ch_strengths[k] >= self.min_strength * 20:
                    best_val = ch_strengths[k]
                    best_idx = k
            if best_idx < 0:
                break
            hh = ch_his[best_idx]
            ll = ch_los[best_idx]
            sr_levels.append((best_val, hh, ll))
            # Zero out overlapping
            for k in range(num_pivots):
                if (ch_his[k] <= hh and ch_his[k] >= ll) \
                        or (ch_los[k] <= hh and ch_los[k] >= ll):
                    used[k] = True
                    ch_strengths[k] = -1

        # Sort desc by strength, cap to max_num_sr
        sr_levels.sort(key=lambda x: -x[0])
        return sr_levels[:self.max_num_sr]

    def update_bar(self, bar: dict) -> dict:
        opn = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        # Append + sync all parallel arrays
        self._opens.append(opn)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._pivot_highs.append(float("nan"))
        self._pivot_lows.append(float("nan"))

        # Capture `_prev` outputs BEFORE recomputing for this bar
        out_prev_top = self._prev_nearest_top
        out_prev_bot = self._prev_nearest_bot

        n = len(self._highs)
        current_idx = n - 1

        # Confirm pivot at index (current - prd) if we have enough history
        confirm_idx = current_idx - self.prd
        # Only check if confirm_idx is in range AND we haven't already confirmed it.
        # The pivot arrays start as NaN so re-running confirm is safe — but to
        # match the original batch we only check once. We track via the buffer
        # length: when current_idx grows, the bar at confirm_idx becomes confirmable.
        if confirm_idx >= self.prd:
            self._confirm_pivot_at(confirm_idx)

        # Build channels for the current bar
        sr_levels = self._build_channels(current_idx)
        num_channels = float(len(sr_levels))

        nearest_top = float("nan")
        nearest_bot = float("nan")
        in_channel = 0.0
        if sr_levels:
            best_dist = float("inf")
            for _, top, bot in sr_levels:
                mid = (top + bot) / 2.0
                dist = abs(close - mid)
                if close <= top and close >= bot:
                    in_channel = 1.0
                if dist < best_dist:
                    best_dist = dist
                    nearest_top = top
                    nearest_bot = bot

        # Breakout detection — compare current close vs prev close against each S/R
        res_broken = 0.0
        sup_broken = 0.0
        if sr_levels and in_channel == 0.0 and len(self._closes) >= 2:
            prev_c = self._closes[-2]
            for _, top, bot in sr_levels:
                if prev_c <= top and close > top:
                    res_broken = 1.0
                if prev_c >= bot and close < bot:
                    sup_broken = 1.0

        # Triggers
        trig_resistance_broken = res_broken == 1.0
        trig_support_broken = sup_broken == 1.0
        trig_enter_zone = (in_channel == 1.0) and (self._prev_in_channel == 0.0)
        trig_exit_zone = (in_channel == 0.0) and (self._prev_in_channel == 1.0)

        # Save state for next bar (and the `_prev` columns)
        self._prev_nearest_top = nearest_top
        self._prev_nearest_bot = nearest_bot
        self._prev_in_channel = in_channel

        # Trim buffer to max size — drop the oldest entry from all parallel arrays
        if n > self._max_buf:
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._opens.pop(0)
            self._pivot_highs.pop(0)
            self._pivot_lows.pop(0)

        return {
            # Indicator columns
            "src_nearest_top": nearest_top,
            "src_nearest_bot": nearest_bot,
            "src_num_channels": num_channels,
            "src_in_channel": in_channel,
            "src_res_broken": res_broken,
            "src_sup_broken": sup_broken,
            "src_nearest_top_prev": out_prev_top,
            "src_nearest_bot_prev": out_prev_bot,
            # Trigger booleans (engine reads via user-pack pickup loop)
            "src_resistance_broken": bool(trig_resistance_broken),
            "src_support_broken": bool(trig_support_broken),
            "src_enter_sr_zone": bool(trig_enter_zone),
            "src_exit_sr_zone": bool(trig_exit_zone),
        }
