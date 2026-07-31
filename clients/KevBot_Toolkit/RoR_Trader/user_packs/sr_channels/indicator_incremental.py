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


# =====================================================================
# FAITHFUL PINE PORT — board #73 step 4.
#
# Selected at pack-load time by the manifest `flag_variants` entry for
# RORT_SR_CHANNELS_PINE (default OFF -> the legacy class above is used
# and nothing here runs). Every method carries the Pine line numbers it
# mirrors; see docs/_active/Audit_SR_Channels_vs_Pine.md for the
# clause-by-clause mapping and the measured gap this closes.
#
# The four load-bearing corrections vs. the legacy class, all named in
# the audit:
#   §2.1 pivot detection -> TradingView's ASYMMETRIC tie-break, not
#        "unique extremum over the whole window" (which is a strict
#        subset: 18% of pivot highs missed at a $0.01 tick).
#   §2.2 pivot array is NEWEST-FIRST (PINE L49 array.unshift). The
#        channel-growing loop is order-dependent, so this is not cosmetic.
#   §2.3 ONE value unshifted per confirmation bar. When a high and a low
#        confirm on the same bar the Pine DISCARDS THE LOW.
#   §2.4 the channel set is `var` (PERSISTENT) and is recomputed ONLY on
#        a bar that confirms a pivot (~6% of bars). Between those the
#        levels are FROZEN and price moves against a standing line. The
#        legacy class rebuilt every bar, so its levels slid under the
#        price mid-cross — the single biggest source of the ~1-in-5
#        spurious break signals, and the reason this port is 5-7x cheaper.
# =====================================================================


class SrChannelsPineIncremental:
    """Faithful port of reference-indicators/sr_channels.pine (v6).

    LonesomeTheBlue, "Support Resistance Channels". Behavioural
    equivalence with the Pine is the contract; see
    src/test_sr_channels_pine_rebuild.py for the differential test
    against tools/sr_channels_pine_reference.py.
    """

    # PINE L41-42: ta.highest(300)/ta.lowest(300) are `na` until 300 bars
    # exist, so the Pine emits NO channels before then. Declared, not
    # silent — the legacy class used a partial window from bar 0 (§2.5).
    WARMUP_BARS = 300

    def __init__(self, **params):
        # PINE L6-L11.
        self.prd = int(params.get("pivot_period", 10))
        self.pivot_source = params.get("pivot_source", "High/Low")
        self.channel_w = int(params.get("channel_width_pct", 5))
        self.min_strength = int(params.get("min_strength", 1))
        # PINE L10: `input.int(defval = 6, ...) - 1` — the input is
        # decremented AT READ TIME, then used as an inclusive loop bound
        # (L163/L174/L181 `for x = 0 to math.min(9, maxnumsr)`). Net
        # channel count is min(10, max_num_sr) either way; keeping the
        # Pine's own arithmetic so the loop bounds read the same.
        self.maxnumsr = int(params.get("max_num_sr", 6)) - 1
        self.loopback = int(params.get("loopback", 290))
        # §2.1: which side of ta.pivothigh/ta.pivotlow admits equality is
        # the ONE clause the audit could not settle offline. Exposed as a
        # parameter so it can be switched without a code change once a
        # live TradingView chart settles it (board #73 step 7). The
        # measured gap moves by under a point either way.
        self.pivot_tie = params.get("pivot_tie", "left_strict")

        # Buffer must cover, all ending at the current bar:
        #   - the 300-bar cwidth window (PINE L41-42)
        #   - the loopback+1 touch-count window (PINE L103)
        #   - the 2*prd+1 pivot window (PINE L33-34)
        # plus prd bars of slack so trim-then-confirm always works.
        self._max_buf = (
            max(300, self.loopback + 1, 2 * self.prd + 1) + self.prd + 2
        )

        self._opens: list = []
        self._highs: list = []
        self._lows: list = []
        self._closes: list = []
        self._trimmed = 0  # bars dropped off the front; buf[j] == bar j+_trimmed

        # PINE L46-47: `var pivotvals` / `var pivotlocs` — persistent,
        # NEWEST-FIRST. `_pivotlocs` holds ABSOLUTE confirmation bar
        # indices, exactly as Pine stores `bar_index` (L50), not the
        # index of the pivot bar itself.
        self._pivotvals: list = []
        self._pivotlocs: list = []

        # PINE L79: `var suportresistance = array.new_float(20, 0)`.
        # PERSISTENT — this is §2.4. Rewritten only inside the
        # `if bool(ph) or bool(pl)` block (L88).
        self._sr: list = [0.0] * 20
        self._stren: list = [0.0] * 10

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

    # ---- PINE L31-32 -------------------------------------------------
    def _src1(self, j):
        if self.pivot_source == "High/Low":
            return self._highs[j]
        return max(self._closes[j], self._opens[j])

    def _src2(self, j):
        if self.pivot_source == "High/Low":
            return self._lows[j]
        return min(self._closes[j], self._opens[j])

    # ---- PINE L33-34: ta.pivothigh / ta.pivotlow ---------------------
    def _pivot_at(self, ctr, hi_side):
        """§2.1 — asymmetric tie-break, NOT "unique over the window".

        One side rejects on equality, the other tolerates it. Both
        variants admit a SUPERSET of what the legacy class admitted.
        """
        if ctr - self.prd < 0 or ctr + self.prd > len(self._highs) - 1:
            return None
        get = self._src1 if hi_side else self._src2
        val = get(ctr)
        left_strict = (self.pivot_tie == "left_strict")

        for j in range(ctr - self.prd, ctr):            # left bars
            v = get(j)
            beats = v > val if hi_side else v < val
            if beats or (left_strict and v == val):
                return None
        for j in range(ctr + 1, ctr + self.prd + 1):    # right bars
            v = get(j)
            beats = v > val if hi_side else v < val
            if beats or ((not left_strict) and v == val):
                return None
        return val

    # ---- PINE L59-76: get_sr_vals ------------------------------------
    def _get_sr_vals(self, ind, cwidth):
        lo = self._pivotvals[ind]
        hi = lo
        numpp = 0
        for y in range(len(self._pivotvals)):
            cpp = self._pivotvals[y]
            wdth = (hi - cpp) if cpp <= hi else (cpp - lo)
            # NaN cwidth (pre-warmup) makes this False, exactly as an
            # `na` comparison is falsy in Pine. PINE L66.
            if wdth <= cwidth:
                if cpp <= hi:
                    lo = min(lo, cpp)
                else:
                    hi = max(hi, cpp)
                numpp += 20   # PINE L74: each pivot point counts 20
        return hi, lo, numpp

    # ---- PINE L88-143: rebuild the channel set (PIVOT BARS ONLY) -----
    def _rebuild_channels(self, i_rel, cwidth):
        npv = len(self._pivotvals)
        supres = []
        self._stren = [0.0] * 10

        for x in range(npv):                                  # L92-96
            hi, lo, strength = self._get_sr_vals(x, cwidth)
            supres.extend([float(strength), hi, lo])

        for x in range(npv):                                  # L99-107
            hh = supres[x * 3 + 1]
            ll = supres[x * 3 + 2]
            s = 0
            for y in range(0, self.loopback + 1):             # L103
                k = i_rel - y
                if k < 0:
                    continue      # Pine: high[y] is `na` -> condition false
                h_k = self._highs[k]
                l_k = self._lows[k]
                # L104 — `and` binds tighter than `or` in Pine, so this
                # is (high in band) or (low in band).
                if (h_k <= hh and h_k >= ll) or (l_k <= hh and l_k >= ll):
                    s += 1
            supres[x * 3] += s

        self._sr = [0.0] * 20                                 # L110
        src = 0
        for _x in range(npv):                                 # L113-136
            stv, stl = -1.0, -1
            for y in range(npv):                              # L116-120
                if supres[y * 3] > stv and \
                        supres[y * 3] >= self.min_strength * 20:
                    stv = supres[y * 3]
                    stl = y
            if stl < 0:
                continue
            hh = supres[stl * 3 + 1]
            ll = supres[stl * 3 + 2]
            self._sr[src * 2] = hh
            self._sr[src * 2 + 1] = ll
            self._stren[src] = supres[stl * 3]
            for y in range(npv):                              # L130-132
                if (supres[y * 3 + 1] <= hh and supres[y * 3 + 1] >= ll) or \
                        (supres[y * 3 + 2] <= hh and supres[y * 3 + 2] >= ll):
                    supres[y * 3] = -1.0
            src += 1
            if src >= 10:                                     # L135-136
                break

        # PINE L138-143 is a bubble sort over `stren` that is DEAD CODE:
        # the greedy pick above is non-increasing by construction, so
        # `stren[y] > stren[x]` never holds for y > x. It is also buggy
        # in the original (`stren[x]` is never assigned `tmp`), so
        # porting it would import an unreachable bug. Deliberately
        # omitted; the non-increasing invariant is asserted by
        # test_greedy_pick_is_non_increasing_so_pine_sort_is_dead.

    def update_bar(self, bar: dict) -> dict:
        self._opens.append(float(bar["open"]))
        self._highs.append(float(bar["high"]))
        self._lows.append(float(bar["low"]))
        self._closes.append(float(bar["close"]))

        i_rel = len(self._highs) - 1
        i_abs = i_rel + self._trimmed

        # `_prev` columns are last bar's values, captured before update.
        out_prev_top = self._prev_nearest_top
        out_prev_bot = self._prev_nearest_bot

        # ---- PINE L33-34 ---------------------------------------------
        ctr = i_rel - self.prd
        ph = pl = None
        if i_abs >= 2 * self.prd and ctr - self.prd >= 0:
            ph = self._pivot_at(ctr, True)
            pl = self._pivot_at(ctr, False)

        # ---- PINE L41-43: cwidth -------------------------------------
        # ta.highest/ta.lowest with no series argument default to
        # high/low. `na` until 300 bars exist -> no channels (§2.5).
        if i_abs + 1 < self.WARMUP_BARS:
            cwidth = float("nan")
        else:
            s = max(0, i_rel - 299)
            cwidth = (max(self._highs[s:i_rel + 1])
                      - min(self._lows[s:i_rel + 1])) * self.channel_w / 100.0

        is_pivot_bar = (ph is not None) or (pl is not None)

        # ---- PINE L48-56: keep pivot levels --------------------------
        if is_pivot_bar:
            # §2.3 / PINE L49 — ONE value per bar. `ph ? ph : pl`: when a
            # high AND a low confirm on the same bar the LOW IS DROPPED.
            # §2.2 / PINE L49-50 — array.unshift == insert at 0, so the
            # array is NEWEST-FIRST.
            self._pivotvals.insert(0, ph if ph is not None else pl)
            self._pivotlocs.insert(0, i_abs)
            # L51-56: drop pivots older than the loopback window.
            while self._pivotlocs and \
                    i_abs - self._pivotlocs[-1] > self.loopback:
                self._pivotvals.pop()
                self._pivotlocs.pop()

            # §2.4 — the WHOLE rebuild sits inside `if bool(ph) or
            # bool(pl)` (L88). On every other bar `_sr` is untouched.
            self._rebuild_channels(i_rel, cwidth)

        # ---- PINE L169-187: in-channel / broken ----------------------
        last = min(9, self.maxnumsr)
        close = self._closes[i_rel]

        not_in_a_channel = True                               # L172-177
        for x in range(0, last + 1):
            if close <= self._sr[x * 2] and close >= self._sr[x * 2 + 1]:
                not_in_a_channel = False

        res_broken = 0.0
        sup_broken = 0.0
        if not_in_a_channel and i_rel >= 1:                   # L180-187
            pc = self._closes[i_rel - 1]
            for x in range(0, last + 1):
                if pc <= self._sr[x * 2] and close > self._sr[x * 2]:
                    res_broken = 1.0
                if pc >= self._sr[x * 2 + 1] and close < self._sr[x * 2 + 1]:
                    sup_broken = 1.0

        in_channel = 0.0 if not_in_a_channel else 1.0

        # ---- Our own columns (no Pine counterpart — audit §1.9) ------
        # Kept so the column contract and `trigger_levels` are unchanged.
        # They are now derived from the FROZEN set, which is what makes
        # the L-type intra-bar level well-defined (§2.7): between pivot
        # bars `*_prev` and the live level are the same number.
        n_ch = 0
        ch_above = 0   # channels entirely ABOVE close (Pine: resistance)
        ch_below = 0   # channels entirely BELOW close (Pine: support)
        nearest_top = float("nan")
        nearest_bot = float("nan")
        best_dist = float("inf")
        for x in range(0, last + 1):
            top = self._sr[x * 2]
            bot = self._sr[x * 2 + 1]
            if top == 0.0:
                continue
            n_ch += 1
            # PINE L158 colour rule: resistance when BOTH bounds are
            # above close, support when BOTH are below, neither
            # otherwise (that is the grey `inch_col` case).
            if top > close and bot > close:
                ch_above += 1
            elif top < close and bot < close:
                ch_below += 1
            dist = abs(close - (top + bot) / 2.0)
            if dist < best_dist:
                best_dist = dist
                nearest_top = top
                nearest_bot = bot

        trig_enter_zone = (in_channel == 1.0) and (self._prev_in_channel == 0.0)
        trig_exit_zone = (in_channel == 0.0) and (self._prev_in_channel == 1.0)

        self._prev_nearest_top = nearest_top
        self._prev_nearest_bot = nearest_bot
        self._prev_in_channel = in_channel

        if len(self._highs) > self._max_buf:
            self._opens.pop(0)
            self._highs.pop(0)
            self._lows.pop(0)
            self._closes.pop(0)
            self._trimmed += 1

        return {
            "src_nearest_top": nearest_top,
            "src_nearest_bot": nearest_bot,
            "src_num_channels": float(n_ch),
            "src_in_channel": in_channel,
            "src_res_broken": res_broken,
            "src_sup_broken": sup_broken,
            "src_nearest_top_prev": out_prev_top,
            "src_nearest_bot_prev": out_prev_bot,
            "src_ch_above": float(ch_above),
            "src_ch_below": float(ch_below),
            "src_resistance_broken": bool(res_broken == 1.0),
            "src_support_broken": bool(sup_broken == 1.0),
            "src_enter_sr_zone": bool(trig_enter_zone),
            "src_exit_sr_zone": bool(trig_exit_zone),
        }
