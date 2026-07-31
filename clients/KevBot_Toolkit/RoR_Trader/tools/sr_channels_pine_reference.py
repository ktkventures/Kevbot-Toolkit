"""Faithful line-by-line transliteration of reference-indicators/sr_channels.pine.

AUDIT INSTRUMENT for board #73 — this is NOT the pack. It exists so the
clause-by-clause mapping can be *measured* rather than asserted: run it and
the shipped pack over identical bars and diff the behaviour.

Every method carries the Pine line numbers it mirrors. Deliberate fidelity
choices (including one original-source bug, kept and asserted-dead) are
marked PINE-BUG / PINE-QUIRK.
"""

import math

NAN = float("nan")


def _isnan(x):
    return x != x


class PineSRChannels:
    """Transliteration of LonesomeTheBlue 'Support Resistance Channels' v6."""

    def __init__(self, prd=10, ppsrc="High/Low", ChannelW=5, minstrength=1,
                 maxnumsr_input=6, loopback=290, pivot_tie="left_strict",
                 require_full_300=True):
        # Pine L6-L11. NOTE L10: the input is decremented by 1 at read time.
        self.prd = prd
        self.ppsrc = ppsrc
        self.ChannelW = ChannelW
        self.minstrength = minstrength
        self.maxnumsr = maxnumsr_input - 1          # PINE L10: `... ) - 1`
        self.loopback = loopback
        self.pivot_tie = pivot_tie
        self.require_full_300 = require_full_300

        self.o, self.h, self.l, self.c = [], [], [], []

        # PINE L46-47: `var` arrays — persist across bars, newest-first.
        self.pivotvals = []
        self.pivotlocs = []
        # PINE L79: `var suportresistance = array.new_float(20, 0)` — the
        # channel set is PERSISTENT and only rewritten on a pivot bar.
        self.suportresistance = [0.0] * 20
        self.stren = [0.0] * 10

        self.sort_ever_fired = False   # proves the L138-143 sort is dead code
        self.pivot_events = 0

    # ---- PINE L31-32 -------------------------------------------------
    def _src1(self, i):
        return self.h[i] if self.ppsrc == "High/Low" else max(self.c[i], self.o[i])

    def _src2(self, i):
        return self.l[i] if self.ppsrc == "High/Low" else min(self.c[i], self.o[i])

    # ---- PINE L33-34: ta.pivothigh / ta.pivotlow ---------------------
    # TradingView's tie-break is ASYMMETRIC (one side allows equality).
    # Which side is the ONE clause this audit could not verify offline, so
    # it is a parameter. Both variants are supersets of the shipped pack's
    # rule (strict-unique over the whole window) — see the audit doc.
    def _pivot_at(self, ctr, hi_side):
        n = len(self.h)
        if ctr - self.prd < 0 or ctr + self.prd > n - 1:
            return None
        val = self._src1(ctr) if hi_side else self._src2(ctr)
        get = self._src1 if hi_side else self._src2

        def worse(a, b):          # a beats b, in the direction we care about
            return a > b if hi_side else a < b

        left_strict = (self.pivot_tie == "left_strict")
        for j in range(ctr - self.prd, ctr):            # left bars
            v = get(j)
            if worse(v, val) or (left_strict and v == val):
                return None
        for j in range(ctr + 1, ctr + self.prd + 1):    # right bars
            v = get(j)
            if worse(v, val) or ((not left_strict) and v == val):
                return None
        return val

    # ---- PINE L59-76: get_sr_vals ------------------------------------
    def _get_sr_vals(self, ind, cwidth):
        lo = self.pivotvals[ind]
        hi = lo
        numpp = 0
        for y in range(len(self.pivotvals)):
            cpp = self.pivotvals[y]
            wdth = (hi - cpp) if cpp <= hi else (cpp - lo)
            if wdth <= cwidth:
                if cpp <= hi:
                    lo = min(lo, cpp)
                else:
                    hi = max(hi, cpp)
                numpp += 20
        return hi, lo, numpp

    # ---- PINE L80-86: changeit ---------------------------------------
    def _changeit(self, x, y):
        sr = self.suportresistance
        sr[y * 2], sr[x * 2] = sr[x * 2], sr[y * 2]
        sr[y * 2 + 1], sr[x * 2 + 1] = sr[x * 2 + 1], sr[y * 2 + 1]

    def update_bar(self, bar):
        self.o.append(float(bar["open"]))
        self.h.append(float(bar["high"]))
        self.l.append(float(bar["low"]))
        self.c.append(float(bar["close"]))
        i = len(self.h) - 1          # bar_index

        ctr = i - self.prd
        ph = self._pivot_at(ctr, True) if ctr >= 0 else None
        pl = self._pivot_at(ctr, False) if ctr >= 0 else None

        # ---- PINE L41-43: cwidth (ta.highest/ta.lowest default to high/low)
        if self.require_full_300 and i + 1 < 300:
            cwidth = NAN
        else:
            s = max(0, i - 299)
            cwidth = (max(self.h[s:i + 1]) - min(self.l[s:i + 1])) * self.ChannelW / 100.0

        # ---- PINE L48-56: keep pivot levels ---------------------------
        if ph is not None or pl is not None:
            # PINE-QUIRK L49: ONE value is unshifted per bar. If a high and a
            # low both confirm on the same bar, the LOW IS DISCARDED.
            self.pivotvals.insert(0, ph if ph is not None else pl)
            self.pivotlocs.insert(0, i)
            while self.pivotlocs and i - self.pivotlocs[-1] > self.loopback:
                self.pivotvals.pop()
                self.pivotlocs.pop()

        # ---- PINE L88-143: recompute S/R — ONLY on a pivot bar ---------
        if ph is not None or pl is not None:
            self.pivot_events += 1
            npv = len(self.pivotvals)
            supres = []
            self.stren = [0.0] * 10

            for x in range(npv):                                  # L92-96
                hi, lo, strength = self._get_sr_vals(x, cwidth)
                supres.extend([float(strength), hi, lo])

            for x in range(npv):                                  # L99-107
                hh = supres[x * 3 + 1]
                ll = supres[x * 3 + 2]
                s = 0
                for y in range(0, self.loopback + 1):
                    k = i - y
                    if k < 0:
                        continue                 # Pine: high[y] is na -> false
                    if (self.h[k] <= hh and self.h[k] >= ll) or \
                       (self.l[k] <= hh and self.l[k] >= ll):
                        s += 1
                supres[x * 3] += s

            self.suportresistance = [0.0] * 20                    # L110
            src = 0
            for _x in range(npv):                                 # L113-136
                stv, stl = -1.0, -1
                for y in range(npv):
                    if supres[y * 3] > stv and supres[y * 3] >= self.minstrength * 20:
                        stv = supres[y * 3]
                        stl = y
                if stl >= 0:
                    hh = supres[stl * 3 + 1]
                    ll = supres[stl * 3 + 2]
                    self.suportresistance[src * 2] = hh
                    self.suportresistance[src * 2 + 1] = ll
                    self.stren[src] = supres[stl * 3]
                    for y in range(npv):                          # L130-132
                        if (supres[y * 3 + 1] <= hh and supres[y * 3 + 1] >= ll) or \
                           (supres[y * 3 + 2] <= hh and supres[y * 3 + 2] >= ll):
                            supres[y * 3] = -1.0
                    src += 1
                    if src >= 10:
                        break

            for x in range(0, 9):                                 # L138-143
                for y in range(x + 1, 10):
                    if self.stren[y] > self.stren[x]:
                        # PINE-BUG: `stren[x]` is never assigned `tmp`. Kept
                        # verbatim; the greedy pick above is non-increasing so
                        # this branch is unreachable — asserted, not assumed.
                        self.sort_ever_fired = True
                        self.stren[y] = self.stren[x]
                        self._changeit(x, y)

        # ---- PINE L169-187: in-channel / broken -----------------------
        last = min(9, self.maxnumsr)
        not_in_a_channel = True
        for x in range(0, last + 1):
            if self.c[i] <= self.suportresistance[x * 2] and \
               self.c[i] >= self.suportresistance[x * 2 + 1]:
                not_in_a_channel = False

        res_broken = False
        sup_broken = False
        if not_in_a_channel and i >= 1:
            pc = self.c[i - 1]
            for x in range(0, last + 1):
                if pc <= self.suportresistance[x * 2] and self.c[i] > self.suportresistance[x * 2]:
                    res_broken = True
                if pc >= self.suportresistance[x * 2 + 1] and self.c[i] < self.suportresistance[x * 2 + 1]:
                    sup_broken = True

        n_ch = sum(1 for x in range(0, last + 1)
                   if self.suportresistance[x * 2] != 0.0)
        return {
            "in_channel": 0.0 if not_in_a_channel else 1.0,
            "res_broken": 1.0 if res_broken else 0.0,
            "sup_broken": 1.0 if sup_broken else 0.0,
            "num_channels": float(n_ch),
        }
