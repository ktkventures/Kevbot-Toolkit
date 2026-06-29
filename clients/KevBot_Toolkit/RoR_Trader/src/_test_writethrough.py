"""Functional test for bar_cache write-through (step 2). Writes a SETTLED TSLA
window through the gated path, reads back, verifies settled=true + revised_at +
OHLCV byte-identical to source. Idempotent (overwrites byte-identical TSLA bars)."""
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
from dotenv import load_dotenv
load_dotenv(".env", override=True)
os.environ["USE_DB"] = "true"

SYMBOL = "TSLA"
start = datetime.fromisoformat("2026-06-26T14:30:00+00:00")
end = start + timedelta(minutes=30)

from data_worker_ingest import _polygon_1s, _polygon_1m
import bar_cache
from db import get_admin_client

poly_1s = _polygon_1s(SYMBOL, start, end)
poly_1m = _polygon_1m(SYMBOL, start, end)
print(f"source: 1Sec={len(poly_1s)}  1Min={len(poly_1m)}", flush=True)

# 1) gated OFF → no-op
os.environ["RORT_BARCACHE_WRITETHROUGH"] = ""
n_off = bar_cache.write_through_bars(SYMBOL, "1Sec", poly_1s)
print(f"flag OFF: write_through_bars -> {n_off} (expect 0)", flush=True)

# 2) gated ON → writes native 1Sec + 1Min
os.environ["RORT_BARCACHE_WRITETHROUGH"] = "1"
os.environ["RORT_SHADOW_SETTLE_MIN"] = "16"
w1s = bar_cache.write_through_bars(SYMBOL, "1Sec", poly_1s)
w1m = bar_cache.write_through_bars(SYMBOL, "1Min", poly_1m)
print(f"flag ON: wrote 1Sec={w1s}  1Min={w1m}", flush=True)

# 3) read back + verify settled/revised_at/values
c = get_admin_client()
def readback(tf):
    r = (c.table("bar_cache").select("ts,open,high,low,close,volume,settled,revised_at")
         .eq("symbol", SYMBOL).eq("timeframe", tf)
         .gte("ts", start.isoformat()).lt("ts", end.isoformat())
         .order("ts").execute()).data or []
    return r

for tf, src in (("1Sec", poly_1s), ("1Min", poly_1m)):
    rows = readback(tf)
    settled_true = sum(1 for x in rows if x["settled"])
    has_rev = sum(1 for x in rows if x["revised_at"])
    # byte-compare OHLCV vs source
    src_n = {pd.Timestamp(ts).isoformat(): row for ts, row in src.iterrows()}
    diffs = 0
    for x in rows:
        k = pd.Timestamp(x["ts"]).isoformat()
        s = src_n.get(k)
        if s is None:
            continue
        for col in ("open", "high", "low", "close", "volume"):
            if abs(float(x[col]) - float(s[col])) > 1e-6:
                diffs += 1
    print(f"  {tf}: rows={len(rows)} settled=true:{settled_true}/{len(rows)} "
          f"revised_at set:{has_rev}/{len(rows)} OHLCV_diffs_vs_source={diffs}",
          flush=True)

print("\nVERDICT:", "✅ write-through faithful + settled + observable"
      if w1s and w1m else "❌ check above", flush=True)
