"""Byte-identity check for the compute_secondary_columns refactor: compute a SECONDARY-gated
strategy's from-cold trades and print count + a stable hash of the keyed trades. Run once with
the refactor stashed (before) and once applied (after); identical hash ⟹ byte-identical.
"""
import os, sys, hashlib
os.environ.setdefault("RORT_SECONDARY_TF_SNAPSHOT", "0")  # force the resample path (no snapshot)
os.environ["USE_DB"] = "true"
from dotenv import load_dotenv
load_dotenv(".env", override=True)
import logging; logging.basicConfig(level=logging.ERROR)
from datetime import datetime, timedelta, timezone
import db
ADMIN = "19d47e46-f718-49a6-af32-5f5407f5b170"
db.set_admin_user_context(ADMIN)
import pack_registry; pack_registry.scan_and_load_all()
import pandas as pd
from db import get_strategy_by_id_admin
from services import prepare_strategy_window_df, get_secondary_tf_map
import general_packs as gp
from unified_engine import run_unified_backtest
from trade_snapshot import KEY_FIELDS, VAL_FIELDS

sids = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [267]
T0 = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
T_end = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)
T_ref = T0 - timedelta(days=10)
gen = gp.get_enabled_general_packs(gp.load_general_packs())

for sid in sids:
    strat = get_strategy_by_id_admin(sid, ADMIN)
    model = strat.get("backtest_model")
    df = prepare_strategy_window_df(strat, T_ref, T_end, warmup_bars=300, data_feed="sip",
                                    model_override=model, no_backfill=False, persist_snapshot=False)
    stm = get_secondary_tf_map(df) or None
    tdf, _ = run_unified_backtest(df, strat, general_packs=gen, secondary_tf_map=stm,
                                  include_open_position=False, last_bar_partial=False)
    keys = []
    if tdf is not None and len(tdf):
        for t in tdf.to_dict("records"):
            keys.append("|".join(str(t.get(f)) for f in KEY_FIELDS) + "##" +
                        "|".join(str(t.get(f)) for f in VAL_FIELDS))
    keys.sort()
    h = hashlib.md5("\n".join(keys).encode()).hexdigest()
    seccols = sorted(c for c in df.columns if "__" in c and not c.startswith("_spec_"))
    print(f"sid={sid} trades={len(keys)} hash={h} sec_cols={seccols}")
