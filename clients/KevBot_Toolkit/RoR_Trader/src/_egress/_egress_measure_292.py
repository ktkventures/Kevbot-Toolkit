"""Board #292 D2: measure read_bars egress bytes AT THE SOCKET.
Production call shape: psycopg3, transaction pooler :6543, prepare_threshold=None,
autocommit=True, one connection per call (bar_cache.py:689)."""
import os, sys, time, json, statistics
sys.path.insert(0, "/tmp/claude-1000/-home-kevin-projects-Kevbot-engine/3215bb93-523e-4ea4-87e8-c6180fe1185e/scratchpad")
from dotenv import load_dotenv
load_dotenv("/home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/src/.env")
from urllib.parse import urlparse
import psycopg
from relay import Relay

raw = os.environ["SUPABASE_CONNECTION_STRING"]
u = urlparse(raw)
UP_HOST, UP_PORT = u.hostname, 6543          # method §3: transaction pooler
r = Relay(UP_HOST, UP_PORT)

def dsn_via_relay():
    netloc = f"{u.username}:{u.password}@127.0.0.1:{r.port}"
    return f"{u.scheme}://{netloc}{u.path}?sslmode=require"

SQL = ("select ts, open, high, low, close, volume from bar_cache "
       "where symbol=%s and timeframe=%s and ts between %s and %s order by ts")

def one_call(symbol, tf, start, end, do_query=True):
    """Exactly bar_cache.read_bars(): fresh connection per call."""
    r.reset()
    t0 = time.time()
    with psycopg.connect(dsn_via_relay(), connect_timeout=20,
                         prepare_threshold=None, autocommit=True) as conn:
        with conn.cursor() as cur:
            if do_query:
                cur.execute(SQL, (symbol, tf, start, end))
                rows = cur.fetchall()
            else:
                cur.execute("select 1")
                rows = cur.fetchall()
    r.wait_idle()
    dt = time.time() - t0
    return {"rows": len(rows), "rx": r.rx, "tx": r.tx, "conns": r.conns, "secs": round(dt, 3)}

if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    out = []
    for s in spec:
        res = one_call(s.get("sym"), s.get("tf"), s.get("start"), s.get("end"),
                       do_query=s.get("q", True))
        res["label"] = s["label"]
        out.append(res)
        print(json.dumps(res), flush=True)
    print("---")
    print(json.dumps(out))
