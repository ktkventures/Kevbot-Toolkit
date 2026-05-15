# Phase E — Flat-File Observable Bars Setup

This document covers the operational setup for Phase E: ingesting Polygon's daily trades flat file, rebuilding 1-second observable bars from `sip_timestamp`, and serving them to the Admin > Parity > Ticks tab.

## What ships in Phase E

- `src/flat_file_ingestion.py` — streaming ingestion module (boto3 + gzip CSV + bar builder + Supabase upsert + retention purge).
- `src/migrations/polygon_observable_bars.sql` — table schema.
- `src/api/routers/admin_parity.py` — new `GET /api/admin/parity/observable` endpoint.
- `frontend/src/charts/ParityObservableComparison.tsx` — three-pane Cache/Observable/Settled view.
- `frontend/src/views/AdminParityPage.tsx` — Ticks tab wired up.

## One-time setup steps (Kevin)

### 1. Apply the SQL migration

Run against your Supabase Postgres instance:

```sql
-- contents of src/migrations/polygon_observable_bars.sql
```

(Supabase SQL editor → paste the file → run.) Creates the `polygon_observable_bars` table with indexes.

### 2. Configure Railway env vars

On the API service (and the worker, if you want the ingestion to run from there):

| Variable | Default | Notes |
|---|---|---|
| `POLYGON_S3_ACCESS_KEY` | *required* | From your Polygon dashboard → S3 credentials |
| `POLYGON_S3_SECRET_KEY` | *required* | Same source |
| `POLYGON_S3_ENDPOINT` | `https://files.polygon.io` | Only set if Polygon changes the URL |
| `POLYGON_S3_BUCKET` | `flatfiles` | Same |
| `FLAT_FILE_SYMBOLS` | `SPY,TSLA` | Comma-separated tickers. Add more later. |
| `FLAT_FILE_RETENTION_DAYS` | `7` | Bars older than this get purged daily |

### 3. Add boto3 to requirements

```
echo "boto3>=1.34" >> requirements.txt
```

(Already imported lazily inside `_build_s3_client` so the rest of the codebase doesn't break if it's absent — but the cron will fail without it.)

### 4. Schedule the daily cron on Railway

Railway → service → Cron Jobs → add:

| Field | Value |
|---|---|
| **Schedule** | `30 17 * * 1-5` (Mon-Fri at 17:30 UTC = 12:30 PM ET during EDT; adjust to `30 18 * * 1-5` during EST) |
| **Command** | `cd /app/clients/KevBot_Toolkit/RoR_Trader/src && python -m flat_file_ingestion` |

Why 12:30 PM ET: Polygon publishes day-N's flat file at ~11 AM ET on day N+1. We add buffer for upstream delays.

### 5. Manual backfill (optional, for immediate testing)

To populate the table for the last 7 days without waiting for the cron:

```bash
# From /app/clients/KevBot_Toolkit/RoR_Trader/src on Railway shell:
for d in 2026-05-08 2026-05-09 2026-05-12 2026-05-13 2026-05-14; do
  python -m flat_file_ingestion "$d"
done
```

Each day takes a few minutes (download + filter + bucket + upsert). Expect ~SPY ~700k trades/day → ~23k 1-sec bars per day.

## Verifying it works

After ingestion runs:

```sql
SELECT ticker, COUNT(*) AS bars, MIN(sip_second_ts) AS oldest, MAX(sip_second_ts) AS newest
FROM polygon_observable_bars
GROUP BY ticker;
```

Should show SPY and TSLA each with thousands of bars covering recent days.

UI verification:
1. Navigate to `/admin/parity`
2. Select a SPY strategy
3. Set window to a recent day's RTH
4. Click the **Ticks** tab
5. Three-pane comparison renders: Cache | Observable | Settled

If "Observable" pane reports "No observable data" → check FLAT_FILE_SYMBOLS includes your test symbol and the cron has run for that date.

## Recorded scoping decisions (2026-05-14)

| Decision | Value | Why |
|---|---|---|
| Initial symbol scope | SPY + TSLA | Diagnostic scope; expand once value confirmed |
| Retention window | 7 days | Bounds storage; daily purge runs after ingest |
| Bar granularity stored | 1-second | Aggregation up to any TF on read; raw resolution preserved |
| Storage estimate | ~40 MB / week per symbol | Negligible vs Supabase Pro's 8GB included tier |
| Cron schedule | Weekdays 12:30 PM ET | After Polygon publishes day-N flat file ~11 AM ET on day N+1 |
| Expansion criteria | Phase G outcome (if live engine confirmed faithful via Cache≈Observable, may not need broader scope; if gaps found, expand to symbols-where-gaps-bite) | TBD by what Phase E reveals |

## What this enables

Once Phase E is running, you can answer this question per symbol/window:

> **Did our live engine's `live_bars` writes match what was actually emitted to subscribers in real time?**

Cache ≈ Observable: yes, live engine is faithful. Algo-vs-bt divergence is downstream of data and Phase G is unnecessary.

Cache ≠ Observable: no, our WS handling has gaps/drops. Phase G fixes the WS pipeline rather than band-aiding via cache backfill.

This decides whether option β (REST backfill) is even the right architectural move.
