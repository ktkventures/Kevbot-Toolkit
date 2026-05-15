"""Polygon flat-file ingestion → observable bars (Phase E, 2026-05-14).

Daily Railway cron job:
  1. Connect to Polygon's S3-compatible flat-file endpoint.
  2. Download yesterday's trades file (or a specified date).
  3. Stream-decompress, filter to watchlist symbols.
  4. Bucket trades into 1-second observable bars using sip_timestamp
     as the bucket key.
  5. Upsert into Supabase `polygon_observable_bars` table.
  6. Purge bars older than 7 days (rolling window).

Usage:
  python -m flat_file_ingestion              # ingest yesterday
  python -m flat_file_ingestion 2026-05-13   # ingest specific date

Env vars required:
  POLYGON_S3_ACCESS_KEY  — from your Polygon dashboard
  POLYGON_S3_SECRET_KEY  — from your Polygon dashboard
  POLYGON_S3_ENDPOINT    — default 'https://files.polygon.io'
  POLYGON_S3_BUCKET      — default 'flatfiles'
  FLAT_FILE_SYMBOLS      — comma-separated tickers (default 'SPY,TSLA')
  FLAT_FILE_RETENTION_DAYS — default 7

  (Supabase creds are picked up from db.SUPABASE_URL / SERVICE_ROLE_KEY.)

Design notes:
  - The trades file is ~3-4 GB compressed (all tickers). We stream
    via boto3's get_object().Body and decompress with gzip.GzipFile,
    so memory usage stays bounded regardless of file size.
  - We filter line-by-line on `ticker == watchlist`. Building bars
    in-memory for our small watchlist (~2 symbols × ~700k trades each
    on SPY) is fine — ~5MB working set.
  - Bucketing: floor(sip_timestamp_nanos / 1e9) → second-aligned epoch
    timestamp. Per-second OHLCV: open = first trade's price, close =
    last, high/low = max/min, volume = sum of sizes.
  - Upsert via Supabase's `upsert()` with on_conflict='ticker,sip_second_ts'
    so re-ingestion of the same day is idempotent.
"""
from __future__ import annotations

import csv
import gzip
import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Standalone CLI usage: load .env from src/ so the script works
# without manual `source .env`. No-op on Railway (env vars come from
# the platform, not a file).
try:
    from dotenv import load_dotenv as _load_dotenv
    import os as _os
    # File now lives at src/flat_file_ingestion.py — .env sits next to it.
    _env_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), '.env')
    if _os.path.exists(_env_path):
        _load_dotenv(_env_path, override=False)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_S3_ENDPOINT = "https://files.polygon.io"
_DEFAULT_S3_BUCKET = "flatfiles"
_DEFAULT_SYMBOLS = "SPY,TSLA"
_DEFAULT_RETENTION_DAYS = 7
_UPSERT_BATCH_SIZE = 1000

# Hardcoded fallback for OHLC-exclusion derived from Polygon's canonical
# /v3/reference/conditions endpoint (snapshot 2026-05-14). Used only when
# the live API fetch fails at startup. Includes:
#   2  Average Price Trade
#   5  Bunched Sold Trade (updates H/L but not O/C; conservative exclude)
#   7  Cash Sale
#   10 Derivatively Priced
#   12 Form T / Extended Hours (Polygon REST excludes from OHLC consolidated)
#   13 Extended Hours (Sold Out Of Sequence)
#   15 Market Center Official Close
#   16 Market Center Official Open
#   20 Next Day
#   21 Price Variation Trade
#   22 Prior Reference Price
#   29 Seller
#   32 Sold (Out Of Sequence)
#   33 Sold (Out of Sequence) and Stopped Stock
#   37 Odd Lot Trade — main culprit for low-volume outlier prints
#   52 Contingent Trade
#   53 Qualified Contingent Trade
_FALLBACK_OHLC_EXCLUDED: set[int] = {
    2, 5, 7, 10, 12, 13, 15, 16, 20, 21, 22, 29, 32, 33, 37, 52, 53,
}
_FALLBACK_VOL_EXCLUDED: set[int] = {15, 16, 38}

# Filled in by _load_polygon_condition_rules() at first call. The dynamic
# fetch replaces the fallback so we always have the up-to-date list if
# Polygon revises it.
_OHLC_EXCLUDED_CACHE: set[int] | None = None
_VOL_EXCLUDED_CACHE: set[int] | None = None


def _load_polygon_condition_rules() -> tuple[set[int], set[int]]:
    """Fetch Polygon's canonical conditions list and derive the
    OHLC-exclusion + volume-exclusion sets. Caches the result for the
    process lifetime. Falls back to the hardcoded snapshot on failure.
    """
    global _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE
    if _OHLC_EXCLUDED_CACHE is not None and _VOL_EXCLUDED_CACHE is not None:
        return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE

    api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "POLYGON_API_KEY not set — using fallback condition exclusion list")
        _OHLC_EXCLUDED_CACHE = set(_FALLBACK_OHLC_EXCLUDED)
        _VOL_EXCLUDED_CACHE = set(_FALLBACK_VOL_EXCLUDED)
        return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE

    try:
        import httpx
        resp = httpx.get(
            "https://api.polygon.io/v3/reference/conditions",
            params={"asset_class": "stocks", "limit": 1000, "apiKey": api_key},
            timeout=30.0,
        )
        resp.raise_for_status()
        rules = resp.json().get("results", [])
        ohlc_excl: set[int] = set()
        vol_excl: set[int] = set()
        for c in rules:
            cid = c.get("id")
            if not isinstance(cid, int):
                continue
            consolidated = (c.get("update_rules") or {}).get("consolidated") or {}
            if (consolidated.get("updates_high_low") is False
                    or consolidated.get("updates_open_close") is False):
                ohlc_excl.add(cid)
            if consolidated.get("updates_volume") is False:
                vol_excl.add(cid)
        if not ohlc_excl:
            logger.warning("Polygon conditions API returned empty exclusion set — using fallback")
            ohlc_excl = set(_FALLBACK_OHLC_EXCLUDED)
            vol_excl = set(_FALLBACK_VOL_EXCLUDED)
        else:
            logger.info(
                "Loaded canonical Polygon condition rules: %d OHLC-excluded, %d vol-excluded",
                len(ohlc_excl), len(vol_excl))
            logger.info("OHLC-excluded codes: %s", sorted(ohlc_excl))
        _OHLC_EXCLUDED_CACHE = ohlc_excl
        _VOL_EXCLUDED_CACHE = vol_excl
    except Exception as e:
        logger.warning(
            "Polygon conditions API fetch failed (%s) — using fallback", e)
        _OHLC_EXCLUDED_CACHE = set(_FALLBACK_OHLC_EXCLUDED)
        _VOL_EXCLUDED_CACHE = set(_FALLBACK_VOL_EXCLUDED)
    return _OHLC_EXCLUDED_CACHE, _VOL_EXCLUDED_CACHE


def _is_eligible_for_ohlc(conditions_str: str | None) -> bool:
    """Parse the flat-file `conditions` column and return True iff the trade
    should update OHLC. Empty / unparseable conditions are treated as
    eligible (fail-open on parsing).

    Polygon flat-file format: comma-separated list of integers, wrapped in
    quotes if multiple values. e.g. `"12,37"`, `"14,12,37,41"`, or empty.

    Exclusion set is loaded from Polygon's canonical /v3/reference/conditions
    endpoint on first call (cached for process lifetime); falls back to a
    hardcoded snapshot if the API is unreachable.
    """
    if not conditions_str:
        return True
    s = conditions_str.strip().strip('"')
    if not s:
        return True
    try:
        codes = {int(c) for c in s.split(',') if c.strip()}
    except ValueError:
        return True  # fail-open on weird formats
    ohlc_excluded, _ = _load_polygon_condition_rules()
    return not (codes & ohlc_excluded)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, "").strip()
    return v if v else default


def _config() -> dict:
    return {
        "access_key": _env("POLYGON_S3_ACCESS_KEY"),
        "secret_key": _env("POLYGON_S3_SECRET_KEY"),
        "endpoint": _env("POLYGON_S3_ENDPOINT", _DEFAULT_S3_ENDPOINT),
        "bucket": _env("POLYGON_S3_BUCKET", _DEFAULT_S3_BUCKET),
        "symbols": [
            s.strip().upper() for s in
            (_env("FLAT_FILE_SYMBOLS", _DEFAULT_SYMBOLS) or "").split(",")
            if s.strip()
        ],
        "retention_days": int(_env("FLAT_FILE_RETENTION_DAYS",
                                    str(_DEFAULT_RETENTION_DAYS))),
    }


# ---------------------------------------------------------------------------
# Observable-bar builder
# ---------------------------------------------------------------------------

@dataclass
class _Bar:
    """1-second OHLCV bar accumulated from streaming trades."""
    second_ts: int         # epoch seconds (UTC)
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int

    def update(self, price: float, size: int) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += size
        self.trade_count += 1

    @classmethod
    def from_trade(cls, second_ts: int, price: float, size: int) -> "_Bar":
        return cls(
            second_ts=second_ts,
            open=price, high=price, low=price, close=price,
            volume=size, trade_count=1,
        )


def _build_observable_bars(
    rows: Iterable[dict],
) -> Iterator[tuple[str, _Bar]]:
    """Group flat-file trade rows into 1-second observable bars.

    Yields (ticker, bar) tuples as each second completes. Assumes input
    is ordered (or near-ordered) by sip_timestamp; out-of-order trades
    within the same second still get bucketed correctly because we key
    by second.
    """
    # Per-ticker current accumulating bar
    current: dict[str, _Bar] = {}
    filtered_conditions = 0
    filtered_corrections = 0

    for row in rows:
        ticker = row["ticker"]
        # Skip corrected/superseded trades — they're replaced by another row
        # with correction=0.
        if (row.get("correction") or "0").strip() not in ("0", ""):
            filtered_corrections += 1
            continue
        # Skip trades that don't qualify for OHLC per Polygon's eligibility
        # rules (Average Price Trade, Cash Sale, Form T excluded? No — Form T
        # IS kept since extended hours matters here; see EXCLUDED_POLYGON_CONDITIONS).
        if not _is_eligible_for_ohlc(row.get("conditions")):
            filtered_conditions += 1
            continue
        # sip_timestamp is in NANOSECONDS in Polygon's flat-file format.
        try:
            ns = int(row["sip_timestamp"])
        except (KeyError, ValueError, TypeError):
            continue
        sec_ts = ns // 1_000_000_000
        try:
            price = float(row["price"])
            size = int(float(row["size"]))
        except (KeyError, ValueError, TypeError):
            continue
        # Skip fractional-share trades (size < 1) — they're settlement
        # micro-prints that don't represent actionable market activity.
        if size < 1:
            continue

        bar = current.get(ticker)
        if bar is None:
            current[ticker] = _Bar.from_trade(sec_ts, price, size)
        elif bar.second_ts == sec_ts:
            bar.update(price, size)
        else:
            # Second rolled — emit prev, start new.
            yield (ticker, bar)
            current[ticker] = _Bar.from_trade(sec_ts, price, size)

    # Flush remaining bars.
    for ticker, bar in current.items():
        yield (ticker, bar)

    if filtered_conditions or filtered_corrections:
        logger.info(
            "trade filters: %d skipped (excluded conditions), %d skipped (corrections)",
            filtered_conditions, filtered_corrections)


# ---------------------------------------------------------------------------
# Flat-file streaming
# ---------------------------------------------------------------------------

def _build_s3_client(cfg: dict):
    """Construct a boto3 S3 client pointed at Polygon's flat-file endpoint."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as e:
        raise RuntimeError(
            "boto3 not installed — `pip install boto3` for flat-file "
            "ingestion") from e
    if not cfg["access_key"] or not cfg["secret_key"]:
        raise RuntimeError(
            "POLYGON_S3_ACCESS_KEY + POLYGON_S3_SECRET_KEY required "
            "(get them from your Polygon dashboard)")
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=BotoConfig(
            signature_version="s3v4",
            # Tuned for streaming 3-4GB files reliably:
            retries={"max_attempts": 5, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=600,  # 10 min — full file is ~9 min to stream
            tcp_keepalive=True,
        ),
    )


def _flat_file_key(date: datetime) -> str:
    """Polygon S3 key for a given trading day's trades file."""
    return (
        f"us_stocks_sip/trades_v1/"
        f"{date.year:04d}/{date.month:02d}/"
        f"{date.year:04d}-{date.month:02d}-{date.day:02d}.csv.gz"
    )


def _stream_filtered_rows(
    s3_client, bucket: str, key: str, symbols: set[str],
) -> Iterator[dict]:
    """Stream the gzipped CSV, yielding only rows matching the watchlist."""
    logger.info("Streaming s3://%s/%s (filter: %s)", bucket, key, sorted(symbols))
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    # Wrap streaming body in a buffered reader → gzip → text reader → csv.
    with gzip.GzipFile(fileobj=body) as gz:
        text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        kept = 0
        scanned = 0
        for row in reader:
            scanned += 1
            if scanned % 1_000_000 == 0:
                logger.info(
                    "scanned=%dM kept=%d", scanned // 1_000_000, kept)
            if row.get("ticker") in symbols:
                kept += 1
                yield row
        logger.info("stream done: scanned=%d kept=%d", scanned, kept)


# ---------------------------------------------------------------------------
# Upsert + purge
# ---------------------------------------------------------------------------

def _upsert_bars(sb_client, bars_iter: Iterator[tuple[str, _Bar]]) -> int:
    """Batch-upsert observable bars into Supabase. Returns inserted count.

    Uses on_conflict='ticker,sip_second_ts' for idempotent re-runs.
    """
    batch: list[dict] = []
    total = 0
    for ticker, bar in bars_iter:
        sec_iso = datetime.fromtimestamp(
            bar.second_ts, tz=timezone.utc).isoformat()
        batch.append({
            "ticker": ticker,
            "sip_second_ts": sec_iso,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "trade_count": bar.trade_count,
        })
        if len(batch) >= _UPSERT_BATCH_SIZE:
            sb_client.table("polygon_observable_bars").upsert(
                batch, on_conflict="ticker,sip_second_ts"
            ).execute()
            total += len(batch)
            batch = []
            if total % 10_000 == 0:
                logger.info("upserted %d bars", total)
    if batch:
        sb_client.table("polygon_observable_bars").upsert(
            batch, on_conflict="ticker,sip_second_ts"
        ).execute()
        total += len(batch)
    logger.info("upsert complete: %d bars", total)
    return total


def _purge_stale(sb_client, retention_days: int) -> int:
    """Delete bars older than `retention_days`. Returns deleted count."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    logger.info("purging polygon_observable_bars before %s", cutoff)
    res = sb_client.table("polygon_observable_bars") \
        .delete().lt("sip_second_ts", cutoff).execute()
    count = len(res.data) if res.data else 0
    logger.info("purge complete: deleted %d rows", count)
    return count


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _clean_date_for_symbols(sb_client, date: datetime, symbols: set[str]) -> int:
    """Delete all observable bars for a specific date + symbols. Used by
    --clean re-ingest path so seconds whose only trades were filtered
    don't linger as stale rows. Returns count of deleted rows.
    """
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    day_end = (date.replace(hour=0, minute=0, second=0, microsecond=0)
               + timedelta(days=1)).isoformat()
    total_deleted = 0
    for sym in symbols:
        res = sb_client.table("polygon_observable_bars").delete() \
            .eq("ticker", sym) \
            .gte("sip_second_ts", day_start) \
            .lt("sip_second_ts", day_end) \
            .execute()
        n = len(res.data) if res.data else 0
        total_deleted += n
        logger.info("--clean: deleted %d rows for %s on %s",
                    n, sym, date.strftime('%Y-%m-%d'))
    return total_deleted


def ingest_day(date_str: Optional[str] = None, clean: bool = False) -> dict:
    """Ingest one day's flat file. Returns summary dict.

    Args:
      date_str: 'YYYY-MM-DD'. Defaults to yesterday (UTC).
      clean: When True, delete all rows for `date_str` + the configured
        symbols BEFORE ingesting. Use this when changing filter rules
        — without it, seconds where ALL trades are now filtered would
        retain their old (bad) values via upsert idempotency.

    Returns:
      {'date': ..., 'symbols': [...], 'bars_inserted': int, 'purged': int,
       'elapsed_s': float}
    """
    cfg = _config()
    if not cfg["symbols"]:
        raise RuntimeError("FLAT_FILE_SYMBOLS env var must list >=1 symbol")
    if date_str:
        date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        date = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)

    symbols = set(cfg["symbols"])
    started = time.time()

    s3 = _build_s3_client(cfg)
    key = _flat_file_key(date)

    # Stream → filter → bar-build → upsert
    from db import get_admin_client
    sb = get_admin_client()

    cleaned = 0
    if clean:
        cleaned = _clean_date_for_symbols(sb, date, symbols)

    rows_iter = _stream_filtered_rows(s3, cfg["bucket"], key, symbols)
    bars_iter = _build_observable_bars(rows_iter)
    inserted = _upsert_bars(sb, bars_iter)

    # Purge stale rows (best-effort; ingest is the priority).
    try:
        purged = _purge_stale(sb, cfg["retention_days"])
    except Exception as e:
        logger.warning("purge failed (continuing): %s", e)
        purged = 0

    elapsed = time.time() - started
    summary = {
        "date": date.strftime("%Y-%m-%d"),
        "symbols": sorted(symbols),
        "cleaned_rows": cleaned,
        "bars_inserted": inserted,
        "purged": purged,
        "elapsed_s": round(elapsed, 1),
    }
    logger.info("ingest summary: %s", summary)
    return summary


def main() -> int:
    """CLI entry: `python -m flat_file_ingestion [YYYY-MM-DD] [--clean]`.

    Flags:
      --clean: delete existing rows for date+symbols before ingest. Use when
               filter rules changed so stale junk-conditioned rows don't
               survive the re-run via upsert idempotency.
    """
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    date_arg = args[0] if args else None
    clean = '--clean' in flags
    try:
        result = ingest_day(date_arg, clean=clean)
        print(result)
        return 0
    except Exception as e:
        logger.exception("ingest_day failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
