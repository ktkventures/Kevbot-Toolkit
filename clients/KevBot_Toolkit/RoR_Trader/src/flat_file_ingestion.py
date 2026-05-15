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

# Phase G (2026-05-15): canonical filter logic extracted to a shared
# module so ralph_engine.py uses the IDENTICAL rules. See
# src/polygon_conditions.py for the implementation.
from polygon_conditions import (
    _classify_eligibility,
    _load_polygon_condition_rules,
    _parse_conditions,
    FALLBACK_OHLC_EXCLUDED as _FALLBACK_OHLC_EXCLUDED,
    FALLBACK_VOL_EXCLUDED as _FALLBACK_VOL_EXCLUDED,
)


def _is_eligible_for_ohlc(conditions_str: str | None) -> bool:
    """Backward-compatible single-flag check (OHLC only). Used by the
    older callsite signature; new code should use _classify_eligibility."""
    ohlc_ok, _ = _classify_eligibility(conditions_str)
    return ohlc_ok


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

    def update(
        self,
        price: float,
        size: int,
        update_ohlc: bool = True,
        update_volume: bool = True,
    ) -> None:
        """Update bar from a single trade.

        Phase F.4: trades are now classified separately for OHLC vs
        volume eligibility per Polygon's actual rules. A vol-only
        trade (Form T, Odd Lot, etc.) contributes to volume but
        doesn't move H/L/C. A rare OHLC-only trade (Corrected
        Consolidated Close, etc.) moves prices but isn't summed
        into volume.
        """
        if update_ohlc:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
            self.close = price
        if update_volume:
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
    filtered_corrections = 0
    skipped_both = 0
    vol_only = 0

    for row in rows:
        ticker = row["ticker"]
        # Skip corrected/superseded trades — they're replaced by another row
        # with correction=0.
        if (row.get("correction") or "0").strip() not in ("0", ""):
            filtered_corrections += 1
            continue

        # Phase F.4: classify the trade. Polygon distinguishes OHLC vs
        # volume eligibility separately, so a Form T or Odd Lot trade
        # may not move price but DOES count for volume.
        ohlc_ok, vol_ok = _classify_eligibility(row.get("conditions"))
        if not ohlc_ok and not vol_ok:
            skipped_both += 1
            continue
        if not ohlc_ok and vol_ok:
            vol_only += 1

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
            if ohlc_ok:
                # Standard start-of-bar: full OHLC + vol from this trade.
                current[ticker] = _Bar.from_trade(sec_ts, price, size)
                if not vol_ok:
                    # OHLC-only edge case (extremely rare): undo the
                    # volume that from_trade added.
                    current[ticker].volume = 0
                    current[ticker].trade_count = 0
            elif vol_ok:
                # Vol-only first trade of a new second — emit a
                # placeholder bar with sentinel zero prices to signal
                # "no OHLC yet, just volume." Downstream we shouldn't
                # write these as OHLC (open/high/low/close all 0) so
                # skip emission later. Track in a separate accumulator
                # would be cleaner but rare enough to handle inline:
                # we DROP these and accept undercounting vol-only-first
                # seconds. A vol-only trade as the FIRST tick of a
                # second is the rare case (typically the open is from
                # a regular trade).
                pass
        elif bar.second_ts == sec_ts:
            bar.update(price, size, update_ohlc=ohlc_ok, update_volume=vol_ok)
        else:
            # Second rolled — emit prev, start new.
            yield (ticker, bar)
            if ohlc_ok:
                current[ticker] = _Bar.from_trade(sec_ts, price, size)
                if not vol_ok:
                    current[ticker].volume = 0
                    current[ticker].trade_count = 0
            else:
                # Vol-only first trade of new second — see comment above.
                # Don't start the new bar; wait for an OHLC-eligible trade.
                current.pop(ticker, None)

    # Flush remaining bars.
    for ticker, bar in current.items():
        yield (ticker, bar)

    logger.info(
        "trade filters: %d skipped (both ineligible), %d vol-only (kept for volume), "
        "%d skipped (corrections)",
        skipped_both, vol_only, filtered_corrections)


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
