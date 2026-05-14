'use client';

/**
 * ParityDivergenceHeatmap — minute × lane grid showing where the three
 * execution models agree / disagree on entries.
 *
 * Columns: live | algo | backtest (in that order)
 * Rows: each minute that has at least one entry in any lane
 *
 * Cell colors:
 *   - green = entry present in this lane at this minute
 *   - gray  = entry NOT present
 *
 * Row tint:
 *   - green   = all 3 lanes have an entry within ±matchToleranceSec
 *   - orange  = 2 of 3 lanes agree
 *   - red     = only 1 lane has an entry (full disagreement)
 *
 * Click any row to see the per-lane records inline.
 */

import { useState } from 'react';
import type { ParitySnapshotAlert, ParitySnapshotTrade } from '@/hooks/queries/useAdminParity';

interface Props {
  liveEntries: ParitySnapshotAlert[];
  algoTrades: ParitySnapshotTrade[];
  btTrades: ParitySnapshotTrade[];
  matchToleranceSec?: number;
}

function toSec(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  return isNaN(ms) ? null : Math.floor(ms / 1000);
}

interface BucketRow {
  bucket_ts_sec: number;
  bucket_iso: string;
  live: ParitySnapshotAlert[];
  algo: ParitySnapshotTrade[];
  bt: ParitySnapshotTrade[];
}

function buildBuckets(
  live: ParitySnapshotAlert[],
  algo: ParitySnapshotTrade[],
  bt: ParitySnapshotTrade[],
  tolSec: number,
): BucketRow[] {
  type Item =
    | { ts: number; iso: string; lane: 'live'; payload: ParitySnapshotAlert }
    | { ts: number; iso: string; lane: 'algo'; payload: ParitySnapshotTrade }
    | { ts: number; iso: string; lane: 'bt'; payload: ParitySnapshotTrade };
  const items: Item[] = [];
  for (const a of live) {
    const ts = toSec(a.fill_ts);
    if (ts != null && a.fill_ts) items.push({ ts, iso: a.fill_ts, lane: 'live', payload: a });
  }
  for (const t of algo) {
    const ts = toSec(t.entry_fill_ts);
    if (ts != null && t.entry_fill_ts) items.push({ ts, iso: t.entry_fill_ts, lane: 'algo', payload: t });
  }
  for (const t of bt) {
    const ts = toSec(t.entry_fill_ts);
    if (ts != null && t.entry_fill_ts) items.push({ ts, iso: t.entry_fill_ts, lane: 'bt', payload: t });
  }
  items.sort((a, b) => a.ts - b.ts);

  const buckets: BucketRow[] = [];
  for (const it of items) {
    const last = buckets[buckets.length - 1];
    if (last && Math.abs(it.ts - last.bucket_ts_sec) <= tolSec) {
      if (it.lane === 'live') last.live.push(it.payload);
      else if (it.lane === 'algo') last.algo.push(it.payload);
      else last.bt.push(it.payload);
    } else {
      const row: BucketRow = {
        bucket_ts_sec: it.ts,
        bucket_iso: it.iso,
        live: [], algo: [], bt: [],
      };
      if (it.lane === 'live') row.live.push(it.payload);
      else if (it.lane === 'algo') row.algo.push(it.payload);
      else row.bt.push(it.payload);
      buckets.push(row);
    }
  }
  return buckets;
}

function laneScore(row: BucketRow): number {
  let n = 0;
  if (row.live.length > 0) n++;
  if (row.algo.length > 0) n++;
  if (row.bt.length > 0) n++;
  return n;
}

function rowTint(row: BucketRow): string {
  const n = laneScore(row);
  if (n === 3) return 'rgba(34, 197, 94, 0.10)';   // green
  if (n === 2) return 'rgba(249, 115, 22, 0.10)';  // orange
  return 'rgba(239, 68, 68, 0.12)';                // red
}

function cellColor(present: boolean): string {
  return present ? 'var(--green)' : 'var(--text-muted)';
}

export default function ParityDivergenceHeatmap({
  liveEntries, algoTrades, btTrades, matchToleranceSec = 60,
}: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const rows = buildBuckets(liveEntries, algoTrades, btTrades, matchToleranceSec);

  const matchCount = rows.filter((r) => laneScore(r) === 3).length;
  const partialCount = rows.filter((r) => laneScore(r) === 2).length;
  const aloneCount = rows.filter((r) => laneScore(r) === 1).length;
  const matchPct = rows.length > 0 ? ((matchCount / rows.length) * 100).toFixed(1) : '—';

  return (
    <div className="space-y-3">
      <div
        className="flex items-baseline gap-6 text-sm p-3 rounded"
        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
      >
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Total entries (any lane): </span>
          <strong>{rows.length}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>3-way match: </span>
          <strong style={{ color: 'var(--green)' }}>{matchCount}</strong>
          <span style={{ color: 'var(--text-muted)' }}> · {matchPct}%</span>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>2-of-3: </span>
          <strong style={{ color: 'var(--orange)' }}>{partialCount}</strong>
        </div>
        <div>
          <span style={{ color: 'var(--text-muted)' }}>Lane-only: </span>
          <strong style={{ color: 'var(--red)' }}>{aloneCount}</strong>
        </div>
      </div>

      <table className="text-xs w-full" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            <th className="text-left px-2 py-1.5">Time (UTC)</th>
            <th className="text-center px-2 py-1.5" style={{ color: 'var(--green)' }}>live</th>
            <th className="text-center px-2 py-1.5" style={{ color: 'var(--orange)' }}>algo</th>
            <th className="text-center px-2 py-1.5" style={{ color: '#3b82f6' }}>backtest</th>
            <th className="text-center px-2 py-1.5">match</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <>
              <tr
                key={`row-${idx}`}
                onClick={() => setExpanded(expanded === idx ? null : idx)}
                style={{
                  background: rowTint(row),
                  borderBottom: '1px solid var(--border)',
                  cursor: 'pointer',
                }}
              >
                <td className="px-2 py-1">{row.bucket_iso.slice(0, 19)}</td>
                <td className="px-2 py-1 text-center" style={{ color: cellColor(row.live.length > 0) }}>
                  {row.live.length > 0 ? '●' : '○'}
                </td>
                <td className="px-2 py-1 text-center" style={{ color: cellColor(row.algo.length > 0) }}>
                  {row.algo.length > 0 ? '●' : '○'}
                </td>
                <td className="px-2 py-1 text-center" style={{ color: cellColor(row.bt.length > 0) }}>
                  {row.bt.length > 0 ? '●' : '○'}
                </td>
                <td className="px-2 py-1 text-center">{laneScore(row)}/3</td>
              </tr>
              {expanded === idx && (
                <tr key={`exp-${idx}`}>
                  <td colSpan={5} style={{ background: 'var(--bg-input)', padding: 12 }}>
                    <div className="grid grid-cols-3 gap-3 text-xs">
                      <div>
                        <strong style={{ color: 'var(--green)' }}>live ({row.live.length})</strong>
                        {row.live.length === 0
                          ? <div style={{ color: 'var(--text-muted)' }}>—</div>
                          : row.live.map((a, i) => (
                              <div key={i} className="mt-1">
                                <div>fill: {a.fill_ts?.slice(11, 19)} {a.exec_type ? `(${a.exec_type})` : ''}</div>
                                <div>price: {a.price ?? '—'}</div>
                                {a.actual_price != null && <div>actual: {a.actual_price}</div>}
                                {a.behavior && <div>behavior: {a.behavior}</div>}
                              </div>
                            ))}
                      </div>
                      <div>
                        <strong style={{ color: 'var(--orange)' }}>algo ({row.algo.length})</strong>
                        {row.algo.length === 0
                          ? <div style={{ color: 'var(--text-muted)' }}>—</div>
                          : row.algo.map((t, i) => (
                              <div key={i} className="mt-1">
                                <div>fill: {t.entry_fill_ts?.slice(11, 19)} {t.exec_type ? `(${t.exec_type})` : ''}</div>
                                <div>entry: {t.entry_price ?? '—'} → exit: {t.exit_price ?? 'open'}</div>
                                {t.stop_price != null && <div>stop: {t.stop_price}</div>}
                                {t.hifi_resolved && <div style={{ color: 'var(--green)' }}>hifi ✓</div>}
                              </div>
                            ))}
                      </div>
                      <div>
                        <strong style={{ color: '#3b82f6' }}>backtest ({row.bt.length})</strong>
                        {row.bt.length === 0
                          ? <div style={{ color: 'var(--text-muted)' }}>—</div>
                          : row.bt.map((t, i) => (
                              <div key={i} className="mt-1">
                                <div>fill: {t.entry_fill_ts?.slice(11, 19)} {t.exec_type ? `(${t.exec_type})` : ''}</div>
                                <div>entry: {t.entry_price ?? '—'} → exit: {t.exit_price ?? 'open'}</div>
                                {t.stop_price != null && <div>stop: {t.stop_price}</div>}
                                {t.hifi_resolved && <div style={{ color: 'var(--green)' }}>hifi ✓</div>}
                              </div>
                            ))}
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
          {rows.length === 0 && (
            <tr>
              <td colSpan={5} className="text-center py-4" style={{ color: 'var(--text-muted)' }}>
                No entries in the selected window.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
