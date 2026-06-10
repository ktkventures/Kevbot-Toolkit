/**
 * spliceAlertLens — reconstruct the `ws_rest_spliced` live view for the Gate
 * Parity "Alert Lens" by combining the first-write and latest cache panes.
 *
 * The live model is ~99% REST (corrected) with a short WS "tip": each bar
 * shows its first-write (WS, decision-time) values for `grace` seconds after
 * close, then heals to the latest (REST-corrected) values. We reproduce that
 * AS OF the replay scrub head: bars whose time is within `grace` of the scrub
 * head use first-write values (the tip); everything older uses latest (the
 * REST body). The gate-ribbon heatmap (a secondary-TF interpreter) uses the
 * longer `graceSecondary` (30s); all other panes use `gracePrimary`.
 *
 * Both `firstPanes` and `latestPanes` are built from the SAME cache rows
 * (live_bars) — they differ only in which columns (first_* vs latest) were
 * used — so they share timestamps and we splice point-for-point by time.
 *
 * Caveat (surfaced on the tab): tip indicator VALUES come from the
 * "all-first-write" computation, not a per-bar recompute on the spliced bar
 * set. For liquid symbols WS≈REST (~$0.01) so this is negligible; the
 * fully-faithful backend recompute is deferred for interactive speed.
 */
import type { PaneConfig } from '@/charts/SyncedChartPane';

function toUnixSec(t: any): number {
  if (t == null) return 0;
  if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : t;
  return Math.floor(new Date(t).getTime() / 1000);
}

export interface SpliceGrace {
  /** WS-tip window for the primary candles + their indicators (live: ~4s). */
  gracePrimary: number;
  /** WS-tip window for the cross-TF gate ribbon / heatmap (live: ~30s). */
  graceSecondary: number;
}

/**
 * Returns a new alert-lens pane array spliced as of `scrubHead` (Unix sec).
 * Falls back to `latestPanes` when the scrub head isn't set.
 */
export function spliceAlertPanes(
  firstPanes: PaneConfig[],
  latestPanes: PaneConfig[],
  scrubHead: number,
  { gracePrimary, graceSecondary }: SpliceGrace,
): PaneConfig[] {
  if (!Array.isArray(latestPanes) || latestPanes.length === 0) return latestPanes;
  if (!Array.isArray(firstPanes) || firstPanes.length === 0) return latestPanes;
  if (!isFinite(scrubHead) || scrubHead <= 0) return latestPanes;

  const firstById = new Map(firstPanes.map((p) => [p.id, p]));

  return latestPanes.map((latestPane) => {
    const firstPane = firstById.get(latestPane.id);
    if (!firstPane) return latestPane;
    const grace = latestPane.id === 'heatmap' ? graceSecondary : gracePrimary;
    const boundary = scrubHead - grace; // bars with time > boundary are the WS tip

    const series = latestPane.series.map((latestSer: any, i: number) => {
      const firstSer: any = firstPane.series[i];
      if (!firstSer || !Array.isArray(latestSer.data)) return latestSer;

      // Index first-write points by time so we can swap tip bars O(1).
      const firstByTime = new Map<number, any>();
      for (const d of firstSer.data || []) firstByTime.set(toUnixSec(d.time ?? d.timestamp), d);

      const data = latestSer.data.map((d: any) => {
        const t = toUnixSec(d.time ?? d.timestamp);
        if (t > boundary && t <= scrubHead) {
          const fd = firstByTime.get(t);
          if (fd) return fd; // WS decision-time value (the tip)
        }
        return d; // REST-corrected value (the body)
      });
      return { ...latestSer, data };
    });

    return { ...latestPane, series };
  });
}
