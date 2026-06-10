'use client';

/**
 * SyncedChartPane — renders multiple synchronized lightweight-charts panes.
 *
 * M8.5 Phase B+ refactor: chart instance + series objects PERSIST across
 * re-renders. The setup useEffect is keyed by `paneStructureKey` (a stable
 * hash of pane id/height/series-types) so it only fires when the structure
 * genuinely changes. A separate data effect runs on every `panes` prop
 * change and pushes new data via `series.setData()` / `series.setMarkers()`
 * without recreating the chart. Visible time range is saved before any
 * structure rebuild and restored after, so the user's zoom/scroll position
 * survives toggling indicators / changing candle count.
 *
 * The forming-bar imperative update (M8.5 Phase B) is unchanged.
 */

import { memo, useEffect, useMemo, useRef, useCallback } from 'react';
import {
  createChart,
  type IChartApi, type ISeriesApi, type SeriesType, type Time,
} from 'lightweight-charts';
import { SessionHighlighting } from './plugins/SessionHighlighting';

/** Convert ISO 8601 string or number to Unix seconds. */
function toUnixTime(t: string | number): number {
  if (typeof t === 'number') return t;
  return Math.floor(new Date(t).getTime() / 1000);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SeriesConfig {
  type: 'Candlestick' | 'Line' | 'Histogram';
  data: Record<string, any>[];
  options?: Record<string, any>;
  markers?: Record<string, any>[];
  /** Opt-in metadata for the live forming-update effect. Series without a
   * `liveRole` are never touched by the forming-bar effect (markers, etc.). */
  liveRole?:
    | 'overlay_line'       // price-pane indicator line driven by formingIndicators[indicatorColumn]
    | 'oscillator_line'    // oscillator pane line
    | 'oscillator_hist'    // oscillator pane histogram (e.g., macd_hist)
    | 'heatmap_cell';      // heatmap cell toggled red/green by forming states
  /** Indicator column name for overlay/oscillator roles. */
  indicatorColumn?: string;
  /** Heatmap role fields: column (interpreter key or cross-TF col) + target state. */
  stateColumn?: string;
  neededState?: string;
  /** Numeric value to paint when the heatmap cell is live (n - idx in the pane). */
  heatmapValue?: number;
  /** Heatmap colors passed back to series.update() so met/unmet swap in place. */
  heatmapMetColor?: string;
  heatmapUnmetColor?: string;
  /** PB cells never receive live updates — their data already reflects the
   * previous-bar state which won't change intra-minute. Only CB cells tick. */
  fidelity?: 'PB' | 'CB';
  /** Whether this state column is cross-TF (look up from stateCrossTf rather
   * than states). Cross-TF indicators don't recompute intra-bar, so the
   * heatmap cell only ticks when the cross-TF state itself changes. */
  crossTf?: boolean;
}

export interface PrimitiveConfig {
  type: 'sessionHighlighting';
  seriesIndex?: number;
  options: any;
}

export interface PaneConfig {
  id: string;
  height: number;
  series: SeriesConfig[];
  primitives?: PrimitiveConfig[];
  /** Hide time axis on this pane (for heatmap/middle panes) */
  hideTimeAxis?: boolean;
}

export interface LiveCandle {
  time: string | number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface SyncedChartPaneProps {
  panes: PaneConfig[];
  /** Candle theme colors */
  upColor?: string;
  downColor?: string;
  upBorderColor?: string;
  gridLines?: boolean;
  /** Number of empty bars to show to the right of the last candle */
  rightOffset?: number;
  /**
   * M8.5: latest forming/completed bar pushed imperatively to the primary
   * pane's candlestick series via series.update(). Does NOT trigger a
   * re-render — the update runs in a dedicated useEffect that only
   * depends on this prop.
   */
  formingBar?: LiveCandle | null;
  /** Intra-bar tentative indicator values, keyed by column name. When
   * provided alongside formingBar, drives `series.update()` on every
   * overlay / oscillator series that declared a matching `liveRole` +
   * `indicatorColumn`. */
  formingIndicators?: Record<string, number> | null;
  /** Intra-bar tentative interpreter states, keyed by interpreter key.
   * Drives live heatmap cell color. */
  formingStates?: Record<string, string> | null;
  /** Cross-TF interpreter states (last-committed). Drives heatmap cells
   * whose `crossTf` flag is true. */
  formingStateCrossTf?: Record<string, string> | null;
  /** IANA or alias timezone string (e.g. 'US/Mountain', 'America/Denver').
   * Drives time-axis tick labels and crosshair tooltip. Falls back to
   * the user's browser locale when absent. */
  timezone?: string | null;
  /** M8.7 M5 (2026-05-04): Replay scrub mode.
   * When set (Unix seconds), every series' data + markers are sliced to
   * entries with time ≤ currentTime before being pushed to LWC. Forming-bar
   * updates are also suppressed so the scrub head is the sole source of
   * truth for the rightmost edge.  Null/undefined = normal live mode (all
   * existing callers unaffected). */
  currentTime?: number | null;
}

// Alias → IANA map matching StrategyDetailPage's TZ_MAP so everything
// formats consistently. Keep in sync if adding/removing entries.
const _TIMEZONE_ALIAS_TO_IANA: Record<string, string> = {
  'US/Eastern': 'America/New_York',
  'US/Central': 'America/Chicago',
  'US/Mountain': 'America/Denver',
  'US/Pacific': 'America/Los_Angeles',
  'US/Alaska': 'America/Anchorage',
  'US/Hawaii': 'Pacific/Honolulu',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Transform raw series data points to LWC format. */
function transformSeriesData(seriesCfg: SeriesConfig): any[] {
  return seriesCfg.data
    .map((d: any) => {
      const time = toUnixTime(d.time ?? d.timestamp) as Time;
      if (!isFinite(time as number)) return null;
      if (seriesCfg.type === 'Candlestick') {
        if (!isFinite(d.open) || !isFinite(d.high) || !isFinite(d.low) || !isFinite(d.close)) return null;
        const candle: any = {
          time, open: Number(d.open), high: Number(d.high),
          low: Number(d.low), close: Number(d.close),
        };
        if (d.color) candle.color = d.color;
        if (d.borderColor) candle.borderColor = d.borderColor;
        if (d.wickColor) candle.wickColor = d.wickColor;
        return candle;
      }
      if (!isFinite(d.value)) return null;
      return d.color ? { time, value: Number(d.value), color: d.color }
                     : { time, value: Number(d.value) };
    })
    .filter(Boolean)
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

/** Transform marker timestamps to LWC format and sort. */
function transformMarkers(markers: Record<string, any>[]): any[] {
  return markers
    .map((m: any) => ({ ...m, time: toUnixTime(m.time) as Time }))
    .filter((m: any) => isFinite(m.time as number))
    .sort((a: any, b: any) => (a.time as number) - (b.time as number));
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function SyncedChartPaneInner({
  panes,
  upColor,
  downColor,
  upBorderColor,
  gridLines = true,
  rightOffset = 3,
  formingBar = null,
  formingIndicators = null,
  formingStates = null,
  formingStateCrossTf = null,
  timezone = null,
  currentTime = null,
}: SyncedChartPaneProps) {
  // Scrub mode is active when currentTime is a finite Unix-seconds number.
  const scrubActive = currentTime != null && isFinite(currentTime);
  // Resolve the user's timezone to a canonical IANA name LWC/Intl accept.
  // Empty string disables the explicit tz (Intl falls back to browser).
  const resolvedTz = useMemo(() => {
    if (!timezone) return '';
    return _TIMEZONE_ALIAS_TO_IANA[timezone] || timezone;
  }, [timezone]);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartsRef = useRef<IChartApi[]>([]);
  const seriesRef = useRef<ISeriesApi<SeriesType>[][]>([]); // [paneIdx][seriesIdx]
  // Last data+markers refs we pushed per (paneIdx, seriesIdx) — lets us skip
  // setData/setMarkers when the upstream reference is unchanged, avoiding
  // redundant work when chartTabData re-runs for an unrelated reason (e.g.
  // alerts refetch updates markers but candle data is stale-referenced).
  const lastDataRef = useRef<unknown[][]>([]);
  const lastMarkersRef = useRef<unknown[][]>([]);
  const syncingRef = useRef(false);
  const primaryCandleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  // Candle count last pushed — lets the data effect detect a real data change
  // (vs a user pan) so it can re-fit a stale out-of-bounds visible range.
  const prevCandleCountRef = useRef<number>(-1);
  // Keep the latest forming bar in a ref so the data effect can re-apply it
  // after setData overwrites the candle series. Without this, every panes
  // re-ref (alerts refetch, live pill timer, etc.) snaps the last candle
  // back to its historical close until the next broadcast ~250ms later,
  // looking like the stream has stopped.
  const formingBarRef = useRef<LiveCandle | null>(null);
  // Track the most-recent timestamp applied to each (paneIdx, seriesIdx) so
  // we skip updates that would violate LWC's "Cannot update oldest data"
  // invariant when the forming bar timestamp hasn't advanced past whatever
  // the historical setData put there.
  const lastAppliedTimesRef = useRef<number[][]>([]);
  // Preserves user's zoom/scroll across rare structure rebuilds (toggle
  // Show Conditions, candle-count change beyond visible range, etc.).
  // Stored as a LOGICAL range (bar-index floats) rather than a time range
  // so it can exceed the data extent — critical for preserving scroll when
  // the user has panned into the right-offset area past the last candle.
  const lastLogicalRangeRef = useRef<{ from: number; to: number } | null>(null);
  // Tracks whether the data effect has run at least once for the current
  // chart instance. Reset by the structure effect on (re)create. Only
  // when false do we call fitContent on data load.
  const hasRenderedInitialDataRef = useRef(false);

  const getThemeColors = useCallback(() => {
    if (typeof window === 'undefined') return { text: '#DDD', grid: '#2B2B2B', border: '#333', up: '#4CAF50', down: '#f44336' };
    const s = getComputedStyle(document.documentElement);
    return {
      text: s.getPropertyValue('--text-secondary').trim() || '#DDD',
      grid: s.getPropertyValue('--border').trim() || '#2B2B2B',
      border: s.getPropertyValue('--border').trim() || '#333',
      up: s.getPropertyValue('--green').trim() || '#4CAF50',
      down: s.getPropertyValue('--red').trim() || '#f44336',
    };
  }, []);

  // M8.5 B+: stable structure hash. Changes only when pane count / heights /
  // series types / primitives genuinely change — NOT when only data changes.
  // Excludes data so live updates don't fire structure rebuilds.
  const paneStructureKey = useMemo(
    () => JSON.stringify(panes.map(p => ({
      id: p.id,
      height: p.height,
      hideTimeAxis: p.hideTimeAxis,
      seriesTypes: p.series.map(s => s.type),
      primitives: (p.primitives || []).map(pr => pr.type),
    }))),
    [panes],
  );

  // ---- STRUCTURE EFFECT ----
  // Creates chart instances + adds empty series. Runs only when structure
  // changes (rare) or theme colors change. Restores visible range from prior
  // chart so the user's zoom/scroll position survives the rebuild.
  useEffect(() => {
    if (!containerRef.current || panes.length === 0) return;

    const colors = getThemeColors();
    const up = upColor || colors.up;
    const down = downColor || colors.down;
    const borderUp = upBorderColor || up;
    const charts: IChartApi[] = [];
    const allSeries: ISeriesApi<SeriesType>[][] = [];
    let primaryCandle: ISeriesApi<'Candlestick'> | null = null;

    // Save scroll position before tearing down old charts
    if (chartsRef.current[0]) {
      try {
        const lr = chartsRef.current[0].timeScale().getVisibleLogicalRange();
        if (lr) lastLogicalRangeRef.current = { from: lr.from as number, to: lr.to as number };
      } catch { /* ignore */ }
    }

    // Tear down any prior charts first (effect re-runs replace, not append)
    for (const c of chartsRef.current) {
      try { c.remove(); } catch { /* ignore */ }
    }
    chartsRef.current = [];
    seriesRef.current = [];
    primaryCandleSeriesRef.current = null;
    // The new chart hasn't received data yet — flag it so the next data
    // effect fitContents (only on first load per chart instance).
    hasRenderedInitialDataRef.current = false;

    const container = containerRef.current;
    container.innerHTML = '';

    for (let pi = 0; pi < panes.length; pi++) {
      const pane = panes[pi];
      const div = document.createElement('div');
      div.style.width = '100%';
      container.appendChild(div);

      // Time formatters honor the user's configured timezone (Settings →
      // Display → Formatting). Tick formatter drives axis labels; time
      // formatter drives the crosshair tooltip. LWC delivers UTC seconds
      // as input, so Intl with the target timezone converts to local
      // display consistently across axis + tooltip.
      const axisTickFormatter = resolvedTz
        ? (time: number) => {
            const d = new Date((time as number) * 1000);
            return d.toLocaleString('en-US', {
              timeZone: resolvedTz,
              month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
              hour12: false,
            });
          }
        : undefined;
      const localization = resolvedTz
        ? {
            timeFormatter: (time: number) => {
              const d = new Date((time as number) * 1000);
              return d.toLocaleString('en-US', {
                timeZone: resolvedTz,
                year: 'numeric', month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit', second: '2-digit',
                hour12: false,
              });
            },
          }
        : undefined;

      const chart = createChart(div, {
        width: container.clientWidth,
        height: pane.height,
        layout: { background: { color: 'transparent' }, textColor: colors.text },
        grid: gridLines
          ? { vertLines: { color: colors.grid, style: 1 }, horzLines: { color: colors.grid, style: 1 } }
          : { vertLines: { visible: false }, horzLines: { visible: false } },
        crosshair: { mode: 0 },
        rightPriceScale: { borderColor: colors.border },
        timeScale: {
          borderColor: colors.border,
          timeVisible: true,
          secondsVisible: true,
          visible: !pane.hideTimeAxis,
          rightOffset: rightOffset,
          shiftVisibleRangeOnNewBar: true,
          ...(axisTickFormatter ? { tickMarkFormatter: axisTickFormatter } : {}),
        },
        ...(localization ? { localization } : {}),
      });
      charts.push(chart);

      // Create empty series instances. Data is pushed by the data effect.
      const paneSeries: ISeriesApi<SeriesType>[] = [];
      for (const seriesCfg of pane.series) {
        let chartSeries: ISeriesApi<SeriesType> | undefined;
        switch (seriesCfg.type) {
          case 'Candlestick':
            chartSeries = chart.addCandlestickSeries({
              upColor: up, downColor: down,
              borderUpColor: borderUp, borderDownColor: down,
              wickUpColor: borderUp, wickDownColor: down,
              ...seriesCfg.options,
            });
            // Capture the first Candlestick across ANY pane — not just
            // pane 0. When a heatmap pane is present, the price pane is
            // pane 1, and pinning capture to pane 0 silently drops the
            // forming-bar update for those strategies.
            if (!primaryCandle) {
              primaryCandle = chartSeries as ISeriesApi<'Candlestick'>;
            }
            break;
          case 'Line':
            chartSeries = chart.addLineSeries({
              priceLineVisible: false, lastValueVisible: false,
              ...seriesCfg.options,
            });
            break;
          case 'Histogram':
            chartSeries = chart.addHistogramSeries({
              priceLineVisible: false,
              ...seriesCfg.options,
            });
            break;
          default:
            continue;
        }
        paneSeries.push(chartSeries);

        // Price lines (set once on creation; rare)
        if ((seriesCfg as any).priceLines && chartSeries) {
          try {
            for (const pl of (seriesCfg as any).priceLines) {
              (chartSeries as any).createPriceLine({
                price: pl.price,
                color: pl.color || 'rgba(255,255,255,0.5)',
                lineWidth: pl.lineWidth || 1,
                lineStyle: pl.lineStyle || 2,
                axisLabelVisible: pl.axisLabelVisible !== false,
                title: pl.title || '',
              });
            }
          } catch (e) {
            console.warn('createPriceLine failed:', e);
          }
        }
      }
      allSeries.push(paneSeries);

      // Attach primitives
      if (pane.primitives) {
        for (const prim of pane.primitives) {
          const target = paneSeries[prim.seriesIndex || 0];
          if (!target) continue;
          try {
            if (prim.type === 'sessionHighlighting') {
              (target as any).attachPrimitive(new SessionHighlighting(prim.options));
            }
          } catch (e) {
            console.warn('Primitive attach failed:', e);
          }
        }
      }
    }

    chartsRef.current = charts;
    seriesRef.current = allSeries;
    primaryCandleSeriesRef.current = primaryCandle;
    // Reset data/markers tracking — new chart instance, nothing pushed yet.
    lastDataRef.current = allSeries.map(pane => new Array(pane.length).fill(null));
    lastMarkersRef.current = allSeries.map(pane => new Array(pane.length).fill(null));
    lastAppliedTimesRef.current = allSeries.map(pane => new Array(pane.length).fill(0));

    // ---- Cross-pane synchronization (logical range) ----
    // Uses LOGICAL range (bar-index floats) rather than time range. The
    // LWC time-range API silently clamps `setVisibleRange` to each chart's
    // data extent, which breaks two things on multi-pane charts:
    //   (a) user pans right past the last candle → clamp on the heatmap
    //       pane boomerangs back to the price pane, snapping the view
    //       to the last data point;
    //   (b) the forming-bar effect updates the price candle before the
    //       heatmap cells advance → cross-pane sync clamps mid-batch.
    // Logical ranges can exceed data extent (they live in bar-index space
    // including the rightOffset region), so both cases are avoided.
    if (charts.length > 1) {
      for (const chart of charts) {
        chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (syncingRef.current || !range) return;
          syncingRef.current = true;
          for (const other of charts) {
            if (other !== chart) {
              try { other.timeScale().setVisibleLogicalRange(range); } catch { /* ignore */ }
            }
          }
          syncingRef.current = false;
          lastLogicalRangeRef.current = { from: range.from as number, to: range.to as number };
        });
      }
    } else if (charts[0]) {
      charts[0].timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) lastLogicalRangeRef.current = { from: range.from as number, to: range.to as number };
      });
    }

    // Resize handler
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      if (w > 0) for (const c of charts) c.applyOptions({ width: w });
    };
    window.addEventListener('resize', handleResize);

    // Observe CONTAINER size changes too — `window.resize` alone misses
    // layout reflows (a sibling lens/1s pane mounting, a scrollbar toggling,
    // a flex/grid column resizing). Without this, a chart created during a
    // transient layout keeps its stale width and renders squished/tiny.
    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver !== 'undefined' && containerRef.current) {
      ro = new ResizeObserver(() => handleResize());
      ro.observe(containerRef.current);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (ro) { try { ro.disconnect(); } catch { /* ignore */ } }
      for (const c of charts) {
        try { c.remove(); } catch { /* ignore */ }
      }
      chartsRef.current = [];
      seriesRef.current = [];
      primaryCandleSeriesRef.current = null;
    };
  }, [paneStructureKey, upColor, downColor, upBorderColor, gridLines, rightOffset, getThemeColors, resolvedTz]);

  // ---- DATA EFFECT ----
  // Runs whenever panes prop changes. Pushes data to existing series via
  // setData / setMarkers — does NOT recreate the chart, so user's zoom/scroll
  // is preserved naturally (LWC v4: setData does not reset visible range;
  // chart auto-tracks the rightmost edge if user is at it, otherwise stays
  // wherever the user panned to).
  //
  // IMPORTANT: do NOT call setVisibleRange here. Doing so would force the
  // view to a snapshotted range every time data updates, overriding LWC's
  // natural "follow rightmost / preserve scroll" behavior. Visible-range
  // restore is only needed across STRUCTURE rebuilds (handled in the
  // structure effect).
  useEffect(() => {
    const charts = chartsRef.current;
    const allSeries = seriesRef.current;
    const wasFirstLoad = !hasRenderedInitialDataRef.current;
    if (charts.length === 0 || allSeries.length !== panes.length) return;

    for (let pi = 0; pi < panes.length; pi++) {
      const pane = panes[pi];
      const paneSeries = allSeries[pi];
      if (!paneSeries || paneSeries.length !== pane.series.length) continue;
      for (let si = 0; si < pane.series.length; si++) {
        const cfg = pane.series[si];
        const series = paneSeries[si];
        if (!series) continue;

        // Skip setData if cfg.data reference is unchanged since last call —
        // chartTabData memoization gives us stable references when the
        // underlying inputs haven't changed. This avoids re-pushing 200+
        // candle points to LWC every time an unrelated dep (e.g. alerts
        // refetch) triggers a chartTabData re-run.
        //
        // Scrub mode bypasses this cache: the same `cfg.data` reference is
        // re-used across scrubs, only the slice changes — so we always
        // re-push when scrubActive.
        const lastData = lastDataRef.current[pi]?.[si];
        const dataChanged = cfg.data !== lastData;
        if (dataChanged || scrubActive) {
          try {
            // M8.7 M5: in scrub mode, slice both data and markers to the
            // current scrub head. transformSeriesData handles transformation
            // post-slice so the LWC payload only contains points up to and
            // including currentTime.
            const sliceByTime = scrubActive
              ? <T extends { time?: any; timestamp?: any }>(arr: T[]) =>
                  arr.filter(d => toUnixTime(d.time ?? d.timestamp) <= (currentTime as number))
              : <T,>(arr: T[]) => arr;
            const slicedCfg: SeriesConfig = scrubActive
              ? { ...cfg, data: sliceByTime(cfg.data) }
              : cfg;
            const data = transformSeriesData(slicedCfg);
            series.setData(data);
            if (lastDataRef.current[pi]) lastDataRef.current[pi][si] = cfg.data;
            // Record the last historical timestamp so the forming-bar
            // effect knows the lower bound for its series.update() calls.
            if (data.length > 0) {
              const lastT = Number((data[data.length - 1] as any).time);
              if (isFinite(lastT) && lastAppliedTimesRef.current[pi]) {
                lastAppliedTimesRef.current[pi][si] = lastT;
              }
            }
            // Re-apply forming bar on the primary candle series — setData
            // just wiped whatever live update was painted there.  Skipped
            // entirely in scrub mode so the scrub head owns the right edge.
            if (!scrubActive
                && series === primaryCandleSeriesRef.current
                && formingBarRef.current) {
              const fb = formingBarRef.current;
              const ft = toUnixTime(fb.time);
              const lastHistMs = data.length > 0
                ? Number((data[data.length - 1] as any).time)
                : 0;
              if (isFinite(ft) && ft >= lastHistMs && isFinite(fb.open) && isFinite(fb.high) &&
                  isFinite(fb.low) && isFinite(fb.close)) {
                try {
                  series.update({
                    time: ft as Time,
                    open: Number(fb.open), high: Number(fb.high),
                    low: Number(fb.low), close: Number(fb.close),
                  });
                  if (lastAppliedTimesRef.current[pi]) {
                    lastAppliedTimesRef.current[pi][si] = ft;
                  }
                } catch { /* timestamp ordering violation, drop */ }
              }
            }
          } catch (e) {
            console.warn('setData failed:', e);
          }
        }

        // ALWAYS call setMarkers — the ref-skip optimization doesn't apply
        // here because (a) markers are cheap to push (handful per series),
        // and (b) the dedup was silently swallowing updates when cfg.markers
        // reference was unstable across chartTabData re-runs. Small cost,
        // guarantees correctness.
        try {
          const rawMarkers = cfg.markers || [];
          const slicedMarkers = scrubActive
            ? rawMarkers.filter(m => toUnixTime(m.time) <= (currentTime as number))
            : rawMarkers;
          const valid = transformMarkers(slicedMarkers);
          (series as any).setMarkers(valid);
        } catch (e) {
          console.warn('[SyncedChartPane] setMarkers failed for pane=%d series=%d type=%s: %o',
                       pi, si, cfg.type, e);
        }
      }
    }

    // First-time data load for this chart instance: either restore the
    // user's prior visible range (if we saved one before a structure
    // rebuild) or fitContent (truly fresh chart). Subsequent data updates:
    // do nothing — LWC preserves the user's view + auto-tracks the right
    // edge naturally.
    // Candle count in this data push (max across panes).
    let candleCount = 0;
    for (const p of panes) for (const s of p.series) {
      if (s.type === 'Candlestick') candleCount = Math.max(candleCount, (s.data || []).length);
    }

    if (wasFirstLoad && charts[0]) {
      const savedRange = lastLogicalRangeRef.current;
      // Restore the saved range ONLY if it still fits the new data. A
      // structure rebuild can arrive with a DIFFERENT bar count (the window
      // recomputed as a sibling lens loaded). Restoring a stale range that's
      // scrolled past / wider than the new data squishes the candles to one
      // side — the exact Gate Parity squish. Otherwise fit to content.
      const rangeFits = !!savedRange && candleCount > 0
        && savedRange.from < candleCount - 0.5
        && savedRange.to <= candleCount + 4;
      if (savedRange && rangeFits) {
        try { charts[0].timeScale().setVisibleLogicalRange(savedRange); }
        catch { try { charts[0].timeScale().fitContent(); } catch { /* ignore */ } }
      } else {
        try { charts[0].timeScale().fitContent(); } catch { /* ignore */ }
      }
      hasRenderedInitialDataRef.current = true;
    } else if (charts[0] && candleCount > 0) {
      // Stale-range guard (2026-06-09): re-fit when the candle COUNT changed
      // OR the current visible range is out of bounds (scrolled past / wider
      // than the data) — both squish the candles to one side (e.g. a sparse
      // 1s pane showing range [0,20] over 8 bars). A user pan WITHIN the data
      // never trips this, so panning is preserved.
      let needRefit = candleCount !== prevCandleCountRef.current;
      if (!needRefit) {
        try {
          const r = charts[0].timeScale().getVisibleLogicalRange();
          if (r && (r.from >= candleCount - 0.5 || r.to > candleCount + 4)) needRefit = true;
        } catch { /* ignore */ }
      }
      if (needRefit) { try { charts[0].timeScale().fitContent(); } catch { /* ignore */ } }
    }
    if (candleCount > 0) prevCandleCountRef.current = candleCount;
    // M8.7 M5: also re-run when scrub head moves so data slices forward.
  }, [panes, currentTime, scrubActive]);

  // ---- LIVE FORMING-BAR EFFECT ----
  // Primary candle + every overlay / oscillator / heatmap cell that declared
  // a `liveRole`. Non-candle series advance FIRST so every pane's data extent
  // reaches the forming-bar timestamp before the candle triggers LWC's
  // `shiftVisibleRangeOnNewBar` on the price pane. Without this ordering the
  // cross-pane sync listener propagates the price pane's advanced range to
  // panes whose data extent hasn't moved yet, LWC silently clamps it back
  // to their last historical bar, and the clamped range boomerangs to the
  // price pane — locking the visible range to the load-time window.
  //
  // The whole batch runs under `syncingRef.current = true` so any intra-batch
  // visible-range-change events the listener sees are ignored. After the
  // batch, every pane's data is aligned, so `shiftVisibleRangeOnNewBar` on
  // each chart auto-advances each pane independently and they stay in sync.
  useEffect(() => {
    formingBarRef.current = formingBar || null;
    // M8.7 M5: scrub mode owns the right edge — never apply live forming-bar
    // updates while scrubbing. The data effect explicitly skips the
    // primary-candle re-apply path too.
    if (scrubActive) return;
    if (!formingBar) return;
    const t = toUnixTime(formingBar.time);
    if (!isFinite(t)) return;

    const prevSyncing = syncingRef.current;
    syncingRef.current = true;
    try {
      // 1. Walk every live-capable series (overlays, oscillators, heatmap CB
      //    AND PB cells). PB cells repeat their last historical color so the
      //    heatmap pane's data extent keeps pace even when no CB cell exists.
      for (let pi = 0; pi < panes.length; pi++) {
        const pane = panes[pi];
        const paneSeries = seriesRef.current[pi];
        if (!paneSeries || paneSeries.length !== pane.series.length) continue;
        for (let si = 0; si < pane.series.length; si++) {
          const cfg = pane.series[si];
          const role = cfg.liveRole;
          if (!role) continue;

          const series = paneSeries[si];
          if (!series) continue;
          const lastT = lastAppliedTimesRef.current[pi]?.[si] ?? 0;
          if (t < lastT) continue;

          try {
            if (role === 'overlay_line' || role === 'oscillator_line' || role === 'oscillator_hist') {
              const col = cfg.indicatorColumn;
              if (!col || !formingIndicators) continue;
              const v = formingIndicators[col];
              if (v == null || !isFinite(v)) continue;
              if (role === 'oscillator_hist') {
                const histColor = v >= 0 ? 'rgba(76,175,80,0.6)' : 'rgba(244,67,54,0.6)';
                series.update({ time: t as Time, value: Number(v), color: histColor } as any);
              } else {
                series.update({ time: t as Time, value: Number(v) } as any);
              }
            } else if (role === 'heatmap_cell') {
              const col = cfg.stateColumn;
              const needed = cfg.neededState;
              const paneVal = cfg.heatmapValue ?? 1;
              const metColor = cfg.heatmapMetColor ?? 'rgba(76,175,80,0.8)';
              const unmetColor = cfg.heatmapUnmetColor ?? 'rgba(244,67,54,0.4)';
              if (!col || !needed) continue;

              let color: string;
              if (cfg.fidelity === 'PB') {
                // PB cells reflect the previous closed bar and don't receive
                // a fresh intra-bar state. Repeat the last historical color
                // at the forming-bar timestamp so the pane's data extent
                // advances without changing the visual.
                const last = cfg.data.length > 0 ? cfg.data[cfg.data.length - 1] as any : null;
                color = (last && typeof last.color === 'string') ? last.color : unmetColor;
              } else {
                const stateMap = cfg.crossTf ? formingStateCrossTf : formingStates;
                if (!stateMap) continue;
                const cur = stateMap[col];
                if (cur == null) continue;
                color = cur === needed ? metColor : unmetColor;
              }
              series.update({ time: t as Time, value: paneVal, color } as any);
            }
            if (lastAppliedTimesRef.current[pi]) {
              lastAppliedTimesRef.current[pi][si] = t;
            }
          } catch { /* drop silently */ }
        }
      }

      // 2. Primary candle LAST — every other pane's data extent is now at t,
      //    so LWC's auto-shift on the price pane will not produce a cross-pane
      //    sync clamp. OHLCV always comes straight off formingBar.
      const candle = primaryCandleSeriesRef.current;
      if (candle &&
          isFinite(formingBar.open) && isFinite(formingBar.high) &&
          isFinite(formingBar.low) && isFinite(formingBar.close)) {
        let candlePi = -1, candleSi = -1;
        for (let pi = 0; pi < seriesRef.current.length; pi++) {
          const paneSeries = seriesRef.current[pi];
          for (let si = 0; si < paneSeries.length; si++) {
            if (paneSeries[si] === candle) { candlePi = pi; candleSi = si; break; }
          }
          if (candlePi >= 0) break;
        }
        const lastT = candlePi >= 0
          ? (lastAppliedTimesRef.current[candlePi]?.[candleSi] ?? 0)
          : 0;
        if (t >= lastT) {
          try {
            candle.update({
              time: t as Time,
              open: Number(formingBar.open), high: Number(formingBar.high),
              low: Number(formingBar.low), close: Number(formingBar.close),
            });
            if (candlePi >= 0 && lastAppliedTimesRef.current[candlePi]) {
              lastAppliedTimesRef.current[candlePi][candleSi] = t;
            }
          } catch { /* LWC rejects out-of-order — drop silently */ }
        }
      }
    } finally {
      syncingRef.current = prevSyncing;
    }
  }, [formingBar, formingIndicators, formingStates, formingStateCrossTf, panes, scrubActive]);

  if (panes.length === 0) {
    return (
      <div style={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '13px' }}>
        No chart data
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: '100%' }} />;
}

// React.memo keeps SyncedChartPane from re-rendering on every parent render.
// Parent (StrategyDetailPage) re-renders ~1Hz from useLiveBar broadcasts —
// without memo, each one would invoke SyncedChartPane's render function,
// diff the virtual DOM, and re-fire any effects whose deps appear changed.
// With memo, we only re-render when an actual prop changes (panes,
// formingBar, theme colors, etc. — all of which are memoized upstream).
const SyncedChartPane = memo(SyncedChartPaneInner);
export default SyncedChartPane;

