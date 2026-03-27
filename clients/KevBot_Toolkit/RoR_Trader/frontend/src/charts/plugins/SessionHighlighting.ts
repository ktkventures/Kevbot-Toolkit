/**
 * SessionHighlighting plugin — draws colored background bands on the chart.
 * Ported from streamlit_lwc_fork for LWC v5.
 *
 * Usage: series.attachPrimitive(new SessionHighlighting({ ranges: [...] }))
 */

import type { Time } from 'lightweight-charts';

interface SessionRange {
  startTime: Time;
  endTime: Time;
  color: string;
}

interface SessionHighlightingOptions {
  ranges: SessionRange[];
}

export class SessionHighlighting {
  private _options: SessionHighlightingOptions;
  private _chart: any;
  private _series: any;

  constructor(options: SessionHighlightingOptions) {
    this._options = options;
    this._chart = null;
    this._series = null;
  }

  attached(param: any): void {
    this._chart = param.chart;
    this._series = param.series;
  }

  detached(): void {
    this._chart = null;
    this._series = null;
  }

  updateAllViews(): void {}

  paneViews() {
    return [{
      zOrder: () => 'bottom' as const,
      renderer: () => {
        if (!this._chart) return null;
        const timeScale = this._chart.timeScale();
        const bands: { x1: number; x2: number; color: string }[] = [];

        for (const range of this._options.ranges) {
          const x1 = timeScale.timeToCoordinate(range.startTime);
          const x2 = timeScale.timeToCoordinate(range.endTime);
          if (x1 !== null && x2 !== null) {
            bands.push({ x1: x1 as number, x2: x2 as number, color: range.color });
          }
        }

        if (bands.length === 0) return null;

        return {
          draw(_target: any) {},
          drawBackground(target: any) {
            target.useMediaCoordinateSpace((scope: any) => {
              const ctx = scope.context;
              const height = scope.mediaSize.height;
              for (const band of bands) {
                ctx.fillStyle = band.color;
                ctx.fillRect(band.x1, 0, band.x2 - band.x1, height);
              }
            });
          },
        };
      },
    }];
  }
}
