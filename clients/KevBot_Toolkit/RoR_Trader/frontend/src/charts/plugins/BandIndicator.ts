/**
 * BandIndicator plugin — draws filled bands between upper/lower price levels.
 * Used for VWAP bands, Bollinger bands, etc.
 * Ported from streamlit_lwc_fork for LWC v5.
 */

import type { Time } from 'lightweight-charts';

interface BandDataPoint {
  time: Time;
  upperValue: number;
  lowerValue: number;
}

interface BandIndicatorOptions {
  fillColor: string;
  data: BandDataPoint[];
}

export class BandIndicator {
  private _options: BandIndicatorOptions;
  private _chart: any;
  private _series: any;

  constructor(options: BandIndicatorOptions) {
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
        if (!this._chart || !this._series) return null;
        const timeScale = this._chart.timeScale();
        const series = this._series;
        const upper: { x: number; y: number }[] = [];
        const lower: { x: number; y: number }[] = [];

        for (const point of this._options.data) {
          const x = timeScale.timeToCoordinate(point.time);
          const yU = series.priceToCoordinate(point.upperValue);
          const yL = series.priceToCoordinate(point.lowerValue);
          if (x !== null && yU !== null && yL !== null) {
            upper.push({ x: x as number, y: yU as number });
            lower.push({ x: x as number, y: yL as number });
          }
        }

        if (upper.length < 2) return null;
        const fillColor = this._options.fillColor;

        return {
          draw(target: any) {
            target.useMediaCoordinateSpace((scope: any) => {
              const ctx = scope.context;
              ctx.beginPath();
              ctx.moveTo(upper[0].x, upper[0].y);
              for (let i = 1; i < upper.length; i++) ctx.lineTo(upper[i].x, upper[i].y);
              for (let j = lower.length - 1; j >= 0; j--) ctx.lineTo(lower[j].x, lower[j].y);
              ctx.closePath();
              ctx.fillStyle = fillColor;
              ctx.fill();
            });
          },
        };
      },
    }];
  }
}
