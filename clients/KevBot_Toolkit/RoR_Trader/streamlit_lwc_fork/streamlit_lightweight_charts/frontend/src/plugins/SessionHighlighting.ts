import {
  ISeriesPrimitivePaneView,
  ISeriesPrimitivePaneRenderer,
  SeriesAttachedParameter,
  Time,
  SeriesType,
  IChartApiBase,
  ISeriesApi,
  SeriesPrimitivePaneViewZOrder,
} from "lightweight-charts";
import { CanvasRenderingTarget2D, MediaCoordinatesRenderingScope } from "fancy-canvas";

interface SessionRange {
  startTime: Time;
  endTime: Time;
  color: string;
}

interface SessionHighlightingOptions {
  ranges: SessionRange[];
}

interface ComputedBand {
  x1: number;
  x2: number;
  color: string;
}

class SessionHighlightingRenderer implements ISeriesPrimitivePaneRenderer {
  private _bands: ComputedBand[];

  constructor(bands: ComputedBand[]) {
    this._bands = bands;
  }

  draw(_target: CanvasRenderingTarget2D): void {
    // all drawing in drawBackground
  }

  drawBackground(target: CanvasRenderingTarget2D): void {
    var bands = this._bands;

    target.useMediaCoordinateSpace((scope: MediaCoordinatesRenderingScope) => {
      var ctx = scope.context;
      var height = scope.mediaSize.height;

      for (var i = 0; i < bands.length; i++) {
        var band = bands[i];
        ctx.fillStyle = band.color;
        ctx.fillRect(band.x1, 0, band.x2 - band.x1, height);
      }
    });
  }
}

class SessionHighlightingPaneView implements ISeriesPrimitivePaneView {
  private _source: SessionHighlighting;

  constructor(source: SessionHighlighting) {
    this._source = source;
  }

  zOrder(): SeriesPrimitivePaneViewZOrder {
    return "bottom";
  }

  renderer(): ISeriesPrimitivePaneRenderer | null {
    return this._source.buildRenderer();
  }
}

export class SessionHighlighting {
  private _options: SessionHighlightingOptions;
  private _chart: IChartApiBase<Time> | null;
  private _series: ISeriesApi<SeriesType, Time> | null;
  private _paneViews: SessionHighlightingPaneView[];

  constructor(options: SessionHighlightingOptions) {
    this._options = options;
    this._chart = null;
    this._series = null;
    this._paneViews = [new SessionHighlightingPaneView(this)];
  }

  attached(param: SeriesAttachedParameter<Time, SeriesType>): void {
    this._chart = param.chart;
    this._series = param.series;
  }

  detached(): void {
    this._chart = null;
    this._series = null;
  }

  updateAllViews(): void {}

  paneViews(): readonly ISeriesPrimitivePaneView[] {
    return this._paneViews;
  }

  buildRenderer(): ISeriesPrimitivePaneRenderer | null {
    if (!this._chart) { return null; }

    var timeScale = this._chart.timeScale();
    var bands: ComputedBand[] = [];

    for (var i = 0; i < this._options.ranges.length; i++) {
      var range = this._options.ranges[i];
      var x1 = timeScale.timeToCoordinate(range.startTime);
      var x2 = timeScale.timeToCoordinate(range.endTime);

      if (x1 !== null && x2 !== null) {
        bands.push({
          x1: x1 as unknown as number,
          x2: x2 as unknown as number,
          color: range.color,
        });
      }
    }

    if (bands.length === 0) { return null; }

    return new SessionHighlightingRenderer(bands);
  }
}
