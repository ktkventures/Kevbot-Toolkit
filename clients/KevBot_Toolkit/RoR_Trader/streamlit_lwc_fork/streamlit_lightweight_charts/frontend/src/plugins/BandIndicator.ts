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

interface BandDataPoint {
  time: Time;
  upperValue: number;
  lowerValue: number;
}

interface BandIndicatorOptions {
  fillColor: string;
  data: BandDataPoint[];
}

interface CoordPoint {
  x: number;
  y: number;
}

class BandIndicatorRenderer implements ISeriesPrimitivePaneRenderer {
  private _upperCoords: CoordPoint[];
  private _lowerCoords: CoordPoint[];
  private _fillColor: string;

  constructor(upperCoords: CoordPoint[], lowerCoords: CoordPoint[], fillColor: string) {
    this._upperCoords = upperCoords;
    this._lowerCoords = lowerCoords;
    this._fillColor = fillColor;
  }

  draw(target: CanvasRenderingTarget2D): void {
    if (this._upperCoords.length < 2) { return; }

    var upperCoords = this._upperCoords;
    var lowerCoords = this._lowerCoords;
    var fillColor = this._fillColor;

    target.useMediaCoordinateSpace((scope: MediaCoordinatesRenderingScope) => {
      var ctx = scope.context;
      ctx.beginPath();

      // Draw upper curve left-to-right
      ctx.moveTo(upperCoords[0].x, upperCoords[0].y);
      for (var i = 1; i < upperCoords.length; i++) {
        ctx.lineTo(upperCoords[i].x, upperCoords[i].y);
      }

      // Draw lower curve right-to-left (close the polygon)
      for (var j = lowerCoords.length - 1; j >= 0; j--) {
        ctx.lineTo(lowerCoords[j].x, lowerCoords[j].y);
      }

      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
    });
  }
}

class BandIndicatorPaneView implements ISeriesPrimitivePaneView {
  private _source: BandIndicator;

  constructor(source: BandIndicator) {
    this._source = source;
  }

  zOrder(): SeriesPrimitivePaneViewZOrder {
    return "bottom";
  }

  renderer(): ISeriesPrimitivePaneRenderer | null {
    return this._source.buildRenderer();
  }
}

export class BandIndicator {
  private _options: BandIndicatorOptions;
  private _chart: IChartApiBase<Time> | null;
  private _series: ISeriesApi<SeriesType, Time> | null;
  private _paneViews: BandIndicatorPaneView[];

  constructor(options: BandIndicatorOptions) {
    this._options = options;
    this._chart = null;
    this._series = null;
    this._paneViews = [new BandIndicatorPaneView(this)];
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
    if (!this._chart || !this._series) { return null; }

    var timeScale = this._chart.timeScale();
    var series = this._series;
    var upperCoords: CoordPoint[] = [];
    var lowerCoords: CoordPoint[] = [];

    for (var i = 0; i < this._options.data.length; i++) {
      var point = this._options.data[i];
      var x = timeScale.timeToCoordinate(point.time);
      var yUpper = series.priceToCoordinate(point.upperValue);
      var yLower = series.priceToCoordinate(point.lowerValue);

      if (x !== null && yUpper !== null && yLower !== null) {
        upperCoords.push({ x: x as unknown as number, y: yUpper as unknown as number });
        lowerCoords.push({ x: x as unknown as number, y: yLower as unknown as number });
      }
    }

    if (upperCoords.length < 2) { return null; }

    return new BandIndicatorRenderer(upperCoords, lowerCoords, this._options.fillColor);
  }
}
