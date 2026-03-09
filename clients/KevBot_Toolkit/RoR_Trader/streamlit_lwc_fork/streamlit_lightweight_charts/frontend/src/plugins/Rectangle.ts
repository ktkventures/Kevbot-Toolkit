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

interface RectanglePoint {
  time: Time;
  price: number;
}

interface RectangleOptions {
  point1: RectanglePoint;
  point2: RectanglePoint;
  fillColor: string;
  borderColor?: string;
  borderWidth?: number;
}

class RectangleRenderer implements ISeriesPrimitivePaneRenderer {
  private _x1: number;
  private _y1: number;
  private _x2: number;
  private _y2: number;
  private _fillColor: string;
  private _borderColor: string;
  private _borderWidth: number;

  constructor(
    x1: number, y1: number, x2: number, y2: number,
    fillColor: string, borderColor: string, borderWidth: number
  ) {
    this._x1 = x1;
    this._y1 = y1;
    this._x2 = x2;
    this._y2 = y2;
    this._fillColor = fillColor;
    this._borderColor = borderColor;
    this._borderWidth = borderWidth;
  }

  draw(target: CanvasRenderingTarget2D): void {
    var x1 = this._x1;
    var y1 = this._y1;
    var x2 = this._x2;
    var y2 = this._y2;
    var fillColor = this._fillColor;
    var borderColor = this._borderColor;
    var borderWidth = this._borderWidth;

    target.useMediaCoordinateSpace((scope: MediaCoordinatesRenderingScope) => {
      var ctx = scope.context;
      var x = Math.min(x1, x2);
      var y = Math.min(y1, y2);
      var w = Math.abs(x2 - x1);
      var h = Math.abs(y2 - y1);

      ctx.fillStyle = fillColor;
      ctx.fillRect(x, y, w, h);

      if (borderWidth > 0) {
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = borderWidth;
        ctx.strokeRect(x, y, w, h);
      }
    });
  }
}

class RectanglePaneView implements ISeriesPrimitivePaneView {
  private _source: Rectangle;

  constructor(source: Rectangle) {
    this._source = source;
  }

  zOrder(): SeriesPrimitivePaneViewZOrder {
    return "bottom";
  }

  renderer(): ISeriesPrimitivePaneRenderer | null {
    return this._source.buildRenderer();
  }
}

export class Rectangle {
  private _options: RectangleOptions;
  private _chart: IChartApiBase<Time> | null;
  private _series: ISeriesApi<SeriesType, Time> | null;
  private _paneViews: RectanglePaneView[];

  constructor(options: RectangleOptions) {
    this._options = options;
    this._chart = null;
    this._series = null;
    this._paneViews = [new RectanglePaneView(this)];
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
    var x1 = timeScale.timeToCoordinate(this._options.point1.time);
    var y1 = this._series.priceToCoordinate(this._options.point1.price);
    var x2 = timeScale.timeToCoordinate(this._options.point2.time);
    var y2 = this._series.priceToCoordinate(this._options.point2.price);

    if (x1 === null || y1 === null || x2 === null || y2 === null) { return null; }

    return new RectangleRenderer(
      x1 as unknown as number,
      y1 as unknown as number,
      x2 as unknown as number,
      y2 as unknown as number,
      this._options.fillColor,
      this._options.borderColor || "transparent",
      this._options.borderWidth || 0
    );
  }
}
