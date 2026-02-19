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

interface AnchoredTextOptions {
  time: Time;
  price: number;
  text: string;
  color?: string;
  fontSize?: number;
  position?: string; // "above" | "below" | "left" | "right"
}

class AnchoredTextRenderer implements ISeriesPrimitivePaneRenderer {
  private _x: number;
  private _y: number;
  private _text: string;
  private _color: string;
  private _fontSize: number;
  private _position: string;

  constructor(x: number, y: number, text: string, color: string, fontSize: number, position: string) {
    this._x = x;
    this._y = y;
    this._text = text;
    this._color = color;
    this._fontSize = fontSize;
    this._position = position;
  }

  draw(target: CanvasRenderingTarget2D): void {
    var x = this._x;
    var y = this._y;
    var text = this._text;
    var color = this._color;
    var fontSize = this._fontSize;
    var position = this._position;

    target.useMediaCoordinateSpace((scope: MediaCoordinatesRenderingScope) => {
      var ctx = scope.context;
      ctx.font = fontSize + "px sans-serif";
      ctx.fillStyle = color;
      ctx.textBaseline = "middle";

      var drawX = x;
      var drawY = y;
      var padding = 4;

      switch (position) {
        case "above":
          ctx.textAlign = "center";
          drawY -= padding + fontSize / 2;
          break;
        case "below":
          ctx.textAlign = "center";
          drawY += padding + fontSize / 2;
          break;
        case "left":
          ctx.textAlign = "right";
          drawX -= padding;
          break;
        case "right":
          ctx.textAlign = "left";
          drawX += padding;
          break;
        default:
          ctx.textAlign = "center";
          drawY -= padding + fontSize / 2;
      }

      ctx.fillText(text, drawX, drawY);
    });
  }
}

class AnchoredTextPaneView implements ISeriesPrimitivePaneView {
  private _source: AnchoredText;

  constructor(source: AnchoredText) {
    this._source = source;
  }

  zOrder(): SeriesPrimitivePaneViewZOrder {
    return "top";
  }

  renderer(): ISeriesPrimitivePaneRenderer | null {
    return this._source.buildRenderer();
  }
}

export class AnchoredText {
  private _options: AnchoredTextOptions;
  private _chart: IChartApiBase<Time> | null;
  private _series: ISeriesApi<SeriesType, Time> | null;
  private _paneViews: AnchoredTextPaneView[];

  constructor(options: AnchoredTextOptions) {
    this._options = options;
    this._chart = null;
    this._series = null;
    this._paneViews = [new AnchoredTextPaneView(this)];
  }

  attached(param: SeriesAttachedParameter<Time, SeriesType>): void {
    this._chart = param.chart;
    this._series = param.series;
  }

  detached(): void {
    this._chart = null;
    this._series = null;
  }

  updateAllViews(): void {
    // coordinates recalculated in buildRenderer
  }

  paneViews(): readonly ISeriesPrimitivePaneView[] {
    return this._paneViews;
  }

  buildRenderer(): ISeriesPrimitivePaneRenderer | null {
    if (!this._chart || !this._series) { return null; }

    var x = this._chart.timeScale().timeToCoordinate(this._options.time);
    var y = this._series.priceToCoordinate(this._options.price);

    if (x === null || y === null) { return null; }

    return new AnchoredTextRenderer(
      x as unknown as number,
      y as unknown as number,
      this._options.text,
      this._options.color || "#ffffff",
      this._options.fontSize || 12,
      this._options.position || "above"
    );
  }
}
