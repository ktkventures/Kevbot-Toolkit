import { useRenderData } from "streamlit-component-lib-react-hooks"
import {
  createChart,
  IChartApi,
  ISeriesApi,
  SeriesType,
} from "lightweight-charts"
import React, { useRef, useEffect } from "react"
import { BandIndicator } from "./plugins/BandIndicator"
import { Rectangle } from "./plugins/Rectangle"
import { SessionHighlighting } from "./plugins/SessionHighlighting"
import { AnchoredText } from "./plugins/AnchoredText"

const LightweightChartsMultiplePanes: React.VFC = () => {

  // returns the renderData passed from Python
  // { args: object, disabled: boolean, theme: object }
  const renderData = useRenderData()
  const chartsData = renderData.args["charts"]

  const chartsContainerRef = useRef<HTMLDivElement>(null)
  const chartElRefs = Array(chartsData.length).fill(useRef<HTMLDivElement>(null))
  const chartRefs = useRef<IChartApi[]>([])

    useEffect(() => {
      if (chartElRefs.find((ref) => !ref.current)) return;

      chartElRefs.forEach((ref, i) => {
        const chart = chartRefs.current[i] = createChart(
          ref.current as HTMLDivElement,{
            height: 300,
            width: chartElRefs[i].current.clientWidth,
            ...chartsData[i].chart,
          }
        );

        // Track created series for primitives that reference by index
        const createdSeries: ISeriesApi<SeriesType>[] = [];

        for (const series of chartsData[i].series){

          let chartSeries: ISeriesApi<SeriesType> | undefined
          switch(series.type) {
            case 'Area':
              chartSeries = chart.addAreaSeries(series.options)
              break
            case 'Bar':
              chartSeries = chart.addBarSeries(series.options )
              break
            case 'Baseline':
              chartSeries = chart.addBaselineSeries(series.options)
              break
            case 'Candlestick':
              chartSeries = chart.addCandlestickSeries(series.options)
              break
            case 'Histogram':
              chartSeries = chart.addHistogramSeries(series.options)
              break
            case 'Line':
              chartSeries = chart.addLineSeries(series.options)
              break
            default:
                return
          }

          if(series.priceScale)
            chart.priceScale(series.options.priceScaleId || '').applyOptions(series.priceScale)

          chartSeries.setData(series.data)

          if(series.markers)
            chartSeries.setMarkers(series.markers)

          // B3: createPriceLine() support
          if(series.priceLines) {
            for (const pl of series.priceLines) {
              chartSeries.createPriceLine(pl)
            }
          }

          createdSeries.push(chartSeries)
        }

        // B4: Primitives dispatcher — attach custom drawing plugins to series
        // Each primitive in the array specifies:
        //   type: string — plugin type identifier
        //   seriesIndex: number — which series to attach to (index into createdSeries)
        //   options: object — plugin-specific configuration
        if (chartsData[i].primitives) {
          for (const prim of chartsData[i].primitives) {
            const targetSeries = createdSeries[prim.seriesIndex || 0]
            if (!targetSeries) continue

            try {
              switch (prim.type) {
                case "bandFill":
                  targetSeries.attachPrimitive(new BandIndicator(prim.options) as any)
                  break
                case "rectangle":
                  targetSeries.attachPrimitive(new Rectangle(prim.options) as any)
                  break
                case "sessionHighlight":
                  targetSeries.attachPrimitive(new SessionHighlighting(prim.options) as any)
                  break
                case "anchoredText":
                  targetSeries.attachPrimitive(new AnchoredText(prim.options) as any)
                  break
                default:
                  console.warn(`Unknown primitive type: ${prim.type}`)
              }
            } catch (err) {
              console.error(`Error attaching primitive ${prim.type}:`, err)
            }
          }
        }

        chart.timeScale().fitContent();

      })

      const charts = chartRefs.current.map((c) => c as IChartApi);

      if(chartsData.length > 1){ // sync charts
        charts.forEach((chart) => {
          if (!chart) return;

          chart.timeScale().subscribeVisibleTimeRangeChange((e) => {
            charts
              .filter((c) => c !== chart)
              .forEach((c) => {
                c.timeScale().applyOptions({
                  rightOffset: chart.timeScale().scrollPosition()
          }) }) })

          chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (range) {
              charts
                .filter((c) => c !== chart)
                .forEach((c) => {
                  c.timeScale().setVisibleLogicalRange({
                    from: range?.from,
                    to: range?.to
          }) }) } })

      }) }

      return () => { // required because how useEffect() works
        charts.forEach((chart) => {
          chart.remove()
        })
      }

    }, [ chartsData, chartElRefs, chartRefs])


    return (
      <div ref={chartsContainerRef}>
        {chartElRefs.map((ref, i) => (
          <div
            ref={ref}
            id={`lightweight-charts-${i}`}
            key={`lightweight-charts-${i}`}
          />
        ))}
      </div>
    )

}

export default LightweightChartsMultiplePanes
