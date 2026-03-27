/**
 * Shared hook for chart display preferences.
 *
 * Reads from the user settings API (same data that Settings > Display saves)
 * and provides typed chart preferences for all chart components.
 */

import { useSettings } from '@/hooks/queries/useSettings';

export interface ChartPrefs {
  // Price chart
  candleStyle: string;
  candleTheme: string;
  visibleCandles: number;
  gridLines: boolean;
  rightOffset: number;
  // Equity curve
  eqBacktestColor: string;
  eqForwardColor: string;
  eqLiveColor: string;
  eqShowZeroLine: boolean;
  eqShowHWM: boolean;
  eqXAxis: 'trade' | 'time';
  // Trade markers
  entryColor: string;
  exitWinColor: string;
  exitLossColor: string;
  exitStopColor: string;
  showLabels: boolean;
}

const DEFAULTS: ChartPrefs = {
  candleStyle: 'Candle',
  candleTheme: 'neutral',
  visibleCandles: 200,
  gridLines: true,
  rightOffset: 3,
  eqBacktestColor: '#2196F3',
  eqForwardColor: '#FF9800',
  eqLiveColor: '#4CAF50',
  eqShowZeroLine: true,
  eqShowHWM: false,
  eqXAxis: 'trade',
  entryColor: '#4CAF50',
  exitWinColor: '#4CAF50',
  exitLossColor: '#F44336',
  exitStopColor: '#FF9800',
  showLabels: true,
};

export function useChartPrefs(): ChartPrefs {
  const { data: settings } = useSettings();
  if (!settings) return DEFAULTS;
  const s = settings as Record<string, any>;
  return {
    candleStyle: s.candleStyle ?? DEFAULTS.candleStyle,
    candleTheme: s.candleTheme ?? DEFAULTS.candleTheme,
    visibleCandles: s.visibleCandles ?? DEFAULTS.visibleCandles,
    gridLines: s.gridLines ?? DEFAULTS.gridLines,
    rightOffset: s.rightOffset ?? DEFAULTS.rightOffset,
    eqBacktestColor: s.eqBacktestColor ?? DEFAULTS.eqBacktestColor,
    eqForwardColor: s.eqForwardColor ?? DEFAULTS.eqForwardColor,
    eqLiveColor: s.eqLiveColor ?? DEFAULTS.eqLiveColor,
    eqShowZeroLine: s.eqShowZeroLine ?? DEFAULTS.eqShowZeroLine,
    eqShowHWM: s.eqShowHWM ?? DEFAULTS.eqShowHWM,
    eqXAxis: s.eqXAxis ?? DEFAULTS.eqXAxis,
    entryColor: s.entryColor ?? DEFAULTS.entryColor,
    exitWinColor: s.exitWinColor ?? DEFAULTS.exitWinColor,
    exitLossColor: s.exitLossColor ?? DEFAULTS.exitLossColor,
    exitStopColor: s.exitStopColor ?? DEFAULTS.exitStopColor,
    showLabels: s.showLabels ?? DEFAULTS.showLabels,
  };
}
