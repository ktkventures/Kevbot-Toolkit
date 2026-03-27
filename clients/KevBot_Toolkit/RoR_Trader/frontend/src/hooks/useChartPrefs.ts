/**
 * Shared hook for chart display preferences.
 *
 * Reads from the user settings API (same data that Settings > Display saves)
 * and provides typed chart preferences for all chart components.
 */

import { useSettings } from '@/hooks/queries/useSettings';

/** Candle theme definitions — mirrors SettingsDisplayPage CANDLE_THEMES */
export const CANDLE_THEMES: Record<string, { up: string; down: string; upBorder?: string }> = {
  theme: { up: 'var(--accent)', down: 'var(--red)' },
  classic: { up: '#26a69a', down: '#ef5350' },
  neutral: { up: '#FFFFFF', down: '#787B86' },
  neutral_hollow: { up: 'transparent', upBorder: '#FFFFFF', down: '#787B86' },
  monochrome: { up: '#d4d4d8', down: '#52525b' },
  neon: { up: '#00ff88', down: '#ff0055' },
};

export interface ChartPrefs {
  // Price chart
  candleStyle: string;
  candleTheme: string;
  candleUp: string;
  candleDown: string;
  candleUpBorder: string;
  visibleCandles: number;
  gridLines: boolean;
  rightOffset: number;
  paneOrder: string[];
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
  exitBarCountColor: string;
  exitHybridColor: string;
  showLabels: boolean;
}

const DEFAULTS: ChartPrefs = {
  candleStyle: 'Candle',
  candleTheme: 'neutral',
  candleUp: '#FFFFFF',
  candleDown: '#787B86',
  candleUpBorder: '#FFFFFF',
  visibleCandles: 200,
  gridLines: true,
  rightOffset: 3,
  paneOrder: ['confluence', 'price', 'oscillators'],
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
  exitBarCountColor: '#26A69A',
  exitHybridColor: '#FF9800',
  showLabels: true,
};

export function useChartPrefs(): ChartPrefs {
  const { data: settings } = useSettings();
  if (!settings) return DEFAULTS;
  const s = settings as Record<string, any>;

  const theme = s.candleTheme ?? DEFAULTS.candleTheme;
  const themeColors = CANDLE_THEMES[theme] || CANDLE_THEMES.neutral;

  return {
    candleStyle: s.candleStyle ?? DEFAULTS.candleStyle,
    candleTheme: theme,
    candleUp: themeColors.up,
    candleDown: themeColors.down,
    candleUpBorder: themeColors.upBorder || themeColors.up,
    visibleCandles: s.visibleCandles ?? DEFAULTS.visibleCandles,
    gridLines: s.gridLines ?? DEFAULTS.gridLines,
    rightOffset: s.rightOffset ?? DEFAULTS.rightOffset,
    paneOrder: s.paneOrder ?? DEFAULTS.paneOrder,
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
    exitBarCountColor: s.exitBarCountColor ?? DEFAULTS.exitBarCountColor,
    exitHybridColor: s.exitHybridColor ?? DEFAULTS.exitHybridColor,
    showLabels: s.showLabels ?? DEFAULTS.showLabels,
  };
}
