/**
 * Shared hook for chart display preferences.
 *
 * Uses inline fetch (not useSettings/apiFetch) to avoid circular
 * dependency that causes "Cannot access before initialization" in
 * production builds.
 */

import { useState, useEffect } from 'react';

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
  // Regional
  timezone: string;
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
  // Alert slippage tolerance (seconds)
  alertSlippage: number;
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
  timezone: 'US/Mountain',
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
  alertSlippage: 5,
  entryColor: '#4CAF50',
  exitWinColor: '#4CAF50',
  exitLossColor: '#F44336',
  exitStopColor: '#FF9800',
  exitBarCountColor: '#26A69A',
  exitHybridColor: '#FF9800',
  showLabels: true,
};

export function useChartPrefs(): ChartPrefs {
  const [settings, setSettings] = useState<Record<string, any> | null>(null);

  useEffect(() => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('ror_access_token') : null;
    if (!token) return;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/api/settings`, { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setSettings(d); })
      .catch(() => {});
  }, []);

  if (!settings) return DEFAULTS;
  const s = settings;

  const theme = s.candleTheme ?? DEFAULTS.candleTheme;
  const themeColors = CANDLE_THEMES[theme] || CANDLE_THEMES.neutral;

  return {
    timezone: s.timezone ?? DEFAULTS.timezone,
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
    alertSlippage: s.alertSlippage ?? DEFAULTS.alertSlippage,
    entryColor: s.entryColor ?? DEFAULTS.entryColor,
    exitWinColor: s.exitWinColor ?? DEFAULTS.exitWinColor,
    exitLossColor: s.exitLossColor ?? DEFAULTS.exitLossColor,
    exitStopColor: s.exitStopColor ?? DEFAULTS.exitStopColor,
    exitBarCountColor: s.exitBarCountColor ?? DEFAULTS.exitBarCountColor,
    exitHybridColor: s.exitHybridColor ?? DEFAULTS.exitHybridColor,
    showLabels: s.showLabels ?? DEFAULTS.showLabels,
  };
}
