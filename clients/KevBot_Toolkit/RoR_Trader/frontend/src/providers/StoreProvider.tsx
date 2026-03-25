'use client';

import { ReactNode } from 'react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

/**
 * Display settings store — persisted to localStorage.
 * Used by chart components, card layouts, and preference controls.
 */
export interface DisplaySettings {
  // Price charts
  chartLibrary: 'lightweight' | 'echarts';
  candleStyle: string;
  candleTheme: string;
  visibleCandles: number;
  rightOffset: number;
  showGrid: boolean;

  // Equity curves
  equityLineStyle: 'solid' | 'smooth' | 'stepped';
  equityXAxis: 'time' | 'trade';
  showGradientFill: boolean;
  showZeroLine: boolean;
  showHWM: boolean;
  showEdgeCheck: boolean;
  showConfidenceBands: boolean;

  // Formatting
  timezone: string;
  dateFormat: string;
  currencySymbol: string;

  // Components
  badgeShape: 'rounded' | 'square';
  showBrackets: boolean;
  execTypeColor: string;
  fidelityColor: string;

  // Tables
  tableDensity: 'compact' | 'comfortable' | 'spacious';
}

const DEFAULT_SETTINGS: DisplaySettings = {
  chartLibrary: 'lightweight',
  candleStyle: 'candle',
  candleTheme: 'theme',
  visibleCandles: 200,
  rightOffset: 10,
  showGrid: true,
  equityLineStyle: 'solid',
  equityXAxis: 'trade',
  showGradientFill: true,
  showZeroLine: true,
  showHWM: false,
  showEdgeCheck: false,
  showConfidenceBands: false,
  timezone: 'America/New_York',
  dateFormat: 'MM/DD/YYYY',
  currencySymbol: '$',
  badgeShape: 'rounded',
  showBrackets: true,
  execTypeColor: '#2196F3',
  fidelityColor: '#00BCD4',
  tableDensity: 'comfortable',
};

interface DisplayStore extends DisplaySettings {
  update: (partial: Partial<DisplaySettings>) => void;
  reset: () => void;
}

export const useDisplayStore = create<DisplayStore>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      update: (partial) => set(partial),
      reset: () => set(DEFAULT_SETTINGS),
    }),
    { name: 'ror-display-settings' }
  )
);

/**
 * StoreProvider — no-op wrapper for now.
 * Zustand stores don't need a provider, but this keeps the layout pattern
 * consistent and gives us a place to add future stores.
 */
export function StoreProvider({ children }: { children: ReactNode }) {
  return <>{children}</>;
}
