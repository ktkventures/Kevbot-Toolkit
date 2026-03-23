'use client';

import { useState, useCallback, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

/* ─────────────────────────── Types ─────────────────────────── */

interface PackParam {
  key: string;
  label: string;
  value: number;
  min: number;
  max: number;
}

interface PackTrigger {
  name: string;
  direction: 'LONG' | 'SHORT' | 'BOTH';
  exec: string;
}

interface UserPack {
  id: string;
  name: string;
  category: string;
  enabled: boolean;
  params: PackParam[];
  outputs: string[];
  triggers: PackTrigger[];
  createdAt: string;
  strategiesUsing: number;
  winRate: number;
  totalSignals: number;
  author: string;
  description: string;
  rating: number;
  downloads: number;
}

/* ─────────────────────────── Mock Data ─────────────────────────── */

const initialPacks: UserPack[] = [
  {
    id: 'my-rsi',
    name: 'My RSI Pack',
    category: 'Momentum',
    enabled: true,
    params: [
      { key: 'period', label: 'Period', value: 14, min: 2, max: 100 },
      { key: 'overbought', label: 'Overbought', value: 70, min: 50, max: 95 },
      { key: 'oversold', label: 'Oversold', value: 30, min: 5, max: 50 },
    ],
    outputs: ['OVERBOUGHT', 'NEUTRAL', 'OVERSOLD'],
    triggers: [
      { name: 'RSI Cross Above OB', direction: 'SHORT', exec: '[C]' },
      { name: 'RSI Cross Below OS', direction: 'LONG', exec: '[C]' },
    ],
    createdAt: '2026-03-18',
    strategiesUsing: 3,
    winRate: 62.4,
    totalSignals: 147,
    author: 'You',
    description: 'Classic RSI with customizable overbought/oversold thresholds',
    rating: 4.5,
    downloads: 0,
  },
  {
    id: 'my-bb',
    name: 'Bollinger Squeeze',
    category: 'Volatility',
    enabled: true,
    params: [
      { key: 'period', label: 'Period', value: 20, min: 5, max: 50 },
      { key: 'std_dev', label: 'Std Dev', value: 2, min: 1, max: 4 },
      { key: 'squeeze_pct', label: 'Squeeze %', value: 5, min: 1, max: 20 },
    ],
    outputs: ['SQUEEZE', 'EXPANSION', 'UPPER_TOUCH', 'LOWER_TOUCH', 'INSIDE'],
    triggers: [
      { name: 'Squeeze Detected', direction: 'BOTH', exec: '[C]' },
      { name: 'Upper Band Touch', direction: 'SHORT', exec: '[C]' },
      { name: 'Lower Band Touch', direction: 'LONG', exec: '[C]' },
    ],
    createdAt: '2026-03-15',
    strategiesUsing: 2,
    winRate: 58.1,
    totalSignals: 89,
    author: 'You',
    description: 'Detects Bollinger Band squeezes for breakout trades',
    rating: 4.2,
    downloads: 0,
  },
  {
    id: 'my-stoch',
    name: 'Stochastic Momentum',
    category: 'Momentum',
    enabled: false,
    params: [
      { key: 'k_period', label: '%K Period', value: 14, min: 5, max: 50 },
      { key: 'd_period', label: '%D Period', value: 3, min: 1, max: 10 },
      { key: 'smooth', label: 'Smooth', value: 3, min: 1, max: 10 },
    ],
    outputs: ['OB_CROSS_DOWN', 'OS_CROSS_UP', 'NEUTRAL'],
    triggers: [
      { name: '%K crosses below %D in OB', direction: 'SHORT', exec: '[C]' },
      { name: '%K crosses above %D in OS', direction: 'LONG', exec: '[C]' },
    ],
    createdAt: '2026-03-10',
    strategiesUsing: 0,
    winRate: 0,
    totalSignals: 0,
    author: 'You',
    description: 'Stochastic oscillator with smoothing for momentum reversals',
    rating: 0,
    downloads: 0,
  },
];

const communityPacks: UserPack[] = [
  {
    id: 'comm-obv',
    name: 'OBV Divergence',
    category: 'Volume',
    enabled: false,
    params: [
      { key: 'lookback', label: 'Lookback', value: 14, min: 5, max: 50 },
      { key: 'threshold', label: 'Threshold', value: 1.5, min: 0.5, max: 5 },
    ],
    outputs: ['BULL_DIV', 'BEAR_DIV', 'CONFIRMING', 'NEUTRAL'],
    triggers: [
      { name: 'Bullish Divergence', direction: 'LONG', exec: '[C]' },
      { name: 'Bearish Divergence', direction: 'SHORT', exec: '[C]' },
    ],
    createdAt: '2026-03-12',
    strategiesUsing: 15,
    winRate: 67.3,
    totalSignals: 423,
    author: 'TraderMike',
    description: 'Detects On-Balance Volume divergences for trend reversals',
    rating: 4.8,
    downloads: 234,
  },
  {
    id: 'comm-adr',
    name: 'ADR Range Filter',
    category: 'Volatility',
    enabled: false,
    params: [
      { key: 'period', label: 'Period', value: 14, min: 5, max: 30 },
      { key: 'pct_used', label: 'ADR % Used', value: 75, min: 50, max: 100 },
    ],
    outputs: ['ROOM_LEFT', 'NEAR_ADR', 'OVEREXTENDED'],
    triggers: [
      { name: 'Near ADR Limit', direction: 'BOTH', exec: '[C]' },
      { name: 'Overextended', direction: 'BOTH', exec: '[C]' },
    ],
    createdAt: '2026-03-08',
    strategiesUsing: 28,
    winRate: 71.2,
    totalSignals: 890,
    author: 'QuantMaster',
    description: 'Filters trades when stock has used most of its average daily range',
    rating: 4.9,
    downloads: 567,
  },
  {
    id: 'comm-ichimoku',
    name: 'Ichimoku Cloud',
    category: 'Trend',
    enabled: false,
    params: [
      { key: 'tenkan', label: 'Tenkan', value: 9, min: 5, max: 20 },
      { key: 'kijun', label: 'Kijun', value: 26, min: 15, max: 52 },
      { key: 'senkou_b', label: 'Senkou B', value: 52, min: 26, max: 104 },
    ],
    outputs: ['ABOVE_CLOUD', 'IN_CLOUD', 'BELOW_CLOUD', 'TK_CROSS_BULL', 'TK_CROSS_BEAR'],
    triggers: [
      { name: 'TK Cross Bullish', direction: 'LONG', exec: '[C]' },
      { name: 'TK Cross Bearish', direction: 'SHORT', exec: '[C]' },
      { name: 'Cloud Breakout Up', direction: 'LONG', exec: '[C]' },
      { name: 'Cloud Breakout Down', direction: 'SHORT', exec: '[C]' },
    ],
    createdAt: '2026-03-01',
    strategiesUsing: 42,
    winRate: 59.8,
    totalSignals: 1247,
    author: 'SamuraiTrader',
    description: 'Full Ichimoku cloud system with TK cross and cloud breakout signals',
    rating: 4.6,
    downloads: 891,
  },
];

/* ─────────────────────────── Helpers ─────────────────────────── */

const categoryColors: Record<string, { color: string; bg: string }> = {
  Momentum: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  Volume: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  Volatility: { color: 'var(--red)', bg: 'var(--red-muted)' },
  Trend: { color: 'var(--green)', bg: 'var(--green-muted)' },
  Custom: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const directionColors: Record<string, { color: string; icon: string }> = {
  LONG: { color: 'var(--green)', icon: '\u2191' },
  SHORT: { color: 'var(--red)', icon: '\u2193' },
  BOTH: { color: 'var(--accent)', icon: '\u2195' },
};

/* ─────────────────────────── SVG Icons ─────────────────────────── */

function IndicatorPreviewIcon({ pack, size = 64 }: { pack: UserPack; size?: number }) {
  if (pack.category === 'Momentum') {
    // RSI-like oscillator
    return (
      <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
        {/* Overbought zone */}
        <rect x="4" y="8" width="56" height="12" rx="1" fill="var(--red)" opacity="0.08" />
        {/* Oversold zone */}
        <rect x="4" y="44" width="56" height="12" rx="1" fill="var(--green)" opacity="0.08" />
        {/* OB/OS lines */}
        <line x1="4" y1="20" x2="60" y2="20" stroke="var(--red)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.5" />
        <line x1="4" y1="44" x2="60" y2="44" stroke="var(--green)" strokeWidth="0.5" strokeDasharray="2 2" opacity="0.5" />
        {/* Center line */}
        <line x1="4" y1="32" x2="60" y2="32" stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.3" />
        {/* Oscillator line */}
        <path d="M4,36 Q10,42 16,48 Q22,52 28,38 Q34,24 40,16 Q46,12 52,28 Q56,36 60,32" stroke="var(--orange)" strokeWidth="2" strokeLinecap="round" />
        {/* Crosses into zones */}
        <circle cx="28" cy="38" r="2.5" fill="var(--green)" opacity="0.8" />
        <circle cx="40" cy="16" r="2.5" fill="var(--red)" opacity="0.8" />
      </svg>
    );
  }
  if (pack.category === 'Volatility') {
    // Bollinger band-like
    return (
      <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
        {/* Bands */}
        <path d="M4,12 Q16,8 28,16 Q40,10 52,14 L60,12" stroke="var(--red)" strokeWidth="1" opacity="0.4" />
        <path d="M4,52 Q16,56 28,48 Q40,54 52,50 L60,52" stroke="var(--red)" strokeWidth="1" opacity="0.4" />
        {/* Fill */}
        <path d="M4,12 Q16,8 28,16 Q40,10 52,14 L60,12 L60,52 Q52,50 40,54 Q28,48 16,56 L4,52 Z" fill="var(--red)" opacity="0.05" />
        {/* Middle band */}
        <path d="M4,32 Q16,28 28,34 Q40,30 52,32 L60,32" stroke="var(--text-secondary)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Price */}
        <path d="M4,30 Q10,36 16,28 Q22,20 28,24 Q34,32 40,26 Q46,18 52,22 Q56,28 60,24" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Squeeze zone indicator */}
        <rect x="18" y="26" width="12" height="12" rx="2" stroke="var(--orange)" strokeWidth="1" strokeDasharray="2 2" fill="var(--orange)" fillOpacity="0.1" />
      </svg>
    );
  }
  if (pack.category === 'Volume') {
    return (
      <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
        {/* Volume bars with divergence highlight */}
        {[8, 12, 6, 18, 14, 22, 10, 28, 16, 8].map((h, i) => (
          <rect key={i} x={4 + i * 6} y={54 - h} width="4" height={h} rx="1" fill={i >= 6 && h > 15 ? 'var(--green)' : 'var(--accent)'} opacity={0.5} />
        ))}
        {/* Divergence line */}
        <line x1="22" y1="36" x2="52" y2="26" stroke="var(--green)" strokeWidth="1.5" strokeDasharray="3 2" />
        {/* Price going opposite */}
        <path d="M22,16 Q32,18 42,22 Q50,26 52,28" stroke="var(--red)" strokeWidth="1.5" strokeLinecap="round" />
        <text x="32" y="12" textAnchor="middle" fontSize="6" fill="var(--green)">DIV</text>
      </svg>
    );
  }
  if (pack.category === 'Trend') {
    // Ichimoku-like cloud
    return (
      <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
        {/* Cloud */}
        <path d="M4,28 Q16,24 28,30 Q40,26 52,28 L60,26 L60,38 Q52,36 40,40 Q28,34 16,38 L4,36 Z" fill="var(--green)" opacity="0.12" />
        {/* Tenkan */}
        <path d="M4,24 Q16,20 28,26 Q40,22 52,24 L60,22" stroke="var(--blue)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Kijun */}
        <path d="M4,30 Q16,28 28,32 Q40,28 52,30 L60,28" stroke="var(--red)" strokeWidth="1.5" strokeLinecap="round" />
        {/* Price breaking above cloud */}
        <path d="M4,34 Q10,32 16,30 Q22,26 28,22 Q34,18 40,16 Q46,14 52,12 L60,10" stroke="var(--text-primary)" strokeWidth="2" strokeLinecap="round" />
        {/* Cross point */}
        <circle cx="18" cy="26" r="2.5" fill="var(--accent)" />
      </svg>
    );
  }
  // Generic fallback
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <circle cx="32" cy="32" r="20" stroke="var(--accent)" strokeWidth="1.5" opacity="0.3" />
      <path d="M20,40 Q28,16 36,36 Q44,48 52,24" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function StarRating({ rating, size = 12 }: { rating: number; size?: number }) {
  const stars = [];
  for (let i = 1; i <= 5; i++) {
    const fill = i <= Math.floor(rating) ? 1 : i - 1 < rating ? 0.5 : 0;
    stars.push(
      <svg key={i} width={size} height={size} viewBox="0 0 24 24" fill="none">
        <polygon
          points="12,2 15,9 22,9 16.5,14 18.5,22 12,17.5 5.5,22 7.5,14 2,9 9,9"
          fill={fill > 0 ? 'var(--orange)' : 'transparent'}
          stroke="var(--orange)"
          strokeWidth="1.5"
          opacity={fill > 0 ? (fill === 1 ? 1 : 0.5) : 0.2}
        />
      </svg>,
    );
  }
  return <div className="flex items-center gap-0.5">{stars}</div>;
}

/* ─────────────────────────── Pack Performance Badge ─────────────────────────── */

function PerformanceBadge({ winRate, signals }: { winRate: number; signals: number }) {
  if (signals === 0) return <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>No data</span>;
  const color = winRate >= 65 ? 'var(--green)' : winRate >= 50 ? 'var(--orange)' : 'var(--red)';
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs font-bold font-mono" style={{ color }}>{winRate.toFixed(1)}%</span>
      <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>({signals} signals)</span>
    </div>
  );
}

/* ─────────────────────────── Leaderboard ─────────────────────────── */

function PackLeaderboard({ packs }: { packs: UserPack[] }) {
  const sorted = [...packs].filter((p) => p.totalSignals > 0).sort((a, b) => b.winRate - a.winRate);

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
          Performance Leaderboard
        </h3>
        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>By win rate</span>
      </div>
      <div className="space-y-2">
        {sorted.length === 0 ? (
          <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No packs with signal data yet</p>
        ) : (
          sorted.map((pack, i) => {
            const medal = i === 0 ? { color: '#FFD700', label: '1st' } : i === 1 ? { color: '#C0C0C0', label: '2nd' } : i === 2 ? { color: '#CD7F32', label: '3rd' } : null;
            return (
              <div
                key={pack.id}
                className="flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all"
                style={{
                  background: i === 0 ? `${medal?.color}08` : 'var(--bg-input)',
                  border: i === 0 ? `1px solid ${medal?.color}30` : '1px solid transparent',
                }}
              >
                {/* Rank */}
                <div className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold" style={{ background: medal ? `${medal.color}20` : 'var(--bg-primary)', color: medal ? medal.color : 'var(--text-muted)' }}>
                  {i + 1}
                </div>
                {/* Icon */}
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-primary)' }}>
                  <IndicatorPreviewIcon pack={pack} size={28} />
                </div>
                {/* Info */}
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{pack.name}</span>
                  <span className="text-[10px] ml-2" style={{ color: 'var(--text-muted)' }}>{pack.author}</span>
                </div>
                {/* Stats */}
                <PerformanceBadge winRate={pack.winRate} signals={pack.totalSignals} />
                {/* Win rate bar */}
                <div className="w-16 h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
                  <div className="h-full rounded-full" style={{ width: `${pack.winRate}%`, background: pack.winRate >= 65 ? 'var(--green)' : pack.winRate >= 50 ? 'var(--orange)' : 'var(--red)' }} />
                </div>
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
}

/* ─────────────────────────── Marketplace Browse ─────────────────────────── */

function MarketplaceBrowser({ packs, onInstall }: { packs: UserPack[]; onInstall: (pack: UserPack) => void }) {
  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>
            Community Packs
          </h3>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Browse and install packs shared by other traders</p>
        </div>
        <span className="text-xs px-2 py-1 rounded-full" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
          {packs.length} available
        </span>
      </div>
      <div className="grid grid-cols-1 gap-3">
        {packs.map((pack) => (
          <div
            key={pack.id}
            className="flex items-center gap-4 px-4 py-3 rounded-xl transition-all"
            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
          >
            {/* Icon */}
            <div className="w-14 h-14 rounded-xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}>
              <IndicatorPreviewIcon pack={pack} size={40} />
            </div>
            {/* Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{pack.name}</span>
                <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: categoryColors[pack.category]?.bg, color: categoryColors[pack.category]?.color }}>
                  {pack.category}
                </span>
              </div>
              <p className="text-[11px] mb-1" style={{ color: 'var(--text-muted)' }}>{pack.description}</p>
              <div className="flex items-center gap-3">
                <StarRating rating={pack.rating} size={10} />
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pack.rating.toFixed(1)}</span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>by {pack.author}</span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pack.downloads} installs</span>
              </div>
            </div>
            {/* Stats */}
            <div className="text-right flex-shrink-0">
              <PerformanceBadge winRate={pack.winRate} signals={pack.totalSignals} />
              <div className="flex items-center gap-1 mt-1 justify-end">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                </svg>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pack.strategiesUsing} using</span>
              </div>
            </div>
            {/* Install button */}
            <button
              onClick={() => onInstall(pack)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex-shrink-0"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Install
            </button>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ─────────────────────────── Pack Detail Modal ─────────────────────────── */

function PackDetailModal({ pack, onClose, onUpdate }: { pack: UserPack; onClose: () => void; onUpdate: (id: string, params: PackParam[]) => void }) {
  const [localParams, setLocalParams] = useState<PackParam[]>(pack.params);
  const [detailTab, setDetailTab] = useState<'overview' | 'params' | 'triggers' | 'performance'>('overview');

  const updateParam = (key: string, val: number) => {
    setLocalParams((prev) => prev.map((p) => (p.key === key ? { ...p, value: val } : p)));
  };

  return (
    <Modal title="" isOpen onClose={onClose} width="750px">
      <div className="flex items-start gap-4 -mt-2 mb-6">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center flex-shrink-0" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
          <IndicatorPreviewIcon pack={pack} size={56} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h2 className="text-xl font-bold">{pack.name}</h2>
            <span className="text-xs px-2 py-0.5 rounded-full" style={{ color: categoryColors[pack.category]?.color, background: categoryColors[pack.category]?.bg }}>
              {pack.category}
            </span>
          </div>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{pack.description}</p>
          <div className="flex items-center gap-3 mt-2">
            <StarRating rating={pack.rating} />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>by {pack.author}</span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Created {pack.createdAt}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-1 mb-5 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
        {(['overview', 'params', 'triggers', 'performance'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setDetailTab(tab)}
            className="flex-1 py-2 text-xs font-medium rounded-lg transition-all"
            style={{
              background: detailTab === tab ? 'var(--bg-card)' : 'transparent',
              color: detailTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: detailTab === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {detailTab === 'overview' && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <MetricCard label="Strategies Using" value={String(pack.strategiesUsing)} delta={pack.strategiesUsing > 0 ? 'Active' : 'Unused'} positive={pack.strategiesUsing > 0} />
            <MetricCard label="Win Rate" value={pack.totalSignals > 0 ? `${pack.winRate.toFixed(1)}%` : '---'} delta={pack.totalSignals > 0 ? `${pack.totalSignals} signals` : 'No data'} positive={pack.winRate >= 50} />
            <MetricCard label="Triggers" value={String(pack.triggers.length)} delta={`${pack.outputs.length} states`} positive />
          </div>
          <div className="rounded-xl p-8 flex items-center justify-center" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
            <IndicatorPreviewIcon pack={pack} size={140} />
          </div>
          <div>
            <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Output States</h4>
            <div className="flex flex-wrap gap-1.5">
              {pack.outputs.map((output) => (
                <span key={output} className="text-xs font-mono px-2.5 py-1 rounded-lg" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
                  {output}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {detailTab === 'params' && (
        <div className="space-y-4">
          {localParams.map((param) => (
            <div key={param.key} className="flex items-center gap-3 px-4 py-3 rounded-xl" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <label className="text-xs w-24" style={{ color: 'var(--text-muted)' }}>{param.label}</label>
              <div className="flex-1 relative h-8 flex items-center">
                <div className="absolute inset-x-0 h-1.5 rounded-full" style={{ background: 'var(--bg-primary)' }} />
                {(() => {
                  const pct = ((param.value - param.min) / (param.max - param.min)) * 100;
                  return (
                    <>
                      <div className="absolute left-0 h-1.5 rounded-full" style={{ width: `${pct}%`, background: 'var(--accent)', transition: 'width 0.15s ease' }} />
                      <input
                        type="range"
                        min={param.min}
                        max={param.max}
                        step={param.max <= 5 ? 0.1 : 1}
                        value={param.value}
                        onChange={(e) => updateParam(param.key, Number(e.target.value))}
                        className="absolute inset-x-0 w-full h-full opacity-0 cursor-pointer"
                        style={{ zIndex: 2 }}
                      />
                      <div className="absolute w-4 h-4 rounded-full border-2 shadow-lg pointer-events-none" style={{ left: `calc(${pct}% - 8px)`, background: 'var(--bg-card)', borderColor: 'var(--accent)', transition: 'left 0.15s ease' }} />
                    </>
                  );
                })()}
              </div>
              <span className="text-sm font-mono w-12 text-center rounded-md py-0.5" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
                {param.value}
              </span>
            </div>
          ))}
          <div className="flex gap-3 pt-2">
            <button className="px-4 py-2 rounded-lg text-sm font-medium flex-1" style={{ background: 'var(--accent)', color: 'white' }} onClick={() => { onUpdate(pack.id, localParams); onClose(); }}>
              Save Parameters
            </button>
            <button className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={() => setLocalParams(pack.params)}>
              Reset
            </button>
          </div>
        </div>
      )}

      {detailTab === 'triggers' && (
        <div className="space-y-2">
          {pack.triggers.map((trigger, i) => {
            const dir = directionColors[trigger.direction];
            return (
              <div key={i} className="flex items-center gap-3 px-4 py-3 rounded-xl" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <div className="w-8 h-8 rounded-lg flex items-center justify-center text-base" style={{ background: `${dir?.color}15`, color: dir?.color }}>
                  {dir?.icon}
                </div>
                <div className="flex-1">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-[10px]" style={{ color: dir?.color }}>{trigger.direction}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>{trigger.exec}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {detailTab === 'performance' && (
        <div className="space-y-4">
          {pack.totalSignals > 0 ? (
            <>
              <div className="grid grid-cols-3 gap-3">
                <MetricCard label="Win Rate" value={`${pack.winRate.toFixed(1)}%`} positive={pack.winRate >= 50} />
                <MetricCard label="Total Signals" value={String(pack.totalSignals)} positive />
                <MetricCard label="Strategies" value={String(pack.strategiesUsing)} positive={pack.strategiesUsing > 0} />
              </div>
              {/* Win rate visualization */}
              <div className="rounded-xl p-4" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Win/Loss Distribution</h4>
                <div className="flex rounded-full overflow-hidden h-6">
                  <div style={{ width: `${pack.winRate}%`, background: 'var(--green)' }} className="flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white">{pack.winRate.toFixed(0)}% W</span>
                  </div>
                  <div style={{ width: `${100 - pack.winRate}%`, background: 'var(--red)' }} className="flex items-center justify-center">
                    <span className="text-[10px] font-bold text-white">{(100 - pack.winRate).toFixed(0)}% L</span>
                  </div>
                </div>
              </div>
              {/* Signal frequency */}
              <div className="rounded-xl p-4" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <h4 className="text-xs font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Signal History (last 14 days)</h4>
                <div className="flex items-end gap-1 h-16">
                  {Array.from({ length: 14 }, (_, i) => {
                    const h = Math.floor(Math.random() * 100);
                    return (
                      <div key={i} className="flex-1 rounded-t" style={{ height: `${h}%`, background: h > 60 ? 'var(--green)' : 'var(--accent)', opacity: 0.5 + h / 200 }} />
                    );
                  })}
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-12">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1" className="mx-auto mb-3">
                <circle cx="12" cy="12" r="10" />
                <path d="M8 15h8M9 9h.01M15 9h.01" />
              </svg>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No performance data yet</p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Add this pack to a strategy and run a backtest to generate data.</p>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}

/* ═══════════════════════════ Main Component ═══════════════════════════ */

export default function UserPacksV4() {
  const [myPacks, setMyPacks] = useState<UserPack[]>(initialPacks);
  const [detailPack, setDetailPack] = useState<UserPack | null>(null);
  const [showNewModal, setShowNewModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'my-packs' | 'marketplace' | 'leaderboard'>('my-packs');
  const [searchQuery, setSearchQuery] = useState('');

  const togglePack = useCallback((id: string) => {
    setMyPacks((prev) => prev.map((p) => (p.id === id ? { ...p, enabled: !p.enabled } : p)));
  }, []);

  const updatePackParams = useCallback((id: string, params: PackParam[]) => {
    setMyPacks((prev) => prev.map((p) => (p.id === id ? { ...p, params } : p)));
  }, []);

  const handleInstall = useCallback((pack: UserPack) => {
    const installed: UserPack = { ...pack, id: `installed-${pack.id}`, author: 'You (installed)', enabled: false, strategiesUsing: 0, winRate: 0, totalSignals: 0, downloads: 0 };
    setMyPacks((prev) => [...prev, installed]);
  }, []);

  const enabledCount = myPacks.filter((p) => p.enabled).length;
  const totalSignals = myPacks.reduce((sum, p) => sum + p.totalSignals, 0);
  const avgWinRate = myPacks.filter((p) => p.totalSignals > 0);
  const overallWinRate = avgWinRate.length > 0 ? avgWinRate.reduce((sum, p) => sum + p.winRate, 0) / avgWinRate.length : 0;

  const filtered = myPacks.filter((p) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return p.name.toLowerCase().includes(q) || p.category.toLowerCase().includes(q);
  });

  return (
    <div>
      <style>{`
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .user-pack-card { animation: fade-in-up 0.4s ease forwards; }
        .user-pack-card:nth-child(1) { animation-delay: 0.0s; }
        .user-pack-card:nth-child(2) { animation-delay: 0.05s; }
        .user-pack-card:nth-child(3) { animation-delay: 0.1s; }
        .user-pack-card:nth-child(4) { animation-delay: 0.15s; }
        .user-pack-card:nth-child(5) { animation-delay: 0.2s; }
      `}</style>

      <PageHeader
        title="User Packs"
        subtitle="Create custom indicator packs, browse community creations, and track performance"
        actions={
          <button
            onClick={() => setShowNewModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'white' }}
          >
            + Create Pack
          </button>
        }
      />

      {/* Summary */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        <MetricCard label="My Packs" value={String(myPacks.length)} delta={`${enabledCount} enabled`} positive />
        <MetricCard label="Total Signals" value={String(totalSignals)} delta="All time" positive />
        <MetricCard label="Avg Win Rate" value={overallWinRate > 0 ? `${overallWinRate.toFixed(1)}%` : '---'} delta={avgWinRate.length > 0 ? `${avgWinRate.length} packs with data` : 'No data'} positive={overallWinRate >= 50} />
        <MetricCard label="Community" value={String(communityPacks.length)} delta="Packs available" positive />
      </div>

      {/* Tab navigation */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl w-fit" style={{ background: 'var(--bg-input)' }}>
        {([
          { key: 'my-packs' as const, label: 'My Packs', count: myPacks.length },
          { key: 'marketplace' as const, label: 'Marketplace', count: communityPacks.length },
          { key: 'leaderboard' as const, label: 'Leaderboard', count: null },
        ]).map(({ key, label, count }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className="px-4 py-2 text-xs font-medium rounded-lg transition-all flex items-center gap-2"
            style={{
              background: activeTab === key ? 'var(--bg-card)' : 'transparent',
              color: activeTab === key ? 'var(--text-primary)' : 'var(--text-muted)',
              boxShadow: activeTab === key ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
            }}
          >
            {label}
            {count !== null && (
              <span className="text-[9px] px-1.5 py-0.5 rounded-full" style={{ background: activeTab === key ? 'var(--accent-muted)' : 'var(--bg-primary)', color: activeTab === key ? 'var(--accent)' : 'var(--text-muted)' }}>
                {count}
              </span>
            )}
          </button>
        ))}
      </div>

      {activeTab === 'my-packs' && (
        <>
          {/* Search */}
          <div className="relative max-w-xs mb-5">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              type="text"
              placeholder="Search packs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
            />
          </div>

          {filtered.length === 0 ? (
            <Card>
              <div className="text-center py-12">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1" className="mx-auto mb-4">
                  <rect x="3" y="3" width="18" height="18" rx="2" />
                  <line x1="12" y1="8" x2="12" y2="16" />
                  <line x1="8" y1="12" x2="16" y2="12" />
                </svg>
                <h3 className="text-lg font-semibold mb-1">No packs yet</h3>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Create your first custom indicator pack or browse the marketplace.</p>
                <div className="flex gap-3 justify-center mt-4">
                  <button onClick={() => setShowNewModal(true)} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>
                    Create Pack
                  </button>
                  <button onClick={() => setActiveTab('marketplace')} className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>
                    Browse Marketplace
                  </button>
                </div>
              </div>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {filtered.map((pack) => (
                <div key={pack.id} className="user-pack-card">
                  <Card className={`relative overflow-hidden ${!pack.enabled ? 'opacity-60' : ''}`}>
                    <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: pack.enabled ? `linear-gradient(90deg, ${categoryColors[pack.category]?.color || 'var(--accent)'}, transparent)` : 'var(--border)' }} />

                    <div className="flex gap-4">
                      <div
                        className="w-16 h-16 rounded-xl flex items-center justify-center flex-shrink-0 cursor-pointer transition-transform hover:scale-105"
                        style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}
                        onClick={() => setDetailPack(pack)}
                      >
                        <IndicatorPreviewIcon pack={pack} size={48} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-1.5">
                          <div>
                            <div className="flex items-center gap-2 mb-0.5">
                              <button onClick={() => setDetailPack(pack)} className="font-semibold text-sm hover:underline" style={{ color: 'var(--text-primary)' }}>
                                {pack.name}
                              </button>
                              <span className="text-[10px] px-1.5 py-0.5 rounded" style={{ background: categoryColors[pack.category]?.bg, color: categoryColors[pack.category]?.color }}>
                                {pack.category}
                              </span>
                            </div>
                            <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{pack.description}</p>
                          </div>
                          <button
                            onClick={() => togglePack(pack.id)}
                            className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
                            style={{ background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)', border: pack.enabled ? 'none' : '1px solid var(--border)' }}
                          >
                            <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: pack.enabled ? 'white' : 'var(--text-muted)', left: pack.enabled ? '22px' : '4px' }} />
                          </button>
                        </div>

                        <div className="flex items-center gap-4 mt-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                          <span>{pack.strategiesUsing} strategies</span>
                          <PerformanceBadge winRate={pack.winRate} signals={pack.totalSignals} />
                          <span>{pack.triggers.length} triggers</span>
                        </div>

                        <div className="flex flex-wrap gap-1.5 mt-2.5">
                          {pack.params.map((p) => (
                            <span key={p.key} className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                              {p.label}: {p.value}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {activeTab === 'marketplace' && (
        <MarketplaceBrowser packs={communityPacks} onInstall={handleInstall} />
      )}

      {activeTab === 'leaderboard' && (
        <PackLeaderboard packs={[...myPacks, ...communityPacks]} />
      )}

      {/* Detail Modal */}
      {detailPack && (
        <PackDetailModal pack={detailPack} onClose={() => setDetailPack(null)} onUpdate={updatePackParams} />
      )}

      {/* New Pack Modal */}
      <Modal title="Create Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="560px">
        <NewPackForm onClose={() => setShowNewModal(false)} />
      </Modal>
    </div>
  );
}

/* ─────────────────────────── New Pack Form ─────────────────────────── */

function NewPackForm({ onClose }: { onClose: () => void }) {
  const [packName, setPackName] = useState('');
  const [category, setCategory] = useState('Momentum');
  const [createMethod, setCreateMethod] = useState<'scratch' | 'template' | 'marketplace'>('scratch');
  const [selectedTemplate, setSelectedTemplate] = useState('');

  const categories = ['Momentum', 'Moving Averages', 'Volume', 'Volatility', 'Trend', 'Custom'];
  const templates = ['RSI', 'Stochastic', 'Bollinger Bands', 'ADX', 'Williams %R', 'CCI'];

  return (
    <div className="space-y-5">
      <div>
        <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Pack Name</label>
        <input
          className="w-full px-3 py-2 rounded-lg text-sm"
          style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
          placeholder="My Custom Pack"
          value={packName}
          onChange={(e) => setPackName(e.target.value)}
        />
      </div>

      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Category</label>
        <div className="flex flex-wrap gap-2">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
              style={{
                background: category === cat ? categoryColors[cat]?.color || 'var(--accent)' : 'var(--bg-input)',
                color: category === cat ? 'white' : 'var(--text-secondary)',
                border: category === cat ? 'none' : '1px solid var(--border)',
              }}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Start From</label>
        <div className="grid grid-cols-3 gap-2">
          {([
            { key: 'scratch' as const, label: 'Blank', desc: 'Start from scratch' },
            { key: 'template' as const, label: 'Template', desc: 'Use a preset' },
            { key: 'marketplace' as const, label: 'Clone', desc: 'Copy community pack' },
          ]).map(({ key, label, desc }) => (
            <button
              key={key}
              onClick={() => setCreateMethod(key)}
              className="px-3 py-3 rounded-xl text-left transition-all"
              style={{
                background: createMethod === key ? 'var(--accent-muted)' : 'var(--bg-input)',
                border: createMethod === key ? '1px solid var(--accent)' : '1px solid var(--border)',
              }}
            >
              <span className="text-sm font-medium block" style={{ color: createMethod === key ? 'var(--accent)' : 'var(--text-primary)' }}>{label}</span>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{desc}</span>
            </button>
          ))}
        </div>
      </div>

      {createMethod === 'template' && (
        <div>
          <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Template</label>
          <div className="grid grid-cols-2 gap-2">
            {templates.map((t) => (
              <button
                key={t}
                onClick={() => setSelectedTemplate(t)}
                className="px-3 py-2 rounded-lg text-sm text-left transition-all"
                style={{
                  background: selectedTemplate === t ? 'var(--accent-muted)' : 'var(--bg-input)',
                  border: selectedTemplate === t ? '1px solid var(--accent)' : '1px solid var(--border)',
                  color: selectedTemplate === t ? 'var(--accent)' : 'var(--text-primary)',
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-3 pt-1">
        <button className="px-4 py-2.5 rounded-lg text-sm font-medium flex-1" style={{ background: 'var(--accent)', color: 'white' }} onClick={onClose}>
          Create Pack
        </button>
        <button className="px-4 py-2.5 rounded-lg text-sm flex-1" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={onClose}>
          Cancel
        </button>
      </div>
    </div>
  );
}
