'use client';

/**
 * Strategy Models admin page (Phase B of plans/synchronous-tickling-yeti.md).
 *
 * Two exported components — `LiveModelsAdminPage` and
 * `BacktestModelsAdminPage` — share a parameterized body driven by `kind`.
 * Style mirrors UserPacksPage: vertical Card stack, click-through
 * detail with a Back button (no modal).
 *
 * Each card shows:
 *   - Model label + ID badge (mono)
 *   - Status pill (available / coming-soon / unknown)
 *   - Default badge (when this is the platform default)
 *   - "N strategies using this" + link affordance
 *   - 1-line description excerpt + chevron right
 *
 * Detail view shows:
 *   - Full description
 *   - Behavior table (status, default flag, strategies-using)
 *   - List of strategies currently selecting the model, with links to
 *     /strategies/{id}
 */

import { useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import {
  useStrategyModelsList,
  useStrategyModelDetail,
  type ModelKind,
  type ModelStatus,
  type StrategyModelRow,
} from '@/hooks/queries/useStrategyModelsAdmin';

interface PageProps {
  kind: ModelKind;
}

const KIND_LABEL: Record<ModelKind, string> = {
  live: 'Live Models',
  backtest: 'Backtest Models',
};

const KIND_DESCRIPTION: Record<ModelKind, string> = {
  live: (
    'Live models govern how the engine sources 1Min bars for real-time '
    + 'alert decisions. Per-strategy selection. Existing strategies always '
    + 'keep whatever they were created with — improving the registry never '
    + 'invalidates older work.'
  ),
  backtest: (
    'Backtest models govern how stored_trades is repopulated and how '
    + 'Chart & Trades loads historical bars. Per-strategy selection. '
    + 'Independent of the live model — same strategy can use different '
    + 'sources for live vs backtest.'
  ),
};

function statusPillStyle(status: ModelStatus): React.CSSProperties {
  if (status === 'available') {
    return { background: 'rgba(76,175,80,0.15)', color: '#7fd081' };
  }
  if (status === 'coming_soon') {
    return { background: 'rgba(255,193,7,0.15)', color: '#ffc107' };
  }
  return { background: 'rgba(120,120,120,0.18)', color: 'var(--text-muted)' };
}

function statusLabel(status: ModelStatus): string {
  if (status === 'available') return 'Available';
  if (status === 'coming_soon') return 'Coming Soon';
  return 'Unknown';
}

function ModelCard({ row, onClick }: {
  row: StrategyModelRow;
  onClick: () => void;
}) {
  const desc = row.description.length > 140
    ? row.description.slice(0, 137) + '...'
    : row.description;
  return (
    <div onClick={onClick} style={{ cursor: 'pointer' }}>
      <Card className="transition-colors">
        <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <h3 className="text-base font-semibold">{row.label}</h3>
            <code className="text-[10px] px-1.5 py-0.5 rounded"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
              {row.id}
            </code>
            <span className="text-[10px] px-2 py-0.5 rounded font-medium"
                  style={statusPillStyle(row.status)}>
              {statusLabel(row.status)}
            </span>
            {row.default && (
              <span className="text-[10px] px-2 py-0.5 rounded font-medium"
                    style={{ background: 'rgba(33,150,243,0.15)', color: '#64b5f6' }}>
                Default
              </span>
            )}
          </div>
          <p className="text-xs mb-2" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {desc}
          </p>
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-primary)' }}>{row.strategies_using}</strong>
            {' '}strateg{row.strategies_using === 1 ? 'y' : 'ies'} using this model
          </p>
        </div>
        <div className="text-xl select-none" style={{ color: 'var(--text-muted)' }}>
          ›
        </div>
        </div>
      </Card>
    </div>
  );
}

function ModelDetail({ kind, id, onBack }: {
  kind: ModelKind; id: string; onBack: () => void;
}) {
  const { data, isLoading, error } = useStrategyModelDetail(kind, id);

  if (isLoading) {
    return (
      <div className="p-4">
        <button onClick={onBack} className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          ← Back
        </button>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading…</p>
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="p-4">
        <button onClick={onBack} className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
          ← Back
        </button>
        <p className="text-sm" style={{ color: 'var(--red)' }}>
          Failed to load: {String((error as any)?.message || error || 'unknown error')}
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <button onClick={onBack} className="text-xs" style={{ color: 'var(--text-muted)', cursor: 'pointer' }}>
        ← Back to {KIND_LABEL[kind]}
      </button>

      <div>
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <h1 className="text-2xl font-semibold">{data.label}</h1>
          <code className="text-xs px-2 py-1 rounded"
                style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
            {data.id}
          </code>
          <span className="text-xs px-2 py-1 rounded font-medium"
                style={statusPillStyle(data.status)}>
            {statusLabel(data.status)}
          </span>
          {data.default && (
            <span className="text-xs px-2 py-1 rounded font-medium"
                  style={{ background: 'rgba(33,150,243,0.15)', color: '#64b5f6' }}>
              Platform Default
            </span>
          )}
        </div>
      </div>

      <Card>
        <h3 className="text-sm font-medium mb-2">Description</h3>
        <p className="text-xs" style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          {data.description}
        </p>
      </Card>

      <Card>
        <h3 className="text-sm font-medium mb-3">
          Strategies using this model
          <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
            ({data.strategies_using})
          </span>
        </h3>
        {data.strategies.length === 0 ? (
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            No strategies are currently using this model.
          </p>
        ) : (
          <div className="space-y-1.5">
            {data.strategies.map(s => (
              <Link key={s.id} href={`/strategies/${s.id}`}
                    className="flex items-center gap-3 text-xs hover:underline"
                    style={{ color: 'var(--text-primary)' }}>
                <code className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>
                  #{s.id}
                </code>
                <span>{s.name || '(unnamed)'}</span>
              </Link>
            ))}
          </div>
        )}
      </Card>

      {data.status === 'coming_soon' && (
        <Card>
          <p className="text-xs" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            <strong style={{ color: '#ffc107' }}>Coming soon:</strong>{' '}
            this model is registered but not yet wired into the engine /
            backtest dispatch path. Strategies cannot select it from the
            ModelsCard yet. See{' '}
            <code style={{ background: 'var(--bg-input)', padding: '1px 4px', borderRadius: 3 }}>
              docs/Live_Model_Decision.md
            </code>{' '}
            for the rollout plan.
          </p>
        </Card>
      )}
    </div>
  );
}

function StrategyModelsAdminBody({ kind }: PageProps) {
  const { data, isLoading, error } = useStrategyModelsList(kind);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  if (selectedId) {
    return <ModelDetail kind={kind} id={selectedId} onBack={() => setSelectedId(null)} />;
  }

  if (isLoading) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading {KIND_LABEL[kind]}…</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="p-6">
        <p className="text-sm" style={{ color: 'var(--red)' }}>
          Failed to load: {String((error as any).message || error)}
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <div>
        <h1 className="text-2xl font-semibold mb-1">{KIND_LABEL[kind]}</h1>
        <p className="text-xs" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {KIND_DESCRIPTION[kind]}
        </p>
      </div>
      <div className="space-y-3">
        {data?.rows.map(row => (
          <ModelCard key={row.id} row={row} onClick={() => setSelectedId(row.id)} />
        ))}
        {(!data || data.rows.length === 0) && (
          <Card>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              No models registered.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}

export function LiveModelsAdminPage() {
  return <StrategyModelsAdminBody kind="live" />;
}

export function BacktestModelsAdminPage() {
  return <StrategyModelsAdminBody kind="backtest" />;
}

export default StrategyModelsAdminBody;
