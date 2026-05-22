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
  backtest: 'Backtest / Algo Models',
};

const KIND_DESCRIPTION: Record<ModelKind, string> = {
  live: (
    'Live models govern how the engine sources 1Min bars for real-time '
    + 'alert decisions. Per-strategy selection. Existing strategies always '
    + 'keep whatever they were created with — improving the registry never '
    + 'invalidates older work.'
  ),
  backtest: (
    'Backtest / Algo models share a single registry — same model entries '
    + 'serve two consumers. The strategy\'s `backtest_model` field controls '
    + 'how stored_trades is repopulated (KPI baseline). The strategy\'s '
    + '`algo_model` field controls how the cron\'s incremental algo-history '
    + 'append runs (live-accountability lane). Both are per-strategy and '
    + 'independent of live_model. Default: backtest_model=rest_hifi, '
    + 'algo_model=cache_locked.'
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

// Default-badge styles — backtest role is blue, algo role is violet so
// the two can sit side by side on a card without ambiguity.
const BACKTEST_BADGE: React.CSSProperties = {
  background: 'rgba(33,150,243,0.15)', color: '#64b5f6',
};
const ALGO_BADGE: React.CSSProperties = {
  background: 'rgba(156,39,176,0.18)', color: '#ce93d8',
};

function ModelCard({ row, kind, onClick }: {
  row: StrategyModelRow;
  kind: ModelKind;
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
                    style={BACKTEST_BADGE}>
                {kind === 'backtest' ? 'Default · Backtest' : 'Default'}
              </span>
            )}
            {row.algo_default && (
              <span className="text-[10px] px-2 py-0.5 rounded font-medium"
                    style={ALGO_BADGE}>
                Default · ALGO
              </span>
            )}
          </div>
          <p className="text-xs mb-2" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            {desc}
          </p>
          {kind === 'backtest' ? (
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              <strong style={{ color: 'var(--text-primary)' }}>{row.strategies_using}</strong>
              {' '}as backtest model
              {'  ·  '}
              <strong style={{ color: 'var(--text-primary)' }}>{row.algo_strategies_using ?? 0}</strong>
              {' '}as algo model
            </p>
          ) : (
            <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
              <strong style={{ color: 'var(--text-primary)' }}>{row.strategies_using}</strong>
              {' '}strateg{row.strategies_using === 1 ? 'y' : 'ies'} using this model
            </p>
          )}
        </div>
        <div className="text-xl select-none" style={{ color: 'var(--text-muted)' }}>
          ›
        </div>
        </div>
      </Card>
    </div>
  );
}

function StrategyLinkList({ title, count, strategies }: {
  title: string;
  count: number;
  strategies: { id: number; name: string }[];
}) {
  return (
    <Card>
      <h3 className="text-sm font-medium mb-3">
        {title}
        <span className="ml-2 text-xs font-normal" style={{ color: 'var(--text-muted)' }}>
          ({count})
        </span>
      </h3>
      {strategies.length === 0 ? (
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          No strategies are currently using this model.
        </p>
      ) : (
        <div className="space-y-1.5">
          {strategies.map(s => (
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
                  style={BACKTEST_BADGE}>
              {kind === 'backtest' ? 'Default · Backtest Model' : 'Platform Default'}
            </span>
          )}
          {data.algo_default && (
            <span className="text-xs px-2 py-1 rounded font-medium"
                  style={ALGO_BADGE}>
              Default · Algo Model
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

      <StrategyLinkList
        title={kind === 'backtest'
          ? 'Strategies using this as backtest model'
          : 'Strategies using this model'}
        count={data.strategies_using}
        strategies={data.strategies}
      />

      {kind === 'backtest' && (
        <StrategyLinkList
          title="Strategies using this as algo model"
          count={data.algo_strategies_using ?? 0}
          strategies={data.algo_strategies ?? []}
        />
      )}

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
          <ModelCard key={row.id} row={row} kind={kind} onClick={() => setSelectedId(row.id)} />
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
