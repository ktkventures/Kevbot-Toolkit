'use client';

// Admin — Trade Snapshots. Capture every backtest-lane trade now, again later,
// and diff to the EXACT added/removed/changed trades (the bit-perfect-vs-before
// check a count/KPI diff can't give). Backs /api/admin/trade-snapshots.
import { useCallback, useEffect, useMemo, useState } from 'react';
import { apiFetch } from '@/lib/api/client';

interface Snapshot {
  id: number;
  label: string;
  status: 'running' | 'ready' | 'failed';
  n_strategies: number;
  n_trades: number;
  error?: string | null;
  created_at: string;
  completed_at?: string | null;
}

interface DiffPer {
  strategy_id: number;
  n_before: number;
  n_after: number;
  added: number;
  removed: number;
  changed: number;
  sample_changed?: Array<{ key: string; diff: Record<string, [unknown, unknown]> }>;
}
interface DiffResult {
  summary: {
    added: number; removed: number; changed: number;
    identical_strats: number; n_strategies: number; changed_strategies: number;
  };
  per_strategy: DiffPer[];
}

export default function TradeSnapshotsAdminPage() {
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [label, setLabel] = useState('');
  const [busy, setBusy] = useState(false);
  const [beforeId, setBeforeId] = useState<number | ''>('');
  const [afterId, setAfterId] = useState<number | ''>('');
  const [diff, setDiff] = useState<DiffResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await apiFetch<{ snapshots: Snapshot[] }>('/api/admin/trade-snapshots');
      setSnapshots(data.snapshots || []);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Poll while any snapshot is still capturing.
  const anyRunning = useMemo(
    () => snapshots.some((s) => s.status === 'running'), [snapshots]);
  useEffect(() => {
    if (!anyRunning) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [anyRunning, load]);

  const take = async () => {
    setBusy(true); setErr(null);
    try {
      await apiFetch('/api/admin/trade-snapshots', {
        method: 'POST',
        body: JSON.stringify({ label: label.trim() || undefined }),
      });
      setLabel('');
      await load();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const runDiff = async () => {
    if (!beforeId || !afterId) return;
    setBusy(true); setErr(null); setDiff(null);
    try {
      const d = await apiFetch<DiffResult>('/api/admin/trade-snapshots/diff', {
        method: 'POST',
        body: JSON.stringify({ before_id: beforeId, after_id: afterId }),
      });
      setDiff(d);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const del = async (id: number) => {
    if (!confirm('Delete this snapshot?')) return;
    try { await apiFetch(`/api/admin/trade-snapshots/${id}`, { method: 'DELETE' }); await load(); }
    catch (e) { setErr(String(e)); }
  };

  const ready = snapshots.filter((s) => s.status === 'ready');

  return (
    <div style={{ padding: 24, maxWidth: 1000 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>Trade Snapshots</h1>
      <p style={{ color: '#666', marginBottom: 16 }}>
        Capture every backtest-lane trade, then diff two snapshots to the exact
        added / removed / changed trades — the bit-perfect fidelity check across
        recompute runs. Snapshot before a fleet recompute, again after, then diff.
      </p>

      {err && <div style={{ color: '#b91c1c', marginBottom: 12 }}>{err}</div>}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="label (e.g. before-fleet-2026-06-28)"
          style={{ flex: 1, padding: 8, border: '1px solid #ccc', borderRadius: 6 }}
        />
        <button onClick={take} disabled={busy}
          style={{ padding: '8px 16px', background: '#111', color: '#fff', borderRadius: 6 }}>
          {busy ? 'Working…' : 'Take snapshot (all strategies)'}
        </button>
      </div>

      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 24 }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #ddd' }}>
            <th>ID</th><th>Label</th><th>Status</th><th>Strategies</th>
            <th>Trades</th><th>Created</th><th></th>
          </tr>
        </thead>
        <tbody>
          {snapshots.map((s) => (
            <tr key={s.id} style={{ borderBottom: '1px solid #f0f0f0' }}>
              <td>{s.id}</td>
              <td>{s.label}</td>
              <td style={{ color: s.status === 'ready' ? '#15803d' : s.status === 'failed' ? '#b91c1c' : '#a16207' }}>
                {s.status}{s.error ? ` (${s.error})` : ''}
              </td>
              <td>{s.n_strategies}</td>
              <td>{s.n_trades.toLocaleString()}</td>
              <td>{new Date(s.created_at).toLocaleString()}</td>
              <td><button onClick={() => del(s.id)} style={{ color: '#b91c1c' }}>delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2 style={{ fontSize: 18, fontWeight: 600 }}>Diff two snapshots</h2>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', margin: '12px 0' }}>
        <select value={beforeId} onChange={(e) => setBeforeId(Number(e.target.value) || '')}
          style={{ padding: 8 }}>
          <option value="">before…</option>
          {ready.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.label}</option>)}
        </select>
        <span>→</span>
        <select value={afterId} onChange={(e) => setAfterId(Number(e.target.value) || '')}
          style={{ padding: 8 }}>
          <option value="">after…</option>
          {ready.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.label}</option>)}
        </select>
        <button onClick={runDiff} disabled={busy || !beforeId || !afterId}
          style={{ padding: '8px 16px', background: '#111', color: '#fff', borderRadius: 6 }}>
          Diff
        </button>
      </div>

      {diff && (
        <div>
          <div style={{ margin: '8px 0', fontWeight: 600 }}>
            {diff.summary.identical_strats}/{diff.summary.n_strategies} strategies
            trade-for-trade identical · added {diff.summary.added} · removed{' '}
            {diff.summary.removed} · changed {diff.summary.changed}
          </div>
          {diff.per_strategy.length === 0 ? (
            <div style={{ color: '#15803d' }}>✓ Every strategy is bit-identical.</div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ textAlign: 'left', borderBottom: '1px solid #ddd' }}>
                  <th>Strategy</th><th>n before→after</th><th>+added</th>
                  <th>−removed</th><th>~changed</th>
                </tr>
              </thead>
              <tbody>
                {diff.per_strategy.map((p) => (
                  <tr key={p.strategy_id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                    <td>{p.strategy_id}</td>
                    <td>{p.n_before} → {p.n_after}</td>
                    <td style={{ color: '#15803d' }}>+{p.added}</td>
                    <td style={{ color: '#b91c1c' }}>−{p.removed}</td>
                    <td style={{ color: '#a16207' }}>~{p.changed}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
