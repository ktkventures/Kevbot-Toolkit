/**
 * Agents Registry — /admin/agents (Spec_Agents_Registry.md Phase 1).
 *
 * The WHO leg of the hub (Tasks = what · Roadmap = why · Agents = who):
 * read-only roster cards grouped by department — letter avatar, status chip,
 * scope/boundaries (expandable), worktree, context docs, and the agent's
 * live queue (open tasks joined client-side from /api/dev-tasks). No
 * dispatch, no editing this phase; M manages rows via the CRUD API.
 * Until the V4.9 dispatcher ships, Session_Charters.md §1 stays the SSOT
 * and this registry mirrors it.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import { Task, RoleChip, roleColor, tagChip, badge } from './taskBoardShared';

export interface Agent {
  id: number;
  letter: string;
  name: string;
  kind: string;
  department: string;
  status: string;
  scope: string;
  boundaries: string;
  worktree: string;
  context_docs: string[];
  prompt_template: string | null;
  notes: string;
  updated_at: string;
}

const STATUS_BG: Record<string, string> = {
  'live-session': '#2e7d32', 'headless': '#2563ab', 'dormant': '#555f6b',
  'retired': '#8a3038', 'ephemeral': '#2aa8a0',
};
const DEPARTMENTS = ['dev', 'builder', 'marketing', 'ops'];

export default function AdminAgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const load = useCallback(async () => {
    try {
      setErr(null);
      const [ag, ts] = await Promise.all([
        apiFetch<Agent[]>('/api/agents'),
        apiFetch<Task[]>('/api/dev-tasks?include_done=false'),
      ]);
      setAgents(ag || []);
      setTasks(ts || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const queues = useMemo(() => {
    const m = new Map<string, Task[]>();
    tasks.forEach((t) => {
      const a = t.assignee || '';
      if (a) m.set(a, [...(m.get(a) || []), t]);
    });
    return m;  // API order is priority order — top of list = "do next"
  }, [tasks]);

  const byDept = useMemo(() => {
    const m = new Map<string, Agent[]>();
    DEPARTMENTS.forEach((d) => m.set(d, []));
    agents.forEach((a) => m.set(a.department, [...(m.get(a.department) || []), a]));
    return m;
  }, [agents]);

  const toggle = (id: number) => setExpanded((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Agents</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 16 }}>
        The team roster — who owns what, what they must not touch, and what each agent is
        working on right now (live from the task board). Read-only; scope changes go
        through the charter. Statuses: live-session · headless · dormant · retired · ephemeral.
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}
      {loading && <Card>Loading…</Card>}

      {!loading && DEPARTMENTS.map((dept) => {
        const rows = byDept.get(dept) || [];
        if (rows.length === 0) return null;
        return (
          <div key={dept} style={{ marginBottom: 20 }}>
            <div style={{
              fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase',
              color: 'var(--text-tertiary)', margin: '0 0 8px 2px',
            }}>{dept}</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
              {rows.map((a) => {
                const queue = queues.get(a.letter) || [];
                const open = expanded.has(a.id);
                return (
                  <div key={a.id} style={{
                    border: `1px solid ${a.kind === 'human' ? roleColor('kevin') : 'var(--border)'}`,
                    borderRadius: 10, padding: 14, background: 'var(--bg-card, var(--bg-input))',
                    display: 'flex', flexDirection: 'column', gap: 8,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <RoleChip role={a.letter} title={a.letter} />
                      <span style={{ fontWeight: 600, fontSize: 14, flex: 1, minWidth: 0 }}>
                        {a.letter} — {a.name}
                      </span>
                      {a.kind === 'human' &&
                        <span style={{ ...tagChip, borderColor: roleColor('kevin'), color: roleColor('kevin') }}>human</span>}
                      <span style={badge(STATUS_BG[a.status] || '#555f6b')}>{a.status}</span>
                    </div>

                    <div style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
                      <span style={{ cursor: 'pointer' }} onClick={() => toggle(a.id)}
                        title="expand scope / boundaries">
                        {open ? '▾' : '▸'} scope & boundaries
                      </span>
                      {open && (
                        <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 6 }}>
                          <div><b style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>OWNS</b>
                            <div style={{ whiteSpace: 'pre-wrap' }}>{a.scope || '—'}</div></div>
                          <div><b style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>MUST NOT TOUCH</b>
                            <div style={{ whiteSpace: 'pre-wrap' }}>{a.boundaries || '—'}</div></div>
                          {a.notes && (
                            <div><b style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>NOTES</b>
                              <div style={{ whiteSpace: 'pre-wrap' }}>{a.notes}</div></div>
                          )}
                        </div>
                      )}
                    </div>

                    {a.worktree && (
                      <div style={{ fontSize: 11.5, color: 'var(--text-tertiary)', fontFamily: 'monospace' }}
                        title="worktree">⌂ {a.worktree}</div>
                    )}
                    {(a.context_docs || []).length > 0 && (
                      <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }} title="standing context docs">
                        {(a.context_docs || []).map((d) => (
                          <div key={d} style={{ fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>📄 {d}</div>
                        ))}
                      </div>
                    )}

                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 8, marginTop: 'auto' }}>
                      <Link href="/admin/tasks" style={{ textDecoration: 'none', color: 'inherit' }}>
                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
                          queue: {queue.length} open{queue.length > 0 ? ' →' : ''}
                        </div>
                        {queue.slice(0, 3).map((t) => (
                          <div key={t.id} style={{
                            fontSize: 12, color: 'var(--text-secondary)', overflow: 'hidden',
                            textOverflow: 'ellipsis', whiteSpace: 'nowrap', padding: '1px 0',
                          }}>
                            <span style={{ color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>#{t.id}</span> {t.title}
                          </div>
                        ))}
                        {queue.length === 0 &&
                          <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>nothing assigned</div>}
                      </Link>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
