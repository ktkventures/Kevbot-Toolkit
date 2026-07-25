/**
 * Agents Registry — /admin/agents (Spec_Agents_Registry.md Phase 1 + the
 * board-#109 Phase 2 additions).
 *
 * The WHO leg of the hub (Tasks = what · Roadmap = why · Agents = who):
 * roster cards grouped by department — letter avatar, status DROPDOWN
 * (dormant/live-session/headless; the headless enrollment flip stays M-only
 * by convention), scope/boundaries (expandable), prompt-template editor
 * (the dispatcher identity prompt, PATCHed in place), recent dispatcher
 * runs (run_history), worktree, context docs, and the agent's live queue
 * (open tasks joined client-side from /api/dev-tasks).
 * Until the V4.9 dispatcher ships, Session_Charters.md §1 stays the SSOT
 * for scope/boundaries and this registry mirrors it.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import { Task, RunRow, RoleChip, roleColor, tagChip, badge, input, OutcomeChip, relTime } from './taskBoardShared';

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
// The operational flips (board #109). Other statuses (retired/ephemeral)
// still render as the current value; they're just not offered as targets.
const STATUS_CHOICES = ['dormant', 'live-session', 'headless'];

export default function AdminAgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  // Prompt-template editor state (board #109): open cards + unsaved drafts.
  const [tplOpen, setTplOpen] = useState<Set<number>>(new Set());
  const [tplDraft, setTplDraft] = useState<Record<number, string>>({});
  const [saving, setSaving] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      setErr(null);
      const [ag, ts, rh] = await Promise.all([
        apiFetch<Agent[]>('/api/agents'),
        apiFetch<Task[]>('/api/dev-tasks?include_done=false'),
        apiFetch<RunRow[]>('/api/run-history?limit=300').catch(() => [] as RunRow[]),
      ]);
      setAgents(ag || []);
      setTasks(ts || []);
      setRuns(rh || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const patchAgent = async (id: number, fields: Partial<Agent>) => {
    try {
      await apiFetch(`/api/agents/${id}`, {
        method: 'PATCH', body: JSON.stringify(fields),
      });
    } catch (e) { setErr(String(e)); }
    await load();
  };
  const saveTpl = async (a: Agent) => {
    const draft = tplDraft[a.id] ?? (a.prompt_template || '');
    setSaving(a.id);
    await patchAgent(a.id, { prompt_template: draft });
    setSaving(null);
  };
  const toggleTpl = (id: number) => setTplOpen((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const runsByAgent = useMemo(() => {
    const m = new Map<string, RunRow[]>();
    runs.forEach((r) => m.set(r.agent_letter, [...(m.get(r.agent_letter) || []), r]));
    return m;  // API order is newest-first
  }, [runs]);

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
        working on right now (live from the task board). Scope changes go through the
        charter; status + prompt template edit here (the &apos;headless&apos; flip is M-only by
        convention). Statuses: live-session · headless · dormant · retired · ephemeral.
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
                      {a.kind === 'human'
                        ? <span style={badge(STATUS_BG[a.status] || '#555f6b')}>{a.status}</span>
                        : (
                          <select value={a.status}
                            title="registry status — 'headless' enrolls the agent for dispatcher runs; that flip stays M-only by convention"
                            style={{
                              border: 'none', borderRadius: 10, padding: '1px 4px',
                              fontSize: 11, fontWeight: 600, color: '#fff', cursor: 'pointer',
                              background: STATUS_BG[a.status] || '#555f6b',
                            }}
                            onChange={(e) => patchAgent(a.id, { status: e.target.value })}>
                            {(STATUS_CHOICES.includes(a.status)
                              ? STATUS_CHOICES : [...STATUS_CHOICES, a.status]).map((s) => (
                              <option key={s} value={s} style={{ color: 'var(--text-primary)', background: 'var(--bg-input)' }}>{s}</option>
                            ))}
                          </select>
                        )}
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

                    {/* Prompt template (board #109) — the dispatcher identity prompt */}
                    {a.kind !== 'human' && (
                      <div style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
                        <span style={{ cursor: 'pointer' }} onClick={() => toggleTpl(a.id)}
                          title="identity prompt the dispatcher builds headless runs from (Phase 2)">
                          {tplOpen.has(a.id) ? '▾' : '▸'} prompt template
                          {!a.prompt_template && <span style={{ color: 'var(--text-tertiary)' }}> (none)</span>}
                        </span>
                        {tplOpen.has(a.id) && (
                          <div style={{ marginTop: 6 }}>
                            <textarea
                              style={{ ...input, width: '100%', minHeight: 120, fontFamily: 'monospace', fontSize: 11.5 }}
                              placeholder="identity prompt the dispatcher prepends for this agent's headless runs…"
                              value={tplDraft[a.id] ?? (a.prompt_template || '')}
                              onChange={(e) => setTplDraft((d) => ({ ...d, [a.id]: e.target.value }))} />
                            <div style={{ display: 'flex', gap: 8, marginTop: 4, alignItems: 'center' }}>
                              <button
                                style={{ ...input, cursor: 'pointer', background: 'var(--blue)', color: '#fff', fontSize: 12 }}
                                disabled={saving === a.id}
                                onClick={() => saveTpl(a)}>
                                {saving === a.id ? 'saving…' : 'save template'}
                              </button>
                              {(tplDraft[a.id] ?? (a.prompt_template || '')) !== (a.prompt_template || '') &&
                                <span style={{ fontSize: 11, color: 'var(--amber, #d98c00)' }}>unsaved</span>}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Recent dispatcher runs (run_history, board #109) */}
                    {(runsByAgent.get(a.letter) || []).length > 0 && (
                      <div style={{ fontSize: 11.5 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)', marginBottom: 2 }}>
                          RECENT RUNS
                        </div>
                        {(runsByAgent.get(a.letter) || []).slice(0, 5).map((r) => (
                          <div key={r.id} style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '1px 0' }}>
                            <OutcomeChip outcome={r.outcome} />
                            <Link href={`/admin/tasks?task=${r.task_id}`}
                              style={{ color: 'inherit', fontVariantNumeric: 'tabular-nums' }}>#{r.task_id}</Link>
                            <span style={{ color: 'var(--text-tertiary)' }}>
                              {relTime(r.finished_at || r.started_at || r.requested_at)}
                              {r.requested_by ? ` · by ${r.requested_by}` : ''}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}

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
