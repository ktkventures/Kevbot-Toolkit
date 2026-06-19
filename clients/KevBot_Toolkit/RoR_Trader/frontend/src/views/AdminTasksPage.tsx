/**
 * Dev Task Tracker — ClickUp-lite shared board for Kevin + Claude.
 *
 * Priority = phase.seq, sorted ascending so 1.1 ("do next") is at top.
 * Urgent is a tag, not a priority. impacts_live / needs_live_validation are
 * badges so the "does this touch trading?" question is visible at a glance.
 * Click a task → modal with description + comments. Talks to /api/dev-tasks.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';

interface Task {
  id: number;
  title: string;
  description: string;
  status: string;
  priority_phase: number;
  priority_seq: number;
  is_urgent: boolean;
  impacts_live: boolean;
  needs_live_validation: boolean;
  area: string;
  assignee: string | null;
  blocked_by: number[];
  tags: string[];
  notes: string;
  updated_at: string;
}
interface Comment { id: number; author: string; body: string; created_at: string; }

const STATUSES = ['Backlog', 'Todo', 'In Progress', 'Blocked', 'Done'];
const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
const ASSIGNEES = ['', 'kevin', 'claude'];
const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Todo': 'var(--blue)',
  'In Progress': 'var(--amber, #d98c00)', 'Blocked': 'var(--red)', 'Done': 'var(--green)',
};

const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
const input: React.CSSProperties = {
  background: 'var(--bg-input)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 6, padding: '4px 6px', fontSize: 13,
};
const badge = (bg: string): React.CSSProperties => ({
  display: 'inline-block', padding: '1px 6px', borderRadius: 10, fontSize: 11,
  fontWeight: 600, background: bg, color: '#fff', marginRight: 4, whiteSpace: 'nowrap',
});

export default function AdminTasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [hideDone, setHideDone] = useState(false);
  const [liveOnly, setLiveOnly] = useState(false);
  const [areaFilter, setAreaFilter] = useState('');
  const [assigneeFilter, setAssigneeFilter] = useState('');
  const [modalId, setModalId] = useState<number | null>(null);
  const [comments, setComments] = useState<Record<number, Comment[]>>({});
  const [draftComment, setDraftComment] = useState('');
  const [nt, setNt] = useState({
    title: '', priority_phase: 1, priority_seq: 99, area: 'other',
    assignee: 'claude', impacts_live: false,
  });

  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<Task[]>('/api/dev-tasks');
      setTasks(data || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  const patch = async (id: number, fields: Partial<Task>) => {
    setTasks((t) => t.map((x) => (x.id === id ? { ...x, ...fields } : x)));
    try {
      await apiFetch(`/api/dev-tasks/${id}`, { method: 'PATCH', body: JSON.stringify(fields) });
    } catch (e) { setErr(String(e)); load(); }
  };
  const createTask = async () => {
    if (!nt.title.trim()) return;
    try {
      await apiFetch('/api/dev-tasks', { method: 'POST', body: JSON.stringify(nt) });
      setNt({ ...nt, title: '' }); load();
    } catch (e) { setErr(String(e)); }
  };
  const del = async (id: number) => {
    if (!confirm('Delete this task?')) return;
    setTasks((t) => t.filter((x) => x.id !== id));
    if (modalId === id) setModalId(null);
    try { await apiFetch(`/api/dev-tasks/${id}`, { method: 'DELETE' }); }
    catch (e) { setErr(String(e)); load(); }
  };
  const openModal = async (id: number) => {
    setModalId(id); setDraftComment('');
    try {
      const cs = await apiFetch<Comment[]>(`/api/dev-tasks/${id}/comments`);
      setComments((m) => ({ ...m, [id]: cs || [] }));
    } catch { /* ignore */ }
  };
  const addComment = async (id: number) => {
    if (!draftComment.trim()) return;
    try {
      await apiFetch(`/api/dev-tasks/${id}/comments`, {
        method: 'POST', body: JSON.stringify({ body: draftComment, author: 'kevin' }),
      });
      const cs = await apiFetch<Comment[]>(`/api/dev-tasks/${id}/comments`);
      setComments((m) => ({ ...m, [id]: cs || [] }));
      setDraftComment('');
    } catch (e) { setErr(String(e)); }
  };

  const visible = useMemo(() => tasks.filter((t) =>
    (!hideDone || t.status !== 'Done') &&
    (!liveOnly || t.impacts_live) &&
    (!areaFilter || t.area === areaFilter) &&
    (!assigneeFilter || (t.assignee || '') === assigneeFilter),
  ), [tasks, hideDone, liveOnly, areaFilter, assigneeFilter]);

  const openCount = tasks.filter((t) => t.status !== 'Done').length;
  const liveCount = tasks.filter((t) => t.impacts_live && t.status !== 'Done').length;
  const modalTask = tasks.find((t) => t.id === modalId) || null;

  const PriCell = ({ t }: { t: Task }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 2, whiteSpace: 'nowrap' }}>
      <input style={{ ...input, width: 34, padding: '4px 3px' }} type="number" value={t.priority_phase}
        onChange={(e) => patch(t.id, { priority_phase: +e.target.value })} />
      <span style={{ color: 'var(--text-tertiary)' }}>.</span>
      <input style={{ ...input, width: 50, padding: '4px 3px' }} type="number" step="0.05" value={t.priority_seq}
        onChange={(e) => patch(t.id, { priority_seq: +e.target.value })} />
    </div>
  );

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Dev Task Tracker</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 16 }}>
        {openCount} open · {liveCount} touch live · sorted by priority (phase.seq, 1.1 first). Legend:
        🟢 offline-ok = fully validatable now (safe to work); ⏳ = needs live-market data to confirm;
        🔴 = touches live engine/trading code (deploy carefully); ⚡ = urgent (a tag, not a priority).
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}

      <Card>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <strong style={{ fontSize: 13 }}>+ New:</strong>
          <input style={{ ...input, flex: 1, minWidth: 240 }} placeholder="task title…"
            value={nt.title} onChange={(e) => setNt({ ...nt, title: e.target.value })}
            onKeyDown={(e) => e.key === 'Enter' && createTask()} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 2, whiteSpace: 'nowrap' }}>
            <input style={{ ...input, width: 40 }} type="number" title="phase"
              value={nt.priority_phase} onChange={(e) => setNt({ ...nt, priority_phase: +e.target.value })} />
            <span>.</span>
            <input style={{ ...input, width: 54 }} type="number" step="0.05" title="seq"
              value={nt.priority_seq} onChange={(e) => setNt({ ...nt, priority_seq: +e.target.value })} />
          </div>
          <select style={input} value={nt.area} onChange={(e) => setNt({ ...nt, area: e.target.value })}>
            {AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <select style={input} value={nt.assignee} onChange={(e) => setNt({ ...nt, assignee: e.target.value })}>
            {ASSIGNEES.map((a) => <option key={a} value={a}>{a || '—'}</option>)}
          </select>
          <label style={{ fontSize: 12 }}><input type="checkbox" checked={nt.impacts_live}
            onChange={(e) => setNt({ ...nt, impacts_live: e.target.checked })} /> 🔴 live</label>
          <button style={{ ...input, cursor: 'pointer', background: 'var(--blue)', color: '#fff' }}
            onClick={createTask}>Add</button>
        </div>
      </Card>

      <Card>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, alignItems: 'center' }}>
          <label><input type="checkbox" checked={hideDone} onChange={(e) => setHideDone(e.target.checked)} /> hide Done</label>
          <label><input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} /> 🔴 live only</label>
          <span>area: <select style={input} value={areaFilter} onChange={(e) => setAreaFilter(e.target.value)}>
            <option value="">all</option>{AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select></span>
          <span>assignee: <select style={input} value={assigneeFilter} onChange={(e) => setAssigneeFilter(e.target.value)}>
            <option value="">all</option>{ASSIGNEES.filter(Boolean).map((a) => <option key={a} value={a}>{a}</option>)}
          </select></span>
          <button style={{ ...input, cursor: 'pointer' }} onClick={load}>↻ refresh</button>
        </div>
      </Card>

      {loading ? <Card>Loading…</Card> : (
        <Card>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)', fontSize: 11, color: 'var(--text-tertiary)' }}>
                <th style={{ ...cell, width: 96 }}>Pri</th><th style={cell}>Task</th><th style={{ ...cell, width: 130 }}>Status</th>
                <th style={{ ...cell, width: 200 }}>Flags</th><th style={{ ...cell, width: 110 }}>Area</th><th style={{ ...cell, width: 100 }}>Who</th><th style={{ ...cell, width: 32 }}></th>
              </tr>
            </thead>
            <tbody>
              {visible.map((t) => (
                <tr key={t.id} style={{ borderBottom: '1px solid var(--border)', opacity: t.status === 'Done' ? 0.55 : 1 }}>
                  <td style={cell}><PriCell t={t} /></td>
                  <td style={{ ...cell, cursor: 'pointer', fontWeight: 500 }} onClick={() => openModal(t.id)}>
                    {t.title}
                    {(t.blocked_by?.length > 0) && <span style={{ color: 'var(--red)', fontSize: 11 }}> ⛔{t.blocked_by.join(',')}</span>}
                  </td>
                  <td style={cell}>
                    <select style={{ ...input, color: STATUS_COLOR[t.status] }} value={t.status}
                      onChange={(e) => patch(t.id, { status: e.target.value })}>
                      {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
                    </select>
                  </td>
                  <td style={cell}>
                    {t.impacts_live && <span style={badge('var(--red)')} title="touches live engine/trading code — deploy carefully">🔴 live</span>}
                    {t.status !== 'Done' && (t.needs_live_validation
                      ? <span style={badge('#a06800')} title="needs live-market data to confirm">⏳ validate</span>
                      : <span style={badge('#2e7d32')} title="fully validatable offline — safe to work now">🟢 offline-ok</span>)}
                    <span style={{ cursor: 'pointer' }} title="toggle urgent" onClick={() => patch(t.id, { is_urgent: !t.is_urgent })}>
                      {t.is_urgent ? <span style={badge('#b00')}>⚡ urgent</span> : <span style={{ opacity: 0.3 }}>⚡</span>}
                    </span>
                  </td>
                  <td style={cell}>
                    <select style={input} value={t.area} onChange={(e) => patch(t.id, { area: e.target.value })}>
                      {AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
                    </select>
                  </td>
                  <td style={cell}>
                    <select style={input} value={t.assignee || ''} onChange={(e) => patch(t.id, { assignee: e.target.value })}>
                      {ASSIGNEES.map((a) => <option key={a} value={a}>{a || '—'}</option>)}
                    </select>
                  </td>
                  <td style={cell}><span style={{ cursor: 'pointer', color: 'var(--red)' }} onClick={() => del(t.id)}>✕</span></td>
                </tr>
              ))}
              {visible.length === 0 && <tr><td colSpan={7} style={{ ...cell, color: 'var(--text-tertiary)' }}>No tasks match the filters.</td></tr>}
            </tbody>
          </table>
        </Card>
      )}

      {/* ── Task modal ─────────────────────────────────────────────── */}
      {modalTask && (
        <div onClick={() => setModalId(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
          display: 'flex', alignItems: 'flex-start', justifyContent: 'center', padding: '6vh 16px',
        }}>
          <div onClick={(e) => e.stopPropagation()} style={{
            background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
            borderRadius: 10, width: '100%', maxWidth: 720, maxHeight: '82vh', overflowY: 'auto',
            padding: 20, boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
              <input style={{ ...input, fontSize: 17, fontWeight: 600, flex: 1, padding: '6px 8px' }}
                value={modalTask.title}
                onChange={(e) => patch(modalTask.id, { title: e.target.value })} />
              <button style={{ ...input, cursor: 'pointer' }} onClick={() => setModalId(null)}>✕ close</button>
            </div>

            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', margin: '12px 0' }}>
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Priority</span>
              <PriCell t={modalTask} />
              <select style={{ ...input, color: STATUS_COLOR[modalTask.status] }} value={modalTask.status}
                onChange={(e) => patch(modalTask.id, { status: e.target.value })}>
                {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              <select style={input} value={modalTask.area} onChange={(e) => patch(modalTask.id, { area: e.target.value })}>
                {AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
              <select style={input} value={modalTask.assignee || ''} onChange={(e) => patch(modalTask.id, { assignee: e.target.value })}>
                {ASSIGNEES.map((a) => <option key={a} value={a}>@{a || '—'}</option>)}
              </select>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.impacts_live}
                onChange={(e) => patch(modalTask.id, { impacts_live: e.target.checked })} /> 🔴 live</label>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.needs_live_validation}
                onChange={(e) => patch(modalTask.id, { needs_live_validation: e.target.checked })} /> ⏳ validate</label>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.is_urgent}
                onChange={(e) => patch(modalTask.id, { is_urgent: e.target.checked })} /> ⚡ urgent</label>
            </div>

            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Description</div>
            <textarea style={{ ...input, width: '100%', minHeight: 90 }} placeholder="description…"
              defaultValue={modalTask.description}
              onBlur={(e) => e.target.value !== modalTask.description && patch(modalTask.id, { description: e.target.value })} />

            <div style={{ fontSize: 12, fontWeight: 600, margin: '14px 0 6px' }}>Comments</div>
            {(comments[modalTask.id] || []).map((cm) => (
              <div key={cm.id} style={{ fontSize: 13, margin: '5px 0', paddingBottom: 5, borderBottom: '1px solid var(--border)' }}>
                <b>{cm.author}</b> <span style={{ color: 'var(--text-tertiary)', fontSize: 11 }}>{cm.created_at?.slice(0, 16).replace('T', ' ')}</span>
                <div>{cm.body}</div>
              </div>
            ))}
            {(comments[modalTask.id] || []).length === 0 && <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>No comments yet.</div>}
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              <input style={{ ...input, flex: 1 }} placeholder="add a comment…" value={draftComment}
                onChange={(e) => setDraftComment(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addComment(modalTask.id)} />
              <button style={{ ...input, cursor: 'pointer', background: 'var(--blue)', color: '#fff' }}
                onClick={() => addComment(modalTask.id)}>Comment</button>
            </div>

            <div style={{ marginTop: 16, textAlign: 'right' }}>
              <button style={{ ...input, cursor: 'pointer', color: 'var(--red)' }} onClick={() => del(modalTask.id)}>Delete task</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
