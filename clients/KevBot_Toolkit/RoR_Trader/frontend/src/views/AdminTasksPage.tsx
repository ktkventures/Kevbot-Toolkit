/**
 * Dev Task Tracker — the multi-session team's board (Spec_Tasks_Team_Board.md).
 *
 * Vision items (top-level tasks tagged 'vision') group their subtasks in the
 * default "By vision" view — title + n/m done rollup — so rabbit-hole fixes
 * stay visibly parented under the big-picture item that spawned them
 * (origin 'discovered' renders the 🔍 chip). One nesting level only; the API
 * rejects deeper trees. "Flat" toggle keeps the original priority-sorted table.
 *
 * Priority = phase.seq, sorted ascending so 1.1 ("do next") is at top.
 * Urgent is a tag, not a priority. impacts_live / needs_live_validation are
 * badges so the "does this touch trading?" question is visible at a glance.
 * Click a task → modal with description + comments. Talks to /api/dev-tasks.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
  parent_id: number | null;
  origin: string;
  updated_at: string;
}
interface Comment { id: number; author: string; body: string; created_at: string; }

const STATUSES = ['Backlog', 'Todo', 'In Progress', 'Blocked', 'Done'];
const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
// Team roles per Session_Charters.md §1 — adding a role is one line here.
// Legacy values ('claude', …) still render: selects append any unknown current
// value as an extra option instead of blanking it.
const ASSIGNEES = ['', 'M', 'E', 'E2', 'F', 'P', 'R', 'kevin'];
const ORIGINS = ['planned', 'discovered', 'kevin'];
const AUTHOR_LS_KEY = 'ror_task_comment_author';
const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Todo': 'var(--blue)',
  'In Progress': 'var(--amber, #d98c00)', 'Blocked': 'var(--red)', 'Done': 'var(--green)',
};

const withLegacy = (list: string[], current: string) =>
  list.includes(current) ? list : [...list, current];

const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
const input: React.CSSProperties = {
  background: 'var(--bg-input)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 6, padding: '4px 6px', fontSize: 13,
};
const badge = (bg: string): React.CSSProperties => ({
  display: 'inline-block', padding: '1px 6px', borderRadius: 10, fontSize: 11,
  fontWeight: 600, background: bg, color: '#fff', marginRight: 4, whiteSpace: 'nowrap',
});
const tagChip: React.CSSProperties = {
  display: 'inline-block', padding: '0px 6px', borderRadius: 8, fontSize: 10.5,
  border: '1px solid var(--border)', color: 'var(--text-secondary)',
  marginLeft: 4, whiteSpace: 'nowrap', verticalAlign: 'middle',
};

export default function AdminTasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<'grouped' | 'flat'>('grouped');
  const [hideDone, setHideDone] = useState(false);
  const [liveOnly, setLiveOnly] = useState(false);
  const [areaFilter, setAreaFilter] = useState('');
  const [assigneeFilter, setAssigneeFilter] = useState('');
  const [modalId, setModalId] = useState<number | null>(null);
  const [comments, setComments] = useState<Record<number, Comment[]>>({});
  const [draftComment, setDraftComment] = useState('');
  const [commentAuthor, setCommentAuthor] = useState('kevin');
  const [nt, setNt] = useState({
    title: '', priority_phase: 1, priority_seq: 99, area: 'other',
    assignee: '', impacts_live: false,
    parent_id: null as number | null, origin: 'planned',
  });
  const titleRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<Task[]>('/api/dev-tasks');
      setTasks(data || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const saved = localStorage.getItem(AUTHOR_LS_KEY);
    if (saved) setCommentAuthor(saved);
  }, []);

  const patch = async (id: number, fields: Partial<Task>) => {
    setTasks((t) => t.map((x) => (x.id === id ? { ...x, ...fields } : x)));
    try {
      await apiFetch(`/api/dev-tasks/${id}`, { method: 'PATCH', body: JSON.stringify(fields) });
    } catch (e) { setErr(String(e)); load(); }
  };
  const createTask = async () => {
    if (!nt.title.trim()) return;
    // No parent selected = a new vision item (auto-tagged); subtasks carry origin.
    const body = nt.parent_id == null
      ? { ...nt, parent_id: null, tags: ['vision'] }
      : { ...nt };
    try {
      await apiFetch('/api/dev-tasks', { method: 'POST', body: JSON.stringify(body) });
      setNt({ ...nt, title: '' }); load();
    } catch (e) { setErr(String(e)); }
  };
  const del = async (id: number) => {
    if (!confirm('Delete this task? Subtasks are deleted with their vision item.')) return;
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
        method: 'POST', body: JSON.stringify({ body: draftComment, author: commentAuthor }),
      });
      const cs = await apiFetch<Comment[]>(`/api/dev-tasks/${id}/comments`);
      setComments((m) => ({ ...m, [id]: cs || [] }));
      setDraftComment('');
    } catch (e) { setErr(String(e)); }
  };
  const pickAuthor = (a: string) => {
    setCommentAuthor(a);
    localStorage.setItem(AUTHOR_LS_KEY, a);
  };
  const startSubtask = (parentId: number) => {
    setNt((n) => ({ ...n, parent_id: parentId, origin: 'planned' }));
    window.scrollTo({ top: 0, behavior: 'smooth' });
    titleRef.current?.focus();
  };
  const parseIds = (s: string): number[] =>
    s.split(',').map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n));
  const parseTags = (s: string): string[] =>
    s.split(',').map((x) => x.trim()).filter(Boolean);

  const matches = (t: Task) =>
    (!hideDone || t.status !== 'Done') &&
    (!liveOnly || t.impacts_live) &&
    (!areaFilter || t.area === areaFilter) &&
    (!assigneeFilter || (t.assignee || '') === assigneeFilter);

  const byParent = useMemo(() => {
    const m = new Map<number, Task[]>();
    tasks.forEach((t) => {
      if (t.parent_id != null) m.set(t.parent_id, [...(m.get(t.parent_id) || []), t]);
    });
    return m;
  }, [tasks]);

  // A "vision item" is top-level and tagged vision — or already has subtasks
  // (so pre-tag data still groups correctly).
  const visionItems = useMemo(() =>
    tasks.filter((t) => t.parent_id == null &&
      ((t.tags || []).includes('vision') || byParent.has(t.id))),
  [tasks, byParent]);
  const looseTasks = useMemo(() =>
    tasks.filter((t) => t.parent_id == null &&
      !((t.tags || []).includes('vision') || byParent.has(t.id))),
  [tasks, byParent]);

  const assigneeOptions = useMemo(() => {
    const known = ASSIGNEES.filter(Boolean);
    const legacy = Array.from(new Set(tasks.map((t) => t.assignee).filter(Boolean) as string[]))
      .filter((a) => !known.includes(a));
    return [...known, ...legacy];
  }, [tasks]);

  const visible = useMemo(() => tasks.filter(matches),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tasks, hideDone, liveOnly, areaFilter, assigneeFilter]);

  const openCount = tasks.filter((t) => t.status !== 'Done').length;
  const liveCount = tasks.filter((t) => t.impacts_live && t.status !== 'Done').length;
  const modalTask = tasks.find((t) => t.id === modalId) || null;
  const ntParent = nt.parent_id != null ? tasks.find((t) => t.id === nt.parent_id) : null;

  const PriCell = ({ t }: { t: Task }) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 2, whiteSpace: 'nowrap' }}>
      <input style={{ ...input, width: 34, padding: '4px 3px' }} type="number" value={t.priority_phase}
        onChange={(e) => patch(t.id, { priority_phase: +e.target.value })} />
      <span style={{ color: 'var(--text-tertiary)' }}>.</span>
      <input style={{ ...input, width: 50, padding: '4px 3px' }} type="number" step="0.05" value={t.priority_seq}
        onChange={(e) => patch(t.id, { priority_seq: +e.target.value })} />
    </div>
  );

  const TagChips = ({ t }: { t: Task }) => (
    <>
      {(t.tags || []).filter((tag) => tag !== 'vision').map((tag) => (
        <span key={tag} style={tagChip}>{tag}</span>
      ))}
    </>
  );

  const TaskRow = ({ t, indent = false, rollup }: {
    t: Task; indent?: boolean; rollup?: { done: number; total: number };
  }) => (
    <tr key={t.id} style={{
      borderBottom: '1px solid var(--border)',
      opacity: t.status === 'Done' ? 0.55 : 1,
      background: rollup ? 'var(--bg-input)' : undefined,
    }}>
      <td style={{ ...cell, color: 'var(--text-tertiary)', fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>#{t.id}</td>
      <td style={cell}><PriCell t={t} /></td>
      <td style={{ ...cell, cursor: 'pointer', fontWeight: rollup ? 600 : 500, paddingLeft: indent ? 28 : cell.padding }}
        onClick={() => openModal(t.id)}>
        {indent && <span style={{ color: 'var(--text-tertiary)' }}>↳ </span>}
        {t.title}
        {rollup && (
          <span style={{
            ...tagChip, marginLeft: 8,
            color: rollup.total > 0 && rollup.done === rollup.total ? 'var(--green)' : 'var(--text-secondary)',
          }} title="subtasks done / total">{rollup.done}/{rollup.total}</span>
        )}
        {t.parent_id != null && t.origin === 'discovered' &&
          <span style={{ ...tagChip, marginLeft: 6 }} title="discovered mid-work (rabbit-hole fix)">🔍</span>}
        {view === 'flat' && t.parent_id != null &&
          <span style={{ ...tagChip, marginLeft: 6 }} title="subtask of this vision item">↳ #{t.parent_id}</span>}
        <TagChips t={t} />
        {(t.blocked_by?.length > 0) && <span style={{ color: 'var(--red)', fontSize: 11 }}> ⛔{t.blocked_by.join(',')}</span>}
        {rollup && (
          <button style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '1px 6px', marginLeft: 8 }}
            title="add a subtask under this vision item"
            onClick={(e) => { e.stopPropagation(); startSubtask(t.id); }}>+ subtask</button>
        )}
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
          {withLegacy(ASSIGNEES, t.assignee || '').map((a) => <option key={a} value={a}>{a || '—'}</option>)}
        </select>
      </td>
      <td style={cell}><span style={{ cursor: 'pointer', color: 'var(--red)' }} onClick={() => del(t.id)}>✕</span></td>
    </tr>
  );

  const groupedRows: React.ReactNode[] = [];
  if (view === 'grouped') {
    visionItems.forEach((v) => {
      const kids = byParent.get(v.id) || [];
      const shownKids = kids.filter(matches);
      if (!matches(v) && shownKids.length === 0) return;
      groupedRows.push(
        <TaskRow key={v.id} t={v}
          rollup={{ done: kids.filter((k) => k.status === 'Done').length, total: kids.length }} />);
      shownKids.forEach((k) => groupedRows.push(<TaskRow key={k.id} t={k} indent />));
    });
    const shownLoose = looseTasks.filter(matches);
    if (shownLoose.length > 0) {
      groupedRows.push(
        <tr key="loose-header">
          <td colSpan={8} style={{ ...cell, color: 'var(--text-tertiary)', fontSize: 11, paddingTop: 14 }}>
            ungrouped — tasks without a vision parent
          </td>
        </tr>);
      shownLoose.forEach((t) => groupedRows.push(<TaskRow key={t.id} t={t} />));
    }
  }

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Dev Task Tracker</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 16 }}>
        {openCount} open · {liveCount} touch live · sorted by priority (phase.seq, 1.1 first). Legend:
        🟢 offline-ok = fully validatable now (safe to work); ⏳ = needs live-market data to confirm;
        🔴 = touches live engine/trading code (deploy carefully); ⚡ = urgent (a tag, not a priority);
        🔍 = discovered mid-work (rabbit-hole fix, parented under its vision item).
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}

      <Card>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <strong style={{ fontSize: 13 }}>+ New:</strong>
          <input ref={titleRef} style={{ ...input, flex: 1, minWidth: 240 }}
            placeholder={ntParent ? `subtask under #${ntParent.id} ${ntParent.title.slice(0, 40)}…` : 'task title…'}
            value={nt.title} onChange={(e) => setNt({ ...nt, title: e.target.value })}
            onKeyDown={(e) => e.key === 'Enter' && createTask()} />
          <select style={input} title="parent vision item (none = new vision item)"
            value={nt.parent_id ?? ''}
            onChange={(e) => setNt({ ...nt, parent_id: e.target.value ? +e.target.value : null })}>
            <option value="">— vision item —</option>
            {visionItems.map((v) => <option key={v.id} value={v.id}>↳ #{v.id} {v.title.slice(0, 40)}</option>)}
          </select>
          {nt.parent_id != null && (
            <select style={input} title="origin — 🔍 discovered = rabbit-hole fix" value={nt.origin}
              onChange={(e) => setNt({ ...nt, origin: e.target.value })}>
              {ORIGINS.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          )}
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
          <span>
            <button style={{ ...input, cursor: 'pointer', fontWeight: view === 'grouped' ? 700 : 400, background: view === 'grouped' ? 'var(--bg-input)' : 'transparent' }}
              onClick={() => setView('grouped')}>By vision</button>
            <button style={{ ...input, cursor: 'pointer', marginLeft: 4, fontWeight: view === 'flat' ? 700 : 400, background: view === 'flat' ? 'var(--bg-input)' : 'transparent' }}
              onClick={() => setView('flat')}>Flat</button>
          </span>
          <label><input type="checkbox" checked={hideDone} onChange={(e) => setHideDone(e.target.checked)} /> hide Done</label>
          <label><input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} /> 🔴 live only</label>
          <span>area: <select style={input} value={areaFilter} onChange={(e) => setAreaFilter(e.target.value)}>
            <option value="">all</option>{AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select></span>
          <span>assignee: <select style={input} value={assigneeFilter} onChange={(e) => setAssigneeFilter(e.target.value)}>
            <option value="">all</option>{assigneeOptions.map((a) => <option key={a} value={a}>{a}</option>)}
          </select></span>
          <button style={{ ...input, cursor: 'pointer' }} onClick={load}>↻ refresh</button>
        </div>
      </Card>

      {loading ? <Card>Loading…</Card> : (
        <Card>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)', fontSize: 11, color: 'var(--text-tertiary)' }}>
                <th style={{ ...cell, width: 44 }}>ID</th><th style={{ ...cell, width: 96 }}>Pri</th><th style={cell}>Task</th><th style={{ ...cell, width: 130 }}>Status</th>
                <th style={{ ...cell, width: 200 }}>Flags</th><th style={{ ...cell, width: 110 }}>Area</th><th style={{ ...cell, width: 100 }}>Who</th><th style={{ ...cell, width: 32 }}></th>
              </tr>
            </thead>
            <tbody>
              {view === 'flat'
                ? visible.map((t) => <TaskRow key={t.id} t={t} />)
                : groupedRows}
              {((view === 'flat' && visible.length === 0) || (view === 'grouped' && groupedRows.length === 0)) &&
                <tr><td colSpan={8} style={{ ...cell, color: 'var(--text-tertiary)' }}>No tasks match the filters.</td></tr>}
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
                {withLegacy(ASSIGNEES, modalTask.assignee || '').map((a) => <option key={a} value={a}>@{a || '—'}</option>)}
              </select>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.impacts_live}
                onChange={(e) => patch(modalTask.id, { impacts_live: e.target.checked })} /> 🔴 live</label>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.needs_live_validation}
                onChange={(e) => patch(modalTask.id, { needs_live_validation: e.target.checked })} /> ⏳ validate</label>
              <label style={{ fontSize: 12 }}><input type="checkbox" checked={modalTask.is_urgent}
                onChange={(e) => patch(modalTask.id, { is_urgent: e.target.checked })} /> ⚡ urgent</label>
            </div>

            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', margin: '0 0 12px' }}>
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Parent</span>
              <select style={input} value={modalTask.parent_id ?? ''}
                onChange={(e) => patch(modalTask.id, { parent_id: e.target.value ? +e.target.value : null })}>
                <option value="">— none (vision item) —</option>
                {visionItems.filter((v) => v.id !== modalTask.id).map((v) =>
                  <option key={v.id} value={v.id}>#{v.id} {v.title.slice(0, 40)}</option>)}
              </select>
              {modalTask.parent_id != null && (
                <>
                  <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Origin</span>
                  <select style={input} value={modalTask.origin || 'planned'}
                    onChange={(e) => patch(modalTask.id, { origin: e.target.value })}>
                    {ORIGINS.map((o) => <option key={o} value={o}>{o === 'discovered' ? '🔍 discovered' : o}</option>)}
                  </select>
                </>
              )}
            </div>

            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', margin: '0 0 12px' }}>
              <label style={{ fontSize: 12, flex: 1, minWidth: 220 }}>tags (comma-separated)
                <input key={`tags-${modalTask.id}`} style={{ ...input, width: '100%', marginTop: 2 }}
                  defaultValue={(modalTask.tags || []).join(', ')}
                  onBlur={(e) => {
                    const tags = parseTags(e.target.value);
                    if (tags.join('|') !== (modalTask.tags || []).join('|')) patch(modalTask.id, { tags });
                  }} />
              </label>
              <label style={{ fontSize: 12, flex: 1, minWidth: 160 }}>⛔ blocked by (task ids)
                <input key={`blk-${modalTask.id}`} style={{ ...input, width: '100%', marginTop: 2 }}
                  defaultValue={(modalTask.blocked_by || []).join(', ')}
                  onBlur={(e) => {
                    const ids = parseIds(e.target.value);
                    if (ids.join('|') !== (modalTask.blocked_by || []).join('|')) patch(modalTask.id, { blocked_by: ids });
                  }} />
              </label>
            </div>

            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>Description</div>
            <textarea key={`desc-${modalTask.id}`} style={{ ...input, width: '100%', minHeight: 90 }} placeholder="description…"
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
              <select style={input} title="comment as" value={commentAuthor}
                onChange={(e) => pickAuthor(e.target.value)}>
                {withLegacy(ASSIGNEES.filter(Boolean), commentAuthor).map((a) => <option key={a} value={a}>@{a}</option>)}
              </select>
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
