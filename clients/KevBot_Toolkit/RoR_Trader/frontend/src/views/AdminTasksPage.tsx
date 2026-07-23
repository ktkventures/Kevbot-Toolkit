/**
 * Dev Task Tracker — the multi-session team's board (Spec_Tasks_Team_Board.md).
 *
 * Vision items (top-level tasks tagged 'vision') group their subtasks in the
 * default "By vision" view — title + n/m done rollup, collapsible via a
 * chevron (persisted in localStorage) — so rabbit-hole fixes stay visibly
 * parented under the big-picture item that spawned them (origin 'discovered'
 * renders the 🔍 chip). One nesting level only; the API rejects deeper trees.
 * "Flat" toggle keeps the original priority-sorted table.
 *
 * Priority = phase.seq, sorted ascending so 1.1 ("do next") is at top.
 * Urgent is a tag, not a priority. impacts_live / needs_live_validation are
 * badges so the "does this touch trading?" question is visible at a glance.
 * Click a task → three-panel detail modal (TaskDetailModal). Status/assignee
 * changes send `actor` so the API can log system activity entries.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import TaskDetailModal from './TaskDetailModal';
import {
  Task, STATUSES, AREAS, ASSIGNEES, ORIGINS, STATUS_COLOR, AUTHOR_LS_KEY,
  COLLAPSED_LS_KEY, withLegacy, cell, input, badge, tagChip,
  NextChip, ProgressBar,
} from './taskBoardShared';

export default function AdminTasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<'grouped' | 'flat'>('grouped');
  const [hideDone, setHideDone] = useState(false);
  const [liveOnly, setLiveOnly] = useState(false);
  const [areaFilter, setAreaFilter] = useState('');
  const [assigneeFilter, setAssigneeFilter] = useState('');
  const [modal, setModal] = useState<{ id: number; sel?: number } | null>(null);
  const [commentAuthor, setCommentAuthor] = useState('kevin');
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set());
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
    try {
      const c = JSON.parse(localStorage.getItem(COLLAPSED_LS_KEY) || '[]');
      if (Array.isArray(c)) setCollapsed(new Set(c.filter((n) => typeof n === 'number')));
    } catch { /* ignore */ }
  }, []);

  const toggleCollapsed = (id: number) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      localStorage.setItem(COLLAPSED_LS_KEY, JSON.stringify(Array.from(next)));
      return next;
    });
  };

  // `actor` rides along on every PATCH (not a task column) so the API can
  // write "status: Todo → In Progress (by F)" system entries.
  const patch = async (id: number, fields: Partial<Task>) => {
    setTasks((t) => t.map((x) => (x.id === id ? { ...x, ...fields } : x)));
    try {
      await apiFetch(`/api/dev-tasks/${id}`, {
        method: 'PATCH', body: JSON.stringify({ ...fields, actor: commentAuthor }),
      });
      if ('status' in fields || 'assignee' in fields) load();
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
    if (modal?.id === id) setModal(null);
    try { await apiFetch(`/api/dev-tasks/${id}`, { method: 'DELETE' }); }
    catch (e) { setErr(String(e)); load(); }
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
  const modalTask = (modal && tasks.find((t) => t.id === modal.id)) || null;
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
        onClick={() => setModal({ id: t.id })}>
        {rollup && (
          <span style={{ cursor: 'pointer', marginRight: 6, color: 'var(--text-tertiary)', fontSize: 12 }}
            title={collapsed.has(t.id) ? 'expand subtasks' : 'collapse subtasks'}
            onClick={(e) => { e.stopPropagation(); toggleCollapsed(t.id); }}>
            {collapsed.has(t.id) ? '▸' : '▾'}
          </span>
        )}
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
        <NextChip t={t} subtasks={byParent.get(t.id) || []} />
        {rollup && <span style={{ marginLeft: 6 }}><ProgressBar done={rollup.done} total={rollup.total} mini /></span>}
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
      if (!collapsed.has(v.id)) {
        shownKids.forEach((k) => groupedRows.push(<TaskRow key={k.id} t={k} indent />));
      }
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

      {modalTask && (
        <TaskDetailModal key={`${modalTask.id}:${modal?.sel ?? ''}`}
          task={modalTask} allTasks={tasks} visionOptions={visionItems}
          initialSelected={modal?.sel}
          patch={patch} del={del}
          onClose={() => setModal(null)}
          onOpenTask={(id, sel) => setModal({ id, sel })}
          commentAuthor={commentAuthor} onPickAuthor={pickAuthor} />
      )}
    </div>
  );
}
