/**
 * Dev Task Tracker — the multi-session team's board (Spec_Tasks_Team_Board.md).
 *
 * CONTAINER rows group their subtasks in the default "By container" view —
 * title + n/m done rollup, collapsible via a chevron (persisted in
 * localStorage) — so rabbit-hole fixes stay visibly parented under the
 * big-picture item that spawned them (origin 'discovered' renders the 🔍 chip).
 * One nesting level only; the API rejects deeper trees. "Flat" toggle keeps the
 * original priority-sorted table.
 *
 * A container is ANY type carrying `children: true` in TASK_TYPES — vision AND
 * goal since board #262, and whatever type three turns out to be. It is NOT
 * "tagged 'vision'", which is what this header claimed until board #295.
 *
 * WHY #295 EXISTED, since the behaviour it "fixed" was already correct: the
 * toggle was labelled "By vision" while `groupBoard` had filtered on
 * `isContainerTask` since #262. Kevin read the label, and reported goal
 * grouping as MISSING on a board that was grouping his goal correctly in front
 * of him. A label that makes a working feature look broken is a real defect —
 * so #295 changed the words and nothing else. Grouping is untouched.
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
import TaskMessagesPanel from './TaskMessagesPanel';
// Board #297 — the board's own manual, GENERATED from taskBoardShared's model
// rather than written beside it. See that file's header for why it must never
// become a hand-typed second copy of the rules.
import TaskHelpModal from './TaskHelpModal';
import {
  Task, RunRow, Mention, AREAS, ASSIGNEES, ORIGINS, STATUS_COLOR, AUTHOR_LS_KEY,
  COLLAPSED_LS_KEY, RUN_REQUESTED_TAG, STATUS_DEF, STATUSES, withLegacy,
  statusOptionsFor, cell, input, badge, tagChip, NextChip, ProgressBar, RunButton,
  RETIRED_TAGS, StampButtons, TwoTouchChip, ImpactSelect, defaultChain,
  groupBoard, StuckChip, isStuckInTodo, HandoffChain,
  AiEligibleToggle, RunStatePill, runState, isFinished, StampMode,
  AgentMeta, AgentRegistryContext, agentRegistryMap, MultiSelectFilter,
  taskTypeOf, TaskTypeIcon, TASK_TYPES, rolesWithGoalLanes,
} from './taskBoardShared';
// Board #246 — multi-select predicates, in a plain-JS module so a test can
// execute them for real rather than source-scan them (see taskFilters.js).
import { taskMatchesSelects, activeSelectCount, UNASSIGNED_LABEL } from './taskFilters';

// Board #198 column headers — the split spelled out where Kevin hovers it.
const AI_COL_TIP = 'AI-eligible — standing permission for a headless agent to claim this task, '
  + 'independent of where it sits on the kanban. Kevin’s Approval stamp arms it; this switch '
  + 'is the manual override. ON can dispatch within one dispatcher poll (~20s).';
const RUN_COL_TIP = 'Run state (what the dispatcher is doing: idle / queued / running / failed, '
  + 'with the last outcome) + ▶ Run once (a one-off queue jump — it does NOT change the '
  + 'AI-eligible switch).';

export default function AdminTasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [view, setView] = useState<'grouped' | 'flat' | 'messages'>('grouped');
  const [hideDone, setHideDone] = useState(false);
  const [liveOnly, setLiveOnly] = useState(false);
  // Board #246 — area / assignee / STATUS are MULTI-SELECT (checkbox lists).
  // An EMPTY array means NO FILTER, never "show nothing" — see taskFilters.js.
  //
  // The status filter REPLACES two booleans that were status filters wearing a
  // different hat (Kevin, 07-30: *"I'm okay with you removing that redundant
  // toggle"*): `reviewOnly` was `status === 'Review'`, and `awaitingOk` was
  // `status === 'Approval'`. Keeping both paths would let the page contradict
  // itself — tick "in Review", also tick "Staged", get an empty board and read
  // it as a bug. `✋ Awaiting my OK` survives as a PRESET over this one filter
  // (it carries a count, which no filter does), not as a second path.
  const [areaFilter, setAreaFilter] = useState<string[]>([]);
  const [assigneeFilter, setAssigneeFilter] = useState<string[]>([]);
  const [statusFilter, setStatusFilter] = useState<string[]>([]);
  // stuckOnly = board #151 tell — Todo tasks whose next actor is Kevin (a
  // contradiction: queue-eligible yet undispatchable). Needs byParent, so its
  // predicate lives after grouping (see `stuck`/`matches`).
  const [stuckOnly, setStuckOnly] = useState(false);
  // Board #198 — the two at-a-glance questions the toggle/pill split makes
  // askable: which tasks are ARMED for an agent, and which are RUNNING now.
  const [armedOnly, setArmedOnly] = useState(false);
  const [runningOnly, setRunningOnly] = useState(false);
  // Column sort applies to the FLAT view only — the grouped view's order IS
  // the pipeline (priority within vision), so it stays fixed.
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  const [modal, setModal] = useState<{ id: number; sel?: number } | null>(null);
  // Board #297 — the help manual. Board-state only, nothing persisted: it is a
  // reference surface, so re-opening it should always start at topic one.
  const [helpOpen, setHelpOpen] = useState(false);
  const [commentAuthor, setCommentAuthor] = useState('kevin');
  const [collapsed, setCollapsed] = useState<Set<number>>(new Set());
  // Assignee/author options come from the agents registry; the const is the
  // fallback so selects never break if the fetch fails (spec §4).
  const [roles, setRoles] = useState<string[]>(ASSIGNEES);
  // Run button state (board #109): registry statuses decide who is
  // headless-enrolled; run_history rows carry requested/running/last-outcome.
  const [agents, setAgents] = useState<AgentMeta[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  // Messages tab (board #143): unread mentions for the current viewer role
  // (whoever you're acting as = commentAuthor) drives the tab's badge. The
  // panel owns the full list; here we only keep the count.
  const [unreadCount, setUnreadCount] = useState(0);
  const [msgSignal, setMsgSignal] = useState(0);
  const [nt, setNt] = useState({
    title: '', priority_phase: 1, priority_seq: 99, area: 'other',
    assignee: '', impacts_live: false,
    parent_id: null as number | null, origin: 'planned',
  });
  const titleRef = useRef<HTMLInputElement>(null);
  // Board #148: the modal's consolidated poll probe hands us the board's
  // max(updated_at); we refetch the 120-row list only when it moves. Seed the
  // baseline from every full load so the first probe compares against truth.
  const lastBoardMax = useRef<string | null>(null);
  const boardMaxInit = useRef(false);

  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<Task[]>('/api/dev-tasks');
      setTasks(data || []);
      let mx: string | null = null;
      for (const t of data || []) if (t.updated_at && (!mx || t.updated_at > mx)) mx = t.updated_at;
      lastBoardMax.current = mx;
      boardMaxInit.current = true;
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    // `kind` rides along for board #222: RoleChip renders a live-session /
    // human owner differently from a headless one, keyed off the registry.
    apiFetch<AgentMeta[]>('/api/agents?active=true')
      .then((rows) => {
        setAgents(rows || []);
        const letters = (rows || []).map((a) => a.letter).filter(Boolean);
        if (letters.length) setRoles(['', ...letters]);
      })
      .catch(() => { /* keep ASSIGNEES fallback */ });
  }, []);
  const loadRuns = useCallback(async () => {
    try { setRuns(await apiFetch<RunRow[]>('/api/run-history?limit=300') || []); }
    catch { /* run states just don't render */ }
  }, []);
  useEffect(() => { loadRuns(); }, [loadRuns]);
  // Unread-mention badge (board #143): count unseen mentions addressed to the
  // current viewer role. Refetched on author change + each board poll tick.
  const loadUnread = useCallback(async () => {
    if (!commentAuthor) { setUnreadCount(0); return; }
    try {
      const m = await apiFetch<Mention[]>(
        `/api/mentions?for=${encodeURIComponent(commentAuthor)}&unseen=true`);
      setUnreadCount((m || []).length);
    } catch { /* badge just doesn't show */ }
  }, [commentAuthor]);
  useEffect(() => { loadUnread(); }, [loadUnread]);
  // Board #222 — one registry lookup for every owner avatar on the page, so
  // "is this lane a live session or a headless agent?" is answered by the
  // registry rather than by a letter check anyone has to remember to update.
  const agentReg = useMemo(() => agentRegistryMap(agents), [agents]);
  // Mirrors the dispatcher's enrollment rule: registry rows with
  // status='headless'; when there are none, its stub fallback is the docs
  // lane's headless half — M-A since the board-#222 split (M is the session).
  const headless = useMemo(() => {
    const h = new Set(agents.filter((a) => a.status === 'headless').map((a) => a.letter));
    if (h.size === 0) h.add('M-A');
    return h;
  }, [agents]);
  const latestRunByTask = useMemo(() => {
    const m = new Map<number, RunRow>();
    runs.forEach((r) => { if (!m.has(r.task_id)) m.set(r.task_id, r); }); // rows arrive newest-first
    return m;
  }, [runs]);
  const requestRun = async (id: number) => {
    try {
      await apiFetch(`/api/dev-tasks/${id}/run-request`, {
        method: 'POST', body: JSON.stringify({ author: commentAuthor }),
      });
    } catch (e) { setErr(String(e)); }
    load(); loadRuns();
  };
  useEffect(() => {
    const id = parseInt(new URLSearchParams(window.location.search).get('task') || '', 10);
    if (Number.isFinite(id)) setModal({ id });
  }, []);
  useEffect(() => {
    const url = new URL(window.location.href);
    if (modal) url.searchParams.set('task', String(modal.id));
    else url.searchParams.delete('task');
    window.history.replaceState(null, '', url);
  }, [modal]);
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
    // No parent selected = a new vision item (auto-tagged); subtasks carry
    // origin and auto-populate the default 4-step process chain (board #134
    // item 2 — editable afterward; visions pipeline via subtasks instead).
    const body = nt.parent_id == null
      ? { ...nt, parent_id: null, tags: ['vision'] }
      : { ...nt, checklist: defaultChain(nt.assignee) };
    try {
      await apiFetch('/api/dev-tasks', { method: 'POST', body: JSON.stringify(body) });
      setNt({ ...nt, title: '' }); load();
    } catch (e) { setErr(String(e)); }
  };
  const del = async (id: number) => {
    if (!confirm('Delete this task? Subtasks are deleted with their parent.')) return;
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

  // Grouping + the CONTAINER predicate live in taskBoardShared.groupBoard — the
  // ONE implementation shared with /admin/roadmap (Spec_Admin_Roadmap.md §2).
  // `visionItems` is a legacy KEY NAME holding every container (vision + goal);
  // see the groupBoard docblock for why it was deliberately not renamed.
  const { byParent, visionItems, looseTasks } = useMemo(() => groupBoard(tasks), [tasks]);

  // The row's TYPE (board #261) — explicit `task_type`, falling back to the
  // pre-#261 derivation. Everything per-type (status set, row glyph) reads this,
  // never a re-derivation. Vision rows are EXEMPT from the
  // Approval/Review/Staged pipeline (board #136) — they track via their subtasks.
  const typeOf = (t: Task) => taskTypeOf(t, byParent.has(t.id));

  // Board #151 contradiction: Todo + next-actor Kevin (needs subtasks).
  const stuck = (t: Task) => isStuckInTodo(t, byParent.get(t.id) || []);
  const stuckCount = tasks.filter(stuck).length;

  // Board #198: armed = the AI-eligible switch is on; busy = the dispatcher is
  // queued/running on it right now (a pure read of run_history via runState).
  const isRunning = (t: Task) => runState(t, latestRunByTask.get(t.id)).key === 'running';
  const armedCount = tasks.filter((t) => t.ai_eligible).length;
  const runningCount = tasks.filter(isRunning).length;

  const matches = (t: Task) =>
    // "hide done" means hide FINISHED (board #232) — Done and Closed alike.
    (!hideDone || !isFinished(t.status)) &&
    (!liveOnly || t.impacts_live) &&
    (!stuckOnly || stuck(t)) &&
    (!armedOnly || !!t.ai_eligible) &&
    (!runningOnly || isRunning(t)) &&
    // Board #246: union WITHIN each multi-select, intersection ACROSS them.
    // Empty selection = no constraint (matchesMulti), so the default view is
    // the whole board, not an empty one.
    taskMatchesSelects(t, { areas: areaFilter, assignees: assigneeFilter, statuses: statusFilter });

  // Board #289 — the assignee vocabulary is the registry lanes PLUS one derived
  // lane per open goal (`G281`). Derived, because a goal session has no registry
  // row by design: that absence is what stops the dispatcher spawning one. Every
  // assignee/owner select on this page reads THIS, not `roles`.
  const roleOptions = useMemo(() => rolesWithGoalLanes(roles, tasks), [roles, tasks]);

  // '' is now a pickable value ("(unassigned)"), not the all-pass sentinel it
  // was under single-select — so unassigned tasks became reachable for free.
  const assigneeOptions = useMemo(() => {
    const known = roleOptions.filter(Boolean);
    const legacy = Array.from(new Set(tasks.map((t) => t.assignee).filter(Boolean) as string[]))
      .filter((a) => !known.includes(a));
    return ['', ...known, ...legacy];
  }, [tasks, roleOptions]);

  const visible = useMemo(() => tasks.filter(matches),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tasks, hideDone, liveOnly, stuckOnly, armedOnly,
      runningOnly, latestRunByTask, areaFilter, assigneeFilter, statusFilter]);

  // How many of the three multi-selects are constraining the view, for the
  // tell beside the controls and the empty-state line. A filtered board must
  // never be mistakable for a board that lost its rows.
  const activeFilters = activeSelectCount({
    areas: areaFilter, assignees: assigneeFilter, statuses: statusFilter,
  });

  // `✋ Awaiting my OK` is now a PRESET over the status filter, not a rival
  // boolean: it is "on" exactly when the status filter is {Approval}, and
  // clicking it sets or clears that. One path, so the two controls can never
  // disagree. Its COUNT keeps the vision exemption (board #136 — vision rows
  // never wait on a stamp); the filtered view is therefore a superset of the
  // count, never a subset, which is the safe direction to be wrong in.
  //
  // Board #261 — the exemption is now read from the TYPE MAP ("does this type
  // even have an Approval status?") instead of a hand-written `!isVision`.
  // Identical today, since `vision` is the only type without it; correct
  // tomorrow, because a `goal` DOES stop at Approval (Kevin authorises it to
  // start) and must not be silently dropped from his stamp inbox.
  const awaitingCount = tasks.filter((t) => t.status === 'Approval'
    && TASK_TYPES[typeOf(t)].statuses.includes('Approval')).length;
  const awaitingOk = statusFilter.length === 1 && statusFilter[0] === 'Approval';

  // THE Approval stamp (board #136, one mode since #232): POST /stamp flips
  // Approval → Todo, arms the task and logs the system comment server-side. It
  // no longer sets kevin_final — the stamp is permission to START.
  const stamp = async (id: number, mode: StampMode) => {
    try {
      await apiFetch(`/api/dev-tasks/${id}/stamp`, {
        method: 'POST', body: JSON.stringify({ mode, author: commentAuthor }),
      });
    } catch (e) { setErr(String(e)); }
    load();
  };

  // Default-chain backfill (board #134 item 2): open LEAF tasks (not vision,
  // no subtasks) with an empty checklist get the universal 4-step chain.
  const chainless = useMemo(() => tasks.filter((t) =>
    !isFinished(t.status) && (t.checklist || []).length === 0 &&
    !((t.tags || []).includes('vision') || byParent.has(t.id))), [tasks, byParent]);
  const backfillChains = async () => {
    if (chainless.length === 0) return;
    if (!confirm(`Add the default 4-step process chain (Build → M review → Kevin approval → Ship) to ${chainless.length} open task(s) without one?`)) return;
    try {
      await Promise.all(chainless.map((t) => apiFetch(`/api/dev-tasks/${t.id}`, {
        method: 'PATCH',
        body: JSON.stringify({ checklist: defaultChain(t.assignee), actor: commentAuthor }),
      })));
    } catch (e) { setErr(String(e)); }
    load();
  };

  // Board-side half of the modal's liveness poll (board #134 item 5). Board
  // #148: the modal hands us the board's max(updated_at) from its consolidated
  // poll probe — refetch the 120-row task list + run states ONLY when it moved
  // (or when called with no arg — a forced refresh, e.g. after a stamp). An
  // open modal no longer drives a full board select every tick.
  // Board #143: the unread-mention badge refreshes on EVERY tick — it is one
  // cheap count query, and a new @-mention does not necessarily move the
  // board's max(updated_at) (a comment insert need not touch dev_tasks).
  const pollRefresh = useCallback((boardMax?: string | null) => {
    loadUnread();
    if (boardMax === undefined) { load(); loadRuns(); return; }          // forced
    if (boardMaxInit.current && boardMax === lastBoardMax.current) return; // unchanged
    const wasInit = boardMaxInit.current;
    lastBoardMax.current = boardMax;
    boardMaxInit.current = true;
    if (wasInit) { load(); loadRuns(); }                                  // changed → refetch
  }, [load, loadRuns, loadUnread]);

  const sortedVisible = useMemo(() => {
    if (!sort) return visible;
    const cmp = (a: Task, b: Task): number => {
      switch (sort.key) {
        case 'id': return a.id - b.id;
        case 'pri': return (a.priority_phase - b.priority_phase) || (a.priority_seq - b.priority_seq);
        case 'title': return a.title.localeCompare(b.title);
        case 'status': return STATUSES.indexOf(a.status) - STATUSES.indexOf(b.status);
        case 'area': return (a.area || '').localeCompare(b.area || '');
        case 'who': return (a.assignee || '').localeCompare(b.assignee || '');
        default: return 0;
      }
    };
    return [...visible].sort((a, b) => cmp(a, b) * sort.dir);
  }, [visible, sort]);

  const clickSort = (key: string) => setSort((prev) =>
    prev?.key === key ? (prev.dir === 1 ? { key, dir: -1 } : null) : { key, dir: 1 });
  const sortMark = (key: string) => sort?.key === key ? (sort.dir === 1 ? ' ▲' : ' ▼') : '';

  // OPEN = not FINISHED (board #232). Counting only `!== 'Done'` would keep
  // every task Kevin has closed out in the "open" number forever.
  const openCount = tasks.filter((t) => !isFinished(t.status)).length;
  const liveCount = tasks.filter((t) => t.impacts_live && !isFinished(t.status)).length;
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
      {/* two-touch marker (board #136) — set by the Approval stamps */}
      <TwoTouchChip t={t} />
      {/* run-requested renders as the Run button's ⏳ state, not a raw chip;
          RETIRED_TAGS (needs-review / needs-approval / kevin-ok) stay hidden
          until M's one-time post-ship tag migration deletes them */}
      {(t.tags || []).filter((tag) =>
        tag !== RUN_REQUESTED_TAG && !RETIRED_TAGS.includes(tag)).map((tag) => (
        <span key={tag} style={tagChip}>{tag}</span>
      ))}
    </>
  );

  const TaskRow = ({ t, indent = false, rollup }: {
    t: Task; indent?: boolean; rollup?: { done: number; total: number };
  }) => (
    <tr key={t.id} style={{
      borderBottom: '1px solid var(--border)',
      opacity: isFinished(t.status) ? 0.55 : 1,
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
        {/* board #261 — the TYPE, visible on the row (Kevin's ask). Container
            types wear a glyph; `action` (the default row) wears none. */}
        <TaskTypeIcon type={typeOf(t)} />
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
          <span style={{ ...tagChip, marginLeft: 6 }} title="subtask of this container">↳ #{t.parent_id}</span>}
        <TagChips t={t} />
        {/* board #169 — the handoff chain as owner avatars, so the process is
            visible on the card instead of hidden behind the Process tab */}
        <HandoffChain task={t} />
        <NextChip t={t} subtasks={byParent.get(t.id) || []} />
        {/* board #151 tell — Todo task whose next actor is Kevin sits undispatched */}
        <StuckChip t={t} subtasks={byParent.get(t.id) || []} />
        {/* Approval-stage stamp buttons (board #136) — Kevin's two-touch
            choice, right in the row so the Awaiting-my-OK view is one-click */}
        <StampButtons t={t} onStamp={stamp} compact />
        {rollup && <span style={{ marginLeft: 6 }}><ProgressBar done={rollup.done} total={rollup.total} mini /></span>}
        {(t.blocked_by?.length > 0) && <span style={{ color: 'var(--red)', fontSize: 11 }}> ⛔{t.blocked_by.join(',')}</span>}
        {rollup && (
          <button style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '1px 6px', marginLeft: 8 }}
            title="add a subtask under this container"
            onClick={(e) => { e.stopPropagation(); startSubtask(t.id); }}>+ subtask</button>
        )}
      </td>
      <td style={cell}>
        <select style={{ ...input, color: STATUS_COLOR[t.status] }} value={t.status}
          title={STATUS_DEF[t.status] || ''}
          onChange={(e) => patch(t.id, { status: e.target.value })}>
          {statusOptionsFor(t, typeOf(t)).map((s) =>
            <option key={s} value={s} title={STATUS_DEF[s]}>{s}</option>)}
        </select>
      </td>
      <td style={cell}>
        {/* blast-radius chip (board #136) — prominent in the Approval view */}
        <ImpactSelect t={t} onPick={(v) => patch(t.id, { impact: v })} compact />
      </td>
      <td style={cell}>
        {t.impacts_live && <span style={badge('var(--red)')} title="touches live engine/trading code — deploy carefully">🔴 live</span>}
        {!isFinished(t.status) && (t.needs_live_validation
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
          {withLegacy(roleOptions, t.assignee || '').map((a) => <option key={a} value={a}>{a || '—'}</option>)}
        </select>
      </td>
      {/* board #198 — INPUT: standing permission for an agent to claim this */}
      <td style={cell}>
        <AiEligibleToggle task={t} compact
          onToggle={(v) => patch(t.id, { ai_eligible: v })} />
      </td>
      {/* board #198 — OUTPUT (pill: what the dispatcher is doing) + ACTION
          (button: run it once now). Deliberately not the same control. */}
      <td style={cell}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 3 }}>
          <RunStatePill task={t} latestRun={latestRunByTask.get(t.id)} compact />
          <RunButton task={t} allTasks={tasks} headless={headless}
            latestRun={latestRunByTask.get(t.id)} onRequest={requestRun} compact />
        </div>
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
          rollup={{ done: kids.filter((k) => isFinished(k.status)).length, total: kids.length }} />);
      if (!collapsed.has(v.id)) {
        shownKids.forEach((k) => groupedRows.push(<TaskRow key={k.id} t={k} indent />));
      }
    });
    const shownLoose = looseTasks.filter(matches);
    if (shownLoose.length > 0) {
      groupedRows.push(
        <tr key="loose-header">
          <td colSpan={11} style={{ ...cell, color: 'var(--text-tertiary)', fontSize: 11, paddingTop: 14 }}>
            ungrouped — tasks without a container parent
          </td>
        </tr>);
      shownLoose.forEach((t) => groupedRows.push(<TaskRow key={t.id} t={t} />));
    }
  }

  return (
    <AgentRegistryContext.Provider value={agentReg}>
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <h1 style={{ fontSize: 22, margin: 0 }}>Dev Task Tracker</h1>
        {/* Board #297 — Kevin's ask: a help button at the TOP of the page, not
            buried in a menu. The manual is generated from the same model this
            page renders from, so it can never describe a board other than the
            one in front of you. */}
        <button style={{ ...input, cursor: 'pointer', fontSize: 12, padding: '3px 9px' }}
          title="how this board works — task types, statuses and assignees, generated from the board's own model (warts included, board #297)"
          onClick={() => setHelpOpen(true)}>✻ Help</button>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 16 }}>
        {openCount} open · {liveCount} touch live · sorted by priority (phase.seq, 1.1 first). Pipeline:
        Backlog → Scoping → Approval → Todo → In Progress → Review → Staged → Done (Blocked = exception;
        vision rows exempt). Legend:
        🟢 offline-ok = fully validatable now (safe to work); ⏳ = needs live-market data to confirm;
        🔴 = touches live engine/trading code (deploy carefully); ⚡ = urgent (a tag, not a priority);
        🔍 = discovered mid-work (rabbit-hole fix, parented under its container);
        🔏 = two-touch (Kevin reviews before Done); Impact = blast radius (contained | app | engine | live).
        🤖 AI = may an agent claim this task (armed by Kevin’s Approval stamp, independent of the
        kanban status) · the Run column reads out what the dispatcher is doing (idle / queued /
        running / failed) next to ▶ Run once, a one-off queue jump.
        Owner avatars: a <b>round</b> avatar is a headless agent the dispatcher can auto-run;
        a <b>squared, ringed</b> one (◧ session) is a live session or human — it never auto-runs,
        so a task parked there is waiting on a person, by design. Read from the agents registry
        (<code>status</code>/<code>kind</code>), not from the letter.
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}

      <Card>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <strong style={{ fontSize: 13 }}>+ New:</strong>
          <input ref={titleRef} style={{ ...input, flex: 1, minWidth: 240 }}
            placeholder={ntParent ? `subtask under #${ntParent.id} ${ntParent.title.slice(0, 40)}…` : 'task title…'}
            value={nt.title} onChange={(e) => setNt({ ...nt, title: e.target.value })}
            onKeyDown={(e) => e.key === 'Enter' && createTask()} />
          <select style={input} title="parent container — vision or goal (none = a new vision item)"
            value={nt.parent_id ?? ''}
            onChange={(e) => setNt({ ...nt, parent_id: e.target.value ? +e.target.value : null })}>
            <option value="">— container —</option>
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
            {roleOptions.map((a) => <option key={a} value={a}>{a || '—'}</option>)}
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
            {/* Board #295 — was "By vision", which named ONE of the two types it
                groups. It groups every CONTAINER (vision ◈ and goal ⚑), and has
                since #262; the old label read as "goals excluded" and got a
                working feature reported as missing. Behaviour is unchanged. */}
            <button style={{ ...input, cursor: 'pointer', fontWeight: view === 'grouped' ? 700 : 400, background: view === 'grouped' ? 'var(--bg-input)' : 'transparent' }}
              title="group subtasks under their container row — vision ◈ and goal ⚑ alike (action tasks are where work happens, never containers)"
              onClick={() => setView('grouped')}>By container</button>
            <button style={{ ...input, cursor: 'pointer', marginLeft: 4, fontWeight: view === 'flat' ? 700 : 400, background: view === 'flat' ? 'var(--bg-input)' : 'transparent' }}
              onClick={() => setView('flat')}>Flat</button>
            {/* Messages tab (board #143) — unread @-mentions for the current
                viewer role; the badge is the unseen count. */}
            <button style={{
              ...input, cursor: 'pointer', marginLeft: 4, position: 'relative',
              fontWeight: view === 'messages' ? 700 : 400,
              background: view === 'messages' ? 'var(--bg-input)' : 'transparent',
              borderColor: unreadCount ? 'var(--blue)' : 'var(--border)',
            }}
              title={`your @-mentions (as ${commentAuthor || '—'}) — ${unreadCount} unread`}
              onClick={() => { setView('messages'); setMsgSignal((n) => n + 1); }}>
              ✉ Messages
              {unreadCount > 0 && (
                <span style={{
                  marginLeft: 6, background: 'var(--blue)', color: '#fff',
                  borderRadius: 9, fontSize: 10.5, fontWeight: 700, padding: '0 6px',
                }}>{unreadCount}</span>
              )}
            </button>
          </span>
          <button style={{
            ...input, cursor: 'pointer', fontWeight: awaitingOk ? 700 : 400, color: '#c9a227',
            borderColor: awaitingOk ? '#c9a227' : 'var(--border)',
          }}
            title="Kevin's inbox — tasks sitting in Approval awaiting a stamp. A PRESET over the status filter (board #246): it sets status = {Approval}; click again to clear. The count excludes vision rows, which are exempt from the pipeline (board #136)."
            onClick={() => setStatusFilter(awaitingOk ? [] : ['Approval'])}>
            ✋ Awaiting my OK ({awaitingCount})</button>
          {/* board #151: a Todo task whose next actor is Kevin can't dispatch —
              surfaced only when the class exists so it never hides again */}
          {(stuckCount > 0 || stuckOnly) && (
            <button style={{
              ...input, cursor: 'pointer', fontWeight: stuckOnly ? 700 : 400,
              color: 'var(--red)', borderColor: stuckOnly ? 'var(--red)' : 'var(--border)',
            }}
              title="contradiction (board #151): Todo tasks whose next actor is Kevin — queue-eligible yet the dispatcher's next-actor gate skips them, so they sit undispatched. Tick the pending Kevin checklist step (or re-stamp) to release."
              onClick={() => setStuckOnly(!stuckOnly)}>
              ⚠ stuck in Todo ({stuckCount})</button>
          )}
          {/* board #198 — the two questions the toggle/pill split makes askable
              at a glance: what is ARMED for an agent, and what is RUNNING now */}
          <button style={{
            ...input, cursor: 'pointer', fontWeight: armedOnly ? 700 : 400, color: 'var(--green)',
            borderColor: armedOnly ? 'var(--green)' : 'var(--border)',
          }}
            title={`tasks armed for AI work (ai_eligible = true) — ${AI_COL_TIP}`}
            onClick={() => setArmedOnly(!armedOnly)}>
            🤖 AI-eligible ({armedCount})</button>
          <button style={{
            ...input, cursor: 'pointer', fontWeight: runningOnly ? 700 : 400, color: 'var(--blue)',
            borderColor: runningOnly ? 'var(--blue)' : 'var(--border)',
          }}
            title="tasks a headless agent is working RIGHT NOW (latest run_history row still open)"
            onClick={() => setRunningOnly(!runningOnly)}>
            ⚙ running ({runningCount})</button>
          <label><input type="checkbox" checked={hideDone} onChange={(e) => setHideDone(e.target.checked)} /> hide Done</label>
          <label><input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} /> 🔴 live only</label>
          {/* Board #246 — three checkbox multi-selects, identical behaviour.
              STATUSES is the SSOT for the status list (never a literal), so a
              twelfth status appears here for free. The old `👀 in Review`
              checkbox is gone: it was this filter with one box ticked. */}
          <MultiSelectFilter label="status" options={STATUSES} selected={statusFilter}
            onChange={setStatusFilter} colors={STATUS_COLOR}
            title="status filter — tick any number (e.g. Review + Staged). None ticked = every status." />
          <MultiSelectFilter label="area" options={AREAS} selected={areaFilter}
            onChange={setAreaFilter}
            title="area filter — tick any number. None ticked = every area." />
          <MultiSelectFilter label="assignee" options={assigneeOptions} selected={assigneeFilter}
            onChange={setAssigneeFilter} allLabel="all"
            title={`assignee filter — tick any number, including ${UNASSIGNED_LABEL}. None ticked = everyone.`} />
          {activeFilters > 0 && (
            <button style={{ ...input, cursor: 'pointer', color: 'var(--blue)', borderColor: 'var(--blue)' }}
              title="clear the status / area / assignee filters — the board is filtered, not empty"
              onClick={() => { setStatusFilter([]); setAreaFilter([]); setAssigneeFilter([]); }}>
              ✕ clear {activeFilters} filter{activeFilters > 1 ? 's' : ''}</button>
          )}
          <button style={{ ...input, cursor: 'pointer' }} onClick={load}>↻ refresh</button>
          {chainless.length > 0 && (
            <button style={{ ...input, cursor: 'pointer' }} onClick={backfillChains}
              title="backfill the default 4-step process chain (Build/investigate → M review → Kevin approval where flagged → Ship/close) onto open tasks that have none">
              ⛓ add default chain ({chainless.length})</button>
          )}
        </div>
      </Card>

      {view === 'messages' ? (
        <Card>
          <TaskMessagesPanel forRole={commentAuthor}
            onOpenTask={(id) => setModal({ id })}
            onSeenChange={loadUnread} refreshSignal={msgSignal} />
        </Card>
      ) : loading ? <Card>Loading…</Card> : (
        <Card>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)', fontSize: 11, color: 'var(--text-tertiary)' }}>
                {view === 'flat' ? (
                  <>
                    <th style={{ ...cell, width: 44, cursor: 'pointer' }} title="sort" onClick={() => clickSort('id')}>ID{sortMark('id')}</th>
                    <th style={{ ...cell, width: 96, cursor: 'pointer' }} title="sort" onClick={() => clickSort('pri')}>Pri{sortMark('pri')}</th>
                    <th style={{ ...cell, cursor: 'pointer' }} title="sort" onClick={() => clickSort('title')}>Task{sortMark('title')}</th>
                    <th style={{ ...cell, width: 130, cursor: 'pointer' }} title="sort" onClick={() => clickSort('status')}>Status{sortMark('status')}</th>
                    <th style={{ ...cell, width: 92 }} title="blast radius: contained | app | engine | live (M-editable)">Impact</th>
                    <th style={{ ...cell, width: 170 }}>Flags</th>
                    <th style={{ ...cell, width: 110, cursor: 'pointer' }} title="sort" onClick={() => clickSort('area')}>Area{sortMark('area')}</th>
                    <th style={{ ...cell, width: 100, cursor: 'pointer' }} title="sort" onClick={() => clickSort('who')}>Who{sortMark('who')}</th>
                    <th style={{ ...cell, width: 56, whiteSpace: 'nowrap' }} title={AI_COL_TIP}>🤖 AI</th>
                    <th style={{ ...cell, width: 118 }} title={RUN_COL_TIP}>Run</th>
                    <th style={{ ...cell, width: 32 }}></th>
                  </>
                ) : (
                  <>
                    <th style={{ ...cell, width: 44 }}>ID</th><th style={{ ...cell, width: 96 }}>Pri</th><th style={cell}>Task</th><th style={{ ...cell, width: 130 }}>Status</th>
                    <th style={{ ...cell, width: 92 }} title="blast radius: contained | app | engine | live (M-editable)">Impact</th>
                    <th style={{ ...cell, width: 170 }}>Flags</th><th style={{ ...cell, width: 110 }}>Area</th><th style={{ ...cell, width: 100 }}>Who</th>
                    <th style={{ ...cell, width: 56, whiteSpace: 'nowrap' }} title={AI_COL_TIP}>🤖 AI</th>
                    <th style={{ ...cell, width: 118 }} title={RUN_COL_TIP}>Run</th>
                    <th style={{ ...cell, width: 32 }}></th>
                  </>
                )}
              </tr>
            </thead>
            <tbody>
              {view === 'flat'
                ? sortedVisible.map((t) => <TaskRow key={t.id} t={t} />)
                : groupedRows}
              {/* Board #246: an empty board must never read as data loss — say
                  how many rows were fetched and what is hiding them. */}
              {((view === 'flat' && visible.length === 0) || (view === 'grouped' && groupedRows.length === 0)) &&
                <tr><td colSpan={11} style={{ ...cell, color: 'var(--text-tertiary)' }}>
                  No tasks match the filters — {tasks.length} loaded
                  {activeFilters > 0 ? `, ${activeFilters} filter${activeFilters > 1 ? 's' : ''} narrowing them` : ''}.
                  {activeFilters > 0 && <> Nothing hidden by a filter is lost; <button
                    style={{ ...input, cursor: 'pointer', marginLeft: 6, fontSize: 11, padding: '2px 6px' }}
                    onClick={() => { setStatusFilter([]); setAreaFilter([]); setAssigneeFilter([]); }}>
                    show all</button></>}
                </td></tr>}
            </tbody>
          </table>
        </Card>
      )}

      {/* Board #297 — fed the live task list so the per-goal assignee lanes it
          shows are the board's ACTUAL open goals, derived the same way the
          assignee selects derive them, not a list typed into the manual. */}
      {helpOpen && <TaskHelpModal tasks={tasks} onClose={() => setHelpOpen(false)} />}

      {modalTask && (
        <TaskDetailModal key={`${modalTask.id}:${modal?.sel ?? ''}`}
          task={modalTask} allTasks={tasks} visionOptions={visionItems}
          initialSelected={modal?.sel}
          roles={roleOptions}
          patch={patch} del={del}
          onClose={() => setModal(null)}
          onOpenTask={(id, sel) => setModal({ id, sel })}
          commentAuthor={commentAuthor} onPickAuthor={pickAuthor}
          headless={headless} onRunRequested={() => { load(); loadRuns(); }}
          onPollTick={pollRefresh} />
      )}
    </div>
    </AgentRegistryContext.Provider>
  );
}
