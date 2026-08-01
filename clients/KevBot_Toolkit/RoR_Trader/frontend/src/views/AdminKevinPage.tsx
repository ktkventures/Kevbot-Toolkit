/**
 * Kevin's dashboard — /admin/kevin (board #257). The OWNER's page.
 *
 * Kevin, 07-31: *"There are generally three fundamental dashboard designs: the
 * manager, who has the most varied and complex work; the session dashboard,
 * which is what you created for R; and the human/owner dashboard — me — who
 * wants to be kept up to date and knows what's needed from him."*
 *
 * So: **M's dashboard answers "what is happening?". This one answers "what is
 * waiting on ME — and can I stop worrying about the rest?"** Ruthlessly
 * filtered to action is the whole brief, and three rules fall out of it:
 *
 *   1 NEVER SHOW A COUNT HE CANNOT ACT ON. If it is not his move it belongs in
 *     the calm strip or nowhere.
 *   2 EVERY ROW IS ACTIONABLE FROM THIS PAGE — stamp, close, reassign, and the
 *     task modal opens IN PLACE. If he has to navigate away to act, this is a
 *     link list, not a dashboard.
 *   3 ZERO IS A GOOD ANSWER AND MUST LOOK LIKE ONE. Empty modules render calm
 *     and affirmative, never blank. R's CARS-WAITING panel read as broken on
 *     07-30 until it was explained; that is a design defect, not a
 *     misunderstanding.
 *
 * IT DOES NOT REPLACE /admin/tasks, and the difference is the design. Kevin:
 * *"tasks is kind of like a shared resource and it's highly filterable. My goal
 * with this dashboard is something that really just pertains to me… a highly
 * executable planner so I don't have to apply a lot of filters."* `/admin/tasks`
 * is the SHARED, FILTERABLE surface; this is the PERSONAL, PRE-FILTERED one —
 * so **it has no filter bar, and must never grow one.** If a module needs
 * filtering to be useful, the module is wrong and should be re-specified rather
 * than given controls.
 *
 * ONE MODAL, TWO CALLERS. `TaskDetailModal` is rendered here with the same
 * props `AdminTasksPage` passes — not copied, not forked, not extracted. It is
 * wrapped in `AgentRegistryContext.Provider` for the same reason the board is:
 * without it, role chips fall back to their registry-outage rendering, which is
 * a quiet visual regression rather than an error.
 *
 * THE DERIVATIONS ARE NOT IN THIS FILE. They live in `kevinDashboard.ts`
 * because `src/test_kevin_dashboard_257.py` lifts and EXECUTES them in node.
 * Nothing about "is this on Kevin's plate" or "is this stuck" is decided here.
 *
 * Production-bundle rails (CLAUDE.md): every hook before any early return, no
 * IIFE initialisation, declare before the useMemo that references it.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import TaskDetailModal from './TaskDetailModal';
import SignOfLife from './SignOfLife';
import {
  AgentMeta, AgentRegistryContext, ASSIGNEES, AUTHOR_LS_KEY, DISPATCHER, RoleChip,
  RolePicker, RunRow, STATUS_COLOR, Task, ageColor, agentRegistryMap, groupBoard,
  input, isFinished, relTime, rolesWithGoalLanes, runIneligibleReason, tagChip,
  utcDay, utcStamp,
} from './taskBoardShared';
import { CapSetting, isLeaseLive, resolveDailyCap, shippingRows } from './AdminDispatchPage';
import {
  CarsPanel, HeartbeatPayload, R_PAGE_ROWS, carsWaitingForATrain,
} from './rSessionSignals';
import { FeedComment, buildFeed } from './mSessionFeed';
import {
  EMPTY_COPY, FEED_ROWS, KEVIN, LOOP_VERSION_NOTE, PLATE_ROWS, Plate, PlateRow,
  STALL_ROWS, StallPanel, buildPlate, buildStalls, decisionsAndOutcomes, loopState,
} from './kevinDashboard';

/** Auto-refresh cadence — matched to /admin/dispatch and /admin/r-session. Two
 *  idioms on one admin section is how a section stops being read. */
const REFRESH_MS = 15000;
const RUN_LIMIT = 400;
const COMMENT_LIMIT = 300;
const CAP_ENDPOINT = `/api/admin/system-settings/${DISPATCHER.DAILY_CAP_SETTING}`;

interface ServicePayload {
  status: string; commit: string | null; branch: string | null;
  service: string | null; scope: string; known_gaps: string[]; generated_at: string;
}
interface DeployEntry {
  day: string | null; time: string; sha: string; wave: number | null; excerpt: string;
}
interface DeployLogPayload {
  entries: DeployEntry[]; waves: number[]; found: boolean;
  resolved_path: string; error: string | null; generated_at: string;
}

type TabKey = 'plate' | 'stuck' | 'team' | 'activity' | 'health';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'plate', label: 'Your Plate' },
  { key: 'stuck', label: 'Stuck' },
  { key: 'team', label: 'Team' },
  { key: 'activity', label: 'Activity' },
  { key: 'health', label: 'Health' },
];

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11.5 };
const dim: React.CSSProperties = { color: 'var(--text-tertiary)' };
const calm: React.CSSProperties = { fontSize: 13, color: 'var(--green)', fontWeight: 600 };

const Panel = ({ title, sub, right, children }: {
  title: string; sub?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode;
}) => (
  <div style={{ marginBottom: 14 }}>
    <Card>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 9 }}>
        {/* board #278 — `minWidth: 0` so a long token in `sub` cannot push
            `right` (the headline number) off the card. */}
        <div style={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
          <div style={{
            fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase',
            color: 'var(--text-tertiary)',
          }}>{title}</div>
          {sub && <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>{sub}</div>}
        </div>
        {right}
      </div>
      {children}
    </Card>
  </div>
);

export default function AdminKevinPage() {
  const [tab, setTab] = useState<TabKey>('plate');
  const [tasks, setTasks] = useState<Task[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [agents, setAgents] = useState<AgentMeta[]>([]);
  const [comments, setComments] = useState<FeedComment[]>([]);
  const [hb, setHb] = useState<HeartbeatPayload | null>(null);
  const [hbErr, setHbErr] = useState<string | null>(null);
  const [svc, setSvc] = useState<ServicePayload | null>(null);
  const [svcErr, setSvcErr] = useState<string | null>(null);
  const [dlog, setDlog] = useState<DeployLogPayload | null>(null);
  const [capRow, setCapRow] = useState<CapSetting | null>(null);
  const [capRead, setCapRead] = useState<string>('loading');
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<number | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  const [modal, setModal] = useState<{ id: number; sel?: number } | null>(null);
  // Every write this page makes is Kevin's — it is his dashboard. The stored
  // author is still honoured so a session acting as someone else does not
  // silently sign his name to a stamp.
  const [author, setAuthor] = useState(KEVIN);
  const [roles, setRoles] = useState<string[]>(ASSIGNEES);
  // Board #289 — registry lanes plus one derived lane per OPEN goal (`G281`).
  // A goal session has no registry row by design, so the registry fetch above
  // can never surface it; every assignee picker here reads THIS, not `roles`.
  const roleOptions = useMemo(() => rolesWithGoalLanes(roles, tasks), [roles, tasks]);

  const load = useCallback(async () => {
    try {
      setErr(null);
      // include_done=true is REQUIRED: the CLOSE bucket is made of Done tasks,
      // and the activity feed references them. Four list GETs joined in memory,
      // never a per-row fetch (the board-#148 poll-efficiency rail).
      const [ts, rh, ag, cs] = await Promise.all([
        apiFetch<Task[]>('/api/dev-tasks?include_done=true'),
        apiFetch<RunRow[]>(`/api/run-history?limit=${RUN_LIMIT}`).catch(() => [] as RunRow[]),
        apiFetch<AgentMeta[]>('/api/agents').catch(() => [] as AgentMeta[]),
        apiFetch<FeedComment[]>(`/api/dev-tasks/comments/recent?limit=${COMMENT_LIMIT}`)
          .catch(() => [] as FeedComment[]),
      ]);
      setTasks(ts || []);
      setRuns(rh || []);
      setAgents(ag || []);
      setComments(cs || []);
      const letters = (ag || []).map((a) => a.letter).filter(Boolean);
      if (letters.length) setRoles(['', ...letters]);
      setFetchedAt(new Date().toISOString());
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);

  /** The environment reads. Each keeps its OWN failure: an unreadable endpoint
   *  and an empty one are different facts, and folding them together is the
   *  false-green shape (#157) every panel here is built against. */
  const loadEnv = useCallback(async () => {
    try {
      const p = await apiFetch<HeartbeatPayload>('/api/r-session/heartbeats');
      setHb(p); setHbErr(null);
    } catch (e) { setHb(null); setHbErr(String(e)); }
    try {
      const s = await apiFetch<ServicePayload>('/api/r-session/service');
      setSvc(s); setSvcErr(null);
    } catch (e) { setSvc(null); setSvcErr(String(e)); }
    try { setDlog(await apiFetch<DeployLogPayload>('/api/r-session/deploy-log')); }
    catch { setDlog(null); }
    try {
      setCapRow(await apiFetch<CapSetting>(CAP_ENDPOINT) || null);
      setCapRead('ok');
    } catch { setCapRow(null); setCapRead('error'); }
  }, []);

  useEffect(() => { load(); loadEnv(); }, [load, loadEnv]);
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) { load(); loadEnv(); }
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [load, loadEnv]);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => { setAuthor(localStorage.getItem(AUTHOR_LS_KEY) || KEVIN); }, []);
  // Deep-linkable, exactly as the board is: /admin/kevin?task=257 opens the
  // modal here rather than bouncing him to /admin/tasks.
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

  const agentReg = useMemo(() => agentRegistryMap(agents), [agents]);
  const headlessLetters = useMemo(() => {
    const l = agents.filter((a) => a.status === 'headless').map((a) => a.letter);
    return l.length ? l : ['M-A'];
  }, [agents]);
  const headless = useMemo(() => new Set(headlessLetters), [headlessLetters]);

  const runsByTask = useMemo(() => {
    const m = new Map<number, RunRow[]>();
    runs.forEach((r) => m.set(r.task_id, [...(m.get(r.task_id) || []), r]));
    return m;   // API order is newest-first and is preserved per bucket
  }, [runs]);

  const pushedKnown = useMemo(
    () => runs.some((r) => Object.prototype.hasOwnProperty.call(r, 'pushed_branch')),
    [runs]);

  const lane = useMemo(
    () => shippingRows(tasks, runsByTask, pushedKnown),
    [tasks, runsByTask, pushedKnown, tick]);          // eslint-disable-line react-hooks/exhaustive-deps
  const cars = useMemo<CarsPanel>(
    () => carsWaitingForATrain(lane, tasks, R_PAGE_ROWS), [lane, tasks]);

  const liveRuns = useMemo(
    () => runs.filter((r) => r.outcome === 'running' && isLeaseLive(r)),
    [runs, tick]);                                    // eslint-disable-line react-hooks/exhaustive-deps
  const lanesFree = Math.max(0, DISPATCHER.CONCURRENCY - liveRuns.length);

  // "Which tasks could the dispatcher take right now?" is answered by the
  // EXISTING resolver (runIneligibleReason), never re-derived here.
  const dispatchableIds = useMemo(() => tasks.filter((t) => t.assignee
    && headless.has(t.assignee) && !isFinished(t.status) && !!t.ai_eligible
    && !liveRuns.some((r) => r.task_id === t.id)
    && !runIneligibleReason(t, tasks, headless)).map((t) => t.id),
  [tasks, headless, liveRuns]);

  const plate = useMemo<Plate>(
    () => buildPlate(tasks, PLATE_ROWS), [tasks, tick]);  // eslint-disable-line react-hooks/exhaustive-deps
  const plateIds = useMemo(
    () => plate.buckets.flatMap((b) => b.rows.map((r) => r.task.id)), [plate]);

  const starved = useMemo(() => (lanesFree > 0
    ? dispatchableIds.filter((id) => {
      const t = tasks.find((x) => x.id === id);
      return !!t && (Date.now() - new Date(t.updated_at).getTime()) / 3600000 >= 0.5;
    }).length
    : 0), [dispatchableIds, tasks, lanesFree, tick]);   // eslint-disable-line react-hooks/exhaustive-deps

  const loop = useMemo(
    () => loopState(runs, starved, Date.now()),
    [runs, starved, tick]);                            // eslint-disable-line react-hooks/exhaustive-deps

  const stalls = useMemo<StallPanel>(() => buildStalls({
    cars, runs, tasks, plateIds, dispatchableIds, lanesFree, loop,
    sessions: hb ? hb.sessions : [], nowMs: Date.now(),
  }, STALL_ROWS), [cars, runs, tasks, plateIds, dispatchableIds, lanesFree, loop, hb, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  const feed = useMemo(
    () => decisionsAndOutcomes(buildFeed(comments, runs), FEED_ROWS), [comments, runs]);

  const capState = useMemo(() => (capRead === 'error'
    ? { cap: DISPATCHER.DAILY_CAP, source: 'constant (settings unreadable)', fromSettings: false }
    : resolveDailyCap(capRow?.db_value)), [capRow, capRead]);
  const startedToday = useMemo(() => {
    const today = new Date().toISOString().slice(0, 10);
    return runs.filter((r) => r.started_at && utcDay(r.started_at) === today).length;
  }, [runs]);

  const visionItems = useMemo(() => groupBoard(tasks).visionItems, [tasks]);
  const modalTask = (modal && tasks.find((t) => t.id === modal.id)) || null;

  /* ── The three inline actions. Every row acts FROM this page. ─────────── */

  const patch = useCallback(async (id: number, fields: Partial<Task>) => {
    setTasks((t) => t.map((x) => (x.id === id ? { ...x, ...fields } : x)));
    try {
      await apiFetch(`/api/dev-tasks/${id}`, {
        method: 'PATCH', body: JSON.stringify({ ...fields, actor: author }),
      });
    } catch (e) { setErr(String(e)); }
    load();
  }, [author, load]);

  const del = useCallback(async (id: number) => {
    if (!confirm('Delete this task? Subtasks are deleted with their vision item.')) return;
    if (modal?.id === id) setModal(null);
    try { await apiFetch(`/api/dev-tasks/${id}`, { method: 'DELETE' }); }
    catch (e) { setErr(String(e)); }
    load();
  }, [load, modal]);

  /** Approve / reject the CURRENT step's stamp — the same server-owned endpoint
   *  the modal uses (POST /steps/stamp). Never a checklist PATCH: the
   *  transition rules and the system comment live in the endpoint. */
  const stampStep = useCallback(async (id: number, decision: string) => {
    setBusy(id);
    try {
      await apiFetch(`/api/dev-tasks/${id}/steps/stamp`, {
        method: 'POST', body: JSON.stringify({ decision, actor: author }),
      });
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
    load();
  }, [author, load]);

  /** Kevin's close — `Done` → `Closed`. His retrospective look at finished
   *  work; it blocks nothing, which is why this is a one-click row action. */
  const closeTask = useCallback((id: number) => patch(id, { status: 'Closed' }), [patch]);

  const closeAll = useCallback(async (ids: number[]) => {
    if (ids.length === 0) return;
    if (!confirm(`Close ${ids.length} finished task(s)? They are already Done — this is your retrospective tick.`)) return;
    setBusy(-1);
    try {
      await Promise.all(ids.map((id) => apiFetch(`/api/dev-tasks/${id}`, {
        method: 'PATCH', body: JSON.stringify({ status: 'Closed', actor: author }),
      })));
    } catch (e) { setErr(String(e)); }
    finally { setBusy(null); }
    load();
  }, [author, load]);

  /* ── Row + panel rendering ────────────────────────────────────────────── */

  const TaskTitle = ({ id, title }: { id: number; title: string }) => (
    <button
      onClick={() => setModal({ id })}
      title="open the task here — the same modal /admin/tasks uses, without leaving this page"
      style={{
        background: 'none', border: 'none', padding: 0, cursor: 'pointer', textAlign: 'left',
        color: 'var(--accent)', fontWeight: 600, fontSize: 12.5,
      }}>#{id} · {title}</button>
  );

  const PlateRowView = ({ r }: { r: PlateRow }) => (
    <div style={{
      display: 'flex', gap: 10, alignItems: 'flex-start', padding: '7px 0',
      borderTop: '1px solid var(--border)',
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <TaskTitle id={r.task.id} title={r.task.title} />
        <div style={{ ...dim, fontSize: 11.5, marginTop: 2 }}>
          {r.stepTitle && <span>▶ {r.stepTitle} · </span>}
          <span title={`age measured from ${r.ageFrom}`} style={{
            color: ageColor(r.ageH), fontWeight: 700,
          }}>
            {r.ageH < 1 ? `${Math.max(1, Math.round(r.ageH * 60))}m` : `${Math.round(r.ageH)}h`} on you
          </span>
          {r.unblocks && <span> · unblocks <b style={{ color: 'var(--text-secondary)' }}>{r.unblocks}</b></span>}
          {r.blocking > 0 && (
            <span style={{ color: 'var(--red)', fontWeight: 700 }}>
              {' '}· {r.blocking} task{r.blocking === 1 ? '' : 's'} cannot start
            </span>
          )}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 5, alignItems: 'center', flex: 'none' }}>
        {r.bucket === 'stamp' && (
          <>
            <button disabled={busy === r.task.id} onClick={() => stampStep(r.task.id, 'approve')}
              title="approve this step's stamp — its owner may then complete it"
              style={{ ...input, cursor: 'pointer', fontSize: 11.5, color: 'var(--green)', borderColor: 'var(--green)' }}>
              ✅ Approve</button>
            <button disabled={busy === r.task.id} onClick={() => stampStep(r.task.id, 'reject')}
              title="reject — records it and routes the task back to M, without ticking the step"
              style={{ ...input, cursor: 'pointer', fontSize: 11.5, color: 'var(--red)' }}>
              ✋ Reject</button>
          </>
        )}
        {r.bucket === 'close' && (
          <button disabled={busy === r.task.id} onClick={() => closeTask(r.task.id)}
            title="Close — your retrospective tick on finished work. It blocks nothing."
            style={{ ...input, cursor: 'pointer', fontSize: 11.5, color: '#1f7a4d', borderColor: '#1f7a4d' }}>
            ✓ Close</button>
        )}
        <span title="hand it to a lane — assignee = whoever the task now waits on (board #171)">
          <RolePicker value={r.task.assignee} options={roleOptions.filter(Boolean)}
            allowEmpty pickTitle="assignee"
            onPick={(role) => patch(r.task.id, { assignee: role || null })} />
        </span>
      </div>
    </div>
  );

  const stuckBadge = stalls.count > 0;

  return (
    <AgentRegistryContext.Provider value={agentReg}>
    <div style={{ padding: '18px 20px 60px', maxWidth: 1180, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 4 }}>
        <h1 style={{ fontSize: 19, fontWeight: 700, margin: 0 }}>Your Dashboard</h1>
        <span style={{ ...dim, fontSize: 12 }}>
          what is waiting on <b>you</b> — and whether you can stop worrying about the rest
        </span>
        <span style={{ flex: 1 }} />
        <Link href="/admin/tasks" style={{ fontSize: 12, color: 'var(--accent)' }}>the full board →</Link>
      </div>
      <div style={{ ...dim, fontSize: 11.5, marginBottom: 10 }}>
        {loading ? 'loading…' : `fetched ${fetchedAt ? utcStamp(fetchedAt) : '—'} · refreshes every ${REFRESH_MS / 1000}s`}
        {' '}· this page is <b>pre-filtered by construction</b> and has no filter bar:{' '}
        <Link href="/admin/tasks" style={{ color: 'var(--accent)' }}>/admin/tasks</Link>{' '}
        is the shared, filterable surface.
        {err && <span style={{ color: 'var(--red)' }}> · {err}</span>}
      </div>

      {/* ── THE THIN STRIP — always visible, never scrolls away ──────────── */}
      <Card>
        <SignOfLife hb={hb} error={hbErr} />
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', alignItems: 'center', fontSize: 12 }}>
          <span title={loop.why}>
            <b style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6 }}>LOOP </b>
            <b style={{
              color: loop.state === 'up' ? 'var(--green)'
                : loop.state === 'down' ? 'var(--red)' : 'var(--text-tertiary)',
            }}>{loop.state === 'unknown' ? 'UNKNOWN' : loop.state.toUpperCase()}</b>
          </span>
          <span style={dim} title="the daily cap the loop would enforce right now, read from `system_settings` — not the build-time mirror">
            cap <b style={{ color: 'var(--text-secondary)' }}>{startedToday}/{capState.cap}</b> used today ({capState.source})
          </span>
          <span style={dim} title={`${liveRuns.length} of ${DISPATCHER.CONCURRENCY} configured slots in use`}>
            lanes <b style={{ color: lanesFree > 0 ? 'var(--green)' : 'var(--amber, #d98c00)' }}>
              {lanesFree} free</b> / {DISPATCHER.CONCURRENCY}
          </span>
          <span title={svcErr || 'GET /health on the commit this api is serving'}>
            <b style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6 }}>/HEALTH </b>
            {svcErr
              ? <b style={{ color: 'var(--red)' }}>NO ANSWER</b>
              : svc
                ? <b style={{ color: svc.status === 'ok' ? 'var(--green)' : 'var(--red)' }}>
                  {svc.status === 'ok' ? '200 ok' : svc.status}</b>
                : <span style={dim}>…</span>}
          </span>
          {/* The ONE thing allowed to interrupt him — and only when non-zero. */}
          {stuckBadge && (
            <button onClick={() => setTab('stuck')} style={{
              ...input, cursor: 'pointer', fontWeight: 700, color: 'var(--red)',
              borderColor: 'var(--red)', fontSize: 12,
            }} title="something is stalled — this badge appears only when it is non-zero">
              ⚠ {stalls.count} stalled
            </button>
          )}
        </div>
        <div style={{ ...dim, fontSize: 10.5, marginTop: 5 }}>{LOOP_VERSION_NOTE}</div>
      </Card>

      {/* ── TABS ─────────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 6, margin: '14px 0 12px', flexWrap: 'wrap' }}>
        {TABS.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            ...input, cursor: 'pointer', fontSize: 12.5,
            fontWeight: tab === t.key ? 700 : 400,
            background: tab === t.key ? 'var(--bg-input)' : 'transparent',
            borderColor: tab === t.key ? 'var(--blue)' : 'var(--border)',
            color: tab === t.key ? 'var(--blue)' : 'var(--text-primary)',
          }}>
            {t.label}
            {t.key === 'plate' && plate.total > 0 && (
              <span style={{ ...tagChip, marginLeft: 6, borderColor: '#c9a227', color: '#c9a227', fontWeight: 700 }}>
                {plate.total}</span>
            )}
            {t.key === 'stuck' && stalls.count > 0 && (
              <span style={{ ...tagChip, marginLeft: 6, borderColor: 'var(--red)', color: 'var(--red)', fontWeight: 700 }}>
                {stalls.count}</span>
            )}
          </button>
        ))}
      </div>

      {/* ── A · YOUR PLATE ───────────────────────────────────────────────── */}
      {tab === 'plate' && (
        <div>
          {plate.total === 0 ? (
            <Card>
              <div style={calm}>✓ {EMPTY_COPY.plate}</div>
              <div style={{ ...dim, fontSize: 12, marginTop: 5 }}>
                Every open task is parked on a lane that is not you. Nothing here is
                hidden by a filter — this page has none.
              </div>
            </Card>
          ) : (
            <div style={{ ...dim, fontSize: 11.5, marginBottom: 10 }}>
              {plate.total} item{plate.total === 1 ? '' : 's'} waiting on you · oldest{' '}
              <b style={{ color: ageColor(plate.oldestH) }}>{Math.round(plate.oldestH)}h</b>.
              Grouped by <b>the action required</b>, because “assigned to kevin” is not one
              thing. A task appears in exactly one group.
            </div>
          )}
          {plate.buckets.map((b) => (
            <Panel key={b.key} title={b.label} sub={b.why}
              right={
                <div style={{ textAlign: 'right' }}>
                  <div style={{
                    fontSize: 22, fontWeight: 800, lineHeight: 1,
                    color: b.count > 0 ? '#c9a227' : 'var(--green)',
                  }}>{b.count}</div>
                  {b.key === 'close' && b.count > 1 && (
                    <button disabled={busy === -1} onClick={() => closeAll(b.rows.map((r) => r.task.id))}
                      title="close every row shown here — these are already Done"
                      style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '1px 6px', marginTop: 4 }}>
                      ✓ close all shown</button>
                  )}
                </div>
              }>
              {b.count === 0
                ? <div style={calm}>✓ {b.empty}</div>
                : b.rows.map((r) => <PlateRowView key={r.task.id} r={r} />)}
              {b.count > b.rows.length && (
                <div style={{ ...dim, fontSize: 11, marginTop: 6 }}>
                  showing {b.rows.length} of {b.count} — the rest are counted, not painted (#240).
                </div>
              )}
            </Panel>
          ))}
          <div style={{ ...dim, fontSize: 11, lineHeight: 1.6 }}>
            <b>“Needs your LOOK” reads the chain’s SHIP STEP, not the status.</b> Shipped
            work still sits in <span style={mono}>Staged</span> on this board (#249), so a
            status-based rule would show you already-merged tasks as things awaiting your
            eyes. A chain with no release-lane step at all is unreadable rather than
            unshipped, and falls through to “waiting on you mid-chain”.
          </div>
        </div>
      )}

      {/* ── C · STUCK ────────────────────────────────────────────────────── */}
      {tab === 'stuck' && (
        <Panel title="Is anything stuck"
          sub={<>The negative-space module: its usual state is <b>empty</b>, and that is
            the point — it is what lets you not read everything else. <b>Stand By</b> is
            deliberately parked and is never reported here, nor is anything already on
            your plate.</>}
          right={<div style={{
            fontSize: 26, fontWeight: 800, lineHeight: 1,
            color: stalls.count > 0 ? 'var(--red)' : 'var(--green)',
          }}>{stalls.count}</div>}>
          {stalls.count === 0 ? (
            <div>
              <div style={calm}>✓ {EMPTY_COPY.stuck}</div>
              <div style={{ ...dim, fontSize: 12, marginTop: 5 }}>
                Checked just now: {stalls.checked.join(' · ')}
                {svc && svc.status === 'ok' ? ' · /health 200' : ''}.
              </div>
            </div>
          ) : (
            <div>
              {stalls.rows.map((s, i) => (
                <div key={`${s.kind}:${s.taskId ?? i}`} style={{
                  padding: '7px 0', borderTop: i ? '1px solid var(--border)' : 'none',
                }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                    <span style={{
                      ...tagChip, marginLeft: 0, fontWeight: 700,
                      borderColor: s.severity === 'red' ? 'var(--red)' : 'var(--amber, #d98c00)',
                      color: s.severity === 'red' ? 'var(--red)' : 'var(--amber, #d98c00)',
                    }}>{s.kind}</span>
                    {s.taskId != null
                      ? <TaskTitle id={s.taskId} title={s.title} />
                      : <b style={{ fontSize: 12.5 }}>{s.title}</b>}
                    {s.ageH > 0 && (
                      <span style={{ ...dim, fontSize: 11.5, color: ageColor(s.ageH) }}>
                        {Math.round(s.ageH)}h</span>
                    )}
                  </div>
                  <div style={{ ...dim, fontSize: 11.5, marginTop: 2 }}>{s.detail}</div>
                </div>
              ))}
              {stalls.count > stalls.rows.length && (
                <div style={{ ...dim, fontSize: 11, marginTop: 6 }}>
                  showing {stalls.rows.length} of {stalls.count} — the rest are counted, not painted (#240).
                </div>
              )}
            </div>
          )}
        </Panel>
      )}

      {/* ── B · TEAM ─────────────────────────────────────────────────────── */}
      {tab === 'team' && (
        <div>
          <Panel title="Sign of life — is your team awake"
            sub="Written BY each session; this page never writes a heartbeat on another session's behalf, so a stale pill is the truth.">
            <SignOfLife hb={hb} error={hbErr} />
          </Panel>
          <Panel title="Can anything even run — the machinery"
            sub={<>Heartbeats answer <b>“is my team awake”</b>; this answers <b>“is the
              machinery on”</b>. Both are needed and neither implies the other.</>}>
            <div style={{ fontSize: 12.5, lineHeight: 1.8 }}>
              <div>
                dispatcher loop:{' '}
                <b style={{
                  color: loop.state === 'up' ? 'var(--green)'
                    : loop.state === 'down' ? 'var(--red)' : 'var(--text-tertiary)',
                }}>{loop.state.toUpperCase()}</b>
                <span style={{ ...dim, marginLeft: 6 }}>— {loop.why}</span>
              </div>
              <div>
                cap: <b>{startedToday}</b> of <b>{capState.cap}</b> used today
                <span style={{ ...dim, marginLeft: 6 }}>(source: {capState.source})</span>
              </div>
              <div>
                lanes: <b>{liveRuns.length}</b> busy, <b>{lanesFree}</b> free of{' '}
                {DISPATCHER.CONCURRENCY} configured
              </div>
            </div>
            {liveRuns.length > 0 && (
              <div style={{ marginTop: 8 }}>
                {liveRuns.map((r) => {
                  const t = tasks.find((x) => x.id === r.task_id);
                  return (
                    <div key={r.id} style={{ fontSize: 12, padding: '3px 0' }}>
                      <RoleChip role={r.agent_letter} />{' '}
                      {t ? <TaskTitle id={t.id} title={t.title} /> : <span style={dim}>#{r.task_id}</span>}
                      <span style={{ ...dim, marginLeft: 6 }}>
                        since {utcStamp(r.started_at || r.requested_at)}</span>
                    </div>
                  );
                })}
              </div>
            )}
            {liveRuns.length === 0 && (
              <div style={{ ...calm, marginTop: 8 }}>✓ No lane is busy — nothing is mid-flight.</div>
            )}
            <div style={{ ...dim, fontSize: 10.5, marginTop: 8 }}>{LOOP_VERSION_NOTE}</div>
          </Panel>
        </div>
      )}

      {/* ── E · ACTIVITY ─────────────────────────────────────────────────── */}
      {tab === 'activity' && (
        <Panel title="Decisions and outcomes"
          sub={<>What shipped, what was decided, what failed — <b>not</b> every status
            tick. Status ticks, assignee changes, dispatch claims, step ticks and
            heartbeat chatter are excluded by rule, because an unfiltered feed is noise
            and noise is how a dashboard stops being read.</>}>
          {feed.count === 0 ? (
            <div>
              <div style={calm}>
                ✓ {EMPTY_COPY.activity} {comments.length
                  ? relTime(comments[0].created_at) : 'the last read'}.
              </div>
              <div style={{ ...dim, fontSize: 12, marginTop: 5 }}>
                Nothing has been decided, shipped or failed in the window this page reads.
              </div>
            </div>
          ) : feed.rows.map((e) => (
            <div key={e.key} style={{ padding: '5px 0', borderTop: '1px solid var(--border)', fontSize: 12.5 }}>
              <span style={{ ...dim, ...mono }}>{utcStamp(e.at)}</span>{' '}
              <RoleChip role={e.actor.replace('·auto', '')} size={15} />{' '}
              {e.taskId != null && <TaskTitle id={e.taskId} title="" />}{' '}
              <span>{e.headline}</span>
            </div>
          ))}
          <div style={{ ...dim, fontSize: 11, marginTop: 8 }}>
            {feed.count} decision/outcome event(s) · {feed.dropped} routine event(s) excluded
            by rule · {feed.unrecognised} matched no rule and are not painted here.
            Nothing is lost: <Link href="/admin/tasks" style={{ color: 'var(--accent)' }}>
            the board</Link> and <Link href="/admin/m-session" style={{ color: 'var(--accent)' }}>
            M’s activity log</Link> hold every event.
          </div>
        </Panel>
      )}

      {/* ── D · HEALTH ───────────────────────────────────────────────────── */}
      {tab === 'health' && (
        <div>
          <Panel title="Service — /health and the head this api is serving"
            sub="A train is not finished until /health returns 200 (#238). Deploy status is NOT health: on Wave 24 deploy-watch read 7/7 success while the api served 502 for twenty minutes.">
            {svcErr && (
              <div style={{ fontSize: 12.5, color: 'var(--red)' }}>
                ⚠️ the api did not answer: {svcErr}. That is itself the signal.
              </div>
            )}
            {svc && (
              <div style={{ fontSize: 12.5 }}>
                <b style={{ color: svc.status === 'ok' ? 'var(--green)' : 'var(--red)' }}>
                  {svc.status === 'ok' ? '200 · ok' : svc.status}
                </b> <span style={dim}>({svc.scope})</span>
                <div style={{ ...mono, marginTop: 4, color: 'var(--text-secondary)' }}>
                  commit {svc.commit || '(not reported)'} · branch {svc.branch || '—'}
                </div>
                <ul style={{ ...dim, fontSize: 11.5, marginTop: 8, paddingLeft: 18 }}>
                  {svc.known_gaps.map((g) => <li key={g}>{g}</li>)}
                </ul>
              </div>
            )}
          </Panel>
          <Panel title="Last deploys" sub="read out of the commit this api is serving.">
            {!dlog && <div style={dim}>loading…</div>}
            {dlog && !dlog.found && (
              <div style={{ fontSize: 12.5, color: 'var(--red)' }}>
                ⚠️ {dlog.error} — tried <span style={mono}>{dlog.resolved_path}</span>. A loud
                failure on purpose: an empty log would read as “no deploys have happened”.
              </div>
            )}
            {dlog && dlog.found && dlog.entries.slice(0, 5).map((e, i) => (
              <div key={`${e.sha}-${i}`} style={{
                padding: '4px 0', borderTop: i ? '1px solid var(--border)' : 'none', fontSize: 12,
              }}>
                <span style={{ ...mono, color: 'var(--accent)' }}>{e.sha}</span>{' '}
                <span style={dim}>{e.day} {e.time}</span>
                {e.wave != null && <b style={{ marginLeft: 6 }}>Wave {e.wave}</b>}
                <div style={{ ...dim, fontSize: 11.5 }}>{e.excerpt}</div>
              </div>
            ))}
          </Panel>
          <Panel title="Trading-system pulse"
            sub="You run a trading system; a dashboard that reports on the task board while the engine is down is telling you the wrong truth.">
            <div style={{ fontSize: 12.5 }}>
              This strip is <b>NOT WIRED YET</b>, and says so rather than rendering a
              reassuring blank. The engine-side reads (fleet fidelity, alert lag, worker
              liveness) belong to the E lane and are out of this page’s scope — a green
              tile here that measured nothing would be the #157 dead alarm in a new place.
              For now: <Link href="/admin/strategy-health" style={{ color: 'var(--accent)' }}>
              Strategy Health</Link> · <Link href="/admin/data-health" style={{ color: 'var(--accent)' }}>
              Data Health</Link>.
            </div>
          </Panel>
        </div>
      )}

      <div style={{ ...dim, fontSize: 11, marginTop: 20, lineHeight: 1.6 }}>
        <b>WHAT THIS PAGE CANNOT SEE.</b> The dispatcher publishes no heartbeat and no
        version, so <span style={mono}>LOOP</span> is inferred from run_history and
        <span style={mono}> unknown</span> means “nothing proves it either way” — never a
        green light. Merged-ness is read from #242’s git-gated ship-step tick, never
        computed here. The trading-system pulse is not wired. A dashboard whose blind
        spots are invisible manufactures false confidence, which is the #157 lesson this
        page is not repeating.
      </div>

      {/* ── THE MODAL, IN PLACE ──────────────────────────────────────────── */}
      {/* The EXISTING component, called with the same props /admin/tasks passes.
          One modal, two callers — not copied, not forked. Kevin: "if I click on
          a task, I would prefer it to pop up the modal without moving me from
          the dashboard… the same modal, something I can call in either place." */}
      {modalTask && (
        <TaskDetailModal key={`${modalTask.id}:${modal?.sel ?? ''}`}
          task={modalTask} allTasks={tasks} visionOptions={visionItems}
          initialSelected={modal?.sel}
          roles={roleOptions}
          patch={patch} del={del}
          onClose={() => setModal(null)}
          onOpenTask={(id, sel) => setModal({ id, sel })}
          commentAuthor={author} onPickAuthor={(a) => {
            setAuthor(a); localStorage.setItem(AUTHOR_LS_KEY, a);
          }}
          headless={headless} onRunRequested={load}
          onPollTick={load} />
      )}
    </div>
    </AgentRegistryContext.Provider>
  );
}
