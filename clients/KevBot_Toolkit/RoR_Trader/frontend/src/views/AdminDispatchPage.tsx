/**
 * Dispatch dashboard — /admin/dispatch (board #193 Half 1).
 * Contract: `docs/_active/Spec_Dispatch_Dashboard.md` (M·auto, 07-29).
 *
 * Kevin has been inferring the state of the agent fleet from chat. Everything
 * he asks for — what is running, which STEP it is on, when it was triggered,
 * how many lanes are in use, what happened earlier — is already in
 * `run_history` / `dev_tasks` / `agents`. This page is a READ over those three:
 * zero schema change, zero new endpoint, zero new capture.
 *
 * Four panels (spec §4-§7):
 *   1 LANES     — slots vs the configured CONCURRENCY, plus per-lane busy/free,
 *                 and (board #228) the DAILY CAP as an editable LIVE value
 *   2 IN FLIGHT — one card per live run, with STEP n of N
 *   3 QUEUE     — the two DELIBERATELY SEPARATE controls: the AI-eligible
 *                 TOGGLE (input) and the run-state PILL (output), with the Run
 *                 button still the one-off queue jump it always was (#198)
 *   4 SHIPPING  — built → pushed → PR'd → staged/done, oldest first, AGE as the
 *                 headline: branch age is what turns a clean merge into
 *                 Monday's conflict (board #166)
 *
 * HARD RAIL (spec §3): the app runs on Railway and cannot read the dispatcher's
 * `state.json`, the git worktrees, or GitHub. Anything not capturable —
 * dispatcher liveness, PR/merge state, retry counters — is printed ON the page
 * as a known gap (§8) rather than faked. A dashboard whose blind spots are
 * invisible manufactures false confidence; that is the #157 "0 healthy"
 * dead-alarm lesson, and this page is not repeating it.
 *
 * BOARD #228 adds the ONE write this page did not have: the daily cap. The loop
 * re-reads `system_settings.dispatcher_daily_cap` on every poll, so the cap is
 * changeable without a deploy — but only if there is somewhere to change it.
 * Kevin ruled that somewhere is here (07-30: "yea put it on dispatch"). Two
 * rails, both of which have a test that fails without them: the panel renders
 * the LIVE settings row and never the `DISPATCHER.DAILY_CAP` mirror (the
 * build-time copy that went stale in #219), and an out-of-range value is
 * REFUSED with its reason rather than written for the loop to silently clamp.
 * `effective_daily_cap()` stays the SSOT for both the clamp and the fallback;
 * `resolveDailyCap` below is a mirror of it and must never diverge.
 *
 * Production-bundle rails (CLAUDE.md): every hook before any early return, no
 * IIFE initialisation, and a variable is declared before the useMemo that
 * references it.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import {
  AUTHOR_LS_KEY, AiEligibleToggle, ChecklistStep, DISPATCHER, MODE_DEF,
  OutcomeChip, RoleChip, RunButton, RunRow, RunStatePill,
  STATUS_COLOR, Task, ageColor, ageShort, elapsedShort, hoursSince, input,
  isProcessChain, relTime, roleAbbrev, roleColor, runIneligibleReason, stepOwner,
  stepTitle, tagChip, utcDay, utcStamp,
} from './taskBoardShared';

/** Minimal shape of an /api/agents row — only what the lane panel needs. */
interface AgentRow { letter: string; name: string; status: string }

/** Auto-refresh cadence (spec §10). Three list GETs per tick, joined in
 *  memory — NEVER a per-row fetch (the board-#148 poll-efficiency lesson: a
 *  fan-out over 400 run rows would be worse than the problem it replaced). */
const REFRESH_MS = 15000;
/** Run-log depth: ~19 days at the current ~21 runs/day; the API caps at 500. */
const RUN_LIMIT = 400;
/** Grace on top of the 45m lease before a `running` row is called stale (§4). */
const STALE_GRACE_S = 120;
/**
 * The one endpoint the daily-cap control reads AND writes (board #228).
 *
 * Declared once so the read and the write can never drift onto different keys —
 * and built from the mirrored `DAILY_CAP_SETTING`, so the key the page touches
 * is the key `effective_daily_cap()` reads, by construction.
 */
const CAP_ENDPOINT = `/api/admin/system-settings/${DISPATCHER.DAILY_CAP_SETTING}`;

/* ── Pure derivations (module scope: testable, and out of the render body) ── */

/** Which chain step a task is on — the SAME first-un-done-step rule the
 *  dispatcher uses to build its prompt. If this page and the dispatcher ever
 *  disagree, the dispatcher is right by definition, so the computation is the
 *  shared helpers verbatim (isProcessChain / stepOwner / stepTitle) and not a
 *  re-implementation. */
export interface StepInfo {
  n: number; total: number; title: string; owner: string | null;
  mode: string; chain: boolean; complete: boolean;
}
export function currentStepInfo(task?: Task | null): StepInfo | null {
  const cl: ChecklistStep[] = task?.checklist || [];
  if (cl.length === 0) return null;
  const chain = isProcessChain(cl);
  const idx = cl.findIndex((s) => !s.done);
  if (idx < 0) {
    return { n: cl.length, total: cl.length, title: 'all steps complete',
      owner: null, mode: '', chain, complete: true };
  }
  const s = cl[idx];
  return { n: idx + 1, total: cl.length, title: stepTitle(s), owner: stepOwner(s),
    mode: s.mode || '', chain, complete: false };
}

/**
 * Does this `running` row still hold a lane? (spec §4)
 *
 * Occupies = `running` AND started within the lease + grace. A `running` row
 * older than that is NOT counted: the dispatcher would have reaped it and did
 * not, so counting it would render 3/3 saturation while nothing is running —
 * this page committing the #197 sin (a status that contradicts the truth) in a
 * brand new place. Those rows get a loud banner instead.
 */
export function isLeaseLive(r: RunRow): boolean {
  const t0 = new Date(r.started_at || r.requested_at).getTime();
  if (!Number.isFinite(t0)) return false;
  return (Date.now() - t0) / 1000 < DISPATCHER.RUN_TIMEOUT_S + STALE_GRACE_S;
}

/** Seconds left on a run's 45m lease (negative once it is past). */
export function leaseLeftS(r: RunRow): number {
  const t0 = new Date(r.started_at || r.requested_at).getTime();
  if (!Number.isFinite(t0)) return 0;
  return DISPATCHER.RUN_TIMEOUT_S - (Date.now() - t0) / 1000;
}

/**
 * Consecutive failed runs since the last `ok`, derived from `run_history`.
 * The dispatcher's REAL retry counter lives in its local `state.json`, which
 * the app cannot read — so this is an approximation and is labelled as one on
 * the page (§8.3). Rows arrive newest-first.
 */
export function consecutiveFailures(rows: RunRow[]): number {
  let n = 0;
  for (const r of rows) {
    if (r.outcome === 'ok') break;
    if (r.outcome === 'error' || r.outcome === 'lease-expired') n += 1;
    else if (r.outcome === 'running' || r.outcome === 'requested') continue;
    else break;
  }
  return n;
}

/* ── Daily cap: the LIVE value, not the mirror (board #228) ──────────────── */

/** The `/api/admin/system-settings/{key}` read shape — only what the cap needs. */
export interface CapSetting { key: string; value: unknown; db_value: unknown; source?: string }

/** What the running loop would compute for this row: the cap and WHY. */
export interface CapState { cap: number; source: string; fromSettings: boolean }

/**
 * Mirror of `effective_daily_cap()` in dispatcher.py — deliberately, line for
 * line (board #228).
 *
 * The loop re-reads `system_settings.dispatcher_daily_cap` on EVERY poll, so
 * the cap it enforces is that row, not the `DISPATCHER.DAILY_CAP` mirror above
 * (a build-time copy — the thing that went stale in #219). This page therefore
 * renders the row too, and resolves it with the SAME policy the loop uses, so
 * the number on the panel and the number in the breaker cannot disagree:
 *
 *   • row missing      -> the fallback constant, source "constant (no row)"
 *   • not an int       -> the fallback constant, source NAMES the bad value
 *   • out of range     -> CLAMPED to [1, DAILY_CAP_MAX], original reported
 *
 * The failure legs return the constant AND say so, because a cap that has
 * quietly reverted to 50 looks identical to one that is working — the #157
 * dead-alarm lesson applied to a circuit breaker.
 *
 * dispatcher.py is the SSOT for this policy; this is a mirror of it and must
 * never be the place it is changed.
 */
export function resolveDailyCap(dbValue: unknown): CapState {
  const fallback = DISPATCHER.DAILY_CAP;
  if (dbValue === null || dbValue === undefined) {
    return { cap: fallback, source: 'constant (no row)', fromSettings: false };
  }
  // Coercion follows Python's `int()` because that is what the loop calls: a
  // NUMBER truncates toward zero, a STRING must look like a whole number, and
  // anything else is refused. Not a detail — a value the loop honours and the
  // page refuses (or vice versa) is the exact disagreement this rail exists to
  // prevent, and the test cross-checks both against the same inputs.
  let val = NaN;
  if (typeof dbValue === 'number') val = Number.isFinite(dbValue) ? Math.trunc(dbValue) : NaN;
  else if (typeof dbValue === 'string' && /^[+-]?\d+$/.test(dbValue.trim())) val = parseInt(dbValue.trim(), 10);
  else val = NaN;
  if (!Number.isFinite(val)) {
    return {
      cap: fallback,
      source: `constant (row value ${JSON.stringify(dbValue)} is not an int)`,
      fromSettings: false,
    };
  }
  const { DAILY_CAP_MIN: lo, DAILY_CAP_MAX: hi } = DISPATCHER;
  if (val < lo || val > hi) {
    return {
      cap: Math.max(lo, Math.min(val, hi)),
      source: `settings CLAMPED from ${val} to [${lo},${hi}]`,
      fromSettings: true,
    };
  }
  return { cap: val, source: 'settings', fromSettings: true };
}

/**
 * Gate on the edit box, using the SAME bounds `effective_daily_cap()` clamps to.
 *
 * The loop would not honour an out-of-range value — it would clamp it and run a
 * different cap than the one typed — so the page refuses it outright and names
 * the reason, rather than writing a number the breaker will silently rewrite.
 * `0` and negatives are refused for the same reason they clamp to 1 in the
 * loop: no setting may disable the breaker.
 */
export function validateCapInput(raw: string): { ok: boolean; value?: number; reason?: string } {
  const s = (raw || '').trim();
  const { DAILY_CAP_MIN: lo, DAILY_CAP_MAX: hi } = DISPATCHER;
  if (!s) return { ok: false, reason: 'enter a number' };
  if (!/^-?\d+$/.test(s)) return { ok: false, reason: `"${s}" is not a whole number` };
  const n = parseInt(s, 10);
  if (n < lo) {
    return { ok: false, reason: `${n} is below the minimum of ${lo} — no setting may disable the breaker` };
  }
  if (n > hi) {
    return { ok: false, reason: `${n} is above the maximum of ${hi} (DAILY_CAP_MAX) — the loop would clamp it` };
  }
  return { ok: true, value: n };
}

/** One row of the shipping lane: work that is built but has not shipped. */
export interface ShipRow {
  task: Task;
  builtAt: string | null;
  builtRun: RunRow;
  pushedBranch: string | null;
  pushedAt: string | null;
  ageH: number;
}

/**
 * built → pushed → (PR'd) → staged/done, oldest first (spec §7).
 *
 * "Built" = the task's most recent run that finished `ok`. "Pushed" = the most
 * recent `pushed_branch` on ANY of its runs (#195). A task that is Done has
 * shipped and drops off the panel — this list is the answer to *what is waiting
 * to go out, and for how long*.
 */
export function shippingRows(tasks: Task[], runsByTask: Map<number, RunRow[]>): ShipRow[] {
  const out: ShipRow[] = [];
  tasks.forEach((t) => {
    if (t.status === 'Done') return;
    const rows = runsByTask.get(t.id) || [];
    const builtRun = rows.find((r) => r.outcome === 'ok');
    if (!builtRun) return;
    const pushed = rows.find((r) => !!r.pushed_branch);
    const builtAt = builtRun.finished_at || builtRun.requested_at;
    out.push({
      task: t, builtAt, builtRun,
      pushedBranch: pushed?.pushed_branch || null,
      pushedAt: pushed?.pushed_at || null,
      ageH: hoursSince(builtAt),
    });
  });
  // Oldest first — the top of this list is the next merge conflict.
  return out.sort((a, b) => (b.ageH) - (a.ageH));
}

/**
 * "What is reviewed and waiting to ship, and for how long?" — Kevin, 07-29:
 * *"I don't want it to get forgotten if the status is set to staged."*
 * Measured motivation: PRs #129/#130 sat mergeable ~4h purely because nobody
 * could see they were ready. Age is the number, so the count carries the age of
 * the OLDEST item, not an average.
 *
 * Waiting-since = the task's last completed run (when the work was produced),
 * falling back to `updated_at` (its last board edit). Which one was used is
 * named in the tooltip — an unlabelled proxy is how a dashboard starts lying.
 */
export interface WaitBucket { status: string; count: number; oldest: Task | null; since: string | null; fromRun: boolean }
export function waitBucket(status: string, tasks: Task[], runsByTask: Map<number, RunRow[]>): WaitBucket {
  const rows = tasks.filter((t) => t.status === status);
  let oldest: Task | null = null;
  let since: string | null = null;
  let fromRun = false;
  rows.forEach((t) => {
    const done = (runsByTask.get(t.id) || []).find((r) => !!r.finished_at);
    const at = done?.finished_at || t.updated_at || null;
    if (at && (!since || at < since)) { since = at; oldest = t; fromRun = !!done?.finished_at; }
  });
  return { status, count: rows.length, oldest, since, fromRun };
}

/* ── Small presentational atoms ──────────────────────────────────────────── */

const Panel = ({ title, sub, children }: {
  title: string; sub?: React.ReactNode; children: React.ReactNode;
}) => (
  <div style={{ marginBottom: 16 }}>
    <Card>
      <div style={{ marginBottom: 10 }}>
        <div style={{
          fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase',
          color: 'var(--text-tertiary)',
        }}>{title}</div>
        {sub && <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>{sub}</div>}
      </div>
      {children}
    </Card>
  </div>
);

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11.5 };
const dim: React.CSSProperties = { color: 'var(--text-tertiary)' };

/** A stage cell of the shipping pipeline: reached / unknown / not tracked. */
const Stage = ({ label, state, detail, title }: {
  label: string; state: 'done' | 'unknown' | 'untracked' | 'pending'; detail?: string; title?: string;
}) => {
  const color = state === 'done' ? 'var(--green)'
    : state === 'pending' ? 'var(--text-secondary)' : 'var(--text-tertiary)';
  const mark = state === 'done' ? '✓' : state === 'unknown' ? '?' : state === 'untracked' ? '—' : '·';
  return (
    <span title={title} style={{
      display: 'inline-flex', flexDirection: 'column', minWidth: 92,
      opacity: state === 'untracked' ? 0.45 : 1,
    }}>
      <span style={{ fontSize: 11, fontWeight: 700, color }}>{mark} {label}</span>
      <span style={{ ...dim, fontSize: 10.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 190 }}>
        {detail || ''}
      </span>
    </span>
  );
};

export default function AdminDispatchPage() {
  const [agents, setAgents] = useState<AgentRow[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  // Ticks the relative clocks (elapsed / lease left / age) between fetches so a
  // parked tab's numbers don't freeze mid-minute while the data is still fresh.
  const [tick, setTick] = useState(0);
  const [author, setAuthor] = useState('');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const [agentFilter, setAgentFilter] = useState('all');
  const [openTails, setOpenTails] = useState<Set<number>>(new Set());
  // Daily cap (#228) — the LIVE settings row, kept out of the three-GET poll
  // body on purpose: it is one constant-path read, not a fan-out over rows.
  const [capRow, setCapRow] = useState<CapSetting | null>(null);
  const [capRead, setCapRead] = useState<'pending' | 'ok' | 'error'>('pending');
  const [capDraft, setCapDraft] = useState('');
  const [capMsg, setCapMsg] = useState<{ kind: 'err' | 'ok'; text: string } | null>(null);
  const [capSaving, setCapSaving] = useState(false);

  const load = useCallback(async () => {
    try {
      setErr(null);
      // Spec §10: exactly three list GETs, joined client-side.
      // include_done=true is REQUIRED — the run log references Done tasks and
      // filtering them would render "#NNN (unknown)" all over the history.
      const [ag, ts, rh] = await Promise.all([
        apiFetch<AgentRow[]>('/api/agents').catch(() => [] as AgentRow[]),
        apiFetch<Task[]>('/api/dev-tasks?include_done=true'),
        apiFetch<RunRow[]>(`/api/run-history?limit=${RUN_LIMIT}`).catch(() => [] as RunRow[]),
      ]);
      setAgents(ag || []);
      setTasks(ts || []);
      setRuns(rh || []);
      setFetchedAt(new Date().toISOString());
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);

  /**
   * The live cap (#228) — a SEPARATE single read, not part of the §10 poll.
   *
   * It is deliberately its own call: the poll's rail is "three list GETs joined
   * in memory, never a per-row fetch" (#148), and folding a fourth list-shaped
   * read into it would blur that rail. This is one constant-path GET of one row.
   *
   * A failed read is NOT silently treated as "no row" — that is the failure the
   * whole task exists to kill — so `capRead` distinguishes unreadable from
   * absent, and the panel says which.
   */
  const loadCap = useCallback(async () => {
    try {
      const row = await apiFetch<CapSetting>(CAP_ENDPOINT);
      setCapRow(row || null);
      setCapRead('ok');
    } catch {
      setCapRow(null);
      setCapRead('error');
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { loadCap(); }, [loadCap]);
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) { load(); loadCap(); }   // paused while the tab is hidden (§10)
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [load, loadCap]);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    setAuthor(localStorage.getItem(AUTHOR_LS_KEY) || '');
  }, []);

  // The cap the loop would enforce right now, resolved with the loop's own
  // policy. `capRead === 'error'` is NOT folded into "no row": an unreadable
  // settings endpoint and an absent row are different failures and read
  // differently on the panel.
  const capState = useMemo<CapState>(() => {
    if (capRead === 'error') {
      return { cap: DISPATCHER.DAILY_CAP, source: 'constant (settings unreadable)', fromSettings: false };
    }
    return resolveDailyCap(capRow?.db_value);
  }, [capRow, capRead]);

  const taskById = useMemo(() => {
    const m = new Map<number, Task>();
    tasks.forEach((t) => m.set(t.id, t));
    return m;
  }, [tasks]);

  const runsByTask = useMemo(() => {
    const m = new Map<number, RunRow[]>();
    runs.forEach((r) => m.set(r.task_id, [...(m.get(r.task_id) || []), r]));
    return m;   // API order is newest-first and is preserved per bucket
  }, [runs]);

  const latestRunByTask = useMemo(() => {
    const m = new Map<number, RunRow>();
    runs.forEach((r) => { if (!m.has(r.task_id)) m.set(r.task_id, r); });
    return m;
  }, [runs]);

  // Consecutive failures per task since its last ok — approximated from
  // run_history (the real counter is in the dispatcher's state.json, §8.3).
  const failStreak = useMemo(() => {
    const m = new Map<number, number>();
    runsByTask.forEach((rows, id) => m.set(id, consecutiveFailures(rows)));
    return m;
  }, [runsByTask]);

  // Mirrors the dispatcher's enrollment rule (registry status='headless'), with
  // the same M stub fallback the board uses when the registry is empty.
  const headlessLetters = useMemo(() => {
    const l = agents.filter((a) => a.status === 'headless').map((a) => a.letter);
    return l.length ? l : ['M'];
  }, [agents]);
  const headless = useMemo(() => new Set(headlessLetters), [headlessLetters]);

  // `running` rows split by the lease rule (§4). `tick` is a dependency because
  // a run crosses from live to stale with the clock, not with a fetch.
  const runningRows = useMemo(() => runs.filter((r) => r.outcome === 'running'), [runs]);
  const liveRuns = useMemo(
    () => runningRows.filter(isLeaseLive), [runningRows, tick]);          // eslint-disable-line react-hooks/exhaustive-deps
  const staleRuns = useMemo(
    () => runningRows.filter((r) => !isLeaseLive(r)), [runningRows, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  const queuedRuns = useMemo(() => runs.filter((r) => r.outcome === 'requested'), [runs]);

  // "started today" from run_history, NOT from the dispatcher's own counter:
  // the two can differ by any run that failed to record, and the label says so.
  const startedToday = useMemo(() => {
    const today = new Date().toISOString().slice(0, 10);
    return runs.filter((r) => r.started_at && utcDay(r.started_at) === today).length;
  }, [runs]);

  // One row per headless lane — the dispatcher serialises per agent letter (all
  // of a letter's runs share one worktree), so a free global slot is NOT enough.
  const laneRows = useMemo(() => headlessLetters.map((letter) => {
    const busy = liveRuns.find((r) => r.agent_letter === letter) || null;
    const nextUp = tasks.find((t) => t.assignee === letter && t.status !== 'Done'
      && !runIneligibleReason(t, tasks, headless)) || null;
    const queued = queuedRuns.find((r) => r.agent_letter === letter) || null;
    return { letter, busy, nextUp, queued };
  }), [headlessLetters, liveRuns, queuedRuns, tasks, headless]);

  // Queue panel (§6): everything parked on a headless lane, split into
  // dispatch-eligible-now and waiting-with-the-reason.
  const queueRows = useMemo(() => {
    const rows = tasks.filter((t) => t.assignee && headless.has(t.assignee)
      && t.status !== 'Done'
      && !liveRuns.some((r) => r.task_id === t.id));
    const eligible = rows.filter((t) => !runIneligibleReason(t, tasks, headless));
    const waiting = rows.filter((t) => !!runIneligibleReason(t, tasks, headless));
    return { eligible, waiting };   // API order is priority order
  }, [tasks, headless, liveRuns]);

  // Does the API payload even carry #198's column? An absent key must render as
  // the labelled legacy gate, never as a confident "off" (§6, acceptance 6).
  const aiEligibleKnown = useMemo(
    () => tasks.some((t) => t.ai_eligible !== undefined), [tasks]);
  // Same question for #195's columns (§7 rail 1: unknown ≠ not pushed).
  const pushedKnown = useMemo(
    () => runs.some((r) => r.pushed_branch !== undefined), [runs]);

  const ship = useMemo(
    () => shippingRows(tasks, runsByTask), [tasks, runsByTask, tick]);     // eslint-disable-line react-hooks/exhaustive-deps
  const staged = useMemo(() => waitBucket('Staged', tasks, runsByTask), [tasks, runsByTask]);
  const inReview = useMemo(() => waitBucket('Review', tasks, runsByTask), [tasks, runsByTask]);

  const logAgents = useMemo(() => {
    const s = new Set<string>();
    runs.forEach((r) => s.add(r.agent_letter));
    return Array.from(s).sort();
  }, [runs]);

  const logRows = useMemo(() => runs.filter((r) => {
    if (agentFilter !== 'all' && r.agent_letter !== agentFilter) return false;
    if (outcomeFilter === 'all') return true;
    if (outcomeFilter === 'failed') return r.outcome === 'error' || r.outcome === 'lease-expired';
    return r.outcome === outcomeFilter;
  }), [runs, agentFilter, outcomeFilter]);

  const logDays = useMemo(() => {
    const days: { day: string; rows: RunRow[] }[] = [];
    logRows.forEach((r) => {
      const d = utcDay(r.finished_at || r.started_at || r.requested_at);
      const last = days[days.length - 1];
      if (last && last.day === d) last.rows.push(r);
      else days.push({ day: d, rows: [r] });
    });
    return days;
  }, [logRows]);

  const patchTask = useCallback(async (id: number, fields: Partial<Task>) => {
    try {
      await apiFetch(`/api/dev-tasks/${id}`, { method: 'PATCH', body: JSON.stringify(fields) });
    } catch (e) { setErr(String(e)); }
    load();
  }, [load]);

  const requestRun = useCallback(async (id: number) => {
    try {
      await apiFetch(`/api/dev-tasks/${id}/run-request`, {
        method: 'POST', body: JSON.stringify({ author: author || 'kevin' }),
      });
    } catch (e) { setErr(String(e)); }
    load();
  }, [author, load]);

  /**
   * Write the new cap (#228). The gate runs BEFORE the request: an out-of-range
   * value is refused here with its reason and never reaches the row, because
   * the loop would clamp it and then enforce a cap nobody chose.
   */
  const saveCap = useCallback(async () => {
    const v = validateCapInput(capDraft);
    if (!v.ok) { setCapMsg({ kind: 'err', text: `refused — ${v.reason}` }); return; }
    setCapSaving(true);
    try {
      await apiFetch(CAP_ENDPOINT, { method: 'PATCH', body: JSON.stringify({ value: v.value }) });
      setCapMsg({
        kind: 'ok',
        text: `saved — cap is ${v.value}. The dispatcher re-reads it on its NEXT poll `
          + `(~${DISPATCHER.POLL_S}s), so a run started in the next few seconds is still `
          + `charged against the old number. No deploy and no restart.`,
      });
      setCapDraft('');
      await loadCap();
    } catch (e) {
      setCapMsg({ kind: 'err', text: `save failed — ${String(e)}` });
    } finally { setCapSaving(false); }
  }, [capDraft, loadCap]);

  const toggleTail = (id: number) => setOpenTails((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const taskLabel = (id: number) => {
    const t = taskById.get(id);
    return t ? t.title : '(unknown task)';
  };

  const slots = Array.from({ length: DISPATCHER.CONCURRENCY }, (_, i) => liveRuns[i] || null);
  const starved = headlessLetters.length > DISPATCHER.CONCURRENCY;

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <h1 style={{ fontSize: 22, marginBottom: 4 }}>Dispatch</h1>
        <Link href="/admin/agents" style={{ fontSize: 12, color: 'var(--blue)' }}>→ agents roster</Link>
        <Link href="/admin/tasks" style={{ fontSize: 12, color: 'var(--blue)' }}>→ task board</Link>
        <span style={{ flex: 1 }} />
        <span style={{ ...dim, fontSize: 11.5 }}>
          as of {fetchedAt ? utcStamp(fetchedAt) : '—'}
          {' · auto-refresh 15s (paused when the tab is hidden)'}
        </span>
        <button style={{ ...input, cursor: 'pointer', fontSize: 12 }} onClick={load}>↻ refresh</button>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 14, maxWidth: 940 }}>
        The fleet&apos;s time axis — what is running right now and which STEP it is on, what is
        queued behind it, and what is built but not yet shipped. Every number is a read over
        <code style={{ ...mono, margin: '0 4px' }}>run_history</code>/
        <code style={{ ...mono, margin: '0 4px' }}>dev_tasks</code>; nothing here is
        inferred from git or GitHub (see KNOWN GAPS at the bottom). The dispatcher polls every
        ~{DISPATCHER.POLL_S}s, so a state change can be up to one poll behind what this page shows.
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}
      {loading && <Card>Loading…</Card>}

      {/* ── Stale-runs banner (§4) — loud, and EXCLUDED from the lane count ── */}
      {staleRuns.length > 0 && (
        <div style={{
          border: '1px solid var(--red)', borderRadius: 10, padding: '10px 14px',
          marginBottom: 14, background: 'rgba(200,60,60,0.08)',
        }}>
          <div style={{ color: 'var(--red)', fontWeight: 700, fontSize: 13 }}>
            ⚠ {staleRuns.length} run{staleRuns.length > 1 ? 's' : ''} stuck in <code style={mono}>running</code> past
            {' '}the {Math.round(DISPATCHER.RUN_TIMEOUT_S / 60)}m lease — the dispatcher may be down, or a reap was missed.
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>
            These are <b>not</b> counted against the lane cap below: showing saturation while nothing
            is running would be worse than showing nothing.
          </div>
          {staleRuns.map((r) => (
            <div key={r.id} style={{ fontSize: 12, marginTop: 4, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <RoleChip role={r.agent_letter} />
              <Link href={`/admin/tasks?task=${r.task_id}`} style={{ color: 'inherit' }}>
                #{r.task_id} {taskLabel(r.task_id)}
              </Link>
              <code style={mono}>{r.run_id || '—'}</code>
              <span style={{ color: 'var(--red)' }}>
                open {elapsedShort(r.started_at || r.requested_at)}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ── Panel 1 — Lanes ───────────────────────────────────────────────── */}
      <Panel
        title={`lanes — ${liveRuns.length}/${DISPATCHER.CONCURRENCY} in use`}
        sub={<>
          {DISPATCHER.CONCURRENCY} concurrent slots · {startedToday}/{capState.cap} runs
          {' '}started today — the cap&apos;s window is the <b>UTC day</b>, so it resets at
          {' '}<code style={mono}>00:00Z</code> (18:00 MT): an evening&apos;s runs are charged to the
          {' '}next MT morning (board #219). Counted from <code style={mono}>run_history</code>, not from the
          {' '}dispatcher&apos;s own <code style={mono}>state.json</code> counter — they can differ by any
          {' '}run that failed to record). <b>CONCURRENCY</b> is labelled <i>configured</i> because it is
          {' '}mirrored from <code style={mono}>tools/team_dispatcher/dispatcher.py</code>, which the app
          {' '}cannot read; the <b>cap is not</b> — it is read live from the settings row below (#228).
        </>}>
        {/* ── Daily-cap control (board #228) ───────────────────────────────
            The number here is the LIVE `system_settings.dispatcher_daily_cap`
            row, never the DISPATCHER.DAILY_CAP mirror: the loop re-reads that
            row every poll, so rendering the build-time copy beside an editable
            control would put two different caps on one panel — which is exactly
            how #219 shipped a claim that was never true. */}
        <div style={{
          border: '1px solid var(--border)', borderRadius: 10, padding: '10px 12px',
          marginBottom: 12, display: 'flex', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap',
        }}>
          <div style={{ minWidth: 190 }}>
            <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6 }}>DAILY CAP · live</div>
            <div style={{ fontSize: 20, fontWeight: 700, lineHeight: 1.2 }}>
              {capRead === 'pending' ? '…' : capState.cap}
              <span style={{ ...dim, fontSize: 12, fontWeight: 400 }}> runs / UTC day</span>
            </div>
            <div style={{
              fontSize: 11, marginTop: 2,
              color: capState.fromSettings ? 'var(--green)' : 'var(--amber, #d98c00)',
            }} title="The `source` the dispatcher prints on its own poll line — settings vs the fallback constant.">
              source: <code style={mono}>{capRead === 'pending' ? '…' : capState.source}</code>
            </div>
          </div>

          <div style={{ flex: '1 1 300px', minWidth: 280 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <input
                style={{ ...input, width: 92, fontSize: 13 }}
                inputMode="numeric"
                placeholder={String(capState.cap)}
                aria-label="new daily cap"
                value={capDraft}
                onChange={(e) => { setCapDraft(e.target.value); setCapMsg(null); }}
                onKeyDown={(e) => { if (e.key === 'Enter') saveCap(); }}
              />
              <button
                style={{ ...input, cursor: capSaving ? 'default' : 'pointer', fontSize: 12 }}
                disabled={capSaving}
                onClick={saveCap}
              >{capSaving ? 'saving…' : 'set cap'}</button>
              <span style={{ ...dim, fontSize: 11.5 }}>
                allowed <code style={mono}>[{DISPATCHER.DAILY_CAP_MIN}, {DISPATCHER.DAILY_CAP_MAX}]</code>
              </span>
            </div>
            {capMsg && (
              <div style={{
                fontSize: 12, marginTop: 5,
                color: capMsg.kind === 'err' ? 'var(--red)' : 'var(--green)',
              }}>{capMsg.kind === 'err' ? '⚠ ' : '✓ '}{capMsg.text}</div>
            )}
            <div style={{ ...dim, fontSize: 11.5, marginTop: 5 }}>
              Takes effect on the dispatcher&apos;s <b>next poll (~{DISPATCHER.POLL_S}s)</b> — no deploy, no
              restart, no loop bounce. A change that has not landed yet is one poll behind, not broken.
              {' '}Out of range is <b>refused here</b> with the reason: the loop would clamp it to
              {' '}<code style={mono}>[{DISPATCHER.DAILY_CAP_MIN}, {DISPATCHER.DAILY_CAP_MAX}]</code> and then
              {' '}enforce a cap nobody chose. Kevin&apos;s call to change: this is a circuit breaker, not a
              {' '}ration (board #219).
            </div>
            {!capState.fromSettings && capRead !== 'pending' && (
              <div style={{ fontSize: 12, marginTop: 5, color: 'var(--amber, #d98c00)' }}>
                ⚠ The loop is running on the <b>fallback constant</b> ({DISPATCHER.DAILY_CAP}), not on the
                settings row — <code style={mono}>{capState.source}</code>. Setting a value here writes the
                row and takes the loop off the fallback.
              </div>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
          {slots.map((r, i) => (
            <div key={i} style={{
              flex: '1 1 260px', minWidth: 240, borderRadius: 10, padding: '10px 12px',
              border: `1px solid ${r ? 'var(--blue)' : 'var(--border)'}`,
              background: r ? 'rgba(59,130,246,0.07)' : 'transparent',
            }}>
              <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6 }}>
                SLOT {i + 1} · {r ? 'occupied' : 'free'}
              </div>
              {r ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
                  <RoleChip role={r.agent_letter} />
                  <Link href={`/admin/tasks?task=${r.task_id}`} style={{ color: 'inherit', fontSize: 13, fontWeight: 600 }}>
                    #{r.task_id}
                  </Link>
                  <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1, minWidth: 0,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {taskLabel(r.task_id)}
                  </span>
                  <span style={{ fontSize: 12, color: 'var(--blue)', fontWeight: 700 }}>
                    {elapsedShort(r.started_at || r.requested_at)}
                  </span>
                </div>
              ) : (
                <div style={{ ...dim, fontSize: 13, marginTop: 6 }}>—</div>
              )}
            </div>
          ))}
        </div>

        <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6, marginBottom: 4 }}>
          PER-LANE (one run per agent letter — a letter&apos;s runs all share one worktree,
          so a free slot above is not sufficient)
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {laneRows.map((l) => (
            <div key={l.letter} style={{
              display: 'flex', alignItems: 'center', gap: 7, padding: '5px 10px',
              borderRadius: 8, border: '1px solid var(--border)', fontSize: 12,
              minWidth: 230, flex: '1 1 230px',
            }}>
              <RoleChip role={l.letter} />
              <span style={{
                fontWeight: 700, color: l.busy ? 'var(--blue)' : 'var(--green)',
              }}>{l.busy ? 'busy' : 'free'}</span>
              <span style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-secondary)' }}>
                {l.busy
                  ? <>#{l.busy.task_id} · {elapsedShort(l.busy.started_at || l.busy.requested_at)}</>
                  : l.queued
                    ? <span style={{ color: 'var(--amber, #d98c00)' }}>⏳ #{l.queued.task_id} queued</span>
                    : l.nextUp
                      ? <span title="next dispatch-eligible task on this lane">next: #{l.nextUp.id} {l.nextUp.title}</span>
                      : <span style={dim}>nothing eligible</span>}
              </span>
            </div>
          ))}
        </div>
        {starved && (
          <div style={{ fontSize: 12, color: 'var(--amber, #d98c00)', marginTop: 8 }}>
            {headlessLetters.length} headless lanes vs {DISPATCHER.CONCURRENCY} configured slots — when
            the fleet saturates, at least one lane always waits. That is the cap, not a fault.
          </div>
        )}
      </Panel>

      {/* ── Panel 2 — In flight ───────────────────────────────────────────── */}
      <Panel title={`in flight — ${liveRuns.length} active`}
        sub="One card per live run: the task, the STEP it is on, when it was triggered, and how much lease is left.">
        {liveRuns.length === 0 && (
          <div style={{ ...dim, fontSize: 13 }}>
            no active runs — the fleet is idle right now. (This is not a statement about the
            dispatcher being up; see KNOWN GAPS.)
          </div>
        )}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(390px, 1fr))', gap: 10 }}>
          {liveRuns.map((r) => {
            const t = taskById.get(r.task_id);
            const info = currentStepInfo(t);
            const leftS = leaseLeftS(r);
            return (
              <div key={r.id} style={{
                border: '1px solid var(--blue)', borderRadius: 10, padding: 12,
                background: 'var(--bg-card, var(--bg-input))', display: 'flex',
                flexDirection: 'column', gap: 6,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <RoleChip role={r.agent_letter} title={`${r.agent_letter}·auto`} />
                  <Link href={`/admin/tasks?task=${r.task_id}`}
                    style={{ color: 'inherit', fontWeight: 600, fontSize: 13.5, minWidth: 0,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    #{r.task_id} {taskLabel(r.task_id)}
                  </Link>
                </div>

                <div style={{ fontSize: 12.5 }}>
                  {!info ? <span style={dim}>no chain</span>
                    : !info.chain
                      ? <span style={dim}>no chain — whole-task dispatch</span>
                      : info.complete
                        ? <span style={{ color: 'var(--green)' }}>all {info.total} steps complete</span>
                        : (
                          <span>
                            <b style={{ color: 'var(--blue)' }}>STEP {info.n} of {info.total}</b>
                            {' — '}{info.title}
                            {info.owner && <> · <b style={{ color: roleColor(info.owner) }}>{roleAbbrev(info.owner)}</b></>}
                            {info.mode && (
                              <span style={{ ...tagChip }} title={MODE_DEF[info.mode] || info.mode}>{info.mode}</span>
                            )}
                          </span>
                        )}
                </div>

                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  triggered {utcStamp(r.requested_at)} ({relTime(r.requested_at)}) ·{' '}
                  {r.requested_by
                    ? <span title="the ▶ Run once button">Run button · by {r.requested_by}</span>
                    : <span title="picked up by the dispatcher's own queue scan">organic queue</span>}
                </div>
                <div style={{ fontSize: 12 }}>
                  running for <b>{elapsedShort(r.started_at || r.requested_at)}</b>
                  <span style={dim}> · {Math.round(DISPATCHER.RUN_TIMEOUT_S / 60)}m lease · </span>
                  <span style={{ color: leftS < 300 ? 'var(--amber, #d98c00)' : 'var(--text-secondary)' }}>
                    {Math.max(0, Math.round(leftS / 60))}m left
                  </span>
                </div>
                <div style={{ ...mono, ...dim }}
                  title={`the run's log file on Kevin's machine: tools/team_dispatcher/logs/${r.run_id}.log`}>
                  {r.run_id || '(no run_id)'}
                </div>
              </div>
            );
          })}
        </div>
      </Panel>

      {/* ── Panel 3 — Queue: toggle (input) vs pill (output) ───────────────── */}
      <Panel title={`queue — ${queueRows.eligible.length} eligible now · ${queueRows.waiting.length} waiting`}
        sub={<>
          Two controls, deliberately separate (board #198): <b>🤖 AI-eligible</b> is the
          {' '}<i>input</i> — standing permission for an agent to claim the task; the
          {' '}<b>run-state pill</b> is the <i>output</i> — what the dispatcher is actually doing;
          {' '}<b>▶ Run once</b> is neither, it is a one-off queue jump. The eligibility mirror
          {' '}below is a courtesy: the dispatcher re-gates at claim time.
          {!aiEligibleKnown && (
            <span style={{ color: 'var(--amber, #d98c00)' }}>
              {' '}⚠ this API build does not return <code style={mono}>ai_eligible</code> — the
              {' '}eligibility column falls back to the <b>legacy Todo gate</b>.
            </span>
          )}
        </>}>
        {[{ label: 'DISPATCH-ELIGIBLE NOW', rows: queueRows.eligible, empty: 'nothing is dispatch-eligible' },
          { label: 'WAITING — WITH THE REASON', rows: queueRows.waiting, empty: 'nothing waiting' }].map((grp) => (
          <div key={grp.label} style={{ marginBottom: 10 }}>
            <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6, marginBottom: 3 }}>{grp.label}</div>
            {grp.rows.length === 0 && <div style={{ ...dim, fontSize: 12.5 }}>{grp.empty}</div>}
            {grp.rows.map((t) => {
              const reason = runIneligibleReason(t, tasks, headless);
              return (
                <div key={t.id} style={{
                  display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
                  padding: '4px 0', borderBottom: '1px solid var(--border)', fontSize: 12.5,
                }}>
                  <RoleChip role={t.assignee || '?'} />
                  <Link href={`/admin/tasks?task=${t.id}`} style={{ color: 'inherit', fontWeight: 600 }}>#{t.id}</Link>
                  <span style={{ flex: '1 1 240px', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.title}
                  </span>
                  <span style={{ ...tagChip, borderColor: STATUS_COLOR[t.status], color: STATUS_COLOR[t.status] }}>
                    {t.status}
                  </span>
                  {aiEligibleKnown
                    ? <AiEligibleToggle task={t} compact onToggle={(v) => patchTask(t.id, { ai_eligible: v })} />
                    : <span style={{ ...tagChip }} title="this API build has no ai_eligible column — falling back to the pre-#198 gate, which was the Todo status alone">
                        legacy Todo gate: {t.status === 'Todo' ? 'on' : 'off'}
                      </span>}
                  <RunStatePill task={t} latestRun={latestRunByTask.get(t.id)} compact />
                  {(failStreak.get(t.id) || 0) > 1 && (
                    <span style={{ ...tagChip, borderColor: 'var(--red)', color: 'var(--red)', fontWeight: 700 }}
                      title={`${failStreak.get(t.id)} consecutive failed runs since this task's last ok. `
                        + 'Derived from run_history — the dispatcher’s real retry counter lives in its '
                        + 'local state.json, which the app cannot read, so the two can differ.'}>
                      ×{failStreak.get(t.id)} failed
                    </span>
                  )}
                  <RunButton task={t} allTasks={tasks} headless={headless}
                    latestRun={latestRunByTask.get(t.id)} onRequest={requestRun} compact />
                  {reason && <span style={{ ...dim, fontSize: 11.5 }}>{reason}</span>}
                </div>
              );
            })}
          </div>
        ))}
      </Panel>

      {/* ── Panel 4 — Shipping lane ───────────────────────────────────────── */}
      <Panel title="shipping lane — built → pushed → PR'd → staged/done"
        sub={<>
          Work that is built but has not shipped, <b>oldest first</b>. Age is the headline
          number because branch age is what turns a clean merge into a conflict (board #166
          measured 16-28h). <b>PR&apos;d is NOT tracked</b> — it needs a{' '}
          <code style={mono}>pr_number</code> field and a GitHub credential the API does not
          have (#193 Half 2), so it renders as a greyed dash rather than a false negative.
          {' '}<b>Staged/Done here is board state, not git truth.</b>
        </>}>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 12 }}>
          {[staged, inReview].map((b) => {
            const h = hoursSince(b.since);
            return (
              <div key={b.status} style={{
                flex: '1 1 260px', border: `1px solid ${b.count ? ageColor(h) : 'var(--border)'}`,
                borderRadius: 10, padding: '9px 12px',
              }}>
                <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6 }}>
                  {b.status.toUpperCase()} — {b.status === 'Staged'
                    ? 'reviewed, waiting for a release train'
                    : 'output done, waiting on sign-off'}
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginTop: 3 }}>
                  <span style={{ fontSize: 22, fontWeight: 700 }}>{b.count}</span>
                  {b.count > 0 && (
                    <span style={{ fontSize: 13, fontWeight: 700, color: ageColor(h) }}
                      title={`oldest: #${b.oldest?.id} — waiting since ${utcStamp(b.since)} `
                        + `(${b.fromRun ? 'its last completed run' : 'its last board edit — no completed run recorded'})`}>
                      oldest {ageShort(b.since) || '—'}
                    </span>
                  )}
                </div>
                {b.count > 0 && b.oldest && (
                  <Link href={`/admin/tasks?task=${b.oldest.id}`}
                    style={{ fontSize: 11.5, color: 'var(--text-secondary)', display: 'block',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    #{b.oldest.id} {b.oldest.title}
                  </Link>
                )}
              </div>
            );
          })}
        </div>

        {ship.length === 0 && <div style={{ ...dim, fontSize: 13 }}>nothing built and unshipped.</div>}
        {ship.map((s) => (
          <div key={s.task.id} style={{
            display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap',
            padding: '7px 0', borderBottom: '1px solid var(--border)',
          }}>
            <span style={{
              fontSize: 15, fontWeight: 700, minWidth: 62, color: ageColor(s.ageH),
            }} title={`built ${utcStamp(s.builtAt)} — age is the merge-conflict predictor`}>
              {ageShort(s.builtAt) || '—'}
            </span>
            <RoleChip role={s.builtRun.agent_letter} />
            <Link href={`/admin/tasks?task=${s.task.id}`}
              style={{ color: 'inherit', fontSize: 13, fontWeight: 600, flex: '1 1 230px', minWidth: 0,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              #{s.task.id} {s.task.title}
            </Link>
            <Stage label="built" state="done" detail={utcStamp(s.builtAt)}
              title={`the task's most recent run that finished ok (run ${s.builtRun.run_id || '—'})`} />
            <Stage
              label="pushed"
              state={s.pushedBranch ? 'done' : 'unknown'}
              detail={s.pushedBranch || (pushedKnown ? 'unknown' : 'unknown (pre-#195)')}
              title={s.pushedBranch
                ? `pushed ${utcStamp(s.pushedAt)} — branch ${s.pushedBranch}`
                : 'UNKNOWN, not "not pushed": run_history only records pushed_branch for runs after '
                  + 'board #195, and this task has none. Check the branch by hand before assuming.'} />
            <Stage label="PR'd" state="untracked" detail="not tracked"
              title="Not tracked. Needs a structured pr_number on dev_tasks AND a GitHub token the API does not have — board #193 Half 2, a credential decision rather than a UI one. Never scraped from comment prose: a scraped number goes stale silently." />
            <Stage
              label={s.task.status === 'Staged' ? 'staged' : 'shipped'}
              state={s.task.status === 'Staged' ? 'pending' : 'untracked'}
              detail={s.task.status}
              title="board state, used as a proxy for the merge: Staged = reviewed and awaiting a release train; Done = shipped. Not read from git." />
          </div>
        ))}
      </Panel>

      {/* ── Run log ───────────────────────────────────────────────────────── */}
      <Panel title={`run log — ${logRows.length} of ${runs.length} runs`}
        sub={<>
          Newest first, grouped by UTC day. A <b>failed</b> run is tinted and counted in its day
          header without filtering — after board #197 lands, a failed run restores the task&apos;s
          pre-dispatch status, which makes this log <b>the only place the failure is visible at
          all</b>. Click a row to read its <code style={mono}>log_tail</code> (the last 4000
          chars; full logs live only on Kevin&apos;s machine).
        </>}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <select style={{ ...input, fontSize: 12 }} value={outcomeFilter}
            onChange={(e) => setOutcomeFilter(e.target.value)}>
            <option value="all">all outcomes</option>
            <option value="failed">failed (error + lease-expired)</option>
            <option value="ok">ok</option>
            <option value="running">running</option>
            <option value="requested">requested</option>
            <option value="ignored">ignored</option>
          </select>
          <select style={{ ...input, fontSize: 12 }} value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value)}>
            <option value="all">all agents</option>
            {logAgents.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <span style={{ ...dim, fontSize: 11.5 }}>
            last {RUN_LIMIT} rows (API cap 500)
          </span>
        </div>

        {logDays.map((d) => {
          const fails = d.rows.filter((r) => r.outcome === 'error' || r.outcome === 'lease-expired').length;
          const oks = d.rows.filter((r) => r.outcome === 'ok').length;
          return (
            <div key={d.day} style={{ marginBottom: 10 }}>
              <div style={{
                display: 'flex', gap: 8, alignItems: 'baseline', padding: '4px 0',
                borderBottom: '1px solid var(--border)', marginBottom: 2,
              }}>
                <span style={{ fontSize: 12, fontWeight: 700 }}>{d.day || 'unknown day'}</span>
                <span style={{ ...dim, fontSize: 11.5 }}>{d.rows.length} runs</span>
                <span style={{ fontSize: 11.5, color: 'var(--green)' }}>{oks} ok</span>
                {fails > 0 && <span style={{ fontSize: 11.5, color: 'var(--red)', fontWeight: 700 }}>{fails} failed</span>}
              </div>
              {d.rows.map((r) => {
                const failed = r.outcome === 'error' || r.outcome === 'lease-expired';
                const open = openTails.has(r.id);
                const dur = r.started_at && r.finished_at
                  ? Math.round((new Date(r.finished_at).getTime() - new Date(r.started_at).getTime()) / 60000)
                  : null;
                return (
                  <div key={r.id} style={{
                    padding: '3px 6px', borderRadius: 6,
                    background: failed ? 'rgba(200,60,60,0.10)' : 'transparent',
                  }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', fontSize: 12 }}>
                      <OutcomeChip outcome={r.outcome} />
                      <RoleChip role={r.agent_letter} />
                      <Link href={`/admin/tasks?task=${r.task_id}`} style={{ color: 'inherit', fontWeight: 600 }}>
                        #{r.task_id}
                      </Link>
                      <span style={{ flex: '1 1 220px', minWidth: 0, color: 'var(--text-secondary)',
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {taskLabel(r.task_id)}
                      </span>
                      <span style={{ ...mono, ...dim }} title="run_id — the log filename on Kevin's machine">
                        {r.run_id || '—'}
                      </span>
                      <span style={{ ...dim, fontSize: 11.5 }}
                        title={`requested ${utcStamp(r.requested_at)}`
                          + (r.started_at ? ` · started ${utcStamp(r.started_at)}` : ' · never started')
                          + (r.finished_at ? ` · finished ${utcStamp(r.finished_at)}` : '')}>
                        {utcStamp(r.finished_at || r.started_at || r.requested_at, false)}
                        {dur !== null ? ` · ${dur}m` : ''}
                        {r.requested_by ? ` · by ${r.requested_by}` : ' · queue'}
                      </span>
                      {r.pushed_branch && (
                        <span style={{ ...tagChip, borderColor: 'var(--green)', color: 'var(--green)' }}
                          title={`pushed ${utcStamp(r.pushed_at)}`}>⇧ {r.pushed_branch}</span>
                      )}
                      {r.log_tail && (
                        <button style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '0px 6px' }}
                          onClick={() => toggleTail(r.id)}>{open ? '▾ tail' : '▸ tail'}</button>
                      )}
                    </div>
                    {open && r.log_tail && (
                      <pre style={{
                        ...mono, whiteSpace: 'pre-wrap', margin: '4px 0 6px', padding: 8,
                        borderRadius: 6, background: 'var(--bg-input)', maxHeight: 320,
                        overflow: 'auto', color: 'var(--text-secondary)',
                      }}>{r.log_tail}</pre>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
        {logDays.length === 0 && <div style={{ ...dim, fontSize: 13 }}>no runs match this filter.</div>}
      </Panel>

      {/* ── Known gaps (§8) — printed, not hidden ─────────────────────────── */}
      <Card>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, color: 'var(--text-tertiary)', marginBottom: 6 }}>
          KNOWN GAPS — what this page CANNOT see
        </div>
        <ol style={{ fontSize: 12.5, color: 'var(--text-secondary)', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <li>
            <b>Dispatcher liveness is not observable here.</b> There is no heartbeat row, so the
            newest run event is the most this page can honestly report — <i>last run event</i>,
            never <i>dispatcher alive</i>. A quiet queue emits nothing, so silence proves nothing.
            Last run event: <b>{runs[0] ? relTime(runs[0].finished_at || runs[0].started_at || runs[0].requested_at) : '—'}</b>.
          </li>
          <li>
            <b>PR / merge state is not tracked.</b> The API has no GitHub token and{' '}
            <code style={mono}>dev_tasks</code> has no <code style={mono}>pr_number</code>; the
            shipping lane substitutes board state (Staged/Done) for the merge and labels it.
          </li>
          <li>
            <b>Retry counters live in the dispatcher&apos;s local <code style={mono}>state.json</code>.</b>{' '}
            Any consecutive-failure count shown here is derived from{' '}
            <code style={mono}>run_history</code> and can differ from what the dispatcher believes.
          </li>
        </ol>
      </Card>
    </div>
  );
}
