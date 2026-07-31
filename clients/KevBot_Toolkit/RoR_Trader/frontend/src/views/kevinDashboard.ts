/**
 * Kevin's dashboard — the pure derivations behind /admin/kevin (board #257).
 *
 * THE DESIGN PRINCIPLE, and everything here follows from it:
 *
 *   M's dashboard answers *"what is happening?"*.
 *   Kevin's answers *"what is waiting on ME — and can I stop worrying about
 *   the rest?"*
 *
 * Those are different products. The failure mode is easy to walk into: an
 * owner dashboard that shows EVERYTHING is a second manager dashboard, and it
 * hands the triage back to the person it was built to spare. So the rules below
 * are subtractive by design — a count Kevin cannot act on does not belong on
 * the page, and `/admin/tasks` remains the shared, filterable surface this one
 * deliberately is not.
 *
 * WHY THEY LIVE HERE AND NOT IN THE PAGE. `src/test_kevin_dashboard_257.py`
 * LIFTS AND EXECUTES these functions in node against fixtures. A static
 * substring assertion would pass against a module that returns the wrong rows,
 * and the bucket rules are the part that has to be right. Keep the bodies free
 * of TS-only syntax (no `as const`, no annotated lambda params) so the lift
 * stays honest — what runs in the test is what ships.
 *
 * NOTHING IS RE-IMPLEMENTED HERE THAT ALREADY EXISTS:
 *   • "has this task's code merged?"  → `mergeStateOf` / `SHIP_STEP_OWNERS`
 *     (AdminDispatchPage, board #245) — imported, never restated.
 *   • "is a car waiting for a train?" → `carsWaitingForATrain`
 *     (rSessionSignals, board #251) — the stall list CONSUMES its output.
 *   • "is a run still holding its lease?" → `isLeaseLive` (board #193).
 *   • "is a session alive?" → `/api/r-session/heartbeats` + `SignOfLife` (#251).
 * A second definition of any of those is a second answer to one question, and
 * the two would drift silently.
 */
import {
  ChecklistStep, RunRow, Task, hoursSince, isFinished, stepOwner, stepTitle,
} from './taskBoardShared';
import { SHIP_STEP_OWNERS, isLeaseLive } from './AdminDispatchPage';
import { CarsPanel, HeartbeatRow } from './rSessionSignals';

/** The owner. A literal `'kevin'` scattered through the file is how a rename
 *  quietly half-lands. */
export const KEVIN = 'kevin';

/** DOM bounds (board #240 — the M-session activity log painted ~435 rows and
 *  stalled Kevin's browser). Render less at a time, never KEEP less: every
 *  panel below reports the FULL count next to a sliced row list. */
export const PLATE_ROWS = 12;
export const STALL_ROWS = 20;
export const FEED_ROWS = 40;

/* ══ A · YOUR PLATE ═══════════════════════════════════════════════════════ */

/**
 * LEGITIMATELY QUIET — never on the plate, and never a stall.
 *
 * `Stand By` is the load-bearing one: board #232 created it so Kevin could park
 * something deliberately, and its own definition says *"no action needed from
 * anyone"*. Measured against the live board on 07-31, FIVE of twenty plate
 * items were `Stand By` — four of them under "Needs your STAMP", which reads as
 * *work cannot start*. That is a count he cannot act on, sitting in the one
 * module that must be perfect, and it is exactly the "second manager dashboard"
 * failure this page is built against. `Backlog` is captured-not-next. Both are
 * decisions, not queues.
 */
export const CALM_STATUSES = ['Stand By', 'Backlog'];

/**
 * The four buckets, in EVALUATION ORDER. First match wins, and a task appears
 * in EXACTLY ONE.
 *
 * Without an explicit order the same task lands in two buckets and the page
 * starts double-counting — which is the class of bug that started this whole
 * thread (a count that lied). The order is M's, board #257 step 2, and it is
 * urgency order: a pending stamp means work CANNOT START, which is the
 * highest-latency state in the system.
 */
export const PLATE_BUCKETS = [
  {
    key: 'stamp',
    label: 'Needs your STAMP',
    why: 'a chain step owned by you gates on an approval that is still pending — '
      + 'work cannot start until you stamp it. The highest-latency item in the system.',
    empty: 'Nothing is waiting on your stamp.',
  },
  {
    key: 'close',
    label: 'Needs your CLOSE',
    why: 'M marked it Done and handed it to you. Closing is your retrospective '
      + 'look at finished work — near-zero thought, and it blocks nothing.',
    empty: 'Nothing is waiting to be closed.',
  },
  {
    key: 'look',
    label: 'Needs your LOOK',
    why: 'the work has SHIPPED (its release-lane step is ticked) and the chain is '
      + 'now parked on a step you own.',
    empty: 'Nothing shipped is waiting on your look.',
  },
  {
    key: 'midchain',
    label: 'Waiting on you mid-chain',
    why: 'the chain is parked on a step you own while the task still reads as if it '
      + 'were in flight. Today’s invisible case — nothing about the status says it is you.',
    empty: 'No chain is parked on you.',
  },
];

/** Every module renders CALM AND AFFIRMATIVE when empty, never blank and never
 *  a bare `0`. R's CARS-WAITING panel read as broken on 07-30 until it was
 *  explained; that is a design defect, not a misunderstanding. */
export const EMPTY_COPY = {
  plate: 'Nothing is waiting on you.',
  stuck: 'Nothing is stalled.',
  activity: 'Quiet since',
};

/** The chain step the task is parked on: the first one not done. Null for a
 *  chainless task or a fully-ticked chain. Same first-un-done-step rule the
 *  dispatcher and /admin/dispatch use — not a re-implementation. */
export function firstOpenStep(task: Task): ChecklistStep | null {
  const cl = (task && Array.isArray(task.checklist)) ? task.checklist : [];
  for (let i = 0; i < cl.length; i += 1) {
    if (cl[i] && !cl[i].done) return cl[i];
  }
  return null;
}

/** Is the first open step owned by Kevin? */
export function parkedOnKevin(task: Task): boolean {
  const s = firstOpenStep(task);
  if (!s) return false;
  return String(stepOwner(s) || '').toLowerCase() === KEVIN;
}

/** Does the first open step gate on a stamp that is still PENDING? A step whose
 *  stamp is already approved is not waiting on Kevin — its owner is. */
export function pendingStamp(task: Task): boolean {
  const s = firstOpenStep(task);
  if (!s) return false;
  const st = s.stamp || null;
  if (!st || !st.required) return false;
  const state = st.state || 'pending';
  return state === 'pending';
}

/**
 * HAS THIS TASK SHIPPED? — read from the chain's SHIP STEP, never from `status`.
 *
 * This is the #249 rail and it is not optional. Shipped work still sits in
 * `Staged` on this board, so a status-based "needs your look" would have shown
 * Kevin six already-merged tasks as things awaiting his eyes — the exact bug he
 * found by looking at a count that lied.
 *
 * The ship step is the honest signal because board #242 gates its tick on
 * `git merge-base --is-ancestor <branch> origin/dev` after a real fetch. A
 * ticked release-lane step is therefore a git-verified merge.
 *
 * THREE ANSWERS, NEVER TWO. A chain with no release-lane step at all is
 * `'none'` — UNREADABLE, not "unshipped" — and such a task falls through to the
 * mid-chain bucket rather than being guessed at in either direction.
 */
export function shipStepState(task: Task): string {
  const cl = (task && Array.isArray(task.checklist)) ? task.checklist : [];
  const ship = cl.filter(
    (s) => SHIP_STEP_OWNERS.indexOf(String(stepOwner(s) || '').toUpperCase()) >= 0);
  if (ship.length === 0) return 'none';
  return ship.every((s) => !!s.done) ? 'done' : 'open';
}

/**
 * Which bucket a task belongs to, or null when it is not on Kevin's plate.
 *
 * FIRST MATCH WINS, in `PLATE_BUCKETS` order. `Closed` never appears: it is the
 * status Kevin sets when he is finished with a task, so showing it back to him
 * is the page asking for the same action twice.
 */
export function plateBucketOf(task: Task): string | null {
  if (!task) return null;
  const status = String(task.status || '');
  // `Closed` is the state THIS PAGE'S actions produce — it never comes back, in
  // any bucket, or the page asks him for the same tick twice.
  if (status === 'Closed') return null;
  // Deliberately parked is not waiting. See CALM_STATUSES.
  if (CALM_STATUSES.indexOf(status) >= 0) return null;
  // 1 — a pending stamp on a Kevin step. Tested before everything, `Done`
  // included: if work cannot START, that outranks work that is finished. A
  // `Done` task with an un-stamped Kevin step is a contradiction, and surfacing
  // it under STAMP is how he finds out.
  if (parkedOnKevin(task) && pendingStamp(task)) return 'stamp';
  // 2 — M marked it Done and handed it over. Exactly `Done`; `Closed` is the
  // state this action PRODUCES.
  if (status === 'Done' && String(task.assignee || '').toLowerCase() === KEVIN) return 'close';
  if (isFinished(status)) return null;
  if (!parkedOnKevin(task)) return null;
  // 3 — shipped, and now parked on him. Derived from the SHIP STEP (#249).
  if (shipStepState(task) === 'done') return 'look';
  // 4 — everything else parked on him: mid-chain, and invisible today.
  return 'midchain';
}

/**
 * HOW LONG HAS IT BEEN HIS MOVE — not how old the task is.
 *
 * The question this page answers is *"how long have I been the holdup"*, and
 * #246 (four hours parked, nothing broken, nobody watching) is why it is the
 * headline number on every row. Measured from the most recently COMPLETED step
 * — the moment the chain handed him the ball — falling back to `updated_at`.
 *
 * `from` names which stamp answered, because an unlabelled proxy is how a
 * dashboard starts lying.
 */
export function landedOnKevin(task: Task): { at: string | null; from: string } {
  const cl = (task && Array.isArray(task.checklist)) ? task.checklist : [];
  let best: string | null = null;
  cl.forEach((s) => {
    if (s && s.done && s.completed_at && (!best || s.completed_at > best)) best = s.completed_at;
  });
  if (best) return { at: best, from: 'the step that handed it to you' };
  return { at: (task && task.updated_at) || null, from: 'the task’s last board edit' };
}

export interface PlateRow {
  task: Task;
  bucket: string;
  ageH: number;
  ageFrom: string;
  /** Who moves NEXT once Kevin acts — "→ F". The point of the row: this is what
   *  his stamp/look/close releases. */
  unblocks: string | null;
  /** How many OPEN tasks name this one in `blocked_by`. #246 rendered as a
   *  number: "1 task cannot start". */
  blocking: number;
  /** The step title he is parked on, for the row's second line. */
  stepTitle: string;
}

export interface PlateBucket {
  key: string; label: string; why: string; empty: string;
  rows: PlateRow[];
  /** ALWAYS the full count, even when `rows` is sliced (#240). */
  count: number;
}

export interface Plate {
  buckets: PlateBucket[];
  total: number;
  /** Age of the oldest thing on his plate, in hours. */
  oldestH: number;
}

/** Who acts after Kevin's step — the owner of the next step in the chain. */
export function unblocksWho(task: Task): string | null {
  const cl = (task && Array.isArray(task.checklist)) ? task.checklist : [];
  let seen = false;
  for (let i = 0; i < cl.length; i += 1) {
    if (!cl[i] || cl[i].done) continue;
    if (!seen) { seen = true; continue; }
    return stepOwner(cl[i]) || null;
  }
  return null;
}

/**
 * THE PLATE — grouped by the ACTION REQUIRED, because "assigned to kevin" is
 * not one thing.
 *
 * Rows are OLDEST FIRST inside each bucket: the top of a list is the thing he
 * has been the holdup on longest, which is the number the page exists for.
 */
export function buildPlate(tasks: Task[], limit: number): Plate {
  const all = tasks || [];
  const openIds = all.filter((t) => !isFinished(t.status));
  const buckets = PLATE_BUCKETS.map((b) => {
    const rows = all.filter((t) => plateBucketOf(t) === b.key).map((t) => {
      const landed = landedOnKevin(t);
      const step = firstOpenStep(t);
      return {
        task: t,
        bucket: b.key,
        ageH: hoursSince(landed.at),
        ageFrom: landed.from,
        unblocks: unblocksWho(t),
        blocking: openIds.filter(
          (o) => o.id !== t.id && (o.blocked_by || []).indexOf(t.id) >= 0).length,
        stepTitle: step ? stepTitle(step) : '',
      };
    });
    rows.sort((a, b2) => b2.ageH - a.ageH);
    return {
      key: b.key, label: b.label, why: b.why, empty: b.empty,
      rows: rows.slice(0, limit), count: rows.length,
    };
  });
  let oldest = 0;
  buckets.forEach((b) => b.rows.forEach((r) => { if (r.ageH > oldest) oldest = r.ageH; }));
  return {
    buckets,
    total: buckets.reduce((n, b) => n + b.count, 0),
    oldestH: oldest,
  };
}

/* ══ B · CAN ANYTHING EVEN RUN — the dispatcher loop ══════════════════════ */

/** A run request the loop has not claimed within this many seconds is evidence
 *  the loop is not polling. The loop runs `--poll 20`, so five minutes is
 *  fifteen missed polls — generous on purpose, because a false DOWN on this
 *  strip is the alarm that stops being believed. */
export const LOOP_CLAIM_GRACE_S = 300;
/** A run claimed within this window is positive evidence the loop is alive. */
export const LOOP_UP_WINDOW_S = 30 * 60;

/** The dispatcher writes no heartbeat and reports no version anywhere the app
 *  can read (checked against `tools/team_dispatcher/dispatcher.py`, V4.29 — the
 *  version lives only in its docstring). So the strip INFERS liveness from
 *  run_history and says so, rather than printing a version it cannot verify. */
export const LOOP_VERSION_NOTE =
  'the dispatcher publishes no heartbeat and no version the app can read — this '
  + 'state is INFERRED from run_history, and `unknown` means exactly that.';

export interface LoopReading { state: string; why: string; lastClaimAt: string | null }

/**
 * IS THE MACHINERY ON? — three answers, and the third is the one that matters.
 *
 * `up`      — a run was claimed recently. Positive evidence.
 * `down`    — there was work the loop should have taken and it did not.
 * `unknown` — nothing has been claimable, so nothing proves it either way.
 *
 * `unknown` is NOT a green light and must never render as one. The #157 dead
 * alarm was an absent signal rendered as a confident measurement; a quiet
 * board and a dead loop look identical from run_history alone, and saying so is
 * the only honest option available to a page with no heartbeat to read.
 */
export function loopState(runs: RunRow[], starvedCount: number, nowMs: number): LoopReading {
  const rows = runs || [];
  let lastClaim: string | null = null;
  rows.forEach((r) => {
    const at = r.started_at || null;
    if (at && (!lastClaim || at > lastClaim)) lastClaim = at;
  });
  if (lastClaim) {
    const ageS = (nowMs - Date.parse(lastClaim)) / 1000;
    if (Number.isFinite(ageS) && ageS >= 0 && ageS < LOOP_UP_WINDOW_S) {
      return {
        state: 'up',
        why: `a run was claimed ${Math.round(ageS / 60)}m ago — the loop is polling`,
        lastClaimAt: lastClaim,
      };
    }
  }
  const unclaimed = rows.filter((r) => {
    if (r.outcome !== 'requested') return false;
    const ageS = (nowMs - Date.parse(r.requested_at)) / 1000;
    return Number.isFinite(ageS) && ageS > LOOP_CLAIM_GRACE_S;
  });
  if (unclaimed.length > 0) {
    return {
      state: 'down',
      why: `${unclaimed.length} run request(s) unclaimed for over `
        + `${Math.round(LOOP_CLAIM_GRACE_S / 60)}m — the loop polls every 20s, so it is not running`,
      lastClaimAt: lastClaim,
    };
  }
  if (starvedCount > 0) {
    return {
      state: 'down',
      why: `${starvedCount} armed, dispatchable task(s) idle past the starvation `
        + 'threshold with a free lane — nothing is picking work up',
      lastClaimAt: lastClaim,
    };
  }
  return {
    state: 'unknown',
    why: 'no run has been claimable recently, so run_history proves neither up nor '
      + 'down. Not a green light.',
    lastClaimAt: lastClaim,
  };
}

/* ══ C · IS ANYTHING STUCK ════════════════════════════════════════════════ */

/** Idle-with-a-free-lane threshold, in hours. Board #257 step 2: 30 minutes. */
export const STARVED_H = 0.5;
/** How far back a failed run is still news. */
export const FAILED_WINDOW_H = 12;
/** Run outcomes that mean the run did not deliver. */
export const FAILED_OUTCOMES = ['error', 'lease-expired', 'ignored'];

/* The stall rules share `CALM_STATUSES` with the plate — one definition of
 * "deliberately parked", declared at the top of this file. A task that is not
 * worth putting on his plate is not worth alarming him about either. */

export interface Stall {
  kind: string;
  taskId: number | null;
  title: string;
  detail: string;
  ageH: number;
  /** `red` interrupts; `amber` is worth a look. Nothing here is informational —
   *  if it does not want a reaction it does not belong in this module. */
  severity: string;
}

export interface StallPanel {
  rows: Stall[];
  count: number;
  /** The reassuring facts printed BESIDE "Nothing is stalled." — an empty panel
   *  has to say what it checked, or it reads as broken rather than as calm. */
  checked: string[];
}

export interface StallInput {
  cars: CarsPanel | null;
  runs: RunRow[];
  tasks: Task[];
  /** Ids the plate already owns — his own queue is not a stall. */
  plateIds: number[];
  /** Tasks the EXISTING dispatch resolver says are eligible right now
   *  (`runIneligibleReason` === null). Not re-derived here. */
  dispatchableIds: number[];
  lanesFree: number;
  loop: LoopReading;
  sessions: HeartbeatRow[];
  nowMs: number;
}

/**
 * THE NEGATIVE-SPACE MODULE — what lets Kevin NOT read everything else.
 *
 * Its usual state is EMPTY, and that must feel like reassurance rather than
 * absence, which is why `checked` exists: the panel says what it looked at.
 *
 * Anything already on his plate is excluded. His queue is his queue; repeating
 * it here as a "stall" would double-count the one number the page is for.
 */
export function buildStalls(input: StallInput, limit: number): StallPanel {
  const plate = new Set(input.plateIds || []);
  const byId = new Map<number, Task>();
  (input.tasks || []).forEach((t) => byId.set(t.id, t));
  const calm = (id: number | null) => {
    if (id == null) return false;
    const t = byId.get(id);
    return !!t && CALM_STATUSES.indexOf(String(t.status || '')) >= 0;
  };
  const rows: Stall[] = [];

  // 1 · A car waiting for a train — R's definition, IMPORTED not restated
  // (rSessionSignals.carsWaitingForATrain, board #251). This is the 07-30
  // four-hour stall, and it is the reason this module leads with it.
  const cars = input.cars;
  if (cars && cars.count > 0) {
    cars.rows.forEach((c) => {
      if (plate.has(c.task.id) || calm(c.task.id)) return;
      rows.push({
        kind: 'car-waiting',
        taskId: c.task.id,
        title: c.task.title,
        detail: 'built, reviewed and parked in Staged with an unmerged branch, and no '
          + 'open train is carrying it — a train is OWED',
        ageH: c.ageH,
        severity: 'red',
      });
    });
  }

  // 2 · A run past its 45m lease. `isLeaseLive` is the #193 resolver — a second
  // definition of "still holding a lane" would drift from the dispatch page's.
  (input.runs || []).forEach((r) => {
    if (r.outcome !== 'running') return;
    if (isLeaseLive(r)) return;
    const t = byId.get(r.task_id);
    rows.push({
      kind: 'run-overrun',
      taskId: r.task_id,
      title: t ? t.title : `task #${r.task_id}`,
      detail: `${r.agent_letter}·auto has been running past its 45-minute lease and was `
        + 'not reaped — the lane it holds is not really working',
      ageH: hoursSince(r.started_at || r.requested_at),
      severity: 'red',
    });
  });

  // 3 · Starvation — armed and dispatchable, a lane free, and nothing happened.
  if (input.lanesFree > 0) {
    (input.dispatchableIds || []).forEach((id) => {
      if (plate.has(id) || calm(id)) return;
      const t = byId.get(id);
      if (!t) return;
      const ageH = hoursSince(t.updated_at);
      if (ageH < STARVED_H) return;
      rows.push({
        kind: 'starved',
        taskId: id,
        title: t.title,
        detail: `armed and dispatch-eligible with ${input.lanesFree} lane(s) free, and `
          + 'nothing has claimed it',
        ageH,
        severity: 'amber',
      });
    });
  }

  // 4 · Runs that died in the last 12h.
  (input.runs || []).forEach((r) => {
    if (FAILED_OUTCOMES.indexOf(r.outcome) < 0) return;
    const at = r.finished_at || r.started_at || r.requested_at;
    const ageH = hoursSince(at);
    if (ageH < 0 || ageH > FAILED_WINDOW_H) return;
    if (calm(r.task_id)) return;
    const t = byId.get(r.task_id);
    rows.push({
      kind: 'run-failed',
      taskId: r.task_id,
      title: t ? t.title : `task #${r.task_id}`,
      detail: `a dispatched run ended \`${r.outcome}\` — there is no output to sign off`,
      ageH,
      severity: 'amber',
    });
  });

  // 5 · The machinery itself. `unknown` is NOT reported as a stall — it is an
  // absence of evidence, and reporting it would make the panel cry wolf on
  // every quiet evening.
  if (input.loop && input.loop.state === 'down') {
    rows.push({
      kind: 'loop-down',
      taskId: null,
      title: 'The dispatcher loop is not picking work up',
      detail: input.loop.why,
      ageH: 0,
      severity: 'red',
    });
  }

  // 6 · A tracked session that is not running. `never` and `down` are BOTH
  // reported and keep different words: "never started" and "stopped" want
  // different reactions (#157), but neither is anybody watching.
  (input.sessions || []).forEach((s) => {
    if (!s || !s.tracked) return;
    if (s.state === 'fresh' || s.state === 'stale') return;
    rows.push({
      kind: 'session-down',
      taskId: null,
      title: `${s.letter} session is NOT RUNNING`,
      detail: s.state === 'never'
        ? 'it has never written a heartbeat — this session does not exist'
        : 'its heartbeat went stale past the down threshold — it stopped',
      ageH: s.age_s != null ? s.age_s / 3600 : 0,
      severity: 'red',
    });
  });

  rows.sort((a, b) => {
    if (a.severity !== b.severity) return a.severity === 'red' ? -1 : 1;
    return b.ageH - a.ageH;
  });
  const checked = [
    `loop ${input.loop ? input.loop.state : 'unknown'}`,
    `${input.lanesFree} lane(s) free`,
    `${(input.runs || []).filter((r) => r.outcome === 'running').length} run(s) in flight`,
    `${(input.sessions || []).filter((s) => s && (s.state === 'fresh' || s.state === 'stale')).length}`
      + ' session(s) alive',
  ];
  return { rows: rows.slice(0, limit), count: rows.length, checked };
}

/* ══ E · ACTIVITY — filtered to DECISIONS AND OUTCOMES ════════════════════ */

/**
 * The minimal shape this module needs out of `buildFeed` (mSessionFeed, board
 * #220). Declared structurally rather than imported as `FeedEvent` so the
 * filter can be lifted and executed on plain fixtures.
 */
export interface Feedish {
  key: string; at: string; actor: string; source: string; kind: string;
  taskId: number | null; headline: string;
}

/**
 * DROPPED FIRST — M named these, and what the feed EXCLUDES is the spec.
 *
 * An unfiltered feed is noise, and noise is how a dashboard stops being read
 * (#240 had to bound M's activity log for the same reason). Note that M's own
 * board chatter is already gone before this runs: `commentEvents` drops it at
 * the source.
 */
export const KEVIN_FEED_DROP = [
  {
    id: 'status-tick',
    why: 'an ordinary board transition — status ticks are the bulk of the log and none of them are decisions',
    test: (e: Feedish) => /^status: /.test(e.headline) && !/→ (Done|Closed|Blocked)\b/.test(e.headline),
  },
  {
    id: 'assignee-change',
    why: 'a hand-off between lanes — the board records it; it is not news to the owner',
    test: (e: Feedish) => /^assignee: /.test(e.headline),
  },
  {
    id: 'dispatched',
    why: 'the dispatcher claimed a task. Expected, high-volume, and the OUTCOME is the event that matters',
    test: (e: Feedish) => /^dispatched to /.test(e.headline),
  },
  {
    id: 'step-tick',
    why: 'a chain step ticked — the chain records it',
    test: (e: Feedish) => /^step \d+ .*complete/.test(e.headline),
  },
  {
    id: 'heartbeat-poll',
    why: 'heartbeat and poll output — machine chatter',
    test: (e: Feedish) => /heartbeat|poll tick/i.test(e.headline),
  },
];

/** KEPT — decisions and outcomes, and nothing else. */
export const KEVIN_FEED_KEEP = [
  {
    id: 'stamp',
    why: 'a stamp — the decision that starts or stops work',
    test: (e: Feedish) => /^stamp\b/i.test(e.headline) || /^stamp:/i.test(e.headline),
  },
  {
    id: 'finished',
    why: 'a task reached Done or Closed — an outcome',
    test: (e: Feedish) => /^status: .*→ (Done|Closed)\b/.test(e.headline),
  },
  {
    id: 'blocked',
    why: 'a task went Blocked — work stopped and a human has to clear it',
    test: (e: Feedish) => /^status: .*→ Blocked\b/.test(e.headline),
  },
  {
    id: 'run-failed',
    why: 'a run did not deliver',
    test: (e: Feedish) => e.source === 'run' && e.kind !== 'run-ok',
  },
  {
    id: 'train',
    why: 'a release train landed or failed',
    test: (e: Feedish) => /release train|\bwave \d+/i.test(e.headline),
  },
  {
    id: 'deploy',
    why: 'a deploy',
    test: (e: Feedish) => /\bdeploy(ed|ing|-log)?\b/i.test(e.headline),
  },
  {
    id: 'flag-armed',
    why: 'a flag was armed — the change went live',
    test: (e: Feedish) => /\barmed\b/i.test(e.headline),
  },
  {
    id: 'issue-raised',
    why: 'an agent hit something the chain did not cover',
    test: (e: Feedish) => /^⚠️ issue raised/.test(e.headline),
  },
  {
    id: 'kevin',
    why: 'Kevin wrote it — his own decisions, back in front of him',
    test: (e: Feedish) => String(e.actor || '').toLowerCase() === KEVIN,
  },
];

export interface FeedFiltered {
  rows: Feedish[];
  /** Total kept, before the render slice. */
  count: number;
  /** How many the DROP rules removed, by rule — printed, never silent. */
  dropped: number;
  /** Matched neither list. NOT shown (this feed is include-only), but COUNTED
   *  and stated, because a silently discarded event is how a feed starts
   *  lying about what happened. */
  unrecognised: number;
}

/**
 * The awareness feed: DECISIONS AND OUTCOMES only.
 *
 * Deliberately the INVERSE of mSessionFeed's posture. That feed fails LOUD —
 * an unrecognised event is `attention`, because M must react to everything.
 * This one fails QUIET — an unrecognised event is not painted — because the
 * owner must react to almost nothing, and a feed he stops reading is worth
 * less than no feed. The compensating rail is that the count is printed: he
 * can always see how much was left out, and `/admin/tasks` holds all of it.
 */
export function decisionsAndOutcomes(events: Feedish[], limit: number): FeedFiltered {
  const kept: Feedish[] = [];
  let dropped = 0;
  let unrecognised = 0;
  (events || []).forEach((e) => {
    if (KEVIN_FEED_DROP.some((r) => { try { return r.test(e); } catch { return false; } })) {
      dropped += 1;
      return;
    }
    if (KEVIN_FEED_KEEP.some((r) => { try { return r.test(e); } catch { return false; } })) {
      kept.push(e);
      return;
    }
    unrecognised += 1;
  });
  return { rows: kept.slice(0, limit), count: kept.length, dropped, unrecognised };
}
