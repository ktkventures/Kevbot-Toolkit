/**
 * R-session signals — the pure derivations behind /admin/r-session (board #251).
 *
 * Kept OUT of the page component for one reason: they are the thing that has to
 * be right, and `src/test_r_session_dashboard_251.py` LIFTS AND EXECUTES them in
 * node. A static substring assertion would pass against a module that returns
 * the wrong rows.
 *
 * THE LEAD SIGNAL — CARS WAITING FOR A TRAIN
 * ------------------------------------------
 * M's definition (board #251 step 2), implemented verbatim. A task is a car
 * waiting for a train when ALL THREE hold:
 *
 *   1 `status === 'Staged'` — terminal to the dispatcher. Nothing dispatches
 *     out of it, which is exactly why work parks there unnoticed.
 *   2 it has a branch that has NOT merged into `dev`.
 *   3 NO OPEN TRAIN TASK is carrying it.
 *
 * Clause 2 is NOT re-implemented here. It is `shippingRows` from
 * AdminDispatchPage — the #245 resolver, already tested by
 * `test_shipping_lane_unmerged_branch_245.py`. Writing a second definition of
 * "has unmerged code" is how two answers to one question appear, and this
 * module is the consumer, never the author, of that answer.
 *
 * Clause 3 is the ONE thing the shipping lane does not have, and it is what
 * makes this module actionable rather than merely informative: the lane says
 * *"is there unmerged code?"*, this says *"…and is anyone coming for it?"*
 *
 * The rail this exists to satisfy: feed it #246's 07-30 shape — `Staged`,
 * branch `feat/tasks-multiselect-filters-246` unmerged, no train task in
 * existence — and it MUST flag it. A version of this module that would not have
 * caught the four-hour stall is not worth building.
 *
 * KNOWN LIMIT, inherited from the resolver and accepted: a branch named for a
 * SIBLING task cannot be resolved (#238 shipped that way and was correctly
 * unresolvable). One task per branch.
 */
import { Task, RunRow, isFinished, stepOwner, stepTitle, hoursSince } from './taskBoardShared';
import { ShipLane, ShipRow, SHIP_STEP_OWNERS } from './AdminDispatchPage';

/** DOM bound for every list on the page (board #240 — the M-session activity
 *  log painted ~435 rows and stalled Kevin's browser). Render less at a time,
 *  never keep less: nothing is dropped from the counts, only from the slice. */
export const R_PAGE_ROWS = 25;

/** `R · Release train Wave 26 (07-30) — #240, #242` — the title `brief_gen.py`
 *  writes (`r_task_row`). Matched case-insensitively; the wave number is the
 *  join key against the deploy log. */
export const TRAIN_TITLE_RE = /release train wave\s+(\d+)/i;

/** `#246` anywhere in a train's title, body or step titles = that car is aboard. */
export const CAR_REF_RE = /#(\d+)/g;

/** A hand-authored train that does not follow the generator's title still has
 *  to SAY it is a train. See `isTrainTask` for why step-1 ownership alone is
 *  not enough. */
export const TRAIN_WORD_RE = /\btrains?\b/i;

/** The wave number a train task carries, or null if its title does not name one. */
export function trainWave(task: Task): number | null {
  const m = TRAIN_TITLE_RE.exec(String(task.title || ''));
  return m ? Number(m[1]) : null;
}

/**
 * Is this task a TRAIN (rather than a car that merely has a ship step)?
 *
 * M's step-2 definition was *"chain step 1 owned by `R-A`"*, and the first half
 * of it is right: every CAR carries an R/R-A ship step too, but never at index
 * 0, because a car is built and reviewed first. Matching "any R step" would
 * classify every car as its own train and report that everything already has
 * one — a false green with exactly the shape of the bug this page exists to
 * catch.
 *
 * **But step-1 ownership alone is NOT sufficient, and the live board says so.**
 * Run against the real 223-task board on 07-30, four tasks match it and are not
 * trains: #234 (an M hand-off doc), #235 (`dev` is RED), #237 (the api-502
 * outage) and #238 (a release-brief change) — all release-lane work whose first
 * step is legitimately R/R-A. #238 is `Staged` and OPEN, so under the loose rule
 * it would count as an open train, and any car id appearing in its body would be
 * silently marked "covered". That is the precise false negative this module must
 * never produce: a hidden car is an invisible stall, which is the 07-30 failure
 * again.
 *
 * So a train must also SAY it is one: the generator's canonical title
 * (`R · Release train Wave N …`, from `brief_gen.r_task_row`), or — for a
 * hand-authored train — step-1 release ownership AND the word "train".
 *
 * The asymmetry is deliberate. An unrecognised train produces a car that shows
 * as waiting when someone is in fact coming for it: a visible extra row that R
 * resolves in one glance. An over-recognised train HIDES a car. Wrong-loud beats
 * wrong-quiet, every time, on this panel.
 */
export function isTrainTask(task: Task): boolean {
  const title = String(task.title || '');
  if (TRAIN_TITLE_RE.test(title)) return true;
  const cl = Array.isArray(task.checklist) ? task.checklist : [];
  const first = cl.length > 0 ? cl[0] : null;
  const owner = first ? String(stepOwner(first) || '').toUpperCase() : '';
  if (!owner || SHIP_STEP_OWNERS.indexOf(owner) < 0) return false;
  return TRAIN_WORD_RE.test(title);
}

/**
 * Which car ids a train is carrying: every `#<id>` in its title, its
 * description, and its step titles.
 *
 * Deliberately GENEROUS. A false "this car has a train" is the failure this
 * module must not produce... but so is a false "no train": the generator writes
 * the cars into the TITLE (`— #240, #242`) while a hand-written brief names them
 * in the body, and a reader of only one would miss half the trains that exist.
 * Reading all three fields is what makes the clause hold for both.
 */
export function trainCarIds(task: Task): number[] {
  const cl = Array.isArray(task.checklist) ? task.checklist : [];
  const hay = [String(task.title || ''), String(task.description || '')]
    .concat(cl.map((s) => stepTitle(s)))
    .join('\n');
  const out: number[] = [];
  const re = new RegExp(CAR_REF_RE.source, 'g');
  let m = re.exec(hay);
  while (m) {
    const n = Number(m[1]);
    if (n && out.indexOf(n) < 0) out.push(n);
    m = re.exec(hay);
  }
  return out;
}

/** Open trains = train tasks that are not `Done`/`Closed` (the #232 set). */
export function openTrains(tasks: Task[]): Task[] {
  return (tasks || []).filter((t) => isTrainTask(t) && !isFinished(t.status));
}

export interface CarRow {
  task: Task;
  branch: string | null;
  /** From the branch push where known, else the run — NAMED, because an
   *  unlabelled proxy is how a dashboard starts lying (#245). */
  ageFrom: string;
  ageH: number;
  mergeWhy: string;
  /** True when merged-ness is UNREADABLE (no R/R-A step in the chain). Kept
   *  and labelled, never dropped: dropping on unknown is how a lane silently
   *  empties. */
  mergeUnknown: boolean;
}

export interface CarsPanel {
  rows: CarRow[];
  /** Total cars waiting — always the FULL count, even when `rows` is sliced. */
  count: number;
  /** Cars excluded because a train is already carrying them (with which). */
  covered: { id: number; title: string; trainId: number; trainTitle: string }[];
  openTrainIds: number[];
  /** Age of the oldest car, in hours. The headline number: #246 sat 4h. */
  oldestH: number;
  /** Inherited from the shipping lane and RE-STATED, never swallowed. */
  warning: string | null;
  /** True when a train is OWED: cars waiting and no open train to carry them. */
  trainOwed: boolean;
}

/**
 * CARS WAITING FOR A TRAIN — the module this whole page exists for.
 *
 * `lane` MUST be the output of `shippingRows` (the #245 resolver): that is
 * clause 2, already answered, with its own tests and its own degrade path. This
 * function adds clauses 1 and 3 and nothing else.
 *
 * Rows are OLDEST FIRST — the top of the list is the car that has been waiting
 * longest, which is the number Kevin reads this panel for. `rows` is bounded to
 * `limit` for the DOM; `count` is not.
 */
export function carsWaitingForATrain(lane: ShipLane, tasks: Task[], limit: number): CarsPanel {
  const trains = openTrains(tasks || []);
  const carrying = new Map<number, Task>();
  trains.forEach((tr) => {
    trainCarIds(tr).forEach((id) => {
      // A train never counts as carrying ITSELF — its own `#id` appears in its
      // body, and self-coverage would hide a stalled train from its own panel.
      if (id !== tr.id && !carrying.has(id)) carrying.set(id, tr);
    });
  });
  const covered: CarsPanel['covered'] = [];
  const rows: CarRow[] = [];
  (lane?.rows || []).forEach((r: ShipRow) => {
    // Clause 1 — Staged only. The shipping lane is deliberately wider (it shows
    // everything unmerged, at any stage); a train is owed only for work that
    // has finished review and parked.
    if (r.task.status !== 'Staged') return;
    const train = carrying.get(r.task.id);
    if (train) {
      covered.push({ id: r.task.id, title: r.task.title,
        trainId: train.id, trainTitle: train.title });
      return;
    }
    rows.push({
      task: r.task, branch: r.pushedBranch, ageFrom: r.ageFrom, ageH: r.ageH,
      mergeWhy: r.merge.why, mergeUnknown: r.merge.state === 'unknown',
    });
  });
  rows.sort((a, b) => b.ageH - a.ageH);
  return {
    rows: rows.slice(0, limit),
    count: rows.length,
    covered,
    openTrainIds: trains.map((t) => t.id),
    oldestH: rows.length ? rows[0].ageH : 0,
    warning: lane?.warning || null,
    trainOwed: rows.length > 0,
  };
}

export interface TrainRow {
  task: Task;
  wave: number | null;
  cars: number[];
  /** Chain progress — `n of N` steps ticked. */
  done: number;
  total: number;
  /** Outcome of its most recent dispatcher run, or null if it never ran. */
  outcome: string | null;
  finishedAt: string | null;
  /** Is this train's deploy-log entry visible in the served commit? `null` when
   *  the wave is unknown or the log could not be read — three answers, never
   *  two, because "no entry" and "cannot tell" want different reactions. */
  logged: boolean | null;
  ageH: number;
}

/**
 * Recent trains and how they went — newest first, bounded.
 *
 * `loggedWaves` comes from `/api/r-session/deploy-log`, read out of the commit
 * the API is serving. A train whose wave is ABSENT there still owes its
 * deploy-log PR — which is the closest thing to "PR open/merged" a container
 * with no GitHub credential can honestly answer, and it is labelled as such on
 * the page rather than presented as a PR query.
 */
export function recentTrains(
  tasks: Task[], runsByTask: Map<number, RunRow[]>,
  loggedWaves: number[] | null, limit: number,
): TrainRow[] {
  const logged = loggedWaves ? new Set(loggedWaves) : null;
  const rows = (tasks || []).filter(isTrainTask).map((t) => {
    const cl = Array.isArray(t.checklist) ? t.checklist : [];
    const runs = (runsByTask && runsByTask.get(t.id)) || [];
    const last = runs.length ? runs[0] : null;
    const wave = trainWave(t);
    return {
      task: t, wave,
      cars: trainCarIds(t).filter((id) => id !== t.id),
      done: cl.filter((s) => !!s.done).length,
      total: cl.length,
      outcome: last ? last.outcome : null,
      finishedAt: last ? (last.finished_at || null) : null,
      logged: (logged && wave) ? logged.has(wave) : null,
      ageH: hoursSince(t.updated_at),
    };
  });
  // Newest wave first; trains with no wave number sort to the back by age.
  rows.sort((a, b) => {
    if (a.wave != null && b.wave != null) return b.wave - a.wave;
    if (a.wave != null) return -1;
    if (b.wave != null) return 1;
    return a.ageH - b.ageH;
  });
  return rows.slice(0, limit);
}

/** Words that make a `Blocked` task R's problem rather than someone else's. */
export const RELEASE_WORDS = ['release', 'train', 'merge', 'deploy', 'ship', 'pr ', 'wave'];

/**
 * `Blocked` items that name a release — module 5.
 *
 * Matched on TITLE ONLY, deliberately. Descriptions on this board routinely
 * narrate a release in passing ("…blocked until after the next train"), and
 * matching them turned every blocked task into R's business in the shape this
 * panel would then be ignored for. A false positive here is worse than a miss:
 * the panel is a short list Kevin is meant to actually read.
 */
export function releaseBlocked(tasks: Task[], limit: number): { rows: Task[]; count: number } {
  const rows = (tasks || []).filter((t) => {
    if (t.status !== 'Blocked') return false;
    const title = String(t.title || '').toLowerCase();
    return RELEASE_WORDS.some((w) => title.indexOf(w) >= 0);
  }).sort((a, b) => hoursSince(b.updated_at) - hoursSince(a.updated_at));
  return { rows: rows.slice(0, limit), count: rows.length };
}

/** One session's sign of life, as `/api/r-session/heartbeats` returns it. */
export interface HeartbeatRow {
  letter: string; key: string; tracked: boolean;
  state: string;            // fresh | stale | down | never
  at: string | null; at_source: string | null; age_s: number | null;
  actor: string | null; note: string | null; updated_at: string | null;
  why: string;
}

export interface HeartbeatPayload {
  sessions: HeartbeatRow[];
  thresholds: { fresh_max_s: number; stale_max_s: number };
  tracked: string[];
  readable: boolean;
  read_error: string | null;
  generated_at: string;
}

/**
 * How a heartbeat renders: label, colour and whether it reads NOT RUNNING.
 *
 * The STATE is not recomputed here — the API decided it against thresholds the
 * API also returns, and a second set of thresholds in the browser is a second
 * answer to "is R alive". This maps a state to pixels, nothing more.
 *
 * `never` and `down` both render NOT RUNNING because the reaction is the same
 * (nobody is watching), but they keep DIFFERENT labels: "never started" and
 * "stopped" are different problems, and #157 is what conflating them costs.
 */
export function heartbeatDisplay(row: HeartbeatRow): {
  label: string; color: string; notRunning: boolean;
} {
  if (!row) return { label: 'unknown', color: 'var(--text-tertiary)', notRunning: true };
  if (row.state === 'fresh') {
    return { label: 'RUNNING', color: 'var(--green)', notRunning: false };
  }
  if (row.state === 'stale') {
    return { label: 'STALE', color: 'var(--yellow, #d69e2e)', notRunning: false };
  }
  if (row.state === 'never') {
    return { label: 'NOT RUNNING — never started', color: 'var(--red)', notRunning: true };
  }
  return { label: 'NOT RUNNING', color: 'var(--red)', notRunning: true };
}

/** `4m` / `2h 10m` / `—`. Seconds, because heartbeats live at minute scale and
 *  `ageShort`'s hour granularity would render every live session as `1m`. */
export function agoShort(ageS: number | null): string {
  if (ageS == null) return '—';
  if (ageS < 0) return 'in the future (clock skew)';
  if (ageS < 90) return `${Math.round(ageS)}s`;
  if (ageS < 3600) return `${Math.round(ageS / 60)}m`;
  const h = Math.floor(ageS / 3600);
  return `${h}h ${Math.round((ageS % 3600) / 60)}m`;
}
