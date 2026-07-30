/**
 * M-session dashboard — the event model and the CLASSIFIER RULES (board #220).
 *
 * This file is the substance of the task, not the page next to it. Kevin asked
 * earlier whether documentation exists for what the M session should monitor;
 * there is none, and the charter has zero mentions of monitoring. A document
 * would rot. **The rules below ARE that documentation** — every future miss
 * adds a RULE here instead of a resolution to remember harder.
 *
 * Three ideas, kept deliberately separate:
 *
 *   1 A FEED EVENT is a SHAPE, not a table row. Phase 1 fills it from
 *     `dev_task_comments` and `run_history`; Phase 2 posts ENVIRONMENT signals
 *     into the same shape (stranded branches, loop-version drift, preflight
 *     reds) — roughly half of what M actually watches is not in the database.
 *     Nothing here may assume `taskId` is present or that a source is a comment.
 *
 *   2 THE FEED SHOWS WHAT M DID NOT DO. Comments authored by the M SESSION
 *     (author exactly `M`) are excluded: they are not news to M, and measured on
 *     07-30 they were 130 of the last 400 board events at ~1,278 chars each — a
 *     third of the volume and all of the noise M generates for itself.
 *     `M·auto` is NOT the M session: it is a dispatched agent whose reports are
 *     precisely what M must react to, so it stays IN.
 *
 *   3 CLASSIFICATION IS A RULE, CONFIRMATION IS A JUDGEMENT. At this volume,
 *     hand-annotating every status change is theatre, so a rule sorts routine
 *     from attention and M's job is catching what the rule got wrong. That is
 *     only meaningful because the classifier is mechanical and inspectable —
 *     hence a declared table with a `why` on every row, rendered on the page.
 *     The seen/actioned STORE is step 4 and deliberately absent here.
 *
 * FAIL-LOUD DEFAULT: an event no rule recognises is classified ATTENTION, never
 * routine. A new event type must earn its way onto the quiet list; the opposite
 * default is how a monitoring surface goes quiet without anyone deciding it
 * should (the #157 dead-alarm lesson).
 */
import { RunRow, Task } from './taskBoardShared';

/* ── Wire shapes ─────────────────────────────────────────────────────────── */

/** A row from GET /api/dev-tasks/comments/recent (board #220). */
export interface FeedComment {
  id: number;
  task_id: number;
  author: string;
  body: string;
  created_at: string;
  step_order?: number | null;
  /** UNTRUNCATED length — `body.length < body_chars` means the wire elided. */
  body_chars?: number;
}

/* ── The event shape ─────────────────────────────────────────────────────── */

/** Where an event came from. `env` is unused in Phase 1 and exists so the
 *  renderer is already source-agnostic when Phase 2's writer arrives. */
export type FeedSource = 'comment' | 'run' | 'env';
export type FeedClass = 'attention' | 'routine';

export interface FeedEvent {
  /** Stable React key, unique ACROSS sources (`comment:41`, `run:7`). */
  key: string;
  source: FeedSource;
  /** ISO timestamp the event happened. */
  at: string;
  /** Who caused it: `kevin` · `F·auto` · `system` · a role letter. */
  actor: string;
  /** Machine-readable event type, set by the adapter (not the classifier). */
  kind: string;
  /** The task it belongs to, or null — Phase 2 env events have no task. */
  taskId: number | null;
  /** One line, always safe to render on its own. */
  headline: string;
  /** Full-ish body/excerpt; '' when the headline IS the event. */
  body: string;
  /** True untruncated body length (0 when there is no body). */
  bodyChars: number;
  cls: FeedClass;
  /** Which rule decided — rendered, so a wrong call is traceable to a rule. */
  rule: string;
  /** That rule's stated reason. */
  why: string;
}

/** Pre-classification event: everything but the verdict. */
export type ProtoEvent = Omit<FeedEvent, 'cls' | 'rule' | 'why'>;

/* ── Author conventions ──────────────────────────────────────────────────── */

/** The M SESSION's own author string. Exactly `M` — `M·auto` is a different
 *  actor entirely (a dispatched agent) and must never be folded in here. */
export const M_SESSION_AUTHOR = 'M';

/** Is this author a dispatched headless agent (`F·auto`, `M·auto`, …)? The
 *  dispatcher writes its agent reports under `<letter>·auto` (dispatcher.py
 *  reap()), which is the only place that suffix comes from. */
export const isAgentAuthor = (a: string): boolean => /·auto$/.test(a || '');

/** The `system` author covers every automatic board transition: status /
 *  assignee edits (dev_tasks.update_task), step completion + stamps
 *  (steps/*), and the dispatcher's own claim/reap/skip notes. */
export const isSystemAuthor = (a: string): boolean => (a || '') === 'system';

/**
 * The Ask-AI tell. `TaskDetailModal`'s 🤖 Ask AI button posts the comment with
 * a literal `@M ` prefix (board #134 item 4), which also writes a `task_mentions`
 * row. Anchored to the START of the body on purpose: an `@M` buried mid-comment
 * is a mention, not Kevin's fast lane, and only the fast lane outranks
 * everything else on the page.
 *
 * THE HYPHEN IS LOAD-BEARING (caught in the step-3 review, 07-30). The lookahead
 * excludes alphanumerics so `@MA`/`@Meta` do not match — but a hyphen is not an
 * alphanumeric, so the original `(?![A-Za-z0-9])` MATCHED `@M-A build this`.
 * Board #222 introduces `M-A` as a real owner, and agent-directed `@M-A` traffic
 * landing in this inbox would degrade exactly the signal it exists for: the
 * whole value of section 5 is that everything in it is genuinely Kevin waiting.
 * Hence `[-A-Za-z0-9]` — `@M` matches, `@M-A` does not.
 */
export const ASK_AI_RE = /^\s*@M(?![-A-Za-z0-9])/;
/** The reverse direction: M tagging Kevin with a question. Same hyphen rule, so
 *  a future `@kevin-something` token cannot be read as a question to Kevin. */
export const ASK_KEVIN_RE = /^\s*@kevin(?![-A-Za-z0-9])/i;

/* ── Classifier context ──────────────────────────────────────────────────── */

/**
 * What a rule may look at beyond the event itself. Kept explicit (rather than
 * letting rules reach into component state) so the table stays a pure function
 * of its inputs and step 4 can test it without a browser.
 */
export interface FeedContext {
  /** Finish times (ms) of runs that ended `ok`, per task. */
  okRunTimes: Map<number, number[]>;
}

export const emptyContext = (): FeedContext => ({ okRunTimes: new Map() });

/** Build the classifier context from the run log. */
export function buildContext(runs: RunRow[]): FeedContext {
  const okRunTimes = new Map<number, number[]>();
  runs.forEach((r) => {
    if (r.outcome !== 'ok') return;
    const t = new Date(r.finished_at || r.started_at || r.requested_at).getTime();
    if (!Number.isFinite(t)) return;
    okRunTimes.set(r.task_id, [...(okRunTimes.get(r.task_id) || []), t]);
  });
  return { okRunTimes };
}

/** Window for "immediately following" a completed run. The dispatcher posts
 *  the status transition in the same reap pass as the run's terminal outcome,
 *  so this is generous by an order of magnitude — deliberately, because the
 *  cost of the window being too tight is a routine event shown as attention
 *  (noise), and too loose is an attention event shown as routine (a miss). */
export const OK_RUN_WINDOW_MS = 10 * 60 * 1000;

/** Did an `ok` run on this task finish within the window around `atMs`? */
export function followsOkRun(taskId: number | null, atMs: number, ctx: FeedContext): boolean {
  if (taskId == null || !Number.isFinite(atMs)) return false;
  return (ctx.okRunTimes.get(taskId) || [])
    .some((t) => Math.abs(atMs - t) <= OK_RUN_WINDOW_MS);
}

/* ── THE RULES ───────────────────────────────────────────────────────────── */

export interface FeedRule {
  id: string;
  cls: FeedClass;
  /** Why this verdict — rendered on the page, so the watch-list is readable. */
  why: string;
  test: (e: ProtoEvent, ctx: FeedContext) => boolean;
}

const startsWith = (e: ProtoEvent, ...prefixes: string[]) =>
  prefixes.some((p) => e.headline.startsWith(p));

/**
 * FIRST MATCH WINS, so order is meaning: the loud failures are tested before
 * the generic shapes they would otherwise fall into. Adding a rule is the
 * sanctioned response to a miss — put it above the generic rule it corrects.
 */
export const FEED_RULES: FeedRule[] = [
  {
    id: 'ask-ai',
    cls: 'attention',
    why: 'Kevin’s 🤖 Ask AI fast lane — he is waiting, and expects to be waiting briefly.',
    test: (e) => e.source === 'comment' && !isSystemAuthor(e.actor)
      && ASK_AI_RE.test(e.body || e.headline),
  },
  {
    id: 'push-failed',
    cls: 'attention',
    why: 'The work is COMMITTED BUT UNPUSHED on local disk — it is invisible to everyone until someone acts.',
    test: (e) => startsWith(e, '⚠️ auto-push FAILED'),
  },
  {
    id: 'run-died',
    cls: 'attention',
    why: 'A dispatched run died with no output to sign off; the task’s status is a restoration, not a result.',
    test: (e) => startsWith(e, 'dispatch run FAILED', 'dispatch LEASE EXPIRED'),
  },
  {
    id: 'dispatch-refused',
    cls: 'attention',
    why: 'The queue refused or dropped work — a task that looks queued and is going nowhere.',
    test: (e) => startsWith(e, '⛔ dispatch skipped', '⚠️ STALENESS TRIPWIRE', 'run-request ignored'),
  },
  {
    id: 'issue-raised',
    cls: 'attention',
    why: 'An agent hit something the chain did not cover and routed it to M — M owns the chain.',
    test: (e) => startsWith(e, '⚠️ issue raised'),
  },
  {
    id: 'stamp',
    cls: 'attention',
    why: 'Kevin stamped a step: an approval unblocks the owner, a rejection routes back to M. Either way M moves next.',
    test: (e) => startsWith(e, 'stamp approved', 'stamp REJECTED', 'stamp:'),
  },
  {
    id: 'status-blocked',
    cls: 'attention',
    why: 'A task went Blocked — nothing dislodges it until a human names and clears the blocker.',
    test: (e) => /^status: .*→ Blocked/.test(e.headline),
  },
  {
    id: 'status-after-run',
    cls: 'routine',
    why: 'A status change immediately following a completed run is the dispatcher doing its job, not news.',
    test: (e, ctx) => /^status: /.test(e.headline)
      && followsOkRun(e.taskId, new Date(e.at).getTime(), ctx),
  },
  {
    id: 'assignee-to-M',
    cls: 'attention',
    why: 'A handoff TO M: assignee = whoever the task waits on (board #171), so this is now M’s next action.',
    test: (e) => /^assignee: .*→ M\b/.test(e.headline),
  },
  {
    id: 'assignee-change',
    cls: 'routine',
    why: 'A handoff between other lanes — visible, but it is not M’s ball.',
    test: (e) => /^assignee: /.test(e.headline),
  },
  {
    id: 'dispatched',
    cls: 'routine',
    why: 'The dispatcher claimed a task. Expected, high-volume, and the outcome is the event that matters.',
    test: (e) => startsWith(e, 'dispatched to '),
  },
  {
    id: 'step-complete',
    cls: 'routine',
    why: 'A chain step ticked. The chain records it; M reads the report, not the tick.',
    test: (e) => /^step \d+ .*complete/.test(e.headline),
  },
  {
    id: 'status-change',
    cls: 'routine',
    why: 'An ordinary board transition with no completed run behind it — logged, not chased.',
    test: (e) => /^status: /.test(e.headline),
  },
  {
    id: 'agent-report',
    cls: 'attention',
    why: 'A dispatched agent’s result report. This includes M·auto, which is NOT the M session — its reports are exactly what M must react to.',
    test: (e) => isAgentAuthor(e.actor),
  },
  {
    id: 'kevin-comment',
    cls: 'attention',
    why: 'Kevin wrote something. Six of the last 400 events were his; none of them are routine.',
    test: (e) => (e.actor || '').toLowerCase() === 'kevin',
  },
  {
    id: 'run-failed',
    cls: 'attention',
    why: 'A run ended error / lease-expired / ignored — the run log is often the only place the failure is visible.',
    test: (e) => e.source === 'run' && e.kind !== 'run-ok',
  },
  {
    id: 'run-ok',
    cls: 'routine',
    why: 'A run finished cleanly; its agent report is the event M reads.',
    test: (e) => e.source === 'run' && e.kind === 'run-ok',
  },
];

/** The fail-loud default — see the file header. Not a rule in the table,
 *  because it is what happens when the table has nothing to say. */
export const UNCLASSIFIED: Omit<FeedRule, 'test'> = {
  id: 'unclassified',
  cls: 'attention',
  why: 'No rule recognises this event. Unrecognised is NOT routine: it surfaces until a rule is written for it.',
};

/** Apply the table. First match wins; no match = attention. */
export function classify(e: ProtoEvent, ctx: FeedContext): FeedEvent {
  for (const r of FEED_RULES) {
    let hit = false;
    try { hit = r.test(e, ctx); } catch { hit = false; }
    if (hit) return { ...e, cls: r.cls, rule: r.id, why: r.why };
  }
  return { ...e, cls: UNCLASSIFIED.cls, rule: UNCLASSIFIED.id, why: UNCLASSIFIED.why };
}

/* ── Adapters: source rows → the event shape ─────────────────────────────── */

/** First non-empty line of a body, trimmed for a headline. */
export function firstLine(body: string, max = 220): string {
  const line = (body || '').split('\n').map((s) => s.trim()).find(Boolean) || '';
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

/**
 * Comments → events, with the M SESSION's own comments dropped (idea 2 in the
 * header). The drop happens HERE rather than in the classifier so that "M's own
 * output is not news to M" stays a property of the feed itself, and a future
 * rule cannot accidentally re-admit it.
 */
export function commentEvents(comments: FeedComment[]): ProtoEvent[] {
  return comments
    .filter((c) => (c.author || '') !== M_SESSION_AUTHOR)
    .map((c) => ({
      key: `comment:${c.id}`,
      source: 'comment' as FeedSource,
      at: c.created_at,
      actor: c.author || 'unknown',
      kind: isSystemAuthor(c.author) ? 'system' : isAgentAuthor(c.author) ? 'report' : 'comment',
      taskId: c.task_id,
      headline: firstLine(c.body),
      body: c.body || '',
      bodyChars: c.body_chars ?? (c.body || '').length,
    }));
}

/**
 * Terminal runs → events. A run and its agent report are two views of one
 * thing and BOTH stay in the feed on purpose: the report is what M reads, the
 * run row is what carries the run id, the duration and the pushed branch. A
 * run still `requested`/`running` is state, not history — that is
 * /admin/dispatch's question (board #193), and #220 answers history.
 */
export function runEvents(runs: RunRow[]): ProtoEvent[] {
  return runs
    .filter((r) => !['requested', 'running'].includes(r.outcome))
    .map((r) => {
      const at = r.finished_at || r.started_at || r.requested_at;
      const pushed = r.pushed_branch ? ` · pushed ${r.pushed_branch}` : '';
      return {
        key: `run:${r.id}`,
        source: 'run' as FeedSource,
        at,
        actor: `${r.agent_letter}·auto`,
        kind: r.outcome === 'ok' ? 'run-ok' : `run-${r.outcome}`,
        taskId: r.task_id,
        headline: `run ${r.outcome} — ${r.run_id || 'no run id'}${pushed}`,
        body: '',
        bodyChars: 0,
      };
    });
}

/**
 * The feed: every source merged, classified, newest first.
 *
 * `extra` is the Phase-2 seam — an environment writer hands its signals in as
 * already-shaped proto events and they classify and render like anything else.
 * It is a parameter rather than a TODO comment precisely so the feed cannot be
 * quietly hard-wired to `dev_task_comments` in the meantime.
 */
export function buildFeed(
  comments: FeedComment[], runs: RunRow[], extra: ProtoEvent[] = [],
): FeedEvent[] {
  const ctx = buildContext(runs);
  return [...commentEvents(comments), ...runEvents(runs), ...extra]
    .map((e) => classify(e, ctx))
    .sort((a, b) => (a.at < b.at ? 1 : a.at > b.at ? -1 : 0));
}

/* ── Section 5 — the Ask-AI inbox ────────────────────────────────────────── */

/** One open question, in either direction. */
export interface InboxItem {
  comment: FeedComment;
  /** Whose answer is outstanding. */
  awaiting: 'M' | 'kevin';
  /** The answering comment, when there is one. */
  answeredBy?: FeedComment;
}

/**
 * Unanswered `@M` Ask-AI comments, OLDEST FIRST.
 *
 * Age is the whole point of this section — an unanswered Ask AI is the only
 * thing on this page that gets worse with time — so the oldest sits at the top
 * where it cannot age quietly. That is the opposite of every other list here.
 *
 * "Answered" (Phase 1) = a later comment on the same task authored by the M
 * SESSION (`M`). Deliberately narrow, and worth naming on the page: an `F·auto`
 * report on the same thread is NOT an answer to Kevin, and treating it as one
 * would manufacture exactly the false coverage this task exists to prevent.
 *
 * NOTE the feed-window caveat: `comments` is the last N rows board-wide, so an
 * Ask AI older than that window is invisible here. The page prints the window.
 */
export function askAiInbox(comments: FeedComment[]): { open: InboxItem[]; answered: InboxItem[] } {
  const byTask = new Map<number, FeedComment[]>();
  comments.forEach((c) => byTask.set(c.task_id, [...(byTask.get(c.task_id) || []), c]));

  const open: InboxItem[] = [];
  const answered: InboxItem[] = [];
  comments.forEach((c) => {
    if (isSystemAuthor(c.author) || (c.author || '') === M_SESSION_AUTHOR) return;
    if (!ASK_AI_RE.test(c.body || '')) return;
    const reply = (byTask.get(c.task_id) || [])
      .filter((x) => (x.author || '') === M_SESSION_AUTHOR && x.created_at > c.created_at)
      .sort((a, b) => (a.created_at < b.created_at ? -1 : 1))[0];
    (reply ? answered : open).push({ comment: c, awaiting: 'M', answeredBy: reply });
  });
  open.sort((a, b) => (a.comment.created_at < b.comment.created_at ? -1 : 1));   // oldest first
  answered.sort((a, b) => (a.comment.created_at < b.comment.created_at ? 1 : -1));
  return { open, answered };
}

/**
 * The reverse direction — questions M put to Kevin that he has not answered.
 * Both halves of the human/agent conversation belong in one place; a page that
 * only tracks what Kevin is owed still leaves M's asks to rot in a thread.
 *
 * Tell: a comment authored by `M` whose body OPENS with `@kevin`, with no later
 * comment by Kevin on that task.
 */
export function waitingOnKevin(comments: FeedComment[]): InboxItem[] {
  const byTask = new Map<number, FeedComment[]>();
  comments.forEach((c) => byTask.set(c.task_id, [...(byTask.get(c.task_id) || []), c]));

  const out: InboxItem[] = [];
  comments.forEach((c) => {
    if ((c.author || '') !== M_SESSION_AUTHOR) return;
    if (!ASK_KEVIN_RE.test(c.body || '')) return;
    const reply = (byTask.get(c.task_id) || [])
      .find((x) => (x.author || '').toLowerCase() === 'kevin' && x.created_at > c.created_at);
    if (!reply) out.push({ comment: c, awaiting: 'kevin' });
  });
  out.sort((a, b) => (a.comment.created_at < b.comment.created_at ? -1 : 1));
  return out;
}

/* ── Section 1 — M's queue ───────────────────────────────────────────────── */

/**
 * Statuses that take a task off M's plate. `Done` is finished; `Blocked` is
 * parked on a named blocker and, by the board's own definition, is not the next
 * thing M picks up. Everything else M is assigned stays visible.
 */
export const QUEUE_EXCLUDED_STATUSES = ['Done', 'Blocked'];

export type QueueComparator = (a: Task, b: Task) => number;

/**
 * The board's own priority fields. This is the FALLBACK, not the section's
 * purpose: Kevin asked for "the order M INTENDS", which is a plan, and a plan
 * is not derivable from `priority_phase.priority_seq` — those say what the
 * BOARD thinks. Step 4 supplies `intendedOrder` below; this stays as the order
 * for anything M has not yet formed an intention about.
 */
export const boardOrder: QueueComparator = (a, b) =>
  (a.priority_phase - b.priority_phase)
  || (a.priority_seq - b.priority_seq)
  || (a.id - b.id);

/* ── M's own store (board #220 step 4 — migrations/m_session_state.sql) ───── */

/**
 * One row of M's intended ordering. Note what is NOT here: no status, no
 * assignee, no title, no priority. Those come from `dev_tasks` at render time.
 * The type is narrow on purpose — design principle B, enforced at the type
 * level so that shadowing a board field is a compile error on the way to being
 * a 400 from the API.
 */
export interface MOrderRow {
  task_id: number;
  rank: number;
  /** M's one-liner: why it sits there. Answers Kevin's "why is that third?" */
  why?: string | null;
  updated_at?: string | null;
  updated_by?: string | null;
}

/** One seen/actioned mark, keyed by FEED EVENT KEY — not by comment id, so
 *  Phase 2's environment events can be covered by the same record. */
export interface MMark {
  event_key: string;
  state: 'seen' | 'actioned' | 'no-action';
  note?: string | null;
  task_id?: number | null;
  marked_at?: string | null;
  marked_by?: string | null;
}

/** GET /api/m-session/state. `available: false` = the migration is unapplied;
 *  the page says so in red rather than rendering an empty canvas. */
export interface MSessionState {
  available: boolean;
  reason?: string | null;
  canvas: { id?: number; body: string; updated_at?: string | null; updated_by?: string | null } | null;
  order: MOrderRow[];
  marks: MMark[];
}

/**
 * The comparator the queue actually uses once M has stated a plan.
 *
 * Ranked tasks first, in M's rank; everything else after, in board order.
 * Unranked LAST rather than first, because an intention is a stronger claim
 * than a default — and because the boundary between the two is then visible on
 * the page ("below here, M has not decided"), which is the point of the
 * section: Kevin can look and say *"why is that third?"*
 *
 * The store is consulted for RANK ONLY. Nothing in `byTask` may influence what
 * a row SAYS about a task — title, status and assignee are read from the board
 * task object at render time. That is the rail: an ordering store that also
 * carried a status would give Kevin two answers to one question.
 */
export function intendedOrder(order: MOrderRow[]): QueueComparator {
  const rank = new Map<number, number>();
  order.forEach((o) => {
    if (typeof o?.task_id === 'number' && Number.isFinite(o?.rank)) {
      rank.set(o.task_id, o.rank);
    }
  });
  return (a, b) => {
    const ra = rank.has(a.id) ? (rank.get(a.id) as number) : Infinity;
    const rb = rank.has(b.id) ? (rank.get(b.id) as number) : Infinity;
    if (ra !== rb) return ra - rb;
    return boardOrder(a, b);
  };
}

/** Everything waiting on the M session, in `cmp` order. */
export function mQueue(tasks: Task[], cmp: QueueComparator = boardOrder): Task[] {
  return tasks
    .filter((t) => t.assignee === M_SESSION_AUTHOR
      && !QUEUE_EXCLUDED_STATUSES.includes(t.status))
    .slice()
    .sort(cmp);
}

/* ── Section 3 — the rules, rendered FROM SOURCE ─────────────────────────── */

/** One block from GET /api/m-session/rules. `found: false` means the page must
 *  show the failure — a renamed heading reads as "M has no rules" otherwise. */
export interface RuleBlock {
  id: string;
  label: string;
  why: string;
  path: string;
  anchor: string;
  resolved_path: string;
  found: boolean;
  text: string;
  error?: string | null;
}

export interface RulesPayload {
  sources: RuleBlock[];
  missing: number;
  ror_root?: string;
  repo_root?: string;
  generated_at?: string;
}
