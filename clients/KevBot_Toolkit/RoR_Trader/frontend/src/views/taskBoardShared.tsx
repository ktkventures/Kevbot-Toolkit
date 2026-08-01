/**
 * Shared types, constants, style atoms, and small chips for the team-board
 * views (AdminTasksPage table + TaskDetailModal). One source for the role
 * list — adding a FIXED role is one line here (ASSIGNEES + ROLE_COLOR).
 *
 * Board #289 added the one lane class that is NOT fixed: per-goal lanes
 * (`G281`) are DERIVED from the board's open goal rows and coloured by a prefix
 * rule, because goals are created at will and no array can enumerate them. See
 * PER-GOAL ASSIGNEE LANES below before reaching for a new ASSIGNEES entry.
 */
import React from 'react';
// Board #246 — the multi-select PREDICATES live in a dependency-free plain-JS
// module so a test can execute them for real (see taskFilters.js header).
import { filterSummary, toggleSelected, UNASSIGNED_LABEL } from './taskFilters';

/**
 * A per-step approval gate (board #182): a stamp carried in the step's own
 * real estate, with its own state — replaces the task-level `kevin_final` for
 * longer chains ("approve THIS step", not "approve the task").
 */
export interface StepStamp {
  required?: boolean;
  state?: 'pending' | 'approved' | 'rejected' | null;
  by?: string | null;
  at?: string | null;
}

/**
 * A process-chain step (board #182). The legacy shape was {text, done, role};
 * the chain widens it with a stable `id`, an `owner` (role or 'kevin'), a
 * one-line `title` (the substance, shown even when collapsed), a markdown
 * `body` (the SOP), a per-step `mode` (execute|discuss) and `stamp`, plus
 * completion provenance. Every new field is OPTIONAL and `text`/`role` are
 * retained, so the 42 legacy checklists render byte-for-byte — a checklist is
 * treated as a chain only once a step carries a new-shape key (isProcessChain,
 * mirroring the API's opt-in signal). Completion fields are server-owned
 * (POST /steps/complete), never toggled by a checklist PATCH.
 */
export interface ChecklistStep {
  done: boolean;
  // legacy (still emitted by defaultChain / simple tasks)
  text?: string;
  role?: string | null;
  // #182 process chain
  id?: string;
  owner?: string | null;
  title?: string;
  body?: string;
  mode?: 'execute' | 'discuss';
  origin?: 'planned' | 'audible';
  stamp?: StepStamp | null;
  completed_at?: string | null;
  completed_by?: string | null;
  // #184 step deliverables — the evidence this step asks for
  deliverables?: StepDeliverable[];
}

export type DeliverableKind = 'text' | 'link' | 'file';

/**
 * A deliverable declared by a chain step (board #184) — the evidence the step
 * asks for, captured where it is asked for rather than in a comment no surface
 * can see. Mirrors the API's shape (dev_tasks._DELIVERABLE_*).
 *
 * TWO HALVES, and the split IS the design. The SPEC (kind/label/required/hint)
 * is author-owned and edited through the generic checklist PATCH; the FILL
 * (value + provenance) is SERVER-owned and reachable only through
 * /steps/deliverables/* — a PATCH that touches a fill field is a 409. Never
 * write the fill fields from the UI's checklist path.
 */
export interface StepDeliverable {
  id?: string;
  kind: DeliverableKind;
  label: string;
  /** DEFAULT FALSE. Optional-by-default is the ruling (Kevin 07-28): a required
   *  input encodes a guess about the SHAPE of the evidence, usually made before
   *  anyone has run the process. Only a `*` one blocks Complete (T10). */
  required?: boolean;
  hint?: string;
  // ── server-owned (fill state) ──
  value?: string | null;   // text: the answer · link: the URL ·
                           // file: the STORAGE OBJECT PATH (never a URL)
  filename?: string | null;
  size_bytes?: number | null;
  content_type?: string | null;
  filled_at?: string | null;
  filled_by?: string | null;
}

/** Three kinds, no more (spec §8) — the moment this grows selects and dates it
 *  stops being a step and starts being a form builder. */
export const DELIVERABLE_KINDS: {
  key: DeliverableKind; icon: string; label: string; def: string;
}[] = [
  { key: 'text', icon: '✎', label: 'text', def: 'a short written answer (≤ 4000 chars)' },
  { key: 'link', icon: '🔗', label: 'link', def: 'an http(s) URL — a PR, a doc, a dashboard' },
  { key: 'file', icon: '📎', label: 'file', def: 'an upload ≤ 10 MB: png jpg gif webp pdf csv txt md json log zip' },
];
export const deliverableIcon = (k?: string): string =>
  DELIVERABLE_KINDS.find((d) => d.key === k)?.icon || '•';

/** A step's deliverables — [] when it never declared any. ABSENT and EMPTY read
 *  the same, which is what makes every pre-#184 chain render unchanged. */
export const stepDeliverables = (s?: ChecklistStep | null): StepDeliverable[] =>
  (Array.isArray(s?.deliverables) ? s!.deliverables! : []);

export const isDeliverableFilled = (d?: StepDeliverable | null): boolean => !!d?.value;

/** Unfilled deliverables of one requiredness. The two halves of T10: required
 *  BLOCKS Complete; optional only prompts (decision D2 — never a block). */
export const unfilledDeliverables = (
  s: ChecklistStep | null | undefined, required: boolean,
): StepDeliverable[] =>
  stepDeliverables(s).filter((d) => !!d.required === required && !isDeliverableFilled(d));

/** Mint a NEW deliverable spec for the step form. No `id` — the server assigns
 *  a stable one on PATCH (_ensure_deliverable_ids), exactly as it does for a
 *  step — and never a fill value: the API refuses one that arrives filled. */
export const makeDeliverable = (
  kind: DeliverableKind = 'text', label = '',
): StepDeliverable => ({ kind, label, required: false, hint: '' });

/** Human file size for the deliverable rows. */
export const fileSize = (n?: number | null): string => {
  if (n == null) return '';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
};

// A checklist becomes a #182 process chain once ANY step carries a new-shape
// key — the exact opt-in signal the API uses (dev_tasks._NEW_SHAPE_KEYS).
// Legacy {text,role,done} checklists stay legacy and behave as before #182.
const NEW_SHAPE_KEYS: (keyof ChecklistStep)[] = ['id', 'owner', 'title', 'body', 'mode', 'origin', 'stamp', 'deliverables'];
export const isProcessChain = (cl?: ChecklistStep[] | null): boolean =>
  Array.isArray(cl) && cl.some((s) => !!s && NEW_SHAPE_KEYS.some((k) => s[k] != null));

/** Who acts on a step — new `owner`, falling back to legacy `role` (mirrors
 *  the API's _step_owner). */
export const stepOwner = (s: ChecklistStep): string | null => s.owner ?? s.role ?? null;

/** One line of substance — new `title`, falling back to legacy `text`
 *  (mirrors the API's _step_title). */
export const stepTitle = (s: ChecklistStep): string => s.title || s.text || '';

/** Per-step mode tooltips (board #182). Execute steps are dispatchable; a
 *  discuss step is a human review point and must never dispatch to a headless
 *  agent (enforced by the dispatcher, Step 7). */
export const MODE_DEF: Record<string, string> = {
  execute: 'execution step — dispatchable to a headless agent',
  discuss: 'discussion step — a human reviews here; never dispatched to a headless agent',
};

// ── Step 6: slash-command insertion (board #182) ─────────────────────────────
// Editing the chain must be as cheap AND as safe as ticking it — a hand-edited
// JSON checklist is exactly the fragility M hit (dropped a step, duplicated a
// row). These helpers back the ClickUp-style `/` insert menu and the accordion's
// structured insert/edit; every path routes through the API's structural PATCH
// (add/edit/reorder of NOT-done steps is allowed; completion stays server-owned).

/**
 * A chain is "under way" once any step is complete — a step inserted past that
 * point is a mid-flight course-correction and is auto-tagged origin='audible'
 * (Kevin's ruling 5: audibles must be visible). A fresh chain still being
 * authored (nothing done) inserts as origin='planned'.
 */
export const chainStarted = (steps?: ChecklistStep[] | null): boolean =>
  Array.isArray(steps) && steps.some((s) => !!s && s.done);

/**
 * Mint a NEW chain step for a structured insert (slash "process module" or the
 * accordion "+ insert step"). No `id` — the server assigns a stable one on PATCH
 * (dev_tasks._ensure_step_ids); `done:false` always (the API refuses an inserted
 * step that starts complete, T4'). `origin` is audible when the chain is under
 * way. This is the ONLY sanctioned way to add a step — never hand-edit the JSON.
 */
export const makeChainStep = (
  fields: { owner?: string | null; title: string; mode?: 'execute' | 'discuss'; body?: string },
  started: boolean,
): ChecklistStep => ({
  owner: fields.owner || null,
  title: fields.title.trim(),
  body: fields.body || '',
  mode: fields.mode || 'execute',
  origin: started ? 'audible' : 'planned',
  done: false,
});

export type SlashKind = 'step' | 'text';
export interface SlashItem {
  key: string;
  icon: string;
  label: string;
  hint: string;
  kind: SlashKind;
  /** text inserts only: the snippet dropped in; '|' marks where the caret lands. */
  template?: string;
}

// The insert menu (Kevin's ClickUp screenshot is the reference). PROCESS MODULE
// FIRST — that is the headline: inserting a step must be as cheap as ticking one.
// The rest are ordinary markdown content blocks dropped at the caret.
export const SLASH_ITEMS: SlashItem[] = [
  { key: 'process', icon: '⛓', label: 'Process module', kind: 'step',
    hint: 'insert a chain step — owner · title · mode · SOP' },
  { key: 'image', icon: '🖼', label: 'Image', kind: 'text',
    hint: 'embed an image by URL', template: '![|](https://)' },
  { key: 'code', icon: '{ }', label: 'Code block', kind: 'text',
    hint: 'fenced code block', template: '```\n|\n```\n' },
  { key: 'banner', icon: '▤', label: 'Banner', kind: 'text',
    hint: 'callout / note banner', template: '> **Note:** |\n' },
  { key: 'list', icon: '•', label: 'List', kind: 'text',
    hint: 'bulleted list', template: '- |\n- ' },
];

/** Filter the insert menu by the text typed after `/`, matching label or key. */
export const filterSlashItems = (filter: string): SlashItem[] => {
  const f = (filter || '').toLowerCase();
  if (!f) return SLASH_ITEMS;
  return SLASH_ITEMS.filter(
    (it) => it.label.toLowerCase().includes(f) || it.key.includes(f));
};

/**
 * Apply a text-insert slash item to a body string: replace the `/trigger` range
 * [start,end) with the item's template, returning the new text and the caret
 * position (where the template's '|' marker was). Pure — the slash wiring in
 * TaskDetailModal calls this and then restores focus/selection.
 */
export const applySlashInsert = (
  text: string, start: number, end: number, item: SlashItem,
): { text: string; caret: number } => {
  const tpl = item.template || '';
  const marker = tpl.indexOf('|');
  const body = marker >= 0 ? tpl.replace('|', '') : tpl;
  const next = text.slice(0, start) + body + text.slice(end);
  const caret = start + (marker >= 0 ? marker : body.length);
  return { text: next, caret };
};

/**
 * Detect a live slash-command trigger at the caret: a `/` that starts a token
 * (line-start or after whitespace) followed by optional word chars, and nothing
 * after it up to the caret. Returns the trigger's char range + the filter text,
 * or null when there is no active trigger. Board #182 Step 6.
 */
export const detectSlash = (
  value: string, caret: number,
): { start: number; end: number; filter: string } | null => {
  const before = value.slice(0, caret);
  const m = before.match(/(?:^|\s)\/([\w-]*)$/);
  if (!m) return null;
  const filter = m[1] || '';
  return { start: caret - (filter.length + 1), end: caret, filter };
};

/**
 * The slash-command insert menu (board #182 Step 6) — a ClickUp-style dropdown
 * that opens when you type `/` in a task body. Presentational: the parent owns
 * trigger detection, the filter text, and the active index (arrow-key nav on the
 * textarea). Anchored just under the field rather than at the exact caret pixel
 * — robust across fonts, and the menu is short. `onMouseDown`+preventDefault so
 * a click doesn't blur the textarea before the pick handler runs.
 */
export const SlashMenu = ({ items, filter, active, onPick, onClose }: {
  items: SlashItem[];
  filter: string;
  active: number;
  onPick: (item: SlashItem) => void;
  onClose: () => void;
}) => {
  if (items.length === 0) return null;
  return (
    <>
      <span onMouseDown={onClose} style={{ position: 'fixed', inset: 0, zIndex: 1500 }} />
      <div role="listbox" aria-label="insert menu" style={{
        position: 'absolute', top: '100%', left: 0, marginTop: 4, zIndex: 1600,
        minWidth: 264, maxWidth: 360, padding: 4, borderRadius: 8,
        border: '1px solid var(--border)', background: 'var(--bg-card, var(--bg-input))',
        boxShadow: '0 8px 28px rgba(0,0,0,0.45)',
      }}>
        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', padding: '3px 8px', letterSpacing: 0.4 }}>
          INSERT{filter ? ` · /${filter}` : ''}
        </div>
        {items.map((it, i) => (
          <button key={it.key} type="button"
            onMouseDown={(e) => { e.preventDefault(); onPick(it); }}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '6px 8px', borderRadius: 6, cursor: 'pointer', textAlign: 'left',
              border: 'none', color: 'inherit',
              background: i === active ? 'var(--bg-input)' : 'transparent',
            }}>
            <span style={{ width: 24, textAlign: 'center', fontSize: 12, fontFamily: 'monospace' }}>{it.icon}</span>
            <span style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <span style={{ fontSize: 13, fontWeight: it.kind === 'step' ? 700 : 500 }}>
                {it.label}{it.kind === 'step' ? ' ⭐' : ''}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{it.hint}</span>
            </span>
          </button>
        ))}
      </div>
    </>
  );
};

export interface Task {
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
  checklist: ChecklistStep[];
  affected_sids?: number[] | null;
  // Kanban lifecycle (board #136): blast-radius chip + two-touch stamp +
  // reserved standing-approval flag (dev_tasks_lifecycle.sql). Optional so
  // pre-migration rows render.
  impact?: string;
  kevin_final?: boolean;
  standing_approval?: boolean;
  // Explicit task type (board #261, dev_tasks_task_type.sql): action | vision |
  // goal. OPTIONAL so a pre-migration row — or an API not yet serving the
  // column — still renders: `taskTypeOf` falls back to the old derivation
  // rather than defaulting silently to 'action'.
  task_type?: string;
  // Goal parameters (board #262). Ships in #261's migration, read by nothing
  // until then — declared here so the shape is not invented twice.
  goal_params?: Record<string, unknown> | null;
  // May an AI agent run this task (board #198, dev_tasks_ai_eligible.sql).
  // Optional so pre-migration rows render; see AiEligibleToggle for the split
  // this column makes — status says WHERE the task is, this says MAY IT RUN.
  ai_eligible?: boolean;
  updated_at: string;
}

export interface Comment { id: number; author: string; body: string; created_at: string; }

/**
 * An @-mention delivered to a role (board #143). Written at comment-POST time;
 * `seen_at` NULL = still in the inbox. Enriched by GET /api/mentions with the
 * host task's title/status and the comment excerpt for display.
 */
export interface Mention {
  id: number;
  comment_id: number;
  task_id: number;
  mentioned: string;      // the role tagged: M|E|E2|F|P|R|kevin
  author: string;         // who wrote the comment
  created_at: string;
  seen_at: string | null;
  task_title?: string | null;
  task_status?: string | null;
  excerpt?: string | null;
}

// The @-mention token set (board #143): registry letters + kevin. Sourced from
// the agents registry at runtime; this is the fallback + the compose picker's
// order. Adding a role is one line here + one registry row.
export const MENTION_ROLES = ['M', 'M-A', 'E', 'E2', 'F', 'P', 'R', 'R-A', 'kevin'];

/** One dispatcher run of a board task (run_history table, board #109). */
export interface RunRow {
  id: number;
  task_id: number;
  agent_letter: string;
  run_id: string | null;
  requested_by: string | null;   // null = organic queue dispatch (no button)
  requested_at: string;
  started_at: string | null;
  finished_at: string | null;
  outcome: string;               // requested | running | ok | error | lease-expired | ignored
  log_tail: string | null;
  // Board #195 (run_history_pushed_branch.sql): what the run pushed at reap.
  // OPTIONAL — the key is ABSENT on an un-migrated DB and NULL on every run
  // recorded before #195. Absent/NULL means UNKNOWN, never "not pushed"
  // (Spec_Dispatch_Dashboard.md §7 rail 1) — a false red here would send
  // Kevin hunting for a branch that shipped days ago.
  pushed_branch?: string | null;
  pushed_at?: string | null;
}

/**
 * Dispatcher constants, mirrored for the UI (board #193, Spec §4).
 *
 * SSOT: `tools/team_dispatcher/dispatcher.py` (~L127 — CONCURRENCY / DAILY_CAP
 * / RUN_TIMEOUT_S). The app runs on Railway and cannot read that file, so these
 * are declared ONCE here and labelled "configured" wherever they render, so a
 * drift is legible rather than invisible.
 *
 * POLL_S is the OPERATIONAL value the loop is started with (`--loop --poll 20`),
 * not the module default (900) — it is used only to word the "can lag one poll"
 * note, never to compute a state.
 *
 * DAILY_CAP 24 → 40 → 50 (Kevin, 07-30) — a circuit breaker, not a ration.
 * Its window is the UTC DAY, which rolls at 18:00 MT: an evening's runs are
 * charged to the next morning. The lane panel prints that reset time so the cap
 * cannot be spent unknowingly.
 *
 * ⚠ DAILY_CAP IS NO LONGER THE LIVE VALUE (board #228). The live cap lives in
 * `system_settings.dispatcher_daily_cap` and is re-read by the loop on EVERY
 * poll; this mirror is a build-time copy of dispatcher.py's FALLBACK constant
 * and nothing more. It went stale under #219 and needed #226 to fix — which is
 * exactly why the dispatch page renders the settings row, not this number, and
 * falls back to it only when the row is missing or unreadable (and says so).
 *
 * DAILY_CAP_MAX / DAILY_CAP_SETTING mirror the same two names in dispatcher.py.
 * The bound is what `effective_daily_cap()` CLAMPS to, so the UI refuses out of
 * range using this exact number: page and loop cannot be allowed to disagree
 * about what a legal cap is.
 *
 * CONCURRENCY 3 → 4 (same change, board #219) — this mirror was missed when
 * DAILY_CAP was updated beside it, so the dashboard advertised 3 lanes while the
 * loop ran 4, and test_dispatch_dashboard_193.py went RED on dev. Both constants
 * move together: dispatcher.py is the SSOT and this line only mirrors it.
 */
export const DISPATCHER = {
  CONCURRENCY: 4, DAILY_CAP: 50, RUN_TIMEOUT_S: 2700, POLL_S: 20,
  DAILY_CAP_MIN: 1, DAILY_CAP_MAX: 500, DAILY_CAP_SETTING: 'dispatcher_daily_cap',
};

// The agreed pipeline (Kevin+M 07-25, board #136; extended by #232): Backlog →
// Scoping → Approval → Todo → In Progress → Review → Staged → Done → Closed.
// `Blocked` and `Stand By` are the two anywhere-exceptions and they are NOT the
// same thing — see STATUS_DEF. This is the `action` type's status set; the
// container types get their own (TASK_TYPES below).
export const STATUSES = ['Backlog', 'Scoping', 'Approval', 'Todo', 'In Progress', 'Review', 'Staged', 'Blocked', 'Stand By', 'Done', 'Closed'];

/* ── THE TASK-TYPE MODEL (board #261) ───────────────────────────────────────
 * The board already BEHAVED as though tasks had types — `isVisionTask()`
 * derived "vision" from a tag or from having children, and a separate
 * `VISION_STATUSES` list gave those rows their own status set. The type was
 * implicit and the split was frontend-only. #261 makes it explicit
 * (`dev_tasks.task_type`, migration dev_tasks_task_type.sql) and gives it a
 * third member.
 *
 * THIS MAP IS THE ONE SOURCE OF TRUTH for what a type IS. `VISION_STATUSES`
 * folded INTO it and no longer exists beside it — a second exported list is
 * exactly how the two drift. Everything that asks "which statuses may this row
 * have / what icon does it wear / may it have children" reads THIS.
 *
 * Mirrored server-side by `api.routers.dev_tasks.TASK_TYPES`, asserted equal by
 * PARSED VALUE in test_task_type_model_261.py (boards #219/#245: a constant
 * shipped without its mirror put dev red for hours, and a quote-naive
 * string-compare "mirror" read two identical lists as different).
 *
 * STATUSES ARE RENDERED FROM HERE, NOT ENFORCED FROM HERE. The API does not
 * reject an out-of-set status: 236 legacy rows predate this map, which is the
 * whole reason `withLegacy` exists. Enforcement is a later task.
 *
 * `loop` is deliberately NOT a member — Kevin raised it and excluded it. */
export type TaskType = 'action' | 'vision' | 'goal';

export interface TaskTypeDef {
  /** Human label for the type. */
  label: string;
  /** Row glyph so the type is visible at a glance (Kevin's ask). The DEFAULT
   *  type wears none — an icon on every row is an icon on no row. */
  icon: string;
  /** One-liner shown as the icon's tooltip. */
  def: string;
  /** The statuses this type may hold — the dropdown is rendered from this. */
  statuses: string[];
  /** May this type hold subtasks? Containers can; `action` cannot. */
  children: boolean;
  /** Does this type carry its own session? Only `goal` (board #262 builds it). */
  session: boolean;
}

export const TASK_TYPES: Record<TaskType, TaskTypeDef> = {
  // Where the work actually happens — a subtask or a loose task. The default,
  // and the only type that runs the full pipeline.
  action: {
    label: 'Action', icon: '', statuses: STATUSES, children: false, session: false,
    def: 'action — a subtask or loose task; where the work actually happens',
  },
  // The default parent container. Tracks via its subtasks, not the pipeline, so
  // no Approval / Review / Staged — a container has no branch to review or
  // stage. `Stand By` and `Closed` apply exactly as they do to a task: Kevin can
  // park one, and he closes one out himself. THIS LIST IS THE PRE-#261
  // `VISION_STATUSES`, unchanged — no vision row's dropdown moves on landing.
  vision: {
    label: 'Vision', icon: '◈', children: true, session: false,
    statuses: ['Backlog', 'Scoping', 'Todo', 'In Progress', 'Blocked', 'Stand By', 'Done', 'Closed'],
    def: 'vision — the default parent container; tracks via its subtasks',
  },
  // A DIFFERENT kind of parent container, and a SIBLING of vision — a goal is
  // never filed under one. It gets `Approval` where a vision does not: a goal is
  // authorised to START (Kevin stamps it), but a goal never ships — its children
  // do, which is why it has no Review/Staged either. `Scoping` is load-bearing:
  // #262 adds the rule that a goal cannot leave it without a `done_when`.
  goal: {
    label: 'Goal', icon: '⚑', children: true, session: true,
    statuses: ['Backlog', 'Scoping', 'Approval', 'In Progress', 'Blocked', 'Stand By', 'Done', 'Closed'],
    def: 'goal — a parent container that carries its own session',
  },
};

/** The vocabulary, in declaration order. Mirrors the DB CHECK constraint. */
export const TASK_TYPE_VALUES = Object.keys(TASK_TYPES) as TaskType[];
/** The column default — and what a pre-migration row reads as. */
export const DEFAULT_TASK_TYPE: TaskType = 'action';

/**
 * The type a row IS.
 *
 * The explicit `task_type` wins — EXCEPT that it can never DEMOTE a row today's
 * derivation calls a container. That exception is the behaviour-preservation
 * guarantee, not a hedge: the backfill matched `isVisionTask()` exactly at one
 * MOMENT, and a row that gains children (or the vision tag) afterwards is
 * precisely the row it could not cover. Without the rescue such a parent renders
 * as an `action` row, drops out of `visionItems`, and its subtasks — filtered out
 * of `looseTasks` by `parent_id != null` — VANISH from the board entirely.
 *
 * It only ever upgrades action → vision. An explicit `vision`/`goal` is always
 * honoured, and a `goal` needs no rescue: it is already a container.
 */
export const taskTypeOf = (t: Task, hasChildren = false): TaskType => {
  const derivedContainer = (t.tags || []).includes('vision') || hasChildren;
  const explicit = t.task_type || '';
  if (!(explicit in TASK_TYPES)) {
    // Pre-migration row / API not serving the column / a value outside the
    // vocabulary: today's derivation, whole. Never a silent 'action'.
    return derivedContainer ? 'vision' : 'action';
  }
  const type = explicit as TaskType;
  if (derivedContainer && !TASK_TYPES[type].children) return 'vision';
  return type;
};

/** The statuses a type may hold. Unknown type → the full pipeline, never an
 *  empty dropdown (a row must always be able to render its own status). */
export const statusesForType = (type: TaskType): string[] =>
  TASK_TYPES[type]?.statuses || STATUSES;

/* ── THE GOAL TYPE (board #262) ─────────────────────────────────────────────
 * #261 gave `goal` a row in the type map. This gives it its BEHAVIOUR: the
 * parameters that define a goal, and the context block a goal session is
 * started from.
 *
 * WHAT THIS IS NOT: a `/goal` skill, a runner or a scheduler. Kevin uses
 * Claude's BUILT-IN `/goal` (his ruling, 07-31) — so everything here is
 * INPUT to that command, never a replacement for it. The built-in's own prompt
 * cannot be introspected from here, which is exactly why `renderGoalBlock`
 * states the objective, the end condition and the rails OUTRIGHT rather than
 * assuming the command will ask for them.
 *
 * The block is rendered CLIENT-SIDE and pasted. That is deliberate: the render
 * IS the gate (see `renderGoalBlock`), and a goal session cannot start without
 * it. */

/** A goal's autonomy. NOT a `goal_params` key — see `autonomyOf`. */
export type Autonomy = 'approve_each' | 'standing';
export const AUTONOMY_VALUES: Autonomy[] = ['approve_each', 'standing'];
export const AUTONOMY_DEF: Record<Autonomy, string> = {
  approve_each: 'approve_each — Kevin approves each task before it runs; the '
    + 'goal session creates it, assigns it to `kevin`, and waits',
  standing: 'standing — the goal session may create AND dispatch tasks within '
    + 'its scope without a per-task approval; it still reports',
};

/**
 * AUTONOMY LIVES IN THE `standing_approval` COLUMN, NOT IN `goal_params`.
 *
 * `standing_approval` is a real column with a checkbox in the Config tab that
 * NOTHING has ever read (shipped reserved by board #136). A goal's `autonomy`
 * is precisely what it was reserved for, so it is wired — not duplicated. A
 * second `goal_params.autonomy` would be two switches for one decision, and
 * they would disagree the first time one of them was set.
 *
 * The column WINS unconditionally: a stray `goal_params.autonomy` (hand-PATCHed,
 * or copied from M's spec table) is ignored on read and stripped on write.
 */
export const autonomyOf = (t: Task): Autonomy =>
  t.standing_approval ? 'standing' : 'approve_each';

/** The patch that SETS autonomy — the column, never a params key. */
export const autonomyPatch = (v: Autonomy): { standing_approval: boolean } =>
  ({ standing_approval: v === 'standing' });

/**
 * A child's effective autonomy: INHERITED from its goal unless the child
 * overrides it.
 *
 * "Unless overridden" needs a way to tell "not set" from "set to approve_each",
 * and a boolean column cannot carry three states — so the override is explicit:
 * `goal_params.autonomy_override` on the CHILD. Absent = inherit. This is the
 * one place a params key may name autonomy, and it is a child key, not a goal
 * key: it never shadows a goal's own column.
 */
export const inheritedAutonomy = (child: Task, goal: Task | null): Autonomy => {
  const override = String(goalParamsOf(child).autonomy_override || '').trim();
  if (override === 'standing') return 'standing';
  if (override === 'approve_each') return 'approve_each';
  return goal ? autonomyOf(goal) : autonomyOf(child);
};

/** One editable goal parameter. `column` = it is stored on a real column, not
 *  inside the `goal_params` JSONB. */
export interface GoalFieldDef {
  key: string;
  label: string;
  required: boolean;
  kind: 'text' | 'textarea' | 'select' | 'list';
  hint: string;
  options?: string[];
  column?: string;
}

/** Lanes a goal may dispatch to when it names none (M's spec default). */
export const DEFAULT_GOAL_LANES = ['E', 'F', 'M-A'];

/**
 * The `goal_params` shape, field by field — M's step-1 spec, verbatim.
 * DECLARATIVE: the editor and `missingGoalParams` are both rendered from this,
 * so adding a field is one entry here, not an entry plus two hand-edits.
 */
export const GOAL_PARAM_FIELDS: GoalFieldDef[] = [
  { key: 'objective', label: 'objective', required: true, kind: 'text',
    hint: 'one sentence — what "achieved" means' },
  { key: 'done_when', label: 'done_when', required: true, kind: 'textarea',
    hint: 'the OBSERVABLE end condition. A goal without one never ends — the '
      + 'context block refuses to render until this is filled in' },
  { key: 'autonomy', label: 'autonomy', required: true, kind: 'select',
    options: AUTONOMY_VALUES, column: 'standing_approval',
    hint: 'writes the EXISTING standing_approval column — not a params key' },
  { key: 'scope', label: 'scope', required: false, kind: 'textarea',
    hint: 'what it may touch' },
  { key: 'lanes', label: 'lanes', required: false, kind: 'list',
    hint: `which agents it may dispatch to (comma-separated) — default ${DEFAULT_GOAL_LANES.join(', ')}` },
  { key: 'boundaries', label: 'boundaries', required: false, kind: 'textarea',
    hint: 'goal-specific ADDITIONS. The standing rails are emitted whether or '
      + 'not this is filled in — they are not the author’s to omit' },
  { key: 'oversight', label: 'oversight', required: false, kind: 'textarea',
    hint: 'Kevin’s checkpoints beyond M’s standing review' },
];

/** The params keys actually stored in the JSONB (autonomy is a column). */
export const GOAL_PARAM_KEYS = GOAL_PARAM_FIELDS
  .filter((f) => !f.column).map((f) => f.key);

export interface GoalParams {
  objective?: string;
  done_when?: string;
  boundaries?: string;
  scope?: string;
  lanes?: string[] | string;
  oversight?: string;
  /** CHILD-only (see `inheritedAutonomy`). Never read on the goal itself. */
  autonomy_override?: string;
  [k: string]: unknown;
}

/** `goal_params` as an object, never null — a missing column reads as `{}`. */
export const goalParamsOf = (t: Task): GoalParams => {
  const p = t.goal_params;
  return (p && typeof p === 'object' && !Array.isArray(p)) ? p as GoalParams : {};
};

/** The lanes a goal may dispatch to, defaulted DECLARATIVELY (the default is
 *  written into the block, so the session never has to guess). */
export const goalLanes = (t: Task): string[] => {
  const raw = goalParamsOf(t).lanes;
  const list = Array.isArray(raw)
    ? raw : String(raw || '').split(',');
  const clean = list.map((s) => String(s).trim()).filter(Boolean);
  return clean.length ? clean : DEFAULT_GOAL_LANES;
};

/** The write that SETS one params key. Strips `autonomy` unconditionally: it
 *  belongs to the column, and a params copy is the parallel field this task
 *  exists to not create. */
export const goalParamsPatch = (t: Task, key: string, value: unknown) => {
  const next = { ...goalParamsOf(t) };
  if (value === '' || value === null || value === undefined) delete next[key];
  else next[key] = value;
  delete next.autonomy;
  return { goal_params: next };
};

/** One `goal_params` value as trimmed text. A value may be absent, or an array
 *  (`lanes`); the block wants a string either way, and a blank is a blank. */
export const goalVal = (p: GoalParams, k: string): string =>
  String(p[k] ?? '').trim();

/** REQUIRED params that are still blank. `autonomy` can never be missing — the
 *  column is a boolean, so it always resolves to one of the two values. */
export const missingGoalParams = (t: Task): string[] => {
  const p = goalParamsOf(t);
  return GOAL_PARAM_FIELDS
    .filter((f) => f.required && !f.column)
    .filter((f) => !String(p[f.key] ?? '').trim())
    .map((f) => f.key);
};

/**
 * THE NESTING GUARD — a goal is a TOP-LEVEL container.
 *
 * A goal contains action tasks; it does not sit inside another goal, and it
 * does not sit inside a vision. Both directions are one rule: a `goal` row may
 * not carry a `parent_id`. Returns the refusal text, or null when it is legal.
 *
 * Mirrored server-side in `_validate_team_fields` — the UI states the rule, the
 * API enforces it, and neither is the other's only line of defence.
 */
export const goalNestError = (type: TaskType, parentId: number | null | undefined): string | null =>
  (type === 'goal' && parentId != null)
    ? `a goal is a top-level container — it cannot be filed under #${parentId} `
      + '(not under a vision, and not under another goal). Its own children are '
      + 'the action tasks it creates.'
    : null;

/** The rails EVERY goal inherits, emitted whether or not `boundaries` is set —
 *  they are not the author's to omit. Kevin's ruling (a goal drives E for every
 *  Railway action, flag flip and migration) is deliberately FIRST. */
export const GOAL_STANDING_RAILS = [
  '**Every Railway action, flag flip and migration goes through E.** Create a '
  + 'task for E; do not run them yourself. This rail is not widened for goals.',
  'Never force-push. Never reset `dev` or `main`.',
  'Never apply a migration. Author it, hand it to M, Kevin authorizes by name.',
  'Never edit the main checkout — a live dispatcher loop runs from it.',
  'Never mark a task `Closed`. That is Kevin’s alone.',
  'Never restart the dispatcher loop.',
];

export interface GoalBlockResult {
  ok: boolean;
  /** Why it refused. Present iff `ok` is false. */
  refusal: string | null;
  /** The pasteable block. Present iff `ok` is true — NEVER a partial block. */
  block: string | null;
  missing: string[];
}

/**
 * THE CONTEXT BLOCK — the text Kevin pastes into a session running Claude's
 * built-in `/goal`. M's step-1 template, rendered.
 *
 * IT REFUSES WITHOUT `done_when`, AND THAT REFUSAL IS THE WHOLE GATE.
 * A goal with no observable end condition never ends. Enforcing it HERE rather
 * than at the API (M's step-1 decision, argued and kept) puts the gate where it
 * actually bites — a goal session cannot be started without this block — while
 * still letting Kevin SAVE a half-written goal, which is what a draft is. It
 * also keeps #261's fence: no API-level rejection is added to this arc without
 * a deliberate decision (#168 owns that).
 *
 * Refusal is TOTAL: `block` is null, not a block with a hole in it. A
 * half-rendered block is the one output that would get pasted anyway.
 */
export const renderGoalBlock = (
  t: Task, type: TaskType, apiHost = '<api-host>',
): GoalBlockResult => {
  if (type !== 'goal') {
    return { ok: false, block: null, missing: [],
      refusal: `#${t.id} is a \`${type}\` task — only a goal has a goal block.` };
  }
  const missing = missingGoalParams(t);
  if (missing.length) {
    const head = missing.includes('done_when')
      ? 'REFUSED — this goal has no `done_when`. A goal with no observable end '
        + 'condition never ends, so the block that starts one will not render '
        + 'without it.'
      : 'REFUSED — this goal is missing a required parameter.';
    return { ok: false, block: null, missing,
      refusal: `${head} Missing: ${missing.join(', ')}. Fill it in above, then `
        + 'the block renders.' };
  }
  const p = goalParamsOf(t);
  const autonomy = autonomyOf(t);
  const autonomyPara = autonomy === 'standing'
    ? 'You have standing approval to create and dispatch tasks within your\n'
      + 'scope. You do NOT need Kevin’s approval per task. You still report.'
    : 'Kevin approves each task before it runs. Create the task, assign\n'
      + 'it to `kevin`, and WAIT. Do not dispatch work he has not approved.';
  const extra = goalVal(p, 'boundaries');
  const block = [
    `You are the session for GOAL #${t.id} — ${t.title}.`,
    '',
    '## The goal',
    goalVal(p, 'objective'),
    '',
    '## Done when',
    goalVal(p, 'done_when'),
    '',
    'This is the ONLY definition of finished. When it is true, say so and stop. If you',
    'come to believe it is unreachable or wrong, say THAT and stop — do not quietly',
    'substitute a goal you can reach.',
    '',
    '## Your container',
    `Board task #${t.id} is your container. Every task you create in pursuit of this`,
    `goal is a CHILD of it: set \`parent_id\` to ${t.id}. Do not create top-level tasks.`,
    `Board: /admin/tasks · API: ${apiHost}/api/dev-tasks`,
    '',
    `## Autonomy — ${autonomy}`,
    autonomyPara,
    'Children of this goal inherit this autonomy unless their own',
    '`goal_params.autonomy_override` says otherwise.',
    '',
    '## Scope',
    goalVal(p, 'scope') || '(not narrowed — stay inside what the objective plainly implies, and ask if unsure)',
    '',
    '## Lanes you may dispatch to',
    `${goalLanes(t).join(', ')} — assign a task to that letter and the dispatcher picks it up. You do`,
    'not do the deep work yourself; you decompose it and route it.',
    '',
    '## Oversight',
    `M reviews your output and owns the chain structure. ${goalVal(p, 'oversight')}`.trim(),
    'Report to Kevin when the goal is achieved, when it is blocked, and when you',
    'learn something that changes what "achieved" means.',
    '',
    '## Boundaries — never, regardless of what seems reasonable',
    ...GOAL_STANDING_RAILS.map((r) => `- ${r}`),
    ...(extra ? [extra] : []),
    '',
    '## Sign of life',
    // The letter is validated server-side by `parse_letter`
    // (^[A-Z](?:[0-9]|-[A-Z])?$), so `G<id>` would 400 and the session would
    // look dead forever. The KEY stays `G`; the goal id rides in `actor`,
    // which is a free string. See the report on #262.
    `POST ${apiHost}/api/r-session/heartbeat {"letter": "G", "actor": "goal-${t.id}"} each`,
    'cycle. If it goes stale, the dashboards show NOT RUNNING — which is correct.',
    'Do not fake it.',
    '',
    '## Start here',
    'Read your container task and its existing children before creating anything.',
    'Say what you plan to do first, then do it.',
  ].join('\n');
  return { ok: true, refusal: null, block, missing: [] };
};

/* ── THE ONE SHARED "FINISHED" SET (board #232) ─────────────────────────────
 * Two statuses mean a task is OVER: `Done` (M's close) and `Closed` (Kevin's
 * retrospective look, which comes after it). Everything asking "is this task
 * finished?" — open counts, hide-done filters, roll-ups, dependency checks,
 * dimming — reads THIS, never a bare `status === 'Done'`.
 *
 * WHY IT IS A CONSTANT AND NOT 14 COMPARISONS: `Done` meant "finished" in ~14
 * load-bearing places. Adding `Closed` by hand-editing each one is how one gets
 * missed, and the one that matters fails SILENTLY — the dispatcher's `done_ids`
 * resolves `blocked_by`, so a blocker moving Done → Closed would drop out and
 * re-block every dependent task with no error. Mirrored server-side in
 * dispatcher.FINISHED_STATUSES and api/routers/dev_tasks.FINISHED_STATUSES.
 *
 * NOT "is this dispatchable" — `Blocked`, `Staged` and `Stand By` are undispatchable
 * but emphatically NOT finished, and must keep blocking anything that depends
 * on them. Adding status number three should be a one-line change here. */
export const FINISHED_STATUSES = ['Done', 'Closed'];
export const isFinished = (status?: string | null): boolean =>
  !!status && FINISHED_STATUSES.includes(status);
// One-liners VERBATIM from Session_Charters.md §7 (Kevin+M 07-25 kanban
// lifecycle — supersedes the 07-23 set) — shown as tooltips on every status
// select. Charter changes re-sync here.
export const STATUS_DEF: Record<string, string> = {
  'Backlog': 'captured, not next',
  'Scoping': 'M fleshes out purpose/plan/impact so Kevin can judge it',
  'Approval': 'awaiting Kevin\u2019s stamp; NOTHING runs from here. ONE stamp (board #232): \u201capproved to start\u201d \u2014 permission to execute, not a verdict that the outcome looks great',
  'Todo': 'approved and queued; dispatch-eligible',
  'In Progress': 'actively worked or dispatch-claimed',
  'Review': 'output done, awaiting M\u2019s sign-off',
  'Staged': 'reviewed, brief held, waiting for a release train (trains ship everything Staged)',
  'Blocked': 'anywhere-exception, blocker named \u2014 something PREVENTS progress and someone should look',
  'Stand By': 'Kevin has SEEN it, wants it, and chose NOT YET \u2014 deliberately parked, no action needed from anyone. Not Blocked: nothing is preventing it, so it must not sit in the queue of things someone should come look at',
  'Done': 'shipped/closed by M, never self-set by the agent that did the work \u2014 then it goes to Kevin to Close',
  'Closed': 'Kevin\u2019s version of done, AFTER Done \u2014 his retrospective look at finished work, so he sees what shipped and can spawn a follow-up while it is fresh. Blocks NOTHING: the work is already live by then',
};
export const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
// Team roles per Session_Charters.md §1. Legacy values ('claude', …) still
// render: selects append any unknown current value instead of blanking it.
// NOT the whole assignee vocabulary since board #289 — per-goal lanes (`G281`)
// are derived, not listed (see openGoalLanes / rolesWithGoalLanes). This stays
// the FALLBACK the selects use when the agents registry fetch fails.
export const ASSIGNEES = ['', 'M', 'M-A', 'E', 'E2', 'F', 'P', 'R', 'R-A', 'kevin'];
export const ORIGINS = ['planned', 'discovered', 'kevin'];
export const AUTHOR_LS_KEY = 'ror_task_comment_author';
// RETIRED by the board-#136 pipeline: needs-review became the Review STATUS;
// needs-approval/kevin-ok became the Approval stage + stamp buttons. Kept
// only so lingering data rows stay hidden from the chip cluster until M's
// one-time post-ship tag migration removes them.
export const RETIRED_TAGS = ['needs-review', 'needs-approval', 'kevin-ok'];
// Board #109: the Run button is DECLARATIVE — this tag is the request; the
// LOCAL dispatcher --loop polls for it and executes (Railway cannot reach
// Kevin's machine). Cleared by the dispatcher on claim.
export const RUN_REQUESTED_TAG = 'run-requested';
export const COLLAPSED_LS_KEY = 'ror_board_collapsed_visions';
// New-stage colors track who acts there: Approval gold = Kevin's stamp inbox
// (his role color), Review sky = eyes on output, Staged teal = R's
// release-train queue (R's role color).
// Board #232: `Stand By` is a MUTED STEEL BLUE, deliberately calm and as far
// from Blocked's red as the palette gets — the two must never be mistaken for
// each other at a glance, because that is the whole reason the status exists.
// `Closed` is a DEEPER green than Done's: same family (it is finished), one
// step further along.
export const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Scoping': '#a855f7', 'Approval': '#c9a227',
  'Todo': 'var(--blue)', 'In Progress': 'var(--amber, #d98c00)', 'Review': '#0ea5e9',
  'Staged': '#2aa8a0', 'Blocked': 'var(--red)', 'Stand By': '#6b83a8',
  'Done': 'var(--green)', 'Closed': '#1f7a4d',
};
// Impact = blast-radius chip next to status (board #136), M-editable,
// prominent in the Approval view. Colors escalate with radius.
export const IMPACTS = ['contained', 'app', 'engine', 'live'];
export const IMPACT_COLOR: Record<string, string> = {
  contained: 'var(--text-tertiary)', app: 'var(--blue)',
  engine: 'var(--amber, #d98c00)', live: 'var(--red)',
};
export const IMPACT_DEF: Record<string, string> = {
  contained: 'contained — this lane only, no shared surfaces',
  app: 'app — app-wide surface (UI/API), user-visible',
  engine: 'engine — engine code paths, correctness risk',
  live: 'live — touches live trading, deploy carefully',
};
/* ── PER-GOAL ASSIGNEE LANES (board #289) ───────────────────────────────────
 * A goal task carries its OWN SESSION (`TASK_TYPES.goal.session`), and Kevin's
 * ruling 08-01 is that the lane is PER-GOAL — `G281`, not one shared `G`:
 * *"the point of a goal is to have a dedicated session fully focused on that
 * goal."* So the lane vocabulary is UNBOUNDED, and `ASSIGNEES` — a fixed array
 * — structurally cannot express it. The G-lanes are DERIVED from the goal rows
 * the board already holds, and `roleColor` gets a PREFIX rule rather than N map
 * entries, for the same reason.
 *
 * THIS IS THE UI HALF ONLY. They already worked in the DATA: the API whitelists
 * which FIELDS a caller may set and validates `origin`/`task_type`, but it has
 * no `assignee` vocabulary at all — which is how #281's goal session set itself
 * to `G` within minutes of the first goal being spawned, with no UI for it.
 *
 * ⛔ NO `G*` ROW IS ADDED TO THE AGENTS REGISTRY, and that omission is the
 * safety rail, not an oversight. `dispatcher.headless_agents()` selects
 * `status=eq.headless`, and `triage_todo()` refuses any assignee absent from
 * that dict as *"waiting on G281 (not a headless agent)"* — is_stuck=False, so
 * the task sits QUIETLY and the loop never tries to spawn a goal session.
 * KEVIN spawns goals. A registry row is precisely what would arm the loop to do
 * it instead, which is why the fix lives here and not there.
 *
 * The heartbeat key is the precedent, not a contradiction: `session_heartbeat_G`
 * keys on the BARE letter because `parse_letter` (^[A-Z](?:[0-9]|-[A-Z])?$)
 * would 400 on `G281`, so #262 rides the goal id in the free-string `actor`.
 * `assignee` has no such validator, so the lane carries the id outright. */
export const GOAL_LANE_PREFIX = 'G';
/** The lane a goal task's session runs as. */
export const goalLaneFor = (t: Task): string => `${GOAL_LANE_PREFIX}${t.id}`;
/** `G` + digits and nothing else. A BARE `G` is deliberately NOT a lane (that
 *  is the shared-lane shape Kevin ruled out), and neither is some future
 *  registry letter that merely starts with G. */
export const isGoalLane = (role?: string | null): boolean =>
  /^G\d+$/.test(String(role || ''));
/**
 * The G-lanes worth OFFERING: one per UNFINISHED goal on the board, lowest id
 * first. A finished goal drops out — its session is over — but a task still
 * parked on that lane keeps rendering it, because every assignee select wraps
 * its options in `withLegacy`. That is the same treatment a retired role gets,
 * and it is why this list can shrink without blanking a dropdown.
 */
export const openGoalLanes = (tasks: Task[] | null | undefined): string[] =>
  (tasks || [])
    .filter((t) => taskTypeOf(t) === 'goal' && !isFinished(t.status))
    .sort((a, b) => a.id - b.id)
    .map(goalLaneFor);
/** Registry lanes + the board's open goal lanes = what an ASSIGNEE select
 *  offers. Deliberately NOT what the @-mention picker offers: the API only
 *  writes a `task_mentions` row for a real registry letter, so an `@G281` would
 *  post and then silently reach nobody (see TaskDetailModal.mentionRoles). */
export const rolesWithGoalLanes = (
  roles: string[], tasks: Task[] | null | undefined,
): string[] => {
  const add = openGoalLanes(tasks).filter((g) => !roles.includes(g));
  return add.length ? [...roles, ...add] : roles;
};

// A split lane's headless half takes a LIGHTER tint of the same hue (M/M-A,
// R/R-A) — same family reads as "same lane", the tint is only the secondary
// cue. The primary one is shape, set by the registry (see isSessionLane).
export const ROLE_COLOR: Record<string, string> = {
  M: '#7c5cff', 'M-A': '#a48cff', E: '#d9534f', E2: '#e08a3c', F: '#3b82f6',
  P: '#2e9e5b', R: '#2aa8a0', 'R-A': '#5fc9c2', kevin: '#c9a227',
  claude: '#64748b', system: '#556070',
};
/** ONE colour for EVERY per-goal lane (board #289) — magenta, the open slot in
 *  the palette: far from E's red, F's blue and M's violet at chip size. It is a
 *  PREFIX rule below rather than a map entry because there is no finite key set
 *  to enumerate; goals are created at will. Goal lanes do not tint by id: the
 *  hue says "a goal session owns this", the LABEL says which goal. */
export const GOAL_LANE_COLOR = '#d4499b';
export const roleColor = (r: string) =>
  ROLE_COLOR[r] || (isGoalLane(r) ? GOAL_LANE_COLOR : '#64748b');
export const roleAbbrev = (r: string) =>
  r === 'kevin' ? 'K' : r === 'claude' ? 'C' : r === 'system' ? '⚙'
    : r === 'M-A' ? 'MA' : r === 'R-A' ? 'RA' : r;

/* ── Lane kind: session vs headless agent (board #222) ──────────────────────
 * `M` used to mean two actors at once — the long-running manager SESSION and
 * the headless engineer that builds the team's own tooling — and the board
 * could not tell them apart, so twice on 07-30 an ARMED task assigned `M`
 * dispatched `M·auto` into work reserved for the session. The fix is a registry
 * split (M = live-session, M-A = headless); this is the display half of it.
 *
 * The rule is read from the REGISTRY, never from a letter check: any owner whose
 * row says `status='live-session'` (or `kind='human'`) renders as a session. So
 * the next lane that goes live-session inherits the treatment for free, and this
 * file needs no edit for it.
 *
 * Board #229 CASHED THAT CHEQUE: splitting `R` into R (session conductor) +
 * `R-A` (headless executor) needed no change below this comment at all — only a
 * colour, an abbreviation and two list entries above it. Everything that decides
 * session-vs-agent already reads the registry, so the second split cost nothing,
 * which is the evidence the claim above was true rather than merely intended. */
export interface AgentMeta {
  letter: string;
  status: string;                 // live-session | headless | dormant | …
  kind?: string;                  // agent | human
}

/** letter → registry row, for everything that renders an owner. Empty by
 *  default so a component outside the provider still renders (fallback below). */
export const AgentRegistryContext = React.createContext<Map<string, AgentMeta>>(
  new Map());
export const useAgentRegistry = () => React.useContext(AgentRegistryContext);

/** Convenience for pages that already fetched `/api/agents`. */
export const agentRegistryMap = (rows: AgentMeta[] | null | undefined) =>
  new Map((rows || []).filter((a) => a && a.letter).map((a) => [a.letter, a]));

/** Owners that render as a session when the registry is UNREACHABLE. Mirrors
 *  the dispatcher's own stub posture (AdminTasksPage does the same for the
 *  headless set): a fallback for an API outage, not the rule. Kevin is a human;
 *  M is the session the #222 split created and R the one #229 creates — being
 *  wrong in this direction is the safe way to be wrong, because it
 *  under-promises dispatch. This set is ONLY consulted when the registry
 *  answered with nothing, so while a lane's `live-session` flip is still held
 *  the live board keeps reading the true (headless) status from the row. */
const SESSION_FALLBACK = new Set(['kevin', 'M', 'R']);

/** True when this owner is a live session / human — i.e. NEVER auto-dispatched;
 *  a task parked here waits on a person, which is the designed state. */
export const isSessionLane = (role: string,
                              reg: Map<string, AgentMeta>): boolean => {
  // Board #289 — the ONE class the registry cannot answer for, because its
  // absence from the registry IS the design (see PER-GOAL ASSIGNEE LANES). A
  // lookup miss falls through to "agent", whose tooltip reads *"the dispatcher
  // can auto-run this lane"* — the single thing that is never true of a goal:
  // Kevin spawns them, the loop never does. So it is STATED here, since the
  // registry has no row with which to state it. This is not the letter check
  // the comment above forbids: that one asked the wrong source for data the
  // registry HAS. Every registry-backed lane below is untouched.
  if (isGoalLane(role)) return true;
  const row = reg.get(role);
  if (!row) return reg.size === 0 && SESSION_FALLBACK.has(role);
  return row.status === 'live-session' || row.kind === 'human';
};

export const laneKindLabel = (role: string, reg: Map<string, AgentMeta>) =>
  isSessionLane(role, reg) ? 'session' : 'agent';

export const LANE_KIND_TIP: Record<string, string> = {
  session: 'live session — the dispatcher never auto-runs this lane; the task '
    + 'waits on a person at a keyboard (registry status: live-session/human)',
  agent: 'headless agent — the dispatcher can auto-run this lane '
    + '(registry status: headless)',
  // Board #289 — a per-goal lane is a session, but not one the REGISTRY knows:
  // it has no row there by design. Both parentheticals above would therefore be
  // false of it, and "waits on a person at a keyboard" doubly so, so it gets
  // its own line instead of being bent to fit one of theirs.
  goal: 'goal session — the session dedicated to that goal task. The dispatcher '
    + 'never auto-runs it: Kevin spawns a goal, the loop cannot (no registry '
    + 'row, deliberately — board #289)',
};

export const withLegacy = (list: string[], current: string) =>
  list.includes(current) ? list : [...list, current];

/** Status options for a select — the TYPE's set (board #261), always keeping a
 *  legacy/current value renderable.
 *
 *  `withLegacy` STAYS. 236 rows predate the type map and some hold a status no
 *  type lists; dropping it would blank their dropdown, which is also why the API
 *  does not enforce the set. */
export const statusOptionsFor = (t: Task, type: TaskType) =>
  withLegacy(statusesForType(type), t.status);

/**
 * The type glyph on a row (Kevin's ask: the type visible at a glance).
 * Renders NOTHING for `action` — the default row is the baseline, and an icon
 * on every row carries no information.
 */
export const TaskTypeIcon = ({ type }: { type: TaskType }) => {
  const def = TASK_TYPES[type];
  if (!def || !def.icon) return null;
  return (
    <span title={def.def} style={{
      marginRight: 5, fontSize: 12, color: 'var(--text-secondary)',
      verticalAlign: 'middle',
    }}>{def.icon}</span>
  );
};

export const cell: React.CSSProperties = { padding: '7px 8px', verticalAlign: 'middle', fontSize: 13 };
export const input: React.CSSProperties = {
  background: 'var(--bg-input)', color: 'var(--text-primary)',
  border: '1px solid var(--border)', borderRadius: 6, padding: '4px 6px', fontSize: 13,
};
export const badge = (bg: string): React.CSSProperties => ({
  display: 'inline-block', padding: '1px 6px', borderRadius: 10, fontSize: 11,
  fontWeight: 600, background: bg, color: '#fff', marginRight: 4, whiteSpace: 'nowrap',
});
export const tagChip: React.CSSProperties = {
  display: 'inline-block', padding: '0px 6px', borderRadius: 8, fontSize: 10.5,
  border: '1px solid var(--border)', color: 'var(--text-secondary)',
  marginLeft: 4, whiteSpace: 'nowrap', verticalAlign: 'middle',
};

/**
 * Universal default process chain (Kevin 07-25, charter §7): every task
 * carries a checklist; simple tasks get this 4-step handoff pipeline. New
 * subtasks auto-populate it (editable); chain-less open tasks backfill via
 * the board's "add default chain" button.
 */
export const defaultChain = (assignee?: string | null): ChecklistStep[] => [
  { text: 'Build / investigate', done: false, role: assignee || null },
  { text: 'M review', done: false, role: 'M' },
  { text: 'Kevin approval (where flagged)', done: false, role: 'kevin' },
  { text: 'Ship / close', done: false, role: 'M' },
];

/** The stamp mode the UI sends. ONE mode since board #232 — the type is a union
 *  of one on purpose, so a second mode cannot be added without a deliberate edit
 *  here and in api/routers/dev_tasks._STAMP_MODE_ALIASES. */
export type StampMode = 'start';

/**
 * THE Approval-stage stamp button (board #136, reduced to one by #232) —
 * rendered only while the task sits in Approval. It hits POST /stamp, which
 * moves the task to Todo, arms it, ticks the pending Kevin step and logs the
 * system comment.
 *
 * It WAS two buttons: "Approve — M closes" and "Approve + I review before Done"
 * (kevin_final). Kevin's ruling: the stamp is PERMISSION TO START, not a verdict
 * that the outcome looks great, so it says exactly that now. The second button
 * was a PRE-commitment to review, days before the work landed and easy to forget
 * by then; the `Closed` status replaces it with a POST check on finished work
 * that blocks nothing. Keeping both would ask him to review twice.
 * stopPropagation: the board renders it inside the title cell.
 */
export const StampButtons = ({ t, onStamp, compact = false }: {
  t: Task; onStamp: (id: number, mode: StampMode) => void; compact?: boolean;
}) => {
  if (t.status !== 'Approval') return null;
  const base: React.CSSProperties = {
    ...input, cursor: 'pointer', fontWeight: 600, whiteSpace: 'nowrap',
    fontSize: compact ? 11 : 12.5, padding: compact ? '1px 7px' : '3px 10px',
  };
  return (
    <span style={{ display: 'inline-flex', gap: 6, marginLeft: compact ? 6 : 0 }}>
      <button style={{ ...base, color: 'var(--green)', borderColor: 'var(--green)' }}
        title="stamp: approved to start — permission to execute, NOT a verdict that the outcome looks great. Task → Todo and armed; you get your look at the finished work when M moves it to Done and it comes to you to Close."
        onClick={(e) => { e.stopPropagation(); onStamp(t.id, 'start'); }}>
        ✅ Approve to start</button>
    </span>
  );
};

/** LEGACY two-touch marker (board #136): rows stamped "I review before Done"
 *  before board #232 retired that mode. Nothing SETS `kevin_final` any more, but
 *  the column, its rows and the API's close guard all still stand, so the chip
 *  keeps rendering — a task stamped two-touch must keep looking like one. */
export const TwoTouchChip = ({ t }: { t: Task }) => !t.kevin_final ? null : (
  <span style={{ ...tagChip, borderColor: '#c9a227', color: '#c9a227', fontWeight: 700 }}
    title="two-touch: Kevin stamped 'Approve + I review before Done' — only Kevin signs this off Review → Staged/Done">
    🔏 two-touch</span>
);

/** Impact select styled as a colored chip — M-editable in place (like the
 *  status select), colors escalate with blast radius. */
export const ImpactSelect = ({ t, onPick, compact = false }: {
  t: Task; onPick: (impact: string) => void; compact?: boolean;
}) => {
  const val = t.impact || 'contained';
  return (
    <select value={val} title={IMPACT_DEF[val] || 'blast radius'}
      style={{
        ...input, color: IMPACT_COLOR[val] || 'var(--text-secondary)', fontWeight: 600,
        fontSize: compact ? 11 : 13, padding: compact ? '1px 4px' : input.padding,
      }}
      onClick={(e) => e.stopPropagation()}
      onChange={(e) => onPick(e.target.value)}>
      {withLegacy(IMPACTS, val).map((i) => (
        <option key={i} value={i} title={IMPACT_DEF[i]}>{i}</option>
      ))}
    </select>
  );
};

/** Compact "how long has this been running" — 42s, 7m03s, 1h12m. */
export function elapsedShort(iso: string | null | undefined): string {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return '';
  const s = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, '0')}m`;
}

/** "2m ago" style relative timestamp for the activity feed. */
export function relTime(iso: string | undefined): string {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return '';
  const s = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 48) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

/**
 * UTC wall-clock stamp — `07-29 20:14:03Z`. Board #193 renders every timestamp
 * in UTC with an explicit `Z`, paired with relTime()'s "ago": the deploy log,
 * the dispatcher logs and Kevin's own chat are all UTC, so a local-time render
 * would desync this page from every other surface he reads.
 */
export function utcStamp(iso: string | null | undefined, withDate = true): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return '';
  const p = (n: number) => String(n).padStart(2, '0');
  const time = `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}Z`;
  return withDate ? `${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${time}` : time;
}

/** UTC calendar day of a timestamp — `2026-07-29`. Day grouping and the
 *  "started today" count both key off this, never off local midnight. */
export function utcDay(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (!Number.isFinite(d.getTime())) return '';
  return d.toISOString().slice(0, 10);
}

/** Whole hours since a timestamp; -1 when it is missing/unparseable. Age is
 *  the headline number of the shipping lane (board #166: 16-28h branch age is
 *  what turned a clean merge into Monday's conflicts). */
export function hoursSince(iso: string | null | undefined): number {
  if (!iso) return -1;
  const t = new Date(iso).getTime();
  if (!Number.isFinite(t)) return -1;
  return (Date.now() - t) / 3600000;
}

/** Age colour thresholds, calibrated on board #166's measured 16-28h:
 *  <12h neutral · ≥12h amber · ≥24h red. */
export function ageColor(hours: number): string {
  if (hours < 0) return 'var(--text-tertiary)';
  if (hours >= 24) return 'var(--red)';
  if (hours >= 12) return 'var(--amber, #d98c00)';
  return 'var(--text-secondary)';
}

/** Compact age — `4h` / `2d 3h` / `18m`. Empty when unknown. */
export function ageShort(iso: string | null | undefined): string {
  const h = hoursSince(iso);
  if (h < 0) return '';
  if (h < 1) return `${Math.max(1, Math.round(h * 60))}m`;
  if (h < 48) return `${Math.round(h)}h`;
  return `${Math.floor(h / 24)}d ${Math.round(h % 24)}h`;
}

/** Vision predicate — now the explicit `task_type` (board #261), falling back
 *  to the pre-#261 derivation (tagged 'vision' OR already has subtasks) for any
 *  row the backfill missed. See `taskTypeOf` for why the fallback can only ever
 *  promote. Vision rows are pipeline-exempt (board #136). */
export const isVisionTask = (t: Task, byParent: Map<number, Task[]>) =>
  taskTypeOf(t, byParent.has(t.id)) === 'vision';

/**
 * CONTAINER predicate — a row that HOLDS subtasks, read off the type map's
 * `children` flag rather than a hard-coded 'vision' (board #262).
 *
 * WHY THIS EXISTS: `groupBoard` used to ask `isVisionTask`, which is
 * `type === 'vision'` exactly. #261 added a SECOND container type, so a `goal`
 * with children fell through to `looseTasks` while its children — filtered out
 * of `looseTasks` by `parent_id != null` and belonging to no vision — VANISHED
 * from the board. That is the identical failure the `taskTypeOf` rescue exists
 * to prevent, one type over. Adding container type number three should now be
 * a `children: true` in the map and nothing else.
 */
export const isContainerTask = (t: Task, hasChildren: boolean) =>
  !!TASK_TYPES[taskTypeOf(t, hasChildren)]?.children;

/**
 * The board grouping — ONE implementation consumed by /admin/tasks and
 * /admin/roadmap (Spec_Admin_Roadmap.md §2): the two pages must never
 * disagree about what a vision item is. Input order (API priority order)
 * is preserved in every output list.
 *
 * `visionItems` is the CONTAINER list — vision AND goal (board #262). The key
 * keeps its name because three pages destructure it, but the predicate is
 * `isContainerTask`, not `isVisionTask`: a goal renders its children strip
 * exactly as a vision does, and without that its children are invisible.
 */
export function groupBoard(tasks: Task[]): {
  byParent: Map<number, Task[]>; visionItems: Task[]; looseTasks: Task[];
} {
  const byParent = new Map<number, Task[]>();
  tasks.forEach((t) => {
    if (t.parent_id != null) byParent.set(t.parent_id, [...(byParent.get(t.parent_id) || []), t]);
  });
  const visionItems = tasks.filter((t) => t.parent_id == null && isContainerTask(t, byParent.has(t.id)));
  const looseTasks = tasks.filter((t) => t.parent_id == null && !isContainerTask(t, byParent.has(t.id)));
  return { byParent, visionItems, looseTasks };
}

/**
 * Who acts next (spec Phase-3 amendments #3). The checklist is a handoff
 * pipeline: leaf = first un-done step's role (fallback: assignee); vision =
 * first non-Done subtask's assignee (subtasks arrive priority-ordered from
 * the API). Team practice is to REASSIGN at each handoff point, so
 * next !== assignee means a handoff is due (someone forgot to pass the ball).
 */
export function nextActor(t: Task, subtasks: Task[]): { next: string | null; handoff: boolean } {
  let next: string | null;
  if (subtasks.length > 0) {
    // FINISHED, not literally Done (board #232) — a Closed subtask is over too,
    // and treating it as open would make the vision's next-actor chip point at
    // whoever owned work Kevin already closed out.
    const firstOpen = subtasks.find((s) => !isFinished(s.status));
    next = firstOpen ? (firstOpen.assignee || null) : null;
  } else if (t.status === 'Approval') {
    next = 'kevin'; // his stamp is the gate (board #136)
  } else if (t.status === 'Review') {
    // M signs off at Review. LEGACY two-touch rows (stamped before board #232
    // retired that mode) still route to Kevin — nothing sets kevin_final now,
    // but a task stamped two-touch must keep behaving as stamped.
    next = t.kevin_final ? 'kevin' : 'M';
  } else if (t.status === 'Staged') {
    next = 'R'; // the next release train ships everything Staged
  } else {
    const firstStep = (t.checklist || []).find((s) => !s.done);
    next = (firstStep && stepOwner(firstStep)) || t.assignee || null;
  }
  const handoff = !!next && !!t.assignee && next !== t.assignee;
  return { next, handoff };
}

/**
 * Circular role avatar (ClickUp-style) — letters for now; the circle is the
 * slot a profile picture can fill later. Used for activity authors, owner
 * chips, and as the RolePicker trigger/options.
 *
 * Board #222 — SHAPE carries the lane kind, not colour. A headless agent is a
 * CIRCLE (as it always was); a live session / human is a SQUARED avatar with an
 * outer ring. Shape survives colourblindness, greyscale printing and
 * forced-colors mode, where a hue swap would say nothing; the colour and the
 * tooltip are the reinforcement, never the whole signal. Read from the registry
 * via context, so the next live-session lane inherits it with no code change.
 *
 * Board #243 — `size` (default 20, the original) exists so a surface that shows
 * TWO owners can rank them by scale: the actionable one full size, the
 * for-context one visibly demoted. Every dimension derives from `size`, the
 * #222 shape cue included, so a smaller chip is the same chip and not a second
 * implementation drifting out of sync.
 */
export const RoleChip = ({ role, title, size = 20 }: {
  role: string; title?: string; size?: number;
}) => {
  const reg = useAgentRegistry();
  const session = isSessionLane(role, reg);
  const kind = isGoalLane(role) ? 'goal' : session ? 'session' : 'agent';
  const ring = size / 20;   // the ring is part of the shape cue — it scales too
  // Board #289 — a per-goal lane's label is `G281`, not one or two glyphs, and
  // four characters do not fit a fixed 20px avatar. It grows into a PILL rather
  // than truncating, because the digits ARE the identity: two goal lanes that
  // both render "G…" are indistinguishable, which defeats the per-goal ruling.
  // Same shape family (it is already session-squared), same colour, wider box.
  const goal = kind === 'goal';
  return (
    <span title={title ? `${title} · ${LANE_KIND_TIP[kind]}`
      : `${role} — ${LANE_KIND_TIP[kind]}`} style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: goal ? 'auto' : size, minWidth: size, height: size, flex: 'none',
      padding: goal ? `0 ${Math.max(3, Math.round(size * 0.2))}px` : 0,
      borderRadius: session ? Math.max(3, Math.round(size * 0.25)) : '50%',
      boxShadow: session
        ? `0 0 0 ${1.5 * ring}px var(--bg-card, #12161c), 0 0 0 ${3 * ring}px ${roleColor(role)}`
        : undefined,
      fontSize: Math.max(7, Math.round(size * 0.45)),
      fontWeight: 700, letterSpacing: -0.3, color: '#fff',
      background: roleColor(role), whiteSpace: 'nowrap', verticalAlign: 'middle',
    }}>{roleAbbrev(role)}</span>
  );
};

/**
 * The short LABEL half of the #222 cue — for surfaces with room for words
 * (the modal owner row). "session" reads as "this waits on a person; the
 * dispatcher will never pick it up"; "agent" as "this can auto-run".
 */
export const LaneKindChip = ({ role, sessionOnly = false }: {
  role: string; sessionOnly?: boolean;
}) => {
  const reg = useAgentRegistry();
  // Board #289 — a goal lane is a session AND says which kind, so it gets its
  // own label rather than borrowing "◧ session" and the registry parenthetical
  // that comes with it. `laneKindLabel` already returns 'session' for it (via
  // isSessionLane), so this only refines the wording, never the dispatch claim.
  const kind = isGoalLane(role) ? 'goal' : laneKindLabel(role, reg);
  if (!role) return null;
  // `sessionOnly` for dense rows (chain-step lines): say something only when
  // there IS something to say — a step owned by a session is the thing that
  // silently never dispatches, and it is what #195/#219 got wrong at authoring
  // time. An agent-owned step is the unremarkable case; stay quiet.
  if (sessionOnly && kind === 'agent') return null;
  const undispatched = kind !== 'agent';
  return (
    <span title={LANE_KIND_TIP[kind]} style={{
      ...tagChip, marginLeft: 5,
      border: `1px solid ${undispatched ? roleColor(role) : 'var(--border)'}`,
      color: undispatched ? roleColor(role) : 'var(--text-secondary)',
      fontWeight: undispatched ? 700 : 400,
    }}>{kind === 'goal' ? '⚑ goal session' : kind === 'session' ? '◧ session' : '◍ agent'}</span>
  );
};

/** Dashed empty avatar circle (unassigned / "none" option in pickers). */
export const EmptyRoleCircle = ({ label = '＠' }: { label?: string }) => (
  <span style={{
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: 20, height: 20, borderRadius: '50%', flex: 'none', fontSize: 10,
    border: '1px dashed var(--border)', color: 'var(--text-secondary)',
    verticalAlign: 'middle',
  }}>{label}</span>
);

/**
 * Avatar-style role picker: the current role renders as a clickable chip;
 * clicking opens a popover row of role chips to pick from (Kevin 07-22 —
 * replaces the @ dropdowns in the modal header, composer, and step owners).
 */
export const RolePicker = ({ value, options, onPick, allowEmpty = false, pickTitle, up = false }: {
  value: string | null | undefined;
  options: string[];
  onPick: (role: string) => void;
  allowEmpty?: boolean;
  pickTitle: string;
  up?: boolean;
}) => {
  const [open, setOpen] = React.useState(false);
  const bare: React.CSSProperties = { background: 'none', border: 'none', padding: 0, cursor: 'pointer', lineHeight: 1 };
  return (
    <span style={{ position: 'relative', display: 'inline-block', verticalAlign: 'middle' }}>
      <button title={`${pickTitle}: ${value || '—'} — click to change`} style={bare}
        onClick={() => setOpen(!open)}>
        {value ? <RoleChip role={value} title={`${pickTitle}: ${value} — click to change`} />
          : <EmptyRoleCircle />}
      </button>
      {open && (
        <>
          <span onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 1500 }} />
          <span style={{
            position: 'absolute', ...(up ? { bottom: '120%' } : { top: '120%' }), left: 0, zIndex: 1600,
            display: 'flex', gap: 5, padding: '6px 8px', borderRadius: 8,
            border: '1px solid var(--border)', background: 'var(--bg-card, var(--bg-input))',
            boxShadow: '0 6px 20px rgba(0,0,0,0.4)',
          }}>
            {allowEmpty && (
              <button title="pick none" style={bare} onClick={() => { onPick(''); setOpen(false); }}>
                <EmptyRoleCircle label="—" />
              </button>
            )}
            {options.map((r) => (
              <button key={r} title={`pick ${r}`} style={bare}
                onClick={() => { onPick(r); setOpen(false); }}>
                <RoleChip role={r} />
              </button>
            ))}
          </span>
        </>
      )}
    </span>
  );
};

/**
 * Checkbox multi-select filter (board #246 — Kevin: *"can we make those
 * multi-select… you just have a checkbox next to each option"*).
 *
 * TWO invariants it exists to hold, both about not lying to the reader:
 *
 *  1. NOTHING TICKED = NO FILTER. `matchesMulti` owns that (taskFilters.js);
 *     this control simply never emits an "empty means empty" signal, and the
 *     panel footer says so in words. Eleven statuses make single-select close
 *     to useless — the common question is "show me Review AND Staged" — but a
 *     multi-select that starts by hiding everything is worse than the
 *     single-select it replaced.
 *  2. A FILTERED VIEW MUST NOT LOOK LIKE AN EMPTY BOARD. The closed control
 *     always states what it is doing — `all`, the chosen values, or a count —
 *     and goes bold/blue whenever it is constraining anything.
 */
export const MultiSelectFilter = ({
  label, options, selected, onChange, colors, labelFor, allLabel = 'all', title,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  /** optional swatch per option (statuses use STATUS_COLOR) */
  colors?: Record<string, string>;
  /** optional display name per option ('' renders as "(unassigned)") */
  labelFor?: (o: string) => string;
  allLabel?: string;
  title?: string;
}) => {
  const [open, setOpen] = React.useState(false);
  const active = selected.length > 0;
  const shown = (o: string) => (labelFor ? labelFor(o) : o) || UNASSIGNED_LABEL;
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <button
        title={title || `${label} filter — tick any number; none ticked = all ${options.length}`}
        style={{
          ...input, cursor: 'pointer', maxWidth: 260, whiteSpace: 'nowrap',
          overflow: 'hidden', textOverflow: 'ellipsis',
          fontWeight: active ? 700 : 400,
          borderColor: active ? 'var(--blue)' : 'var(--border)',
          color: active ? 'var(--blue)' : 'var(--text-primary)',
        }}
        onClick={() => setOpen(!open)}>
        {label}: {filterSummary(selected, allLabel)} ▾
      </button>
      {open && (
        <>
          <span onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 1500 }} />
          <div style={{
            position: 'absolute', top: '120%', left: 0, zIndex: 1600, minWidth: 190,
            maxHeight: 320, overflowY: 'auto', padding: '7px 9px', borderRadius: 8,
            border: '1px solid var(--border)', background: 'var(--bg-card, var(--bg-input))',
            boxShadow: '0 6px 20px rgba(0,0,0,0.4)', fontSize: 12.5,
          }}>
            {options.map((o) => (
              <label key={o || '(none)'} style={{
                display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0',
                cursor: 'pointer', whiteSpace: 'nowrap',
              }}>
                <input type="checkbox" checked={selected.includes(o)}
                  onChange={() => onChange(toggleSelected(selected, o))} />
                {colors && colors[o] && (
                  <span style={{
                    width: 8, height: 8, borderRadius: 4, background: colors[o],
                    display: 'inline-block', flexShrink: 0,
                  }} />
                )}
                <span>{shown(o)}</span>
              </label>
            ))}
            <div style={{
              marginTop: 6, paddingTop: 5, borderTop: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
            }}>
              <span style={{ color: 'var(--text-tertiary)', fontSize: 11 }}>
                {active ? `${selected.length} of ${options.length}` : `none ticked = all ${options.length}`}
              </span>
              <button style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '2px 6px' }}
                disabled={!active} onClick={() => onChange([])}
                title="clear this filter — an empty selection shows everything">clear</button>
            </div>
          </div>
        </>
      )}
    </span>
  );
};

/**
 * Board #151 contradiction: a task in Todo whose next actor is Kevin. Todo
 * means "queue-eligible", but the dispatcher's eligibility gate skips any task
 * whose next_actor isn't its assignee — and Kevin is never an assignee — so a
 * kevin-next Todo task silently sits undispatched (dispatchable=0), looking
 * approved but going nowhere. The usual cause is a stamp that moved the task to
 * Todo without ticking its Kevin checklist step (now fixed server-side); this
 * tell catches any that slip through by other paths.
 */
export const isStuckInTodo = (t: Task, subtasks: Task[]): boolean =>
  t.status === 'Todo' && nextActor(t, subtasks).next === 'kevin';

/** Loud tell for the isStuckInTodo contradiction — renders wherever chips do. */
export const StuckChip = ({ t, subtasks }: { t: Task; subtasks: Task[] }) =>
  !isStuckInTodo(t, subtasks) ? null : (
    <span title="contradiction: this task is in Todo (queue-eligible) but its next actor is Kevin — the dispatcher's next-actor gate skips it, so it sits undispatched. Tick the pending Kevin checklist step (or re-stamp) to release it."
      style={{
        ...tagChip, border: '1px solid var(--red)', color: 'var(--red)',
        fontWeight: 700,
      }}>⚠ stuck: Kevin-gated in Todo</span>
  );

/** Next-actor chip: quiet when next == assignee, alert-styled when a handoff is due. */
export const NextChip = ({ t, subtasks }: { t: Task; subtasks: Task[] }) => {
  const { next, handoff } = nextActor(t, subtasks);
  if (!next) return null;
  if (!handoff) {
    return (
      <span title={`next actor: ${next}`} style={{
        ...tagChip, color: 'var(--text-secondary)',
      }}>→ <b style={{ color: roleColor(next) }}>{roleAbbrev(next)}</b></span>
    );
  }
  return (
    <span title={`first open item is owned by ${next} but the task is assigned to ${t.assignee} — reassign at the handoff point`} style={{
      ...tagChip, border: `1px solid ${roleColor(next)}`, color: roleColor(next), fontWeight: 700,
    }}>handoff due → {roleAbbrev(next)}</span>
  );
};

/**
 * Handoff-chain avatar sequence (board #169, Kevin 07-27). Draws a leaf task's
 * process checklist as a compact left-to-right row of owner avatars — one per
 * step, in order — so the handoff pipeline is visible on the card at a glance
 * instead of hidden behind the Process tab. An invisible process stops being
 * followed; this is the at-a-glance reminder, the hover tooltip is the detail.
 *
 * It is a PURE READ of `task.checklist` — the same object the Process tab edits
 * and #167's state-machine defines — so the drawing and the chain stay in sync
 * by construction, not by discipline. (The "assignee auto-advances" behavior
 * Kevin asked for is #167's transition rule; it is intentionally NOT wired here
 * until that table exists, so the UI never hard-codes an implicit process.)
 *
 * State encoding (Kevin's "I need to see where to check myself off"):
 *   done      → filled, faded (spent)
 *   current   → filled, full-strength, haloed ring (you-are-here)
 *   upcoming  → hollow outline (not yet)
 * The whole row shares one hover tooltip spelling the chain out in words.
 * Renders nothing without a chain (e.g. vision rows, which pipeline via
 * subtasks rather than a checklist).
 */
export const HandoffChain = ({ task, size = 15 }: { task: Task; size?: number }) => {
  const steps: ChecklistStep[] = task.checklist || [];
  if (steps.length === 0) return null;
  const current = steps.findIndex((s) => !s.done);        // -1 → every step done
  const cur = current >= 0 ? steps[current] : null;
  const curOwner = cur ? stepOwner(cur) : null;
  const nowOwner = curOwner ? roleAbbrev(curOwner) : cur ? '—' : null;
  const tip = [
    'Process chain (hover-only; expand it in the Summary tab):',
    ...steps.map((s, i) => {
      const o = stepOwner(s);
      const owner = o ? roleAbbrev(o) : '—';
      const mark = s.done ? '✓' : i === current ? '▶' : '·';
      return `  ${mark} ${i + 1}. ${stepTitle(s)}  (${owner})`;
    }),
    nowOwner ? `  ▶ now: ${nowOwner}` : '  — all steps complete',
  ].join('\n');

  const base: React.CSSProperties = {
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: size, height: size, borderRadius: '50%', flex: 'none',
    fontSize: Math.round(size * 0.55), fontWeight: 700, letterSpacing: -0.3,
    boxSizing: 'border-box', whiteSpace: 'nowrap',
  };
  return (
    <span title={tip} style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      verticalAlign: 'middle', marginLeft: 6, cursor: 'default',
    }}>
      {steps.map((s, i) => {
        const state = i === current ? 'current' : s.done ? 'done' : 'upcoming';
        const o = stepOwner(s);
        const col = o ? roleColor(o) : '#64748b';
        const letter = o ? roleAbbrev(o) : '·';
        const skin: React.CSSProperties =
          state === 'current'
            ? { background: col, color: '#fff',
                boxShadow: `0 0 0 1.5px var(--bg-card, var(--bg-input)), 0 0 0 3px ${col}` }
            : state === 'done'
              ? { background: col, color: '#fff', opacity: 0.4 }
              : { background: 'transparent', color: col, border: `1.5px solid ${col}`, opacity: 0.85 };
        return <span key={i} style={{ ...base, ...skin }}>{letter}</span>;
      })}
    </span>
  );
};

/* ── Run button + run history (board #109, Registry Phase 2) ──────────── */

export const OUTCOME_STYLE: Record<string, { color: string; icon: string; label: string }> = {
  'requested': { color: 'var(--amber, #d98c00)', icon: '⏳', label: 'requested' },
  'running': { color: 'var(--blue)', icon: '⚙', label: 'running' },
  'ok': { color: 'var(--green)', icon: '✓', label: 'ok' },
  'error': { color: 'var(--red)', icon: '✗', label: 'error' },
  'lease-expired': { color: 'var(--red)', icon: '⏰', label: 'lease-expired' },
  'ignored': { color: 'var(--text-tertiary)', icon: '∅', label: 'ignored' },
};

/** Small outcome chip for run rows / last-run state. */
export const OutcomeChip = ({ outcome, title }: { outcome: string; title?: string }) => {
  const s = OUTCOME_STYLE[outcome] || { color: 'var(--text-tertiary)', icon: '?', label: outcome };
  return (
    <span title={title || `run ${s.label}`} style={{
      ...tagChip, borderColor: s.color, color: s.color, fontWeight: 600,
    }}>{s.icon} {s.label}</span>
  );
};

/* ── AI-eligible toggle + run-state pill (board #198) ─────────────────────── */
//
// One column was doing two unrelated jobs. `status` said BOTH where a task sits
// on the kanban AND whether an AI may touch it (the dispatch gate read
// `status=eq.Todo`), and the Run column mixed the eligibility control with the
// "is it running right now" readout. #198 splits both pairs along the same seam:
//
//   INPUT  — AiEligibleToggle : may an agent run this        (ai_eligible)
//   OUTPUT — RunStatePill     : what the dispatcher is doing (run_history)
//   ACTION — RunButton        : run it once, now             (run-requested tag)
//
// So they render as three visibly different things: a switch, a chip, a button.

/**
 * The AI-eligible switch — standing permission for a headless agent to pick this
 * task up wherever it sits on the board. Kevin's Approval stamp arms it
 * (POST /stamp), and this is the manual override.
 *
 * ARMING = LAUNCHING (Kevin's ruling, 07-29): flipping this on can dispatch
 * within one dispatcher poll (~20s) — "I am okay if flipping the toggle starts
 * to run in twenty seconds". No confirm, no delay, by his explicit choice; the
 * tooltip says so plainly instead.
 *
 * A switch, not a button, precisely because the Run button sits beside it: this
 * is a state you leave set, that is a one-off queue jump. Mistaking one for the
 * other is the confusion #198 exists to kill.
 */
export const AiEligibleToggle = ({ task, onToggle, compact = false }: {
  task: Task;
  onToggle: (next: boolean) => void;
  compact?: boolean;
}) => {
  const on = !!task.ai_eligible;
  const h = compact ? 16 : 18;
  const w = compact ? 30 : 34;
  const knob = h - 6;
  const tip = on
    ? 'AI-eligible: ON — a headless agent may claim this task wherever it sits on the board. '
      + 'The dispatcher can pick it up on its next poll (~20s). Click to disarm.'
    : 'AI-eligible: OFF — no agent claims this on its own. (A task in Todo still dispatches '
      + 'via the status gate, and ▶ Run still forces a one-off run.) Click to arm — it can '
      + 'dispatch within ~20s.';
  return (
    <button type="button" role="switch" aria-checked={on} aria-label="AI-eligible" title={tip}
      onClick={(e) => { e.stopPropagation(); onToggle(!on); }}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 5, verticalAlign: 'middle',
        background: 'none', border: 'none', padding: 0, cursor: 'pointer', color: 'inherit',
      }}>
      {/* The OFF knob uses --text-secondary, not --text-tertiary: on the dark
          board a tertiary knob on a --bg-input track disappeared entirely and
          the switch read as an empty capsule (caught in the #198 local preview,
          07-29). A switch that doesn't look like a switch is the same
          input/output confusion this task is fixing, one layer down. */}
      <span style={{
        display: 'inline-flex', alignItems: 'center', boxSizing: 'border-box',
        width: w, height: h, padding: 2, borderRadius: h, flex: 'none',
        justifyContent: on ? 'flex-end' : 'flex-start',
        background: on ? 'var(--green)' : 'var(--bg-input)',
        border: `1px solid ${on ? 'var(--green)' : 'var(--border)'}`,
      }}>
        <span style={{
          display: 'block', width: knob, height: knob, borderRadius: '50%', flex: 'none',
          background: on ? '#fff' : 'var(--text-secondary)',
        }} />
      </span>
      {!compact && (
        <span style={{
          fontSize: 12, fontWeight: 700, whiteSpace: 'nowrap',
          color: on ? 'var(--green)' : 'var(--text-tertiary)',
        }}>🤖 AI-eligible{on ? '' : ' · off'}</span>
      )}
    </button>
  );
};

export type RunStateKey = 'idle' | 'queued' | 'running' | 'failed';

/** Derived dispatcher state of one task: the pill's label, colour and detail. */
export interface RunState {
  key: RunStateKey;
  color: string;
  icon: string;
  label: string;
  /** short trailing context — elapsed or "ok 2h ago"; '' when there is none. */
  sub: string;
  /** the full sentence, shown as the pill's tooltip. */
  detail: string;
}

/**
 * What the dispatcher is actually doing with this task — idle / queued /
 * running / failed, plus the last outcome and when. A PURE READ of `run_history`
 * (task_id · agent_letter · run_id · requested_at · started_at · finished_at ·
 * outcome) and the run-requested tag: #198 needs no new capture, only the split.
 *
 * `running` is checked before `queued` so a task whose tag hasn't been cleared
 * yet at claim time reads as running (what is true NOW), never as merely queued.
 */
export function runState(task: Task, latest?: RunRow): RunState {
  if (latest?.outcome === 'running') {
    return {
      key: 'running', color: 'var(--blue)', icon: '⚙', label: 'running',
      sub: elapsedShort(latest.started_at || latest.requested_at),
      detail: `${latest.agent_letter}·auto working for `
        + `${elapsedShort(latest.started_at || latest.requested_at)}`
        + (latest.run_id ? ` (run ${latest.run_id})` : ''),
    };
  }
  if ((task.tags || []).includes(RUN_REQUESTED_TAG) || latest?.outcome === 'requested') {
    return {
      key: 'queued', color: 'var(--amber, #d98c00)', icon: '⏳', label: 'queued',
      sub: latest?.outcome === 'requested' ? relTime(latest.requested_at) : '',
      detail: 'run requested — the local dispatcher claims it on its next poll',
    };
  }
  if (!latest) {
    return {
      key: 'idle', color: 'var(--text-tertiary)', icon: '·', label: 'idle',
      sub: 'never run', detail: 'idle — this task has never been dispatched',
    };
  }
  const when = relTime(latest.finished_at || latest.requested_at);
  const by = latest.requested_by ? `requested by ${latest.requested_by}` : 'queue dispatch';
  const full = `last run ${latest.outcome} ${when} · ${latest.agent_letter}·auto · ${by}`
    + (latest.run_id ? ` · run ${latest.run_id}` : '');
  const style = OUTCOME_STYLE[latest.outcome];
  // A failed run must not read as a quiet "idle": these two are the states that
  // need someone's eyes, so they get the red label and keep the outcome word.
  if (latest.outcome === 'error' || latest.outcome === 'lease-expired') {
    return {
      key: 'failed', color: 'var(--red)', icon: style?.icon || '✗',
      label: latest.outcome === 'error' ? 'failed' : 'lease-expired',
      sub: when, detail: full,
    };
  }
  return {
    key: 'idle', color: 'var(--text-tertiary)', icon: style?.icon || '·',
    label: 'idle', sub: `${latest.outcome} ${when}`.trim(), detail: full,
  };
}

/**
 * The run-state pill (board #198) — the OUTPUT half of the old Run column.
 * Kevin: *"one thing I do like about the run status is that it kind of shows if
 * that task is actively being worked on."* That readout now lives in its own
 * chip, so the button next to it can go back to meaning exactly one thing.
 */
export const RunStatePill = ({ task, latestRun, compact = false }: {
  task: Task;
  latestRun?: RunRow;
  compact?: boolean;
}) => {
  const st = runState(task, latestRun);
  const busy = st.key === 'running' || st.key === 'queued';
  return (
    <span title={`run state: ${st.label} — ${st.detail}`} style={{
      ...tagChip, marginLeft: 0, borderColor: st.color, color: st.color,
      fontWeight: busy || st.key === 'failed' ? 700 : 500,
      fontSize: compact ? 10 : 10.5,
    }}>
      {st.icon} {st.label}
      {st.sub && <span style={{ opacity: 0.75, fontWeight: 500 }}> · {st.sub}</span>}
    </span>
  );
};

/**
 * Why a task can't be dispatched right now, or null when it can. Mirrors the
 * dispatcher's run-request gates (dispatcher.py run_requested): description
 * non-empty, not blocked, assignee headless-enrolled OR stub, and not
 * already In Progress / Done / Blocked. The dispatcher re-gates at claim
 * time — this is a courtesy mirror so the button disables with a reason.
 */
export function runIneligibleReason(
  t: Task, allTasks: Task[], headless: Set<string>,
): string | null {
  if (t.status === 'In Progress') return 'already In Progress';
  // FINISHED, not literally Done (board #232) — mirrors the dispatcher's
  // is_finished() refusal in run_requested().
  if (isFinished(t.status)) return `task is ${t.status}`;
  if (t.status === 'Blocked') return 'task is Blocked';
  // Kevin SAW it and chose not yet. The Run button overrides queue ORDER, never
  // a human's explicit "not yet" (board #232).
  if (t.status === 'Stand By') return 'Stand By — parked on purpose';
  // The button overrides queue ORDER, never "not workable yet" (M, #109 review).
  if (t.status === 'Scoping') return 'Scoping — not workable yet';
  // Board #136 pipeline stages — mirrored by the run-request endpoint and the
  // dispatcher's claim-time refusals.
  if (t.status === 'Approval') return 'Approval — awaiting Kevin’s stamp';
  if (t.status === 'Review') return 'Review — output awaiting sign-off';
  if (t.status === 'Staged') return 'Staged — ships with the next release train';
  if ((t.tags || []).includes('needs-scoping')) return 'tagged needs-scoping — not workable yet';
  if (!(t.description || '').trim()) return 'no description — never dispatch unscoped work';
  if (!(t.assignee || '').trim()) return 'no assignee';
  if (!headless.has(t.assignee!)) return `${t.assignee} is not headless-enrolled`;
  // THE RAIL (board #232): a blocker Kevin has since CLOSED is still finished.
  // Spelled `!== 'Done'` this silently re-blocked every dependent task the
  // moment a blocker moved Done → Closed — the same defect as the dispatcher's
  // `done_ids`, and this is the mirror the button's tooltip reads.
  const open = (t.blocked_by || []).filter((b) =>
    !isFinished(allTasks.find((x) => x.id === b)?.status));
  if (open.length > 0) return `blocked by #${open.join(', #')}`;
  return null;
}

/**
 * The Run button — a ONE-OFF QUEUE JUMP: "dispatch this task now", not "may an
 * agent run it" (that is AiEligibleToggle) and not "is it running" (that is
 * RunStatePill). Board #198 renamed it "▶ Run once" for exactly that reason: it
 * was the control Kevin was most likely to mistake for the eligibility switch,
 * and it does not change ai_eligible.
 *
 * It reads "▶ Run once" in EVERY state and only its enabled-ness changes:
 * queued, already running, ineligible → disabled with the why as tooltip. It no
 * longer relabels itself "⏳ requested" / "⚙ running…" and no longer draws the
 * last outcome, because a button that reports status is the exact input/output
 * conflation #198 exists to remove — RunStatePill owns that readout on both
 * surfaces, right next to it.
 */
export const RunButton = ({ task, allTasks, headless, latestRun, onRequest, compact = false }: {
  task: Task;
  allTasks: Task[];
  headless: Set<string>;
  latestRun?: RunRow;
  onRequest: (id: number) => void;
  compact?: boolean;
}) => {
  const requested = (task.tags || []).includes(RUN_REQUESTED_TAG)
    || latestRun?.outcome === 'requested';
  const running = latestRun?.outcome === 'running';
  const reason = runIneligibleReason(task, allTasks, headless);
  const last = latestRun && !['requested', 'running'].includes(latestRun.outcome)
    ? latestRun : null;
  const lastTitle = last
    ? `last run ${last.outcome} ${relTime(last.finished_at || last.requested_at)}`
      + (last.requested_by ? ` (requested by ${last.requested_by})` : ' (queue dispatch)')
    : '';

  const base: React.CSSProperties = {
    ...input, cursor: 'pointer', fontSize: compact ? 11 : 12, fontWeight: 600,
    padding: compact ? '1px 7px' : '3px 10px', whiteSpace: 'nowrap',
  };
  // One label, one meaning. `why` non-null = not clickable right now, and the
  // pill beside it already says which of these it is.
  const why = running
    ? `already running as ${latestRun?.agent_letter}·auto`
      + (latestRun?.run_id ? ` (run ${latestRun.run_id})` : '')
    : requested
      ? 'a run is already queued — the local dispatcher claims it on its next poll'
      : reason ? `can’t dispatch: ${reason}` : null;
  const btn = why
    ? <button disabled style={{ ...base, cursor: 'not-allowed', opacity: 0.4 }}
        title={why}>▶ Run once</button>
    : <button style={{ ...base, color: 'var(--green)', borderColor: 'var(--green)' }}
        title={`run once now: dispatch to ${task.assignee}·auto via the local dispatcher — a `
          + 'one-off queue jump; it does NOT change the 🤖 AI-eligible switch'
          + `${lastTitle ? ` — ${lastTitle}` : ''}`}
        onClick={(e) => { e.stopPropagation(); onRequest(task.id); }}>▶ Run once</button>;
  return <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>{btn}</span>;
};

/** Thin progress bar with n/m at the end; renders nothing when total is 0. */
export const ProgressBar = ({ done, total, width = 90, mini = false }: {
  done: number; total: number; width?: number; mini?: boolean;
}) => {
  if (total === 0) return null;
  const pct = Math.round((done / total) * 100);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, verticalAlign: 'middle' }}
      title={`${done}/${total} done`}>
      <span style={{
        display: 'inline-block', width: mini ? 44 : width, height: mini ? 3 : 5,
        borderRadius: 3, background: 'var(--bg-input)', border: '1px solid var(--border)',
        overflow: 'hidden',
      }}>
        <span style={{
          display: 'block', height: '100%', width: `${pct}%`,
          background: pct === 100 ? 'var(--green)' : 'var(--blue)',
        }} />
      </span>
      {!mini && <span style={{ fontSize: 10.5, color: 'var(--text-tertiary)' }}>{done}/{total}</span>}
    </span>
  );
};
