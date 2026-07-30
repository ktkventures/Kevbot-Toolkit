/**
 * Shared types, constants, style atoms, and small chips for the team-board
 * views (AdminTasksPage table + TaskDetailModal). One source for the role
 * list — adding a role is one line here (ASSIGNEES + ROLE_COLOR).
 */
import React from 'react';

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
}

// A checklist becomes a #182 process chain once ANY step carries a new-shape
// key — the exact opt-in signal the API uses (dev_tasks._NEW_SHAPE_KEYS).
// Legacy {text,role,done} checklists stay legacy and behave as before #182.
const NEW_SHAPE_KEYS: (keyof ChecklistStep)[] = ['id', 'owner', 'title', 'body', 'mode', 'origin', 'stamp'];
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
 * CONCURRENCY 3 → 4 (same change, board #219) — this mirror was missed when
 * DAILY_CAP was updated beside it, so the dashboard advertised 3 lanes while the
 * loop ran 4, and test_dispatch_dashboard_193.py went RED on dev. Both constants
 * move together: dispatcher.py is the SSOT and this line only mirrors it.
 */
export const DISPATCHER = { CONCURRENCY: 4, DAILY_CAP: 50, RUN_TIMEOUT_S: 2700, POLL_S: 20 };

// The agreed pipeline (Kevin+M 07-25, board #136): Backlog → Scoping →
// Approval → Todo → In Progress → Review → Staged → Done; Blocked is the
// anywhere-exception. Vision rows are EXEMPT (VISION_STATUSES below).
export const STATUSES = ['Backlog', 'Scoping', 'Approval', 'Todo', 'In Progress', 'Review', 'Staged', 'Blocked', 'Done'];
// Vision items track via their subtasks, not the pipeline — no Approval /
// Review / Staged on them.
export const VISION_STATUSES = ['Backlog', 'Scoping', 'Todo', 'In Progress', 'Blocked', 'Done'];
// One-liners VERBATIM from Session_Charters.md §7 (Kevin+M 07-25 kanban
// lifecycle — supersedes the 07-23 set) — shown as tooltips on every status
// select. Charter changes re-sync here.
export const STATUS_DEF: Record<string, string> = {
  'Backlog': 'captured, not next',
  'Scoping': 'M fleshes out purpose/plan/impact so Kevin can judge it',
  'Approval': 'awaiting Kevin\u2019s stamp; NOTHING runs from here (dispatcher is Todo-only by construction); Kevin stamps one of two ways: \u201cApprove \u2014 M closes\u201d or \u201cApprove + I review before Done\u201d (kevin_final)',
  'Todo': 'approved and queued; dispatch-eligible',
  'In Progress': 'actively worked or dispatch-claimed',
  'Review': 'output done, awaiting sign-off (M always; Kevin closes iff kevin_final)',
  'Staged': 'reviewed, brief held, waiting for a release train (trains ship everything Staged)',
  'Blocked': 'anywhere-exception, blocker named',
  'Done': 'shipped/closed, never self-set by the agent that did the work',
};
export const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
// Team roles per Session_Charters.md §1. Legacy values ('claude', …) still
// render: selects append any unknown current value instead of blanking it.
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
export const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Scoping': '#a855f7', 'Approval': '#c9a227',
  'Todo': 'var(--blue)', 'In Progress': 'var(--amber, #d98c00)', 'Review': '#0ea5e9',
  'Staged': '#2aa8a0', 'Blocked': 'var(--red)', 'Done': 'var(--green)',
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
// A split lane's headless half takes a LIGHTER tint of the same hue (M/M-A,
// R/R-A) — same family reads as "same lane", the tint is only the secondary
// cue. The primary one is shape, set by the registry (see isSessionLane).
export const ROLE_COLOR: Record<string, string> = {
  M: '#7c5cff', 'M-A': '#a48cff', E: '#d9534f', E2: '#e08a3c', F: '#3b82f6',
  P: '#2e9e5b', R: '#2aa8a0', 'R-A': '#5fc9c2', kevin: '#c9a227',
  claude: '#64748b', system: '#556070',
};
export const roleColor = (r: string) => ROLE_COLOR[r] || '#64748b';
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
};

export const withLegacy = (list: string[], current: string) =>
  list.includes(current) ? list : [...list, current];

/** Status options for a select: full pipeline for leaves, the exempt set for
 *  vision rows — always keeping a legacy/current value renderable. */
export const statusOptionsFor = (t: Task, isVision: boolean) =>
  withLegacy(isVision ? VISION_STATUSES : STATUSES, t.status);

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

/**
 * The two Approval-stage stamp buttons (board #136) — Kevin's two-touch
 * choice, rendered only while the task sits in Approval. 'delegate' =
 * "Approve — M closes" (→ Todo, kevin_final=false); 'final' = "Approve + I
 * review before Done" (→ Todo, kevin_final=true; only Kevin then signs off
 * Review → Staged/Done). Both hit POST /stamp, which logs the system
 * comment. stopPropagation: the board renders them inside the title cell.
 */
export const StampButtons = ({ t, onStamp, compact = false }: {
  t: Task; onStamp: (id: number, mode: 'delegate' | 'final') => void; compact?: boolean;
}) => {
  if (t.status !== 'Approval') return null;
  const base: React.CSSProperties = {
    ...input, cursor: 'pointer', fontWeight: 600, whiteSpace: 'nowrap',
    fontSize: compact ? 11 : 12.5, padding: compact ? '1px 7px' : '3px 10px',
  };
  return (
    <span style={{ display: 'inline-flex', gap: 6, marginLeft: compact ? 6 : 0 }}>
      <button style={{ ...base, color: 'var(--green)', borderColor: 'var(--green)' }}
        title="stamp: Approve — M closes. Task → Todo; M signs it off at Review."
        onClick={(e) => { e.stopPropagation(); onStamp(t.id, 'delegate'); }}>
        ✅ Approve · M closes</button>
      <button style={{ ...base, color: '#c9a227', borderColor: '#c9a227' }}
        title="stamp: Approve + I review before Done (two-touch). Task → Todo; only Kevin signs it off Review → Staged/Done."
        onClick={(e) => { e.stopPropagation(); onStamp(t.id, 'final'); }}>
        ✅👀 Approve · I review</button>
    </span>
  );
};

/** Two-touch marker: Kevin stamped "I review before Done" — his sign-off
 *  gates Review → Staged/Done. Set by the stamps; renders wherever chips do. */
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

/** Vision predicate: tagged 'vision' OR already has subtasks (pre-tag data
 *  still groups correctly). Vision rows are pipeline-exempt (board #136). */
export const isVisionTask = (t: Task, byParent: Map<number, Task[]>) =>
  (t.tags || []).includes('vision') || byParent.has(t.id);

/**
 * The board grouping — ONE implementation consumed by /admin/tasks and
 * /admin/roadmap (Spec_Admin_Roadmap.md §2): the two pages must never
 * disagree about what a vision item is. Input order (API priority order)
 * is preserved in every output list.
 */
export function groupBoard(tasks: Task[]): {
  byParent: Map<number, Task[]>; visionItems: Task[]; looseTasks: Task[];
} {
  const byParent = new Map<number, Task[]>();
  tasks.forEach((t) => {
    if (t.parent_id != null) byParent.set(t.parent_id, [...(byParent.get(t.parent_id) || []), t]);
  });
  const visionItems = tasks.filter((t) => t.parent_id == null && isVisionTask(t, byParent));
  const looseTasks = tasks.filter((t) => t.parent_id == null && !isVisionTask(t, byParent));
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
    const firstOpen = subtasks.find((s) => s.status !== 'Done');
    next = firstOpen ? (firstOpen.assignee || null) : null;
  } else if (t.status === 'Approval') {
    next = 'kevin'; // his stamp is the gate (board #136)
  } else if (t.status === 'Review') {
    // Two-touch stamp: Kevin signs off; otherwise M closes (board #136).
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
 */
export const RoleChip = ({ role, title }: { role: string; title?: string }) => {
  const reg = useAgentRegistry();
  const session = isSessionLane(role, reg);
  const kind = session ? 'session' : 'agent';
  return (
    <span title={title ? `${title} · ${LANE_KIND_TIP[kind]}`
      : `${role} — ${LANE_KIND_TIP[kind]}`} style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: 20, height: 20, flex: 'none',
      borderRadius: session ? 5 : '50%',
      boxShadow: session ? `0 0 0 1.5px var(--bg-card, #12161c), 0 0 0 3px ${roleColor(role)}` : undefined,
      fontSize: 9, fontWeight: 700, letterSpacing: -0.3, color: '#fff',
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
  const kind = laneKindLabel(role, reg);
  if (!role) return null;
  // `sessionOnly` for dense rows (chain-step lines): say something only when
  // there IS something to say — a step owned by a session is the thing that
  // silently never dispatches, and it is what #195/#219 got wrong at authoring
  // time. An agent-owned step is the unremarkable case; stay quiet.
  if (sessionOnly && kind !== 'session') return null;
  return (
    <span title={LANE_KIND_TIP[kind]} style={{
      ...tagChip, marginLeft: 5,
      border: `1px solid ${kind === 'session' ? roleColor(role) : 'var(--border)'}`,
      color: kind === 'session' ? roleColor(role) : 'var(--text-secondary)',
      fontWeight: kind === 'session' ? 700 : 400,
    }}>{kind === 'session' ? '◧ session' : '◍ agent'}</span>
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
  if (t.status === 'Done') return 'task is Done';
  if (t.status === 'Blocked') return 'task is Blocked';
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
  const open = (t.blocked_by || []).filter((b) =>
    allTasks.find((x) => x.id === b)?.status !== 'Done');
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
