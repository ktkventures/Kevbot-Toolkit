/**
 * Shared types, constants, style atoms, and small chips for the team-board
 * views (AdminTasksPage table + TaskDetailModal). One source for the role
 * list — adding a role is one line here (ASSIGNEES + ROLE_COLOR).
 */
import React from 'react';

export interface ChecklistStep { text: string; done: boolean; role: string | null; }

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
  updated_at: string;
}

export interface Comment { id: number; author: string; body: string; created_at: string; }

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
}

export const STATUSES = ['Backlog', 'Scoping', 'Todo', 'In Progress', 'Blocked', 'Done'];
// One-liners VERBATIM from Session_Charters.md §7 (Kevin+M, 07-23) — shown as
// tooltips on every status select.
export const STATUS_DEF: Record<string, string> = {
  'Backlog': 'real but not next, don\u2019t start',
  'Scoping': 'needs a human conversation to define, next actor Kevin/M',
  'Todo': 'scoped and workable, queue-eligible (loops/dispatcher may claim)',
  'In Progress': 'actively worked or dispatch-claimed',
  'Blocked': 'can\u2019t proceed, blocker named in thread/blocked_by',
  'Done': 'verified complete, never self-set by the agent that did the work',
};
export const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
// Team roles per Session_Charters.md §1. Legacy values ('claude', …) still
// render: selects append any unknown current value instead of blanking it.
export const ASSIGNEES = ['', 'M', 'E', 'E2', 'F', 'P', 'R', 'kevin'];
export const ORIGINS = ['planned', 'discovered', 'kevin'];
export const AUTHOR_LS_KEY = 'ror_task_comment_author';
// Tag convention: work is finished and waiting on a human — agents skip these,
// Kevin filters to them. Rendered as a distinct chip, toggled like ⚡urgent.
export const NEEDS_REVIEW_TAG = 'needs-review';
// Batch-review protocol (Kevin 07-25, charter §7): needs-approval = M requests
// Kevin's eyes BEFORE execution (amber); Kevin pre-approves by flipping it to
// kevin-ok (green) = run when bandwidth allows. needs-review stays post-run.
export const NEEDS_APPROVAL_TAG = 'needs-approval';
export const KEVIN_OK_TAG = 'kevin-ok';
// Board #109: the Run button is DECLARATIVE — this tag is the request; the
// LOCAL dispatcher --loop polls for it and executes (Railway cannot reach
// Kevin's machine). Cleared by the dispatcher on claim.
export const RUN_REQUESTED_TAG = 'run-requested';
export const COLLAPSED_LS_KEY = 'ror_board_collapsed_visions';
export const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Scoping': '#a855f7', 'Todo': 'var(--blue)',
  'In Progress': 'var(--amber, #d98c00)', 'Blocked': 'var(--red)', 'Done': 'var(--green)',
};
export const ROLE_COLOR: Record<string, string> = {
  M: '#7c5cff', E: '#d9534f', E2: '#e08a3c', F: '#3b82f6', P: '#2e9e5b',
  R: '#2aa8a0', kevin: '#c9a227', claude: '#64748b', system: '#556070',
};
export const roleColor = (r: string) => ROLE_COLOR[r] || '#64748b';
export const roleAbbrev = (r: string) =>
  r === 'kevin' ? 'K' : r === 'claude' ? 'C' : r === 'system' ? '⚙' : r;

export const withLegacy = (list: string[], current: string) =>
  list.includes(current) ? list : [...list, current];

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
 * Next tags after a pre-approve click: needs-approval → kevin-ok, kevin-ok →
 * back to needs-approval (undo). Null = neither tag present (no button).
 */
export function nextApprovalTags(t: Task): string[] | null {
  const tags = t.tags || [];
  if (tags.includes(KEVIN_OK_TAG))
    return [...tags.filter((x) => x !== KEVIN_OK_TAG && x !== NEEDS_APPROVAL_TAG), NEEDS_APPROVAL_TAG];
  if (tags.includes(NEEDS_APPROVAL_TAG))
    return [...tags.filter((x) => x !== NEEDS_APPROVAL_TAG && x !== KEVIN_OK_TAG), KEVIN_OK_TAG];
  return null;
}

/**
 * Pre-approve control (board #134 item 3) — the chip IS the button. Amber
 * "needs-approval" flips to green "kevin-ok" on click and back on a second
 * click. Renders nothing when neither tag is present (add needs-approval via
 * Config tags). stopPropagation: it lives inside the clickable title cell.
 */
export const ApprovalChip = ({ t, onToggle }: { t: Task; onToggle: (t: Task) => void }) => {
  const tags = t.tags || [];
  const ok = tags.includes(KEVIN_OK_TAG);
  if (!ok && !tags.includes(NEEDS_APPROVAL_TAG)) return null;
  return ok ? (
    <span style={{ ...tagChip, cursor: 'pointer', borderColor: 'var(--green)', color: 'var(--green)', fontWeight: 700 }}
      title="pre-approved by Kevin — run when bandwidth allows, no further check-ins absent surprises. Click to revert to needs-approval."
      onClick={(e) => { e.stopPropagation(); onToggle(t); }}>✓ kevin-ok</span>
  ) : (
    <span style={{ ...tagChip, cursor: 'pointer', borderColor: 'var(--amber, #d98c00)', color: 'var(--amber, #d98c00)', fontWeight: 700 }}
      title="M requests Kevin's eyes BEFORE execution. Click to pre-approve (flips to kevin-ok)."
      onClick={(e) => { e.stopPropagation(); onToggle(t); }}>✋ needs-approval</span>
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
  } else {
    const firstStep = (t.checklist || []).find((s) => !s.done);
    next = (firstStep && firstStep.role) || t.assignee || null;
  }
  const handoff = !!next && !!t.assignee && next !== t.assignee;
  return { next, handoff };
}

/**
 * Circular role avatar (ClickUp-style) — letters for now; the circle is the
 * slot a profile picture can fill later. Used for activity authors, owner
 * chips, and as the RolePicker trigger/options.
 */
export const RoleChip = ({ role, title }: { role: string; title?: string }) => (
  <span title={title || role} style={{
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    width: 20, height: 20, borderRadius: '50%', flex: 'none',
    fontSize: 9, fontWeight: 700, letterSpacing: -0.3, color: '#fff',
    background: roleColor(role), whiteSpace: 'nowrap', verticalAlign: 'middle',
  }}>{roleAbbrev(role)}</span>
);

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
 * The Run button. States (from run_history + tags):
 *   run-requested tag → "⏳ requested" (disabled; dispatcher will claim)
 *   latest run running → "⚙ running…" (disabled)
 *   eligible → enabled "▶ Run"; ineligible → disabled with the reason as
 *   tooltip. A terminal latest run renders as a small outcome chip alongside
 *   (compact mode folds it into the button title instead).
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
  let btn: React.ReactNode;
  if (requested) {
    btn = <button disabled style={{ ...base, cursor: 'default', color: 'var(--amber, #d98c00)', borderColor: 'var(--amber, #d98c00)' }}
      title="run requested — the local dispatcher claims it on its next poll">⏳ requested</button>;
  } else if (running) {
    btn = <button disabled style={{ ...base, cursor: 'default', color: 'var(--blue)', borderColor: 'var(--blue)' }}
      title={`running as ${latestRun?.agent_letter}·auto (run ${latestRun?.run_id || '?'})`}>⚙ running…</button>;
  } else if (reason) {
    btn = <button disabled style={{ ...base, cursor: 'not-allowed', opacity: 0.4 }}
      title={`can't dispatch: ${reason}`}>▶ Run</button>;
  } else {
    btn = <button style={{ ...base, color: 'var(--green)', borderColor: 'var(--green)' }}
      title={`dispatch to ${task.assignee}·auto via the local dispatcher${lastTitle ? ` — ${lastTitle}` : ''}`}
      onClick={(e) => { e.stopPropagation(); onRequest(task.id); }}>▶ Run</button>;
  }
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {btn}
      {!compact && last && <OutcomeChip outcome={last.outcome} title={lastTitle} />}
    </span>
  );
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
