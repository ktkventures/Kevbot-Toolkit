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
  updated_at: string;
}

export interface Comment { id: number; author: string; body: string; created_at: string; }

export const STATUSES = ['Backlog', 'Todo', 'In Progress', 'Blocked', 'Done'];
export const AREAS = ['engine', 'backtest', 'frontend', 'infra', 'data', 'docs', 'other'];
// Team roles per Session_Charters.md §1. Legacy values ('claude', …) still
// render: selects append any unknown current value instead of blanking it.
export const ASSIGNEES = ['', 'M', 'E', 'E2', 'F', 'P', 'R', 'kevin'];
export const ORIGINS = ['planned', 'discovered', 'kevin'];
export const AUTHOR_LS_KEY = 'ror_task_comment_author';
// Tag convention: work is finished and waiting on a human — agents skip these,
// Kevin filters to them. Rendered as a distinct chip, toggled like ⚡urgent.
export const NEEDS_REVIEW_TAG = 'needs-review';
export const COLLAPSED_LS_KEY = 'ror_board_collapsed_visions';
export const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Todo': 'var(--blue)',
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
