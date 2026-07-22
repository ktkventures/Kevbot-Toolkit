/**
 * Shared types, constants, and style atoms for the team-board views
 * (AdminTasksPage table + TaskDetailModal). One source for the role list —
 * adding a role is one line here.
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
export const COLLAPSED_LS_KEY = 'ror_board_collapsed_visions';
export const STATUS_COLOR: Record<string, string> = {
  'Backlog': 'var(--text-tertiary)', 'Todo': 'var(--blue)',
  'In Progress': 'var(--amber, #d98c00)', 'Blocked': 'var(--red)', 'Done': 'var(--green)',
};

export const withLegacy = (list: string[], current: string) =>
  list.includes(current) ? list : [...list, current];

export const isVision = (t: Task, childrenOf: Map<number, Task[]>) =>
  t.parent_id == null && ((t.tags || []).includes('vision') || childrenOf.has(t.id));

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
