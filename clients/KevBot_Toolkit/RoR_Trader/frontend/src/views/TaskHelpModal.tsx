/**
 * The board's own manual (board #297) — opened from the ✻ Help button on
 * /admin/tasks. Left nav = topics, right panel = detail.
 *
 * ── WHY THIS FILE IS WRITTEN THE WAY IT IS ─────────────────────────────────
 * The rules that govern this board live in FIVE places already: the code model
 * below, a 381-line `Session_Charters.md`, ten `.claude/skills`, M's memory
 * files, and task descriptions. A HAND-TYPED help modal becomes a SIXTH, and it
 * drifts inside a week — at which point it is worse than no manual at all,
 * because it looks authoritative and is wrong, and Kevin has explicitly said he
 * wants the agents to trust it.
 *
 * So nothing here is typed twice. Every row this modal shows is DERIVED from
 * `TASK_TYPES` / `STATUSES` / `STATUS_DEF` / `ASSIGNEES` and friends in
 * taskBoardShared.tsx. Add a task type, a status or a role there and this
 * manual grows a row on its own; change a `def` string there and the paragraph
 * here changes with it. Drift is not "avoided by discipline", it is impossible
 * by construction — which is the only version of this that survives.
 *
 * The three builders below (`helpTypeRows`, `helpStatusRows`,
 * `helpAssigneeRows`) are deliberately PURE and JSX-free so that
 * test_task_help_modal_297.py can lift them out of this file and EXECUTE them
 * in node against the real model, rather than grep for a promise. A future
 * hand-edit that hardcodes a row fails that test.
 *
 * ── AND WHY IT DOCUMENTS THE WARTS ─────────────────────────────────────────
 * Kevin's framing, 08-02: *"create the modal based on the rules and features
 * that govern everything TODAY — I can better wrap my head around what needs to
 * be changed."* This is a DIAGNOSTIC, not a brochure. Where the board's actual
 * behaviour contradicts what a reader would reasonably assume from the model,
 * `HELP_WARTS` says so outright, with the board ref and the file that proves
 * it. A manual that hides them is worse than no manual, because the V2 redesign
 * is about to be argued against exactly these.
 *
 * DELIBERATELY NOT HERE (this pass): process chains, stamps, the release/train
 * lifecycle. Those need prose, prose is a sixth copy, and #304 may invalidate
 * it. This pass is only the parts that render themselves.
 */
'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Task, AgentMeta, ASSIGNEES, STATUSES, STATUS_DEF, STATUS_COLOR,
  TASK_TYPES, TASK_TYPE_VALUES, TaskType, DEFAULT_TASK_TYPE, FINISHED_STATUSES,
  GOAL_LANE_PREFIX, LANE_KIND_TIP, isGoalLane, laneKindLabel,
  openGoalLanes, roleAbbrev, roleColor, useAgentRegistry, input, badge,
} from './taskBoardShared';

/* ── DERIVED ROWS ───────────────────────────────────────────────────────────
 * Pure, JSX-free, and exported so the acceptance test can execute them. If you
 * are tempted to inline one of these as literal JSX, that is the drift this
 * whole file exists to prevent — the test will stop you. */

export interface HelpTypeRow {
  type: string; label: string; icon: string; def: string;
  statuses: string[]; children: boolean; session: boolean;
  isDefault: boolean;
  /** Statuses the FULL pipeline has that this type does not — the interesting
   *  half of a container type, and a subtraction, never a second list. */
  missing: string[];
}

/** One row per member of the type vocabulary, in declaration order. */
export const helpTypeRows = (): HelpTypeRow[] => TASK_TYPE_VALUES.map((t) => {
  const d = TASK_TYPES[t];
  return {
    type: t,
    label: d.label,
    icon: d.icon,
    def: d.def,
    statuses: d.statuses,
    children: d.children,
    session: d.session,
    isDefault: t === DEFAULT_TASK_TYPE,
    missing: STATUSES.filter((s) => !d.statuses.includes(s)),
  };
});

export interface HelpStatusRow {
  status: string; def: string; color: string;
  /** Which types may hold it — computed, so a type whose set changes moves
   *  here on its own. A status no type lists is a LEGACY status and says so. */
  types: string[];
  finished: boolean;
  legacy: boolean;
}

/**
 * One row per status, pipeline order.
 *
 * The list is `STATUSES` (the `action` set = the full pipeline) UNIONED with
 * every status any other type names. Today that union adds nothing — every
 * container status is a subset — and that is exactly why it is a union and not
 * a copy of `STATUSES`: the day a type introduces a status of its own, this
 * table grows the row without anyone remembering to come here.
 */
export const helpStatusRows = (): HelpStatusRow[] => {
  const all = [...STATUSES];
  TASK_TYPE_VALUES.forEach((t) => TASK_TYPES[t].statuses.forEach((s) => {
    if (!all.includes(s)) all.push(s);
  }));
  return all.map((s) => {
    const types = TASK_TYPE_VALUES.filter(
      (t) => TASK_TYPES[t].statuses.includes(s));
    return {
      status: s,
      def: STATUS_DEF[s] || '',
      color: STATUS_COLOR[s] || 'var(--text-tertiary)',
      types,
      finished: FINISHED_STATUSES.includes(s),
      legacy: types.length === 0,
    };
  });
};

export interface HelpAssigneeRow {
  role: string; abbrev: string; color: string;
  /** 'session' | 'agent', read from the agents registry — never from the
   *  letter. The registry is the source; this only renders it. */
  kind: string;
  kindTip: string;
  /** A per-goal lane (`G281`), derived from a goal row rather than listed. */
  derived: boolean;
  /** The goal task this lane belongs to, when it is one. */
  goalId: number | null;
}

/**
 * One lane's row. A per-goal lane gets its own `kind` rather than one of the
 * registry's two answers, because BOTH of theirs would be false of it: it has
 * no registry row at all, and that omission is the safety rail keeping the
 * dispatcher from ever spawning a goal session (board #289).
 *
 * Module-level rather than nested inside `helpAssigneeRows` so the acceptance
 * test can lift it — a helper hidden in a closure is one the suite cannot
 * execute, and executing it is the entire point.
 */
export const helpAssigneeRow = (
  role: string, derived: boolean, reg: Map<string, AgentMeta>,
): HelpAssigneeRow => ({
  role,
  abbrev: roleAbbrev(role),
  color: roleColor(role),
  kind: isGoalLane(role) ? 'goal' : laneKindLabel(role, reg),
  kindTip: isGoalLane(role) ? LANE_KIND_TIP.goal
    : LANE_KIND_TIP[laneKindLabel(role, reg)] || '',
  derived,
  goalId: isGoalLane(role)
    ? Number(role.slice(GOAL_LANE_PREFIX.length)) : null,
});

/**
 * One row per assignable lane: the FIXED roles from `ASSIGNEES`, then the
 * board's own open per-goal lanes.
 *
 * The two halves are built differently on purpose, and that difference IS the
 * lesson the panel is there to teach: a fixed role is one array entry, a goal
 * lane cannot be — goals are created at will, so the vocabulary is unbounded
 * and has to be derived from the rows themselves (board #289).
 */
export const helpAssigneeRows = (
  reg: Map<string, AgentMeta>, tasks: Task[] | null | undefined,
): HelpAssigneeRow[] => [
  ...ASSIGNEES.filter((r) => r).map((r) => helpAssigneeRow(r, false, reg)),
  ...openGoalLanes(tasks).map((r) => helpAssigneeRow(r, true, reg)),
];

/* ── THE WARTS ──────────────────────────────────────────────────────────────
 * Behaviour a reader would NOT predict from the model above. Each one names the
 * board ticket it belongs to AND the file + symbol that proves it, because a
 * wart is the one thing here that cannot be rendered from the model — the model
 * is precisely what it contradicts. The citation is what keeps it honest: the
 * acceptance test opens every `source` and checks the named `symbol` is still
 * there, so a rename turns this into a red test rather than a lie in a modal.
 *
 * Paths are relative to the RoR_Trader root. */
export interface HelpWart {
  id: string;
  /** Which panel it belongs under. */
  section: string;
  title: string;
  body: string;
  /** Board refs, e.g. `#296`. */
  refs: string[];
  source: string;
  symbol: string;
}

export const HELP_WARTS: HelpWart[] = [
  {
    id: 'backlog-does-not-hold',
    section: 'statuses',
    title: 'Backlog does NOT hold a task',
    body: 'Reading the pipeline top-to-bottom suggests a task waits in the '
      + 'first column until someone advances it. It does not. The dispatcher '
      + 'gate is an OR — AI-eligible OR the queued status — so a task with the '
      + 'AI toggle on is dispatchable from ANY non-terminal status, the first '
      + 'column included. The status column and the dispatch decision are two '
      + 'different switches that merely look like one.',
    refs: ['#296', '#198'],
    source: 'tools/team_dispatcher/dispatcher.py',
    symbol: 'GATE_FILTER',
  },
  {
    id: 'staged-is-terminal',
    section: 'statuses',
    title: 'Staged is terminal — nothing moves work out of it',
    body: 'It reads like a waypoint between review and shipping, and it is not '
      + 'one: the dispatcher lists it among the states it refuses to run, so '
      + 'work parked there stays there until a human builds a release train '
      + 'for it. An idle loop and a non-empty column at once is not a stall, '
      + 'it is the designed state — and it means a train is owed.',
    refs: ['#198'],
    source: 'tools/team_dispatcher/dispatcher.py',
    symbol: 'TERMINAL_STATUSES',
  },
  {
    id: 'statuses-not-enforced',
    section: 'statuses',
    title: 'Statuses are DECLARATIVE, not enforced',
    body: 'The per-type status sets above drive what the dropdowns OFFER. They '
      + 'are not a constraint: the API validates origin, impact and task type, '
      + 'and says nothing at all about status, so any string reaches the '
      + 'column. That is deliberate rather than missed — a few hundred rows '
      + 'predate the type model and hold statuses no type lists, which is the '
      + 'entire reason every select wraps its options to keep an unknown '
      + 'current value renderable. Rows like that show up here as legacy.',
    refs: ['#261'],
    source: 'src/api/routers/dev_tasks.py',
    symbol: '_validate_team_fields',
  },
  {
    id: 'assignee-unvalidated',
    section: 'assignees',
    title: 'assignee is not validated at all',
    body: 'There is no assignee vocabulary anywhere on the server — not a '
      + 'whitelist, not a check constraint, nothing. Any string is accepted. '
      + 'That is not a hole to plug, it is load-bearing: it is how a goal '
      + 'session set its own lane to a per-goal value minutes after the first '
      + 'goal was ever spawned, months before any dropdown could offer one. '
      + 'The list below is what the UI OFFERS, which is a strictly smaller '
      + 'thing than what the column ACCEPTS.',
    refs: ['#289', '#281'],
    source: 'src/api/routers/dev_tasks.py',
    symbol: '_validate_team_fields',
  },
  {
    id: 'type-can-be-overridden',
    section: 'types',
    title: 'A row’s stored type can be overruled by its children',
    body: 'The type column is not the last word. A row that has subtasks (or '
      + 'carries the legacy container tag) renders as a container even when '
      + 'the column says otherwise — the derivation only ever upgrades, never '
      + 'demotes. It exists because the backfill matched one MOMENT, and a row '
      + 'that gained children afterwards is exactly the row it could not '
      + 'cover; without the rescue its subtasks vanish off the board '
      + 'entirely. So the glyph on a row and the value in its column can '
      + 'legitimately disagree.',
    refs: ['#261'],
    source: 'frontend/src/views/taskBoardShared.tsx',
    symbol: 'taskTypeOf',
  },
];

/* ── CITED WRITTEN SOURCES ──────────────────────────────────────────────────
 * The rules this pass deliberately does NOT restate. Naming where they live is
 * the whole of the promise: a section either renders from the model or points
 * at its ONE written source by path. Copying them in here is what would make
 * this file the sixth place. */
export interface HelpSource { label: string; path: string; note: string; }

export const HELP_SOURCES: HelpSource[] = [
  {
    label: 'Roles — what each lane is actually FOR',
    path: 'docs/_active/Session_Charters.md',
    note: '§1 is the roster and §7 the lifecycle. The model below knows '
      + 'a lane’s colour and whether the dispatcher may run it; what the lane '
      + 'is RESPONSIBLE for is prose, and prose lives there, once.',
  },
  {
    label: 'The task-type model itself',
    path: 'frontend/src/views/taskBoardShared.tsx',
    note: 'Every table in this manual is rendered from the constants in this '
      + 'file. Mirrored server-side and asserted equal by parsed value, so the '
      + 'two halves cannot drift apart silently.',
  },
  {
    label: 'What the dispatcher will and will not run',
    path: 'tools/team_dispatcher/dispatcher.py',
    note: 'The gate filter and the terminal-status set. The board displays a '
      + 'status; this file decides what one MEANS for dispatch, and the two '
      + 'are not the same question.',
  },
];

/* ── PANELS ─────────────────────────────────────────────────────────────────*/

const TOPICS = [
  { id: 'types', label: 'Task types', icon: '◈' },
  { id: 'statuses', label: 'Statuses', icon: '●' },
  { id: 'assignees', label: 'Assignees', icon: '◑' },
  { id: 'sources', label: 'Where the rules live', icon: '§' },
];

const H2: React.CSSProperties = {
  fontSize: 15, fontWeight: 700, margin: '0 0 4px',
};
const LEAD: React.CSSProperties = {
  fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.55,
  margin: '0 0 14px',
};
const TH: React.CSSProperties = {
  textAlign: 'left', fontSize: 11, textTransform: 'uppercase',
  letterSpacing: 0.5, color: 'var(--text-tertiary)', fontWeight: 600,
  padding: '4px 8px', borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap',
};
const TD: React.CSSProperties = {
  padding: '7px 8px', fontSize: 12.5, verticalAlign: 'top',
  borderBottom: '1px solid var(--border)', lineHeight: 1.5,
};

/** A wart callout. Loud on purpose — it is the reason the manual is useful. */
const Wart = ({ w }: { w: HelpWart }) => (
  <div style={{
    border: '1px solid var(--amber, #d98c00)', borderLeftWidth: 3,
    borderRadius: 6, padding: '8px 10px', marginBottom: 8,
    background: 'rgba(217,140,0,0.06)',
  }}>
    <div style={{ fontSize: 12.5, fontWeight: 700, marginBottom: 3 }}>
      <span style={{ marginRight: 6 }}>{'⚠'}</span>{w.title}
    </div>
    <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
      {w.body}
    </div>
    <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 5 }}>
      {w.refs.join(' · ')} {'· '}
      <code>{w.source}</code> {'→ '}<code>{w.symbol}</code>
    </div>
  </div>
);

const wartsFor = (section: string) =>
  HELP_WARTS.filter((w) => w.section === section);

const YesNo = ({ v, yes, no }: { v: boolean; yes: string; no: string }) => (
  <span style={{ color: v ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>
    {v ? yes : no}
  </span>
);

const TypesPanel = () => {
  const rows = helpTypeRows();
  return (
    <div>
      <h2 style={H2}>Task types</h2>
      <p style={LEAD}>
        Every row on the board is exactly one of these. The type decides which
        statuses the row may hold, whether it may contain subtasks, and whether
        it carries a session of its own. The table is generated from the type
        map — if a fourth type is ever added there, it appears here unprompted.
      </p>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 14 }}>
        <thead><tr>
          <th style={TH}>Type</th>
          <th style={TH}>What it is</th>
          <th style={TH}>Subtasks</th>
          <th style={TH}>Session</th>
          <th style={TH}>Statuses</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.type}>
              <td style={{ ...TD, whiteSpace: 'nowrap' }}>
                <b>{r.icon ? `${r.icon} ` : ''}{r.label}</b>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                  <code>{r.type}</code>{r.isDefault ? ' · default' : ''}
                </div>
                {!r.icon && (
                  <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                    no glyph
                  </div>
                )}
              </td>
              <td style={TD}>{r.def}</td>
              <td style={TD}><YesNo v={r.children} yes="yes" no="no" /></td>
              <td style={TD}><YesNo v={r.session} yes="yes" no="no" /></td>
              <td style={TD}>
                <div>{r.statuses.map((s) => (
                  <span key={s} style={{
                    ...badge(STATUS_COLOR[s] || 'var(--text-tertiary)'),
                    marginBottom: 3,
                  }}>{s}</span>
                ))}</div>
                {r.missing.length > 0 && (
                  <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 3 }}>
                    not available: {r.missing.join(', ')}
                  </div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {wartsFor('types').map((w) => <Wart key={w.id} w={w} />)}
    </div>
  );
};

const StatusesPanel = () => {
  const rows = helpStatusRows();
  return (
    <div>
      <h2 style={H2}>Statuses</h2>
      <p style={LEAD}>
        In pipeline order, with the types each one applies to. Every definition
        below is the one the board already shows as that status&rsquo;s tooltip —
        the same string, read from the same place, so a status can never mean
        one thing in a dropdown and another in the manual.
      </p>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 14 }}>
        <thead><tr>
          <th style={TH}>Status</th>
          <th style={TH}>Applies to</th>
          <th style={TH}>What it means</th>
        </tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.status}>
              <td style={{ ...TD, whiteSpace: 'nowrap' }}>
                <span style={badge(r.color)}>{r.status}</span>
                {r.finished && (
                  <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 3 }}>
                    counts as finished
                  </div>
                )}
              </td>
              <td style={{ ...TD, whiteSpace: 'nowrap' }}>
                {r.legacy ? (
                  <span style={{ color: 'var(--text-tertiary)' }}>
                    legacy — no type lists it
                  </span>
                ) : r.types.map((t) => (
                  <div key={t} style={{ fontSize: 11.5 }}>
                    {TASK_TYPES[t as TaskType].icon || '·'}{' '}
                    {TASK_TYPES[t as TaskType].label}
                  </div>
                ))}
              </td>
              <td style={TD}>{r.def || (
                <span style={{ color: 'var(--text-tertiary)' }}>
                  no definition in the model
                </span>
              )}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {wartsFor('statuses').map((w) => <Wart key={w.id} w={w} />)}
    </div>
  );
};

const AssigneesPanel = ({ tasks }: { tasks: Task[] }) => {
  const reg = useAgentRegistry();
  const rows = useMemo(() => helpAssigneeRows(reg, tasks), [reg, tasks]);
  const fixed = rows.filter((r) => !r.derived);
  const derived = rows.filter((r) => r.derived);
  const table = (list: HelpAssigneeRow[]) => (
    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
      <thead><tr>
        <th style={TH}>Lane</th>
        <th style={TH}>Kind</th>
        <th style={TH}>What that means</th>
      </tr></thead>
      <tbody>
        {list.map((r) => (
          <tr key={r.role}>
            <td style={{ ...TD, whiteSpace: 'nowrap' }}>
              <span style={badge(r.color)}>{r.abbrev}</span>
              <code style={{ fontSize: 11.5 }}>{r.role}</code>
            </td>
            <td style={{ ...TD, whiteSpace: 'nowrap' }}>{r.kind}</td>
            <td style={TD}>{r.kindTip}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
  return (
    <div>
      <h2 style={H2}>Assignees</h2>
      <p style={LEAD}>
        The assignee is <b>whoever the task is waiting on</b> — the dispatcher
        routes on it and nothing else. Whether a lane can be auto-run is read
        from the agents registry, never guessed from the letter, so the kind
        column below reflects what that registry says right now.
      </p>
      <div style={{ fontSize: 11.5, color: 'var(--text-tertiary)', marginBottom: 4 }}>
        FIXED LANES — one array entry each
      </div>
      {table(fixed)}
      <div style={{ fontSize: 11.5, color: 'var(--text-tertiary)', marginBottom: 4 }}>
        PER-GOAL LANES — derived from the board&rsquo;s own open goal rows, one
        per goal, because goals are created at will and no fixed list can
        enumerate them
      </div>
      {derived.length
        ? table(derived)
        : (
          <p style={{ ...LEAD, marginBottom: 12 }}>
            None right now — there is no unfinished goal on the board, so there
            is no goal lane to offer. Create a goal and its lane appears here.
          </p>
        )}
      {wartsFor('assignees').map((w) => <Wart key={w.id} w={w} />)}
    </div>
  );
};

const SourcesPanel = () => (
  <div>
    <h2 style={H2}>Where the rules live</h2>
    <p style={LEAD}>
      This manual only shows what it can generate. Everything else is a pointer,
      on purpose: the rules already live in several places, and a hand-written
      copy here would become one more of them and go stale first. If something
      you need is not in the panels on the left, it is in one of these.
    </p>
    {HELP_SOURCES.map((s) => (
      <div key={s.path} style={{
        border: '1px solid var(--border)', borderRadius: 6,
        padding: '8px 10px', marginBottom: 8,
      }}>
        <div style={{ fontSize: 12.5, fontWeight: 600 }}>{s.label}</div>
        <div style={{ fontSize: 11.5, margin: '2px 0 4px' }}>
          <code>{s.path}</code>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.55 }}>
          {s.note}
        </div>
      </div>
    ))}
    <p style={{ ...LEAD, marginTop: 12, marginBottom: 0 }}>
      Not covered in this pass: process chains, step stamps, and the
      release/train lifecycle. They are prose rather than model, so writing them
      here would create exactly the duplicate this manual exists to avoid —
      they wait on the V2 discussion (#304).
    </p>
  </div>
);

/* ── SHELL ──────────────────────────────────────────────────────────────────*/

export default function TaskHelpModal(
  { tasks, onClose }: { tasks: Task[]; onClose: () => void },
) {
  const [topic, setTopic] = useState(TOPICS[0].id);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  if (!mounted) return null;

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '4vh 16px',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))',
        border: '1px solid var(--border)', borderRadius: 12,
        width: '90vw', maxWidth: 1200, height: '86vh',
        display: 'flex', flexDirection: 'column', padding: 16,
        boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', gap: 12, marginBottom: 10,
        }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700 }}>
              {'✻'} How this board works
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-tertiary)' }}>
              Generated from the board&rsquo;s own model, warts included — not a
              second copy of the rules
            </div>
          </div>
          <button style={{ ...input, cursor: 'pointer' }} onClick={onClose}>
            {'✕'} close
          </button>
        </div>

        <div style={{ display: 'flex', gap: 14, flex: 1, minHeight: 0 }}>
          <nav style={{
            width: 190, flexShrink: 0, borderRight: '1px solid var(--border)',
            paddingRight: 10, overflowY: 'auto',
          }}>
            {TOPICS.map((t) => (
              <button key={t.id} onClick={() => setTopic(t.id)} style={{
                display: 'block', width: '100%', textAlign: 'left',
                background: topic === t.id ? 'var(--bg-input)' : 'transparent',
                color: topic === t.id ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: '1px solid ' + (topic === t.id ? 'var(--border)' : 'transparent'),
                borderRadius: 6, padding: '6px 8px', marginBottom: 3,
                fontSize: 13, fontWeight: topic === t.id ? 600 : 400,
                cursor: 'pointer',
              }}>
                <span style={{ marginRight: 7, color: 'var(--text-tertiary)' }}>
                  {t.icon}
                </span>{t.label}
              </button>
            ))}
            <div style={{
              fontSize: 11, color: 'var(--text-tertiary)', marginTop: 12,
              paddingTop: 10, borderTop: '1px solid var(--border)',
              lineHeight: 1.5,
            }}>
              {HELP_WARTS.length} known gotchas are called out inline, each with
              its board ref and the file that proves it.
            </div>
          </nav>

          <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', paddingRight: 4 }}>
            {topic === 'types' && <TypesPanel />}
            {topic === 'statuses' && <StatusesPanel />}
            {topic === 'assignees' && <AssigneesPanel tasks={tasks} />}
            {topic === 'sources' && <SourcesPanel />}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
