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
 * THE RULE THIS FILE RUNS ON — CITE YOUR SOURCE. Every section does exactly one
 * of two things:
 *
 *   RENDER — the section is generated from the live model (`TASK_TYPES`,
 *            `STATUSES`, `STATUS_DEF`, `ASSIGNEES`, `MODE_DEF`,
 *            `DELIVERABLE_KINDS` in taskBoardShared.tsx). Add a task type, a
 *            status or a role there and this manual grows a row on its own;
 *            change a `def` string and the paragraph here changes with it.
 *            Drift is not "avoided by discipline", it is impossible by
 *            construction.
 *
 *   CITE   — the rule is not in the model (it is enforcement, sequencing or
 *            policy), so the paragraph carries a VISIBLE `path:line` pointing at
 *            the code that enforces it. That is what separates this from a sixth
 *            copy: a cited claim is auditable — you can open the line and check
 *            it, and when it drifts the acceptance test says so, naming the
 *            claim and the file.
 *
 * A section that does NEITHER does not belong here. Nothing in this file states
 * a rule it cannot render or cite.
 *
 * The pure builders below (`helpTypeRows`, `helpStatusRows`, `helpAssigneeRows`,
 * `helpModeRows`, `helpDeliverableRows`) are deliberately JSX-free so that
 * test_task_help_modal_297.py can lift them out of this file and EXECUTE them in
 * node against the real model, rather than grep for a promise. A future
 * hand-edit that hardcodes a row fails that test, and so does a citation whose
 * line number has moved.
 *
 * ── AND WHY IT DOCUMENTS THE WARTS ─────────────────────────────────────────
 * Kevin's framing, 08-02: *"create the modal based on the rules and features
 * that govern everything TODAY — I can better wrap my head around what needs to
 * be changed."* This is a DIAGNOSTIC, not a brochure. Where the board's actual
 * behaviour contradicts what a reader would reasonably assume, `HELP_WARTS` says
 * so outright, with the board ref and the exact line that proves it. A manual
 * that reads tidier than the system defeats the purpose, because the V2
 * redesign (#304) is about to be argued against precisely these.
 *
 * OUT OF SCOPE, deliberately: nothing here is editable. Describing the system is
 * the job; becoming the system is a much larger decision.
 */
'use client';

import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Task, AgentMeta, ASSIGNEES, STATUSES, STATUS_DEF, STATUS_COLOR,
  TASK_TYPES, TASK_TYPE_VALUES, TaskType, DEFAULT_TASK_TYPE, FINISHED_STATUSES,
  GOAL_LANE_PREFIX, LANE_KIND_TIP, isGoalLane, laneKindLabel,
  openGoalLanes, roleAbbrev, roleColor, useAgentRegistry, input, badge,
  MODE_DEF, DELIVERABLE_KINDS,
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

export interface HelpModeRow { mode: string; def: string; }

/** A step's `mode` vocabulary, straight out of the model's own tooltips. The
 *  chain section is mostly CITED prose — this half of it does not have to be,
 *  because the model already carries the definitions. */
export const helpModeRows = (): HelpModeRow[] =>
  Object.keys(MODE_DEF).map((m) => ({ mode: m, def: MODE_DEF[m] }));

export interface HelpDeliverableRow {
  kind: string; icon: string; label: string; def: string;
}

/** The evidence kinds a step may ask for (board #184) — same argument as
 *  `helpModeRows`: the model carries `def`, so nothing is retyped. */
export const helpDeliverableRows = (): HelpDeliverableRow[] =>
  DELIVERABLE_KINDS.map((d) => ({
    kind: d.key, icon: d.icon, label: d.label, def: d.def,
  }));

/* ── CITATIONS ──────────────────────────────────────────────────────────────
 * The other half of the rule. A claim that cannot be rendered from the model
 * names the LINE that enforces it, and the acceptance test opens that file and
 * checks the `anchor` is still on that exact line. When the code moves the test
 * goes red with the new number, which is the whole difference between a citation
 * and a footnote nobody maintains.
 *
 * Paths are relative to the RoR_Trader root. */
export interface HelpCite {
  /** Repo-relative path, rendered verbatim so it can be pasted into an editor. */
  path: string;
  /** 1-indexed line. Pinned by the suite — if it drifts, the test names it. */
  line: number;
  /** The text that must be ON that line. This is what makes the pin checkable
   *  rather than decorative. */
  anchor: string;
}

export const HELP_CITES: Record<string, HelpCite> = {
  // ── chains ──
  chainShape: {
    path: 'src/api/routers/dev_tasks.py', line: 191,
    anchor: '_NEW_SHAPE_KEYS = (',
  },
  chainOptIn: {
    path: 'src/api/routers/dev_tasks.py', line: 240,
    anchor: 'def _is_process_chain(checklist) -> bool:',
  },
  chainOptInUi: {
    path: 'frontend/src/views/taskBoardShared.tsx', line: 135,
    anchor: 'export const isProcessChain = ',
  },
  chainServerOwned: {
    path: 'src/api/routers/dev_tasks.py', line: 195,
    anchor: '_STEP_COMPLETION_FIELDS = (',
  },
  chainStrictOrder: {
    path: 'src/api/routers/dev_tasks.py', line: 259,
    anchor: 'def _current_step_index(checklist)',
  },
  chainHandoff: {
    path: 'src/api/routers/dev_tasks.py', line: 268,
    anchor: 'def _next_assignee(checklist, current):',
  },
  chainTick: {
    path: 'src/api/routers/dev_tasks.py', line: 1233,
    anchor: '/steps/complete',
  },
  chainPatchRefused: {
    path: 'src/api/routers/dev_tasks.py', line: 502,
    anchor: 'completion is managed by /steps/complete',
  },
  chainLegacyUntouched: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 756,
    anchor: 'Only for process chains; legacy checklists',
  },
  chainDiscuss: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 767,
    anchor: '== "discuss":',
  },
  chainDeliverableSplit: {
    path: 'src/api/routers/dev_tasks.py', line: 209,
    anchor: '_DELIVERABLE_FILL_FIELDS = (',
  },
  chainDeliverableGate: {
    path: 'src/api/routers/dev_tasks.py', line: 963,
    anchor: 'missing = _unfilled(step, required=True)',
  },
  chainConvention: {
    path: 'docs/_active/Session_Charters.md', line: 272,
    anchor: '## 7. Team coordination',
  },

  // ── stamps ──
  stampShape: {
    path: 'frontend/src/views/taskBoardShared.tsx', line: 21,
    anchor: 'export interface StepStamp',
  },
  stampStates: {
    path: 'src/api/routers/dev_tasks.py', line: 190,
    anchor: '_STAMP_STATES = ',
  },
  stampGate: {
    path: 'src/api/routers/dev_tasks.py', line: 954,
    anchor: 'stamp.get("required") and stamp.get("state") !=',
  },
  stampEndpoint: {
    path: 'src/api/routers/dev_tasks.py', line: 1521,
    anchor: '/steps/stamp',
  },
  stampWrite: {
    path: 'src/api/routers/dev_tasks.py', line: 1548,
    anchor: 'stamp["state"] = ',
  },
  stampPatchRefused: {
    path: 'src/api/routers/dev_tasks.py', line: 509,
    anchor: 'stamp state is managed by /steps/stamp',
  },
  stampDerivesKevinFinal: {
    path: 'src/api/routers/dev_tasks.py', line: 279,
    anchor: 'def _derive_kevin_final(checklist, existing)',
  },
  stampReject: {
    path: 'src/api/routers/dev_tasks.py', line: 1556,
    anchor: '# T9 escalation',
  },
  stampTaskLevel: {
    path: 'src/api/routers/dev_tasks.py', line: 1029,
    anchor: '_STAMP_LABEL = ',
  },
  stampArms: {
    path: 'src/api/routers/dev_tasks.py', line: 1080,
    anchor: 'ai_eligible": True',
  },

  // ── status vs the chain ──
  gateFilter: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 323,
    anchor: 'GATE_FILTER = ',
  },
  gateTerminal: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 366,
    anchor: 'TERMINAL_STATUSES = (',
  },
  gateTriage: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 682,
    anchor: 'def triage_todo(agents, done_ids, st=None):',
  },
  gateDisagree: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 761,
    anchor: 'if owner != a:',
  },
  gateArmSelector: {
    path: 'tools/team_dispatcher/dispatcher.py', line: 784,
    anchor: 'legacy_todo_arm = ',
  },
  gateNoStatusValidation: {
    path: 'src/api/routers/dev_tasks.py', line: 589,
    anchor: 'def _validate_team_fields(',
  },
  gateCompletionWrites: {
    path: 'src/api/routers/dev_tasks.py', line: 977,
    anchor: '"assignee": new_assignee,',
  },
  gateActorRule: {
    path: 'docs/_active/Session_Charters.md', line: 168,
    anchor: '## 4. Shared guardrails',
  },

  // ── the release lifecycle ──
  relEntryPoint: {
    path: 'tools/release_brief/brief_gen.py', line: 1153,
    anchor: 'dev_tasks?status=eq.Staged',
  },
  relCarIsAStep: {
    path: 'tools/release_brief/brief_gen.py', line: 265,
    anchor: 'first incomplete step when that step is RELEASE-LANE-owned',
  },
  relShipOwners: {
    path: 'tools/release_brief/release_lane.py', line: 39,
    anchor: 'SHIP_OWNERS = (',
  },
  relNoShipStep: {
    path: 'tools/release_brief/brief_gen.py', line: 552,
    anchor: 'but its chain names NO ',
  },
  relMergedGate: {
    path: 'tools/release_brief/brief_gen.py', line: 1037,
    anchor: '"merge-base", "--is-ancestor"',
  },
  relSweep: {
    path: 'tools/release_brief/car_ship_tick.py', line: 185,
    anchor: 'def plan_ticks(tasks, resolve_branch, is_merged):',
  },
  relShipStepIndex: {
    path: 'tools/release_brief/car_ship_tick.py', line: 161,
    anchor: 'def ship_step_index(task):',
  },
  relStepIsNotStatus: {
    path: 'tools/release_brief/car_ship_tick.py', line: 44,
    anchor: 'THE STEP IS NOT THE STATUS',
  },
  relProtocol: {
    path: 'docs/_active/Session_Charters.md', line: 335,
    anchor: '## 8. Release protocol',
  },
};

/* ── THE CITED CLAIMS ───────────────────────────────────────────────────────
 * Sections 4-7. Each claim is one behaviour, in prose, with the lines that
 * enforce it. Written as data rather than JSX so the suite can walk every claim
 * and prove that not one of them is uncited. */
export interface HelpClaim {
  id: string;
  /** Which panel it belongs under. */
  section: string;
  title: string;
  body: string;
  /** Keys into HELP_CITES. Never empty — an uncited claim is the sixth copy. */
  cites: string[];
}

export const HELP_CLAIMS: HelpClaim[] = [
  /* ── 4. PROCESS CHAINS ─────────────────────────────────────────────────── */
  {
    id: 'chain-optin',
    section: 'chains',
    title: 'A checklist BECOMES a chain the moment a step carries one of the '
      + 'new fields — and the switch is behavioural',
    body: 'There is no flag, no migration and no per-task setting. A checklist '
      + 'of the old shape is a list of ticks anyone may toggle in any order. '
      + 'The instant ANY step in it carries an id, owner, title, body, mode, '
      + 'origin, stamp or deliverables, the whole checklist is treated as a '
      + 'process chain and three behaviours switch on at once: strict-order '
      + 'completion, server-owned completion state, and automatic hand-off to '
      + 'the next step’s owner. The same opt-in test is written twice on '
      + 'purpose — once server-side and once in the board’s own code — because '
      + 'both halves must agree on what they are looking at.',
    cites: ['chainOptIn', 'chainOptInUi', 'chainShape'],
  },
  {
    id: 'chain-step-fields',
    section: 'chains',
    title: 'What a step carries — and which half of it you may edit',
    body: 'An author owns the structure: owner (who acts), title (the one line '
      + 'of substance), body (the SOP the dispatcher actually sends that '
      + 'agent), mode, origin, and any stamp or deliverables the step declares. '
      + 'The server owns the outcome: done, completed_at and completed_by are '
      + 'never writable through the ordinary checklist edit. Reordering, '
      + 'editing and inserting not-yet-done steps all go through the normal '
      + 'edit; ticking one does not, and a PATCH that tries is a 409 naming the '
      + 'endpoint it should have used.',
    cites: ['chainShape', 'chainServerOwned', 'chainPatchRefused'],
  },
  {
    id: 'chain-strict-order',
    section: 'chains',
    title: 'Strict order: only the FIRST incomplete step is completable',
    body: 'The current step is defined as the first step not yet done — there '
      + 'is no way to reach past it and complete a later one, and the '
      + 'completion endpoint takes no step argument at all. Course-correcting '
      + 'mid-chain is done by INSERTING a step (marked as an audible, so the '
      + 'change is visible) rather than by skipping one, and a completed step '
      + 'cannot be deleted afterwards because it is part of the audit trail.',
    cites: ['chainStrictOrder', 'chainTick', 'chainPatchRefused'],
  },
  {
    id: 'chain-handoff',
    section: 'chains',
    title: 'Completing a step REASSIGNS the task to the next step’s owner',
    body: 'This is the part that makes a chain worth having: the hand-off is '
      + 'not a convention agents are asked to remember, it is what the '
      + 'completion call does. Tick a step and the task moves to whoever owns '
      + 'the next incomplete one. When the last step is ticked the task goes to '
      + 'the manager lane rather than to Kevin — closing is follow-through, not '
      + 'a review. An empty owner leaves the assignee where it was.',
    cites: ['chainHandoff', 'chainTick'],
  },
  {
    id: 'chain-discuss',
    section: 'chains',
    title: 'A discussion step is never handed to a headless agent',
    body: 'Mode is the safety rail on the whole dispatch system. An execution '
      + 'step may be sent to an agent; a discussion step wants a human reply, '
      + 'and the dispatcher refuses to dispatch one — classified as waiting '
      + 'rather than stuck, so it never trips the staleness alarm. An agent '
      + 'handed a discussion step would build something nobody asked for, which '
      + 'is why this is enforced in the queue rather than trusted to a prompt.',
    cites: ['chainDiscuss', 'chainConvention'],
  },
  {
    id: 'chain-legacy',
    section: 'chains',
    title: 'Legacy checklists still work, untouched',
    body: 'Chains were added without a flag day: a checklist that never opted '
      + 'in keeps its exact earlier behaviour, including in the dispatcher, '
      + 'which applies the chain gates only to real chains and otherwise routes '
      + 'on the assignee alone. Nothing was batch-converted; a task gets a real '
      + 'chain when someone touches it. So two tasks on this board can behave '
      + 'differently from each other, correctly, and the shape of their '
      + 'checklist is the tell.',
    cites: ['chainLegacyUntouched', 'chainConvention'],
  },
  {
    id: 'chain-deliverables',
    section: 'chains',
    title: 'A step may ask for evidence, and a required ask BLOCKS completion',
    body: 'A step can declare deliverables — the evidence it wants, captured on '
      + 'the step instead of in a comment no surface can see. The same split as '
      + 'everywhere else applies: the ASK is author-owned, the ANSWER is '
      + 'server-owned and only reachable through its own endpoints. An unfilled '
      + 'REQUIRED deliverable refuses completion; an unfilled optional one never '
      + 'does — it is recorded on the thread and the step proceeds.',
    cites: ['chainDeliverableSplit', 'chainDeliverableGate'],
  },

  /* ── 5. STAMPS ─────────────────────────────────────────────────────────── */
  {
    id: 'stamp-shape',
    section: 'stamps',
    title: 'A stamp is four fields on a step',
    body: 'required says the step gates on approval at all; state is one of '
      + 'pending, approved or rejected; by and at record who decided and when. '
      + 'A step with no stamp object gates on nothing and completes normally — '
      + 'stamps are opt-in per step, not a lifecycle every step passes through.',
    cites: ['stampShape', 'stampStates'],
  },
  {
    id: 'stamp-gates-completion',
    section: 'stamps',
    title: 'A required stamp that is not approved REFUSES completion',
    body: 'The completion path checks the stamp before it checks anything else '
      + 'and returns a 409 naming the step and the endpoint to use. This is a '
      + 'real gate, not a UI nicety: it fires the same way for the board, for '
      + 'an agent holding the service key, and for a curl. An agent that hits '
      + 'it is expected to report the refusal rather than route around it.',
    cites: ['stampGate', 'chainTick'],
  },
  {
    id: 'stamp-owns-state',
    section: 'stamps',
    title: 'Only the stamp endpoint may change a stamp’s state',
    body: 'Approving or rejecting goes through one endpoint, which writes '
      + 'state, by and at together and posts a system comment. The ordinary '
      + 'checklist edit is refused with a 409 if it so much as changes a '
      + 'stamp’s state — same argument as completion: two paths that write the '
      + 'same field will drift, so there is one. That endpoint is also the one '
      + 'part of the step machinery that requires a human login rather than a '
      + 'service key, because a stamp is authority and evidence is not.',
    cites: ['stampEndpoint', 'stampWrite', 'stampPatchRefused'],
  },
  {
    id: 'stamp-derives-kevin-final',
    section: 'stamps',
    title: 'A kevin-owned step with a required stamp derives the task-level '
      + 'two-touch flag',
    body: 'It is derived, never typed: if any step owned by kevin carries a '
      + 'required stamp, the task is flagged as one Kevin sees again before it '
      + 'closes. The derivation never CLEARS an existing flag — dropping a '
      + 'review gate silently is the failure it is written to avoid — so the '
      + 'flag is sticky by design, which is a wart if you expected it to track '
      + 'the chain both ways.',
    cites: ['stampDerivesKevinFinal'],
  },
  {
    id: 'stamp-reject',
    section: 'stamps',
    title: 'Rejecting does not tick anything — it routes to the manager lane',
    body: 'A rejection records who rejected it and when, leaves the step open, '
      + 'and reassigns the task to the manager lane as the escalation path. The '
      + 'chain does not advance and no work is marked done, which is what makes '
      + 'rejection safe to use rather than something people avoid.',
    cites: ['stampReject', 'stampEndpoint'],
  },
  {
    id: 'stamp-task-level',
    section: 'stamps',
    title: 'The TASK-level stamp is a different thing, and it means '
      + '“approved to start”',
    body: 'Separate from per-step stamps, a task carries one approval stamp at '
      + 'the Approval stage. It is permission to EXECUTE, not a verdict on the '
      + 'result: stamping arms the task for dispatch and moves it to the queued '
      + 'status in one write, and it runs the same completion path as any other '
      + 'tick so a stamped kevin step gets real completion provenance instead '
      + 'of a hand-rolled one.',
    cites: ['stampTaskLevel', 'stampArms', 'chainTick'],
  },

  /* ── 6. STATUS vs THE CHAIN ────────────────────────────────────────────── */
  {
    id: 'status-is-not-derived',
    section: 'dispatch',
    title: 'Status is a SEPARATE field, and nothing derives it',
    body: 'This is the single most surprising fact about the board, and the one '
      + 'most worth holding in mind before redesigning it. Completing a step '
      + 'writes the checklist, the assignee and the two-touch flag — and not '
      + 'the status. The server never validates a status either. So the '
      + 'lifecycle column and the chain are two independent descriptions of the '
      + 'same task that are kept in agreement by people and by the dispatcher’s '
      + 'own bookkeeping, not by the data model. Every time they disagree, '
      + 'something in this manual’s warts list is the reason.',
    cites: ['gateCompletionWrites', 'gateNoStatusValidation'],
  },
  {
    id: 'dispatch-gate',
    section: 'dispatch',
    title: 'The dispatch gate is an OR — armed, or in the queued column',
    body: 'A task is dispatchable if it is AI-eligible OR it sits in the queued '
      + 'status. Eligibility and lifecycle were deliberately separated: the '
      + 'arming switch decides whether anything may run, the status merely '
      + 'describes where the work is. Reading the columns left to right and '
      + 'concluding that a task must be advanced into the queue before it can '
      + 'run gets this exactly backwards.',
    cites: ['gateFilter', 'gateTriage'],
  },
  {
    id: 'dispatch-terminal',
    section: 'dispatch',
    title: 'Some statuses stop dispatch — as FINALITY, not as permission',
    body: 'A short list of statuses is refused outright however the task was '
      + 'armed: the finished ones, the blocked one, the parked one and the '
      + 'release-staged one. The distinction matters — these are not "not '
      + 'allowed yet", they are "there is nothing HERE to hand an agent". Most '
      + 'are skipped quietly, because a finished or deliberately parked task is '
      + 'not a stalled queue and nagging about it daily would be noise.',
    cites: ['gateTerminal', 'gateTriage'],
  },
  {
    id: 'dispatch-arm-selector',
    section: 'dispatch',
    title: 'Status IS read once inside the queue — as an arm selector',
    body: 'A task in the queued column keeps its older behaviour to the letter, '
      + 'including a fully-ticked chain still dispatching on the assignee '
      + 'alone; an armed task outside that column gets the tighter rails '
      + '(a complete chain has nothing left to hand out, and a step whose run '
      + 'already finished is not re-dispatched). It is the one status read in '
      + 'the queue and it chooses a behaviour, never a permission.',
    cites: ['gateArmSelector', 'gateTriage'],
  },
  {
    id: 'dispatch-disagreement',
    section: 'dispatch',
    title: 'When the chain’s current owner and the assignee disagree, the task '
      + 'is REFUSED',
    body: 'The assignee is authoritative for routing — the checklist is '
      + 'advisory display, and a chain that overrode the assignee once made the '
      + 'entire queue undispatchable for about 41 hours with no error at all. '
      + 'But a real chain still has to agree with it: if the current step is '
      + 'owned by someone other than the assignee, the dispatcher refuses the '
      + 'task and flags it as stuck, because one of the two is wrong and '
      + 'guessing which would be worse than stopping.',
    cites: ['gateDisagree', 'gateTriage', 'gateActorRule'],
  },
  {
    id: 'dispatch-actor',
    section: 'dispatch',
    title: 'A status change names its actor, or it is refused',
    body: 'Moving a task through the API requires saying who moved it; the '
      + 'field is not stored on the row, it writes the attributed line on the '
      + 'thread. Writing status straight to the database bypasses that and logs '
      + 'nothing at all — not anonymous, absent — which is how most finished '
      + 'tasks on this board have no record of who finished them. Both paths '
      + 'now refuse the anonymous version instead of accepting it silently.',
    cites: ['gateActorRule'],
  },

  /* ── 7. THE RELEASE LIFECYCLE ──────────────────────────────────────────── */
  {
    id: 'release-entry',
    section: 'releases',
    title: 'The release pipeline starts at ONE query: tasks in the staged '
      + 'status',
    body: 'The brief generator’s entry point is the staged column and nothing '
      + 'else. A finished branch that was never marked staged is invisible to '
      + 'the release lane no matter how ready it is — no PR scan, no branch '
      + 'sweep, no heuristic. Marking a reviewed task staged is therefore the '
      + 'act that puts it in front of the release lane at all.',
    cites: ['relEntryPoint', 'relProtocol'],
  },
  {
    id: 'release-car',
    section: 'releases',
    title: 'A CAR is a staged task whose current step is release-lane-owned',
    body: 'Among staged tasks, the ones that ship are those whose FIRST '
      + 'INCOMPLETE step is owned by the release lane — that step is the ship '
      + 'step, and it is what makes the task a car. Ownership is one shared '
      + 'definition covering both the release session and its headless '
      + 'executor, kept in a single module precisely because it was once three '
      + 'copies and one of them went blind to the newer name. A staged task '
      + 'whose chain names no such step is not a car and is refused by name in '
      + 'the brief rather than shipped.',
    cites: ['relCarIsAStep', 'relShipStepIndex', 'relShipOwners',
      'relNoShipStep'],
  },
  {
    id: 'release-merge',
    section: 'releases',
    title: 'Who may merge — and who explicitly may not',
    body: 'The release lane merges. A working session finishes a branch, review '
      + 'marks it staged, and stops there; the release session decides what '
      + 'ships together and in what order and authors the brief; the headless '
      + 'executor runs one train and hands back rather than improvising. The '
      + 'manager lane does not self-merge even when it would be faster, and an '
      + 'ambiguous go-ahead means "ship it through the release lane", never '
      + '"do it yourself".',
    cites: ['relProtocol', 'relShipOwners'],
  },
  {
    id: 'release-sweep',
    section: 'releases',
    title: 'After the merges, a sweep ticks each car’s ship step — gated on git',
    body: 'A train is its own board task with its own chain, and it ticks its '
      + 'own steps; nothing ticks the CARS’. Left alone, every shipped car '
      + 'keeps an open ship step and renders like work that never went '
      + 'anywhere, which happened three times before the sweep existed. The '
      + 'sweep never takes anyone’s word for what shipped: each candidate is '
      + 'gated on the branch actually being an ancestor of the deploy branch '
      + 'after a fresh fetch, so a car that failed to merge cannot be ticked.',
    cites: ['relSweep', 'relMergedGate'],
  },
  {
    id: 'release-step-vs-status',
    section: 'releases',
    title: 'Ticking the step is only half the record — the status moves too',
    body: 'The step tick records what happened; the status records where the '
      + 'work now is, and shipping one without the other produced a board that '
      + 'contradicted itself — a staged tile counting seven while five of them '
      + 'had already shipped. Every count built on status inherits that lie, '
      + 'which is why the sweep moves both.',
    cites: ['relStepIsNotStatus', 'relSweep'],
  },
];

/* ── THE WARTS ──────────────────────────────────────────────────────────────
 * Behaviour a reader would NOT predict from the model or from the claims above.
 * Each one names the board ticket it belongs to AND the file + line + symbol
 * that proves it, because a wart is the one thing here that cannot be rendered
 * from the model — the model is precisely what it contradicts. The citation is
 * what keeps it honest: the acceptance test opens every `source` and checks the
 * named `symbol` is still on the named line, so a rename or a move turns this
 * into a red test rather than a lie in a modal.
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
  line: number;
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
    line: 323,
    symbol: 'GATE_FILTER',
  },
  {
    id: 'staged-is-terminal',
    section: 'statuses',
    title: 'Staged is terminal — nothing moves work out of it automatically',
    body: 'It reads like a waypoint between review and shipping, and it is not '
      + 'one: the dispatcher lists it among the states it refuses to run, so '
      + 'work parked there stays there until a release train is built for it. '
      + 'An idle loop and a non-empty column at once is not a stall, it is the '
      + 'designed state — and it means a train is owed. The other half of the '
      + 'same wart is that a shipped car does not leave the column by itself '
      + 'either: a post-merge sweep had to be written to tick the ship step and '
      + 'move the status, and before it existed the board double-counted work '
      + 'that had already gone out.',
    refs: ['#198', '#242', '#249'],
    source: 'tools/team_dispatcher/dispatcher.py',
    line: 366,
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
    line: 589,
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
    line: 589,
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
    line: 529,
    symbol: 'taskTypeOf',
  },
  {
    id: 'complete-ticks-the-first-step',
    section: 'chains',
    title: 'Completing a step ticks the FIRST incomplete one — not the one you '
      + 'were looking at',
    body: 'The endpoint takes no step argument. Whatever step you had open, the '
      + 'call ticks the current one, so an agent or a human who has drifted a '
      + 'step out of sync ticks the wrong thing and hands the task to the wrong '
      + 'owner — twice in one day on 08-01. It is strict order working exactly '
      + 'as designed, and it is still a trap, because nothing in the request '
      + 'says which step you meant and nothing can therefore refuse a mismatch. '
      + 'Read the current step before you tick.',
    refs: ['#202', '#182'],
    source: 'src/api/routers/dev_tasks.py',
    line: 1238,
    symbol: 'Ticks the FIRST incomplete step only',
  },
  {
    id: 'chain-assignee-disagreement-stops-work',
    section: 'dispatch',
    title: 'A chain/assignee disagreement stops the task dead',
    body: 'Both fields say who acts next, and nothing keeps them in step except '
      + 'the completion call — so a manual reassignment that does not match the '
      + 'current step, or a chain edited around a step that already moved, '
      + 'leaves the task refused every pass. The dispatcher does flag it as '
      + 'stuck rather than dropping it silently, which is the fix that came out '
      + 'of the 41-hour outage. But nothing repairs it: on the board the task '
      + 'simply sits, looking assigned and ready, and the only evidence is in '
      + 'the dispatcher’s skip report.',
    refs: ['#73', '#171'],
    source: 'tools/team_dispatcher/dispatcher.py',
    line: 761,
    symbol: 'if owner != a:',
  },
  {
    id: 'no-ship-step-never-ships',
    section: 'releases',
    title: 'A chain with no release-lane step never becomes a car at all',
    body: 'Being staged is necessary and not sufficient. The generator only '
      + 'treats a staged task as shippable if its current step is owned by the '
      + 'release lane, so a task marked staged with a chain that names no such '
      + 'step is not a car and will never be picked up — it is refused. Two of '
      + 'five staged tasks were dropped this way, silently, before the refusal '
      + 'was made loud in the brief; the refusal is visible now, but the task '
      + 'still does not ship until someone adds the step.',
    refs: ['#302', '#269'],
    source: 'tools/release_brief/brief_gen.py',
    line: 552,
    symbol: 'chain names NO',
  },
];

/* ── CITED WRITTEN SOURCES ──────────────────────────────────────────────────
 * The single written home of each rule that is policy rather than code. Naming
 * where they live is the whole of the promise: copying them in here is what
 * would make this file the sixth place. */
export interface HelpSource { label: string; path: string; note: string; }

export const HELP_SOURCES: HelpSource[] = [
  {
    label: 'Roles — what each lane is actually FOR',
    path: 'docs/_active/Session_Charters.md',
    note: '§1 is the roster, §4 the shared guardrails, §7 the board lifecycle '
      + 'and §8 the release protocol. The model below knows a lane’s colour '
      + 'and whether the dispatcher may run it; what the lane is RESPONSIBLE '
      + 'for is prose, and prose lives there, once.',
  },
  {
    label: 'The task-type model itself',
    path: 'frontend/src/views/taskBoardShared.tsx',
    note: 'Every generated table in this manual is rendered from the constants '
      + 'in this file. Mirrored server-side and asserted equal by parsed value, '
      + 'so the two halves cannot drift apart silently.',
  },
  {
    label: 'What the dispatcher will and will not run',
    path: 'tools/team_dispatcher/dispatcher.py',
    note: 'The gate filter, the terminal-status set and the chain gates. The '
      + 'board displays a status; this file decides what one MEANS for '
      + 'dispatch, and the two are not the same question.',
  },
  {
    label: 'Steps, stamps and everything the server owns',
    path: 'src/api/routers/dev_tasks.py',
    note: 'The chain opt-in test, strict-order completion, the hand-off, the '
      + 'stamp gates and the refusals that keep completion state out of a '
      + 'generic edit. Where a rule in this manual is enforced rather than '
      + 'described, it is enforced here.',
  },
  {
    label: 'How a staged task turns into a shipped one',
    path: 'tools/release_brief/brief_gen.py',
    note: 'The release lane’s entry point and its definition of a car. Its '
      + 'companion `car_ship_tick.py` is the post-merge sweep, and '
      + '`release_lane.py` is the one definition of who owns a ship step.',
  },
];

/* ── PANELS ─────────────────────────────────────────────────────────────────*/

const TOPICS = [
  { id: 'types', label: 'Task types', icon: '◈', kind: 'rendered' },
  { id: 'statuses', label: 'Statuses', icon: '●', kind: 'rendered' },
  { id: 'assignees', label: 'Assignees & lanes', icon: '◑', kind: 'rendered' },
  { id: 'chains', label: 'Process chains', icon: '⛓', kind: 'cited' },
  { id: 'stamps', label: 'Stamps', icon: '✓', kind: 'cited' },
  { id: 'dispatch', label: 'Status vs the chain', icon: '⇄', kind: 'cited' },
  { id: 'releases', label: 'Release lifecycle', icon: '🚆', kind: 'cited' },
  { id: 'sources', label: 'Where the rules live', icon: '§', kind: 'cited' },
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

/** How a panel earns its place: generated, or pinned to a line. Shown on every
 *  panel so a reader always knows which promise they are being given. */
const Provenance = ({ kind }: { kind: string }) => (
  <div style={{
    display: 'inline-block', fontSize: 10.5, letterSpacing: 0.4,
    textTransform: 'uppercase', fontWeight: 700, borderRadius: 4,
    padding: '2px 6px', marginBottom: 8,
    border: '1px solid var(--border)', color: 'var(--text-tertiary)',
  }}>
    {kind === 'rendered'
      ? 'rendered from the model'
      : 'cited — every claim carries its source line'}
  </div>
);

/** One `path:line` pin. Deliberately ugly and copy-pasteable: the point is that
 *  you can open it and check, not that it looks like a footnote. */
const Cite = ({ id }: { id: string }) => {
  const c = HELP_CITES[id];
  if (!c) return null;
  return (
    <span title={c.anchor} style={{
      display: 'inline-block', fontSize: 10.5, lineHeight: 1.4,
      border: '1px solid var(--border)', borderRadius: 4,
      padding: '1px 5px', marginRight: 5, marginTop: 4,
      color: 'var(--text-tertiary)', whiteSpace: 'nowrap',
      fontFamily: 'ui-monospace, monospace',
    }}>
      {c.path}:{c.line}
    </span>
  );
};

const Claim = ({ c }: { c: HelpClaim }) => (
  <div style={{
    borderLeft: '2px solid var(--border)', paddingLeft: 10, marginBottom: 14,
  }}>
    <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 3 }}>
      {c.title}
    </div>
    <div style={{
      fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.6,
    }}>
      {c.body}
    </div>
    <div>{c.cites.map((k) => <Cite key={k} id={k} />)}</div>
  </div>
);

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
      <code>{w.source}:{w.line}</code> {'→ '}<code>{w.symbol}</code>
    </div>
  </div>
);

const wartsFor = (section: string) =>
  HELP_WARTS.filter((w) => w.section === section);

const claimsFor = (section: string) =>
  HELP_CLAIMS.filter((c) => c.section === section);

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
      <Provenance kind="rendered" />
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
      <Provenance kind="rendered" />
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
      <h2 style={H2}>Assignees &amp; lanes</h2>
      <Provenance kind="rendered" />
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

const ChainsPanel = () => (
  <div>
    <h2 style={H2}>Process chains</h2>
    <Provenance kind="cited" />
    <p style={LEAD}>
      A chain is a task&rsquo;s real hand-off sequence, written into the task
      before the work starts. Each step names its owner and carries the SOP the
      dispatcher sends that owner, which is the whole payoff: the context lives
      in the task rather than being hand-written at dispatch time.
    </p>
    {claimsFor('chains').map((c) => <Claim key={c.id} c={c} />)}

    <h3 style={{ ...H2, fontSize: 13.5, marginTop: 18 }}>
      Step modes <span style={{
        fontSize: 10.5, fontWeight: 600, color: 'var(--text-tertiary)',
      }}>— rendered from the model</span>
    </h3>
    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 14 }}>
      <thead><tr>
        <th style={TH}>Mode</th>
        <th style={TH}>What it means</th>
      </tr></thead>
      <tbody>
        {helpModeRows().map((r) => (
          <tr key={r.mode}>
            <td style={{ ...TD, whiteSpace: 'nowrap' }}><code>{r.mode}</code></td>
            <td style={TD}>{r.def}</td>
          </tr>
        ))}
      </tbody>
    </table>

    <h3 style={{ ...H2, fontSize: 13.5 }}>
      Deliverable kinds a step may ask for <span style={{
        fontSize: 10.5, fontWeight: 600, color: 'var(--text-tertiary)',
      }}>— rendered from the model</span>
    </h3>
    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 14 }}>
      <thead><tr>
        <th style={TH}>Kind</th>
        <th style={TH}>What it accepts</th>
      </tr></thead>
      <tbody>
        {helpDeliverableRows().map((r) => (
          <tr key={r.kind}>
            <td style={{ ...TD, whiteSpace: 'nowrap' }}>
              {r.icon} <code>{r.label}</code>
            </td>
            <td style={TD}>{r.def}</td>
          </tr>
        ))}
      </tbody>
    </table>
    {wartsFor('chains').map((w) => <Wart key={w.id} w={w} />)}
  </div>
);

const StampsPanel = () => (
  <div>
    <h2 style={H2}>Stamps</h2>
    <Provenance kind="cited" />
    <p style={LEAD}>
      A stamp is an approval attached to one step — &ldquo;approve THIS step&rdquo;
      rather than &ldquo;approve the task&rdquo;. It is the mechanism behind every
      place the board waits for a human before letting work continue.
    </p>
    {claimsFor('stamps').map((c) => <Claim key={c.id} c={c} />)}
    {wartsFor('stamps').map((w) => <Wart key={w.id} w={w} />)}
  </div>
);

const DispatchPanel = () => (
  <div>
    <h2 style={H2}>Status vs the chain</h2>
    <Provenance kind="cited" />
    <p style={LEAD}>
      Two things describe where a task is: the lifecycle column, and the chain&rsquo;s
      current step. They are separate fields, neither derives the other, and most
      confusing board behaviour is the gap between them. This is the panel to
      read before changing anything.
    </p>
    {claimsFor('dispatch').map((c) => <Claim key={c.id} c={c} />)}
    {wartsFor('dispatch').map((w) => <Wart key={w.id} w={w} />)}
  </div>
);

const ReleasesPanel = () => (
  <div>
    <h2 style={H2}>Release lifecycle</h2>
    <Provenance kind="cited" />
    <p style={LEAD}>
      How finished work becomes deployed work: a reviewed task is marked staged,
      the release lane turns the staged queue into a train of cars, the train
      merges them, and a sweep reconciles the board with what git actually
      contains.
    </p>
    {claimsFor('releases').map((c) => <Claim key={c.id} c={c} />)}
    {wartsFor('releases').map((w) => <Wart key={w.id} w={w} />)}
  </div>
);

const SourcesPanel = () => (
  <div>
    <h2 style={H2}>Where the rules live</h2>
    <Provenance kind="cited" />
    <p style={LEAD}>
      This manual shows what it can generate and cites the rest. Nothing here is
      a private copy, on purpose: the rules already live in several places, and a
      hand-written copy in this file would become one more of them and go stale
      first. These are the files every claim above points into.
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
      Not covered, deliberately: extracting this system into its own repo, any
      public-facing surface, and editing the rules from this modal. Describing
      the system is in scope; becoming the system is a much larger decision.
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
              Every section is either generated from the board&rsquo;s own model
              or pinned to the line that enforces it — warts included, and not a
              second copy of the rules
            </div>
          </div>
          <button style={{ ...input, cursor: 'pointer' }} onClick={onClose}>
            {'✕'} close
          </button>
        </div>

        <div style={{ display: 'flex', gap: 14, flex: 1, minHeight: 0 }}>
          <nav style={{
            width: 200, flexShrink: 0, borderRight: '1px solid var(--border)',
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
                <span style={{
                  float: 'right', fontSize: 9.5, letterSpacing: 0.3,
                  color: 'var(--text-tertiary)', textTransform: 'uppercase',
                  lineHeight: '18px',
                }}>{t.kind === 'rendered' ? 'gen' : 'cite'}</span>
              </button>
            ))}
            <div style={{
              fontSize: 11, color: 'var(--text-tertiary)', marginTop: 12,
              paddingTop: 10, borderTop: '1px solid var(--border)',
              lineHeight: 1.5,
            }}>
              {HELP_WARTS.length} known gotchas are called out inline, each with
              its board ref and the line that proves it.
            </div>
          </nav>

          <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', paddingRight: 4 }}>
            {topic === 'types' && <TypesPanel />}
            {topic === 'statuses' && <StatusesPanel />}
            {topic === 'assignees' && <AssigneesPanel tasks={tasks} />}
            {topic === 'chains' && <ChainsPanel />}
            {topic === 'stamps' && <StampsPanel />}
            {topic === 'dispatch' && <DispatchPanel />}
            {topic === 'releases' && <ReleasesPanel />}
            {topic === 'sources' && <SourcesPanel />}
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
