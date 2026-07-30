/**
 * Multi-select filter primitives for the team board (board #246).
 *
 * WHY A PLAIN .js MODULE IN A TYPESCRIPT APP — this is the one piece of the
 * tasks page whose BEHAVIOUR has to be tested rather than source-scanned, and
 * the frontend has no JS test runner (no jest/vitest, no `test` script). A
 * dependency-free ES-module .js file runs under bare `node`, so
 * `src/test_tasks_multiselect_filters_246.py` executes THIS FILE — not a
 * Python re-implementation of it, which would pass happily while the shipped
 * code did something else. tsconfig's `allowJs` + JSDoc types keep it
 * type-checked and Next-bundled like everything else.
 *
 * KEEP IT DEPENDENCY-FREE (no imports, no React, no DOM) or the test can no
 * longer run it and the rails below go back to being untested prose.
 */

/**
 * THE rail of board #246: an EMPTY selection means NO FILTER — show
 * everything — never "show nothing".
 *
 * This is the behaviour most likely to be got wrong by accident, and the cost
 * of getting it wrong is not cosmetic: an empty board reads as data loss.
 * Kevin twice on 07-30 concluded from a display artefact that work was being
 * dropped (the shipping-lane avatar, the stale ship steps). A filter that
 * defaults to hiding every row is a third one waiting to happen.
 *
 * @param {string[]} selected  chosen values; [] = no constraint
 * @param {string|null|undefined} value  the row's value ('' when unset)
 * @returns {boolean}
 */
export const matchesMulti = (selected, value) =>
  !selected || selected.length === 0 || selected.includes(value || '');

/**
 * Tick / untick one option. Returns a NEW array (the state is React state).
 * @param {string[]} selected
 * @param {string} value
 * @returns {string[]}
 */
export const toggleSelected = (selected, value) => {
  const cur = selected || [];
  return cur.includes(value) ? cur.filter((v) => v !== value) : [...cur, value];
};

/**
 * What the CLOSED control says. A filtered view must never look like an empty
 * board, so the control always states what it is doing: `all` when nothing is
 * ticked, the values themselves while they still fit, a count past that.
 *
 * @param {string[]} selected
 * @param {string} [allLabel='all']
 * @param {number} [maxNamed=2]
 * @returns {string}
 */
export const filterSummary = (selected, allLabel = 'all', maxNamed = 2) => {
  const cur = selected || [];
  if (cur.length === 0) return allLabel;
  if (cur.length <= maxNamed) return cur.map((v) => v || UNASSIGNED_LABEL).join(', ');
  return `${cur.length} selected`;
};

/** '' is a real assignee value (nobody), not "all" — multi-select frees it. */
export const UNASSIGNED_LABEL = '(unassigned)';

/**
 * The three multi-selects, combined: UNION within a filter (tick Review and
 * Staged → see both), INTERSECTION across filters (Review ∧ frontend).
 *
 * @param {{area?: string|null, assignee?: string|null, status?: string|null}} task
 * @param {{areas?: string[], assignees?: string[], statuses?: string[]}} sel
 * @returns {boolean}
 */
export const taskMatchesSelects = (task, sel) =>
  matchesMulti(sel.areas || [], task.area)
  && matchesMulti(sel.assignees || [], task.assignee)
  && matchesMulti(sel.statuses || [], task.status);

/**
 * How many of the three multi-selects are constraining the view. Drives the
 * "N filters active" tell next to the controls — the same
 * never-look-like-an-empty-board reason as filterSummary.
 * @param {{areas?: string[], assignees?: string[], statuses?: string[]}} sel
 * @returns {number}
 */
export const activeSelectCount = (sel) =>
  [sel.areas, sel.assignees, sel.statuses].filter((s) => s && s.length > 0).length;
