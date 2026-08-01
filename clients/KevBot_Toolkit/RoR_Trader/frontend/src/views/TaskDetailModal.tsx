/**
 * Three-panel task detail modal (Spec_Tasks_Team_Board.md Phase 3 + the
 * 07-22 design-review amendments, which supersede 3B where they differ).
 *
 * Layout: Context panel top-left with tab chips — leaf = [Summary][Process]
 * [Config], vision = [Summary][Config] — Activity thread right with the
 * composer pinned at the bottom, and (visions only) the subtask-selector
 * strip along the bottom-left. Leaf modals have NO bottom strip: their
 * checklist lives in the Process tab as a HANDOFF PIPELINE (steps
 * {text, done, role}; role = step owner; whole-array PATCH — partial JSONB
 * wipes). A step that needs its own thread gets promoted to a real subtask.
 *
 * The vision strip is a SELECTOR: picking a subtask re-scopes BOTH the
 * Context panel and the Activity thread to it in place — a posted comment
 * lands on the SELECTED subtask. "Summary" (first position) selects the
 * vision itself; "open full task ↗" swaps the modal in place (never stacks);
 * subtask modals get a "← parent" breadcrumb that returns with the selection
 * preserved.
 *
 * Board #134 (polish round 2): comments/descriptions render markdown-lite
 * (shared taskMarkdown, remark-breaks); the open modal polls its thread +
 * run state on a liveness tick (pulsing agent-working badge, comments slide in
 * live, onPollTick refreshes the board's chips); Ask AI posts with an @M
 * marker. Board #148: that poll is ONE consolidated /poll call every 15s,
 * paused while the tab is hidden, and it board-refetches only on real change.
 *
 * Board #136 (kanban lifecycle): Approval-stage stamp buttons (two-touch),
 * impact chip next to status, vision rows exempt from the pipeline stages.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch } from '@/lib/api/client';
import {
  Task, Comment, ChecklistStep, StepStamp, RunRow, AREAS, ASSIGNEES, ORIGINS,
  STATUS_COLOR, STATUS_DEF, statusOptionsFor, withLegacy, input, tagChip, relTime, elapsedShort,
  RoleChip, EmptyRoleCircle, roleColor, NextChip, ProgressBar, RolePicker, RunButton, OutcomeChip,
  StampButtons, TwoTouchChip, ImpactSelect, IMPACT_DEF, defaultChain, StuckChip,
  MENTION_ROLES, HandoffChain, isProcessChain, stepOwner, stepTitle, MODE_DEF,
  SlashItem, SlashMenu, filterSlashItems, applySlashInsert, detectSlash,
  makeChainStep, chainStarted, AiEligibleToggle, RunStatePill, LaneKindChip,
  isFinished, StampMode, taskTypeOf, TaskTypeIcon, TASK_TYPES,
  StepDeliverable, DeliverableKind, DELIVERABLE_KINDS, deliverableIcon,
  stepDeliverables, isDeliverableFilled, unfilledDeliverables, makeDeliverable,
  fileSize,
  GOAL_PARAM_FIELDS, AUTONOMY_DEF, Autonomy, autonomyOf, autonomyPatch,
  goalParamsOf, goalParamsPatch, missingGoalParams, renderGoalBlock,
  goalNestError, GoalFieldDef, inheritedAutonomy, isGoalLane,
} from './taskBoardShared';
import { Md, MD_CSS } from './taskMarkdown';

// `minHeight: 0` / `minWidth: 0` are BOTH load-bearing, and for the same reason
// (board #278): a flex item's `min-*: auto` refuses to shrink it below its
// content, so without these a panel's content dictates its box instead of the
// other way round. The height half was already here; the width half was not, and
// one long token in the Activity thread starved the Context panel next to it.
const panel: React.CSSProperties = {
  border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-card, var(--bg-input))',
  display: 'flex', flexDirection: 'column', minHeight: 0, minWidth: 0,
};
/**
 * Board #278 — the readability floor for the left column.
 *
 * `minWidth: 0` alone stops the starvation but says nothing about how narrow is
 * too narrow, so the two columns would just share the squeeze. The left panel is
 * what Kevin is reading when he authorises something, so it gets an explicit
 * floor: 340px, or 45% of the row on a viewport too small to give it 340. The
 * `min()` keeps the floor from becoming its own overflow bug on a phone.
 */
const CONTEXT_MIN_WIDTH = 'min(340px, 45%)';
const panelHead: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)',
  padding: '8px 12px', borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
  display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
};
const chipBtn = (active: boolean): React.CSSProperties => ({
  ...input, cursor: 'pointer', fontSize: 10.5, padding: '1px 8px', borderRadius: 10,
  fontWeight: active ? 700 : 400,
  borderColor: active ? 'var(--blue)' : 'var(--border)',
  color: active ? 'var(--blue)' : 'var(--text-secondary)',
});
const cfgLabel: React.CSSProperties = { fontSize: 12, color: 'var(--text-secondary)', display: 'block', marginBottom: 2 };
const cfgHint: React.CSSProperties = { fontSize: 11, color: 'var(--text-tertiary)' };

// Open-modal liveness cadence (board #148). Was 7s firing 3-4 calls/tick incl.
// a full 120-row board refetch; now 15s, ONE consolidated /poll call, and it
// skips ticks entirely while the tab is hidden.
const POLL_MS = 15000;

type Tab = 'summary' | 'process' | 'config' | 'goal' | 'deliverables';

/** ≤ 10 MB, mirrored from the API's _DELIVERABLE_FILE_MAX_BYTES. Checked here
 *  so an over-size file is refused BEFORE a 13 MB base64 round-trip, never
 *  INSTEAD of the server check — the API refusal is the real gate. */
const DELIVERABLE_MAX_BYTES = 10 * 1024 * 1024;

/**
 * Board #184 §7 (invariant I1) — the deliverables strip's ENTIRE vertical
 * budget, fixed. One header line plus one row of square cards, and the row
 * scrolls sideways rather than wrapping, so 3 deliverables and 30 cost the
 * column exactly the same 56px.
 *
 * The card is 40, not the spec's "~44": the slim scroll affordance has to live
 * INSIDE the 56 rather than push it, and a fixed strip that quietly grew by a
 * scrollbar's height would be the very thing I1 forbids. 40px still reads as a
 * square icon card at 18px glyphs.
 */
const STRIP_H = 56;
const STRIP_HEAD_H = 12;
const STRIP_CARD = 40;

/**
 * The hover detail for a strip card (spec §7). One `title` string — the modal's
 * existing tooltip convention — carrying what the card's icon cannot: which ask
 * it is, which step asked, what was delivered, and by whom. No tooltip library
 * for a strip of 40px squares.
 */
const deliverableTip = (
  d: StepDeliverable, step: ChecklistStep, idx: number,
): string => {
  const owner = stepOwner(step);
  const filled = isDeliverableFilled(d);
  const value = !filled ? 'not provided'
    : d.kind === 'file' ? `📎 ${d.filename || '(file)'} (${fileSize(d.size_bytes)})`
      : (d.value || '').slice(0, 200);
  return [
    `${d.label || '(unlabelled)'}${d.required ? ' *' : ''}`,
    `step ${idx + 1}${owner ? ` (${owner})` : ''}${step.done ? ' ✓' : ''}`,
    value,
    filled ? `${d.filled_by || '?'}${d.filled_at ? ` · ${relTime(d.filled_at)}` : ''}` : '',
  ].filter(Boolean).join(' · ');
};

/** One consolidated modal-poll round-trip (board #148, GET .../poll). */
interface PollResponse {
  comments: Comment[];
  runs: RunRow[];
  header_runs: RunRow[] | null;
  board_max_updated_at: string | null;
}

interface Props {
  task: Task;
  allTasks: Task[];
  visionOptions: Task[];
  roles?: string[];
  initialSelected?: number;
  patch: (id: number, fields: Partial<Task>) => Promise<void> | void;
  del: (id: number) => void;
  onClose: () => void;
  onOpenTask: (id: number, selectedSubtask?: number) => void;
  commentAuthor: string;
  onPickAuthor: (a: string) => void;
  /** Registry letters that are headless-enrolled (board #109 Run button). */
  headless?: Set<string>;
  /** Board refresh hook after a run request lands (tags changed server-side). */
  onRunRequested?: () => void;
  /** Fired on each liveness poll tick (board #134 item 5) so the board list
   *  can refresh its run / needs-review chips. Board #148: receives the board's
   *  max(updated_at) from the modal's consolidated poll probe — the board
   *  refetches its full task list ONLY when that value moved. Called with no
   *  arg to FORCE a refresh (e.g. after a stamp). */
  onPollTick?: (boardMaxUpdatedAt?: string | null) => void;
}

export default function TaskDetailModal({
  task, allTasks, visionOptions, roles = ASSIGNEES, initialSelected, patch, del, onClose, onOpenTask,
  commentAuthor, onPickAuthor, headless = new Set(['M-A']), onRunRequested, onPollTick,
}: Props) {
  const subtasks = useMemo(
    () => allTasks.filter((t) => t.parent_id === task.id), [allTasks, task.id]);
  // Board #261 — the row's TYPE, from the ONE map. This used to be a second,
  // hand-written copy of `isVisionTask`'s derivation; the modal and the board
  // must never disagree about what a row IS, so both now call `taskTypeOf`.
  const taskType = taskTypeOf(task, subtasks.length > 0);
  // CONTAINER, not literally vision (board #262). Everything below that used to
  // ask `taskType === 'vision'` meant "does this row hold subtasks" — which is
  // now two types. Read off the map's `children` flag so a goal gets the same
  // selector strip, the same roll-up counts and the same pipeline exemption a
  // vision gets; without this a goal renders as a LEAF and its children have
  // nowhere on the board to appear.
  const vision = !!TASK_TYPES[taskType]?.children;
  const isGoal = taskType === 'goal';
  const parent = task.parent_id != null ? allTasks.find((t) => t.id === task.parent_id) : undefined;

  // Selector state: which item the Context + Activity panels are scoped to.
  const [selected, setSelected] = useState<'summary' | number>(initialSelected ?? 'summary');
  const [tab, setTab] = useState<Tab>('summary');
  const scoped = useMemo(() => {
    if (vision && selected !== 'summary') {
      const sub = allTasks.find((t) => t.id === selected);
      if (sub) return sub;
    }
    return task;
  }, [vision, selected, allTasks, task]);

  const [threads, setThreads] = useState<Record<number, Comment[]>>({});
  const [draftComment, setDraftComment] = useState('');
  const [ctxEdit, setCtxEdit] = useState(false);
  const [ctxPreview, setCtxPreview] = useState(false);
  const [draftDesc, setDraftDesc] = useState('');
  const [newStep, setNewStep] = useState('');
  // @-mention compose picker (board #143): inserts an @token into the draft so
  // a mention row gets written server-side and lands in the peer's Messages
  // inbox. Simple insert — no inline autocomplete needed (spec).
  const [mentionOpen, setMentionOpen] = useState(false);
  const feedRef = useRef<HTMLDivElement>(null);
  // Board #182 Step 6 — the ClickUp-style `/` insert menu in the description
  // editor, and the structured step form (process-module insert / step edit).
  // descRef restores the caret after a content insert replaces the `/trigger`.
  const descRef = useRef<HTMLTextAreaElement>(null);
  const [slash, setSlash] = useState<{ open: boolean; filter: string; start: number; end: number; active: number }>(
    { open: false, filter: '', start: 0, end: 0, active: 0 });
  const [stepForm, setStepForm] = useState<{ mode: 'insert' | 'edit'; index: number } | null>(null);

  const loadThread = useCallback(async (id: number) => {
    try {
      const cs = await apiFetch<Comment[]>(`/api/dev-tasks/${id}/comments`);
      setThreads((m) => ({ ...m, [id]: cs || [] }));
    } catch { /* ignore */ }
  }, []);

  // Dispatcher run history (board #109) — per-task, newest first; keyed like
  // threads so the vision/subtask selector re-scopes it too.
  const [runsMap, setRunsMap] = useState<Record<number, RunRow[]>>({});
  const [runErr, setRunErr] = useState<string | null>(null);
  const loadRuns = useCallback(async (id: number) => {
    try {
      const rs = await apiFetch<RunRow[]>(`/api/run-history?task_id=${id}&limit=50`);
      setRunsMap((m) => ({ ...m, [id]: rs || [] }));
    } catch { /* panel just stays empty */ }
  }, []);
  useEffect(() => { loadRuns(task.id); }, [task.id, loadRuns]);
  const requestRun = async (id: number) => {
    setRunErr(null);
    try {
      await apiFetch(`/api/dev-tasks/${id}/run-request`, {
        method: 'POST', body: JSON.stringify({ author: commentAuthor }),
      });
    } catch (e) { setRunErr(String(e)); }
    loadRuns(id); loadThread(id);
    onRunRequested?.();
  };

  // THE Approval stamp (board #136, one mode since #232 — "approved to start"):
  // the server flips Approval → Todo, arms the task and logs the system comment
  // — refresh the thread here and FORCE a board refresh via onPollTick() with no
  // arg (board #148: the status changed, so don't wait on the probe).
  const stampTask = async (id: number, mode: StampMode) => {
    try {
      await apiFetch(`/api/dev-tasks/${id}/stamp`, {
        method: 'POST', body: JSON.stringify({ mode, author: commentAuthor }),
      });
    } catch (e) { setRunErr(String(e)); }
    loadThread(id);
    onPollTick?.();
  };

  // ── Process-chain steps (board #182) ───────────────────────────────────────
  // Which accordion sections are expanded. Default-expand is re-derived on
  // scope/viewer change (the effect below); a manual toggle or a checklist edit
  // does NOT yank sections open again.
  const [openSteps, setOpenSteps] = useState<Set<string>>(new Set());
  const toggleStep = useCallback((key: string) => {
    setOpenSteps((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }, []);
  // Step 4: the step whose owner matches the acting role expands by default;
  // every other section stays freely collapsible. Fall back to the current
  // (first-incomplete) step so something is always open — Kevin: "I'm not
  // saying we should collapse it and lock it so no one else can access it."
  useEffect(() => {
    const cl = scoped.checklist || [];
    if (!isProcessChain(cl)) { setOpenSteps(new Set()); return; }
    const next = new Set<string>();
    const cur = cl.findIndex((s) => !s.done);
    cl.forEach((s, i) => {
      if (!s.done && stepOwner(s) && stepOwner(s) === commentAuthor) next.add(s.id || String(i));
    });
    if (next.size === 0 && cur >= 0) next.add(cl[cur].id || String(cur));
    setOpenSteps(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scoped.id, commentAuthor]);

  // Step 5: the three server-owned step actions (complete-and-hand-off, stamp
  // approve/reject, raise-issue → M). Completion/hand-off/stamps are NOT a
  // checklist PATCH — they hit dedicated endpoints whose transition rules are
  // API refusals. After any of them, refresh the thread and FORCE a board
  // refetch (no-arg onPollTick) so the task prop's checklist + reassigned
  // assignee update in place.
  const [stepBusy, setStepBusy] = useState(false);
  const [stepErr, setStepErr] = useState<string | null>(null);
  const stepAction = async (
    action: 'complete' | 'stamp' | 'raise-issue', body: Record<string, unknown> = {},
  ) => {
    setStepErr(null); setStepBusy(true);
    try {
      await apiFetch(`/api/dev-tasks/${scoped.id}/steps/${action}`, {
        method: 'POST', body: JSON.stringify({ ...body, actor: commentAuthor }),
      });
      loadThread(scoped.id);
      onPollTick?.();
    } catch (e) { setStepErr(String(e)); }
    finally { setStepBusy(false); }
  };
  const stampStep = (decision: 'approve' | 'reject') => stepAction('stamp', { decision });
  const raiseIssue = (issueBody: string) => {
    if (issueBody.trim()) stepAction('raise-issue', { body: issueBody.trim() });
  };

  // ── Step deliverables (board #184) ─────────────────────────────────────────
  // Fills are SERVER-owned exactly as completion is: they go to dedicated
  // endpoints, never through the checklist PATCH (which 409s on a fill field).
  // Specs, by contrast, ARE authored through the checklist PATCH — that split
  // is why `setSteps` handles the StepForm's deliverable rows and these
  // handlers handle the values.
  const deliverableAction = async (
    path: string, init: RequestInit,
  ): Promise<boolean> => {
    setStepErr(null); setStepBusy(true);
    try {
      await apiFetch(`/api/dev-tasks/${scoped.id}/steps/deliverables/${path}`, init);
      loadThread(scoped.id);
      onPollTick?.();
      return true;
    } catch (e) { setStepErr(String(e)); return false; }
    finally { setStepBusy(false); }
  };
  const fillDeliverable = (id: string, value: string) => deliverableAction(id, {
    method: 'POST', body: JSON.stringify({ value, actor: commentAuthor }),
  });
  const clearDeliverable = (id: string) => deliverableAction(
    `${id}/value?actor=${encodeURIComponent(commentAuthor)}`, { method: 'DELETE' });
  /** Read the picked file as base64 and POST it. JSON+base64, not multipart —
   *  `apiFetch` pins Content-Type: application/json on every request, and the
   *  API has no multipart dependency (see upload_deliverable's docstring). */
  const uploadDeliverable = async (id: string, file: File) => {
    if (file.size > DELIVERABLE_MAX_BYTES) {
      setStepErr(`"${file.name}" is ${Math.round(file.size / 1024 / 1024)} MB — the limit is 10 MB`);
      return;
    }
    const b64 = await new Promise<string>((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = () => reject(new Error('could not read that file'));
      fr.onload = () => resolve(String(fr.result).split(',')[1] || '');
      fr.readAsDataURL(file);
    }).catch((e) => { setStepErr(String(e)); return ''; });
    if (!b64) return;
    await deliverableAction(`${id}/upload`, {
      method: 'POST',
      body: JSON.stringify({
        filename: file.name, content_type: file.type || null, b64,
        actor: commentAuthor,
      }),
    });
  };
  /** Signed read URL, minted ON CLICK (spec §3.4) — the bucket is private and
   *  the board fetch must stay cheap, so nothing is signed on load. */
  const openDeliverableFile = async (id: string) => {
    setStepErr(null);
    try {
      const res = await apiFetch<{ url: string }>(
        `/api/dev-tasks/${scoped.id}/steps/deliverables/${id}/url`);
      if (res?.url) window.open(res.url, '_blank', 'noopener');
    } catch (e) { setStepErr(String(e)); }
  };

  /**
   * The strip card's ONE click (spec §7: "the natural action … one click, no
   * menu"). The action follows the card's STATE, not a menu: a filled file
   * opens its signed URL, a filled link opens the URL, filled text expands into
   * the deliverables tab.
   *
   * An UNFILLED card has no value to open, and the spec's stated fallback (the
   * tab) is a dead end there — the tab is read-only and can only say
   * "— not provided". So an unfilled deliverable on a still-open step routes to
   * the place that can actually satisfy it: the process accordion, with that
   * step already expanded. On a COMPLETED step nothing is fillable any more
   * (fills 409 there), so the read-only tab is the right destination after all.
   */
  const openDeliverableCard = (
    d: StepDeliverable, step: ChecklistStep, idx: number,
  ) => {
    if (isDeliverableFilled(d)) {
      if (d.kind === 'file' && d.id) { openDeliverableFile(d.id); return; }
      if (d.kind === 'link' && d.value) { window.open(d.value, '_blank', 'noopener'); return; }
      setTab('deliverables');
      return;
    }
    if (step.done) { setTab('deliverables'); return; }
    setTab('process');
    setOpenSteps((prev) => new Set(prev).add(step.id || String(idx)));
  };

  // T10's UI half. Required-and-unfilled BLOCKS (the server refuses it anyway —
  // this only makes the wall visible before the click); optional-and-unfilled
  // asks for a confirm and is never a block (decision D2). The confirm is the
  // volatile half of "visible, not silent"; the durable half is the system
  // comment the server appends, which outlives any dialog.
  const completeStep = () => {
    const cl = scoped.checklist || [];
    const cur = cl.find((s) => !s.done);
    const skipped = unfilledDeliverables(cur, false);
    if (skipped.length && !window.confirm(
      `${skipped.length} optional deliverable${skipped.length === 1 ? '' : 's'} `
      + `${skipped.length === 1 ? 'is' : 'are'} unfilled:\n\n`
      + skipped.map((d) => `  • ${d.label}`).join('\n')
      + '\n\nComplete the step anyway? (they will be named on the thread)')) return;
    stepAction('complete');
  };

  useEffect(() => { loadThread(scoped.id); loadRuns(scoped.id); setCtxEdit(false); setCtxPreview(false); }, [scoped.id, loadThread, loadRuns]);
  useEffect(() => { setDraftDesc(scoped.description || ''); }, [scoped.id, scoped.description]);
  const thread = threads[scoped.id] || [];

  // Real-time activity (board #134 item 5, reworked board #148): while the
  // modal is open, poll the scoped thread + run state — new comments land
  // live, the agent-working badge tracks run_history, and onPollTick lets the
  // board refresh its chips. ONE consolidated /poll call per tick replaces the
  // old 3-4 (comments + runs(scoped) + runs(task) + a full board refetch); it
  // skips ticks entirely while the tab is hidden (resuming on focus), and the
  // board-change probe (board_max_updated_at) means the board's 120-row list
  // is refetched only when it actually changed. The interval is torn down on
  // unmount, so a closed modal polls nothing.
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      if (cancelled) return;
      // No point polling a modal nobody is looking at.
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
      const sid = scoped.id;
      try {
        const res = await apiFetch<PollResponse>(
          `/api/dev-tasks/${task.id}/poll${sid !== task.id ? `?scoped_id=${sid}` : ''}`);
        if (cancelled || !res) return;
        setThreads((m) => ({ ...m, [sid]: res.comments || [] }));
        setRunsMap((m) => {
          const next = { ...m, [sid]: res.runs || [] };
          if (res.header_runs) next[task.id] = res.header_runs;
          return next;
        });
        onPollTick?.(res.board_max_updated_at ?? null);
      } catch { /* transient — the next tick retries */ }
    };
    const iv = setInterval(poll, POLL_MS);
    // Resume promptly when the tab regains focus (ticks are skipped while hidden).
    const onVis = () => { if (document.visibilityState === 'visible') poll(); };
    document.addEventListener('visibilitychange', onVis);
    return () => {
      cancelled = true;
      clearInterval(iv);
      document.removeEventListener('visibilitychange', onVis);
    };
  }, [scoped.id, task.id, onPollTick]);

  // Auto-scroll: jump to the bottom on scope change; on live growth only
  // follow when the reader is already near the bottom (don't yank them out
  // of history mid-read).
  const lastScrollScope = useRef<number | null>(null);
  useEffect(() => {
    const el = feedRef.current;
    if (!el) return;
    const scopeChanged = lastScrollScope.current !== scoped.id;
    lastScrollScope.current = scoped.id;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 300;
    if (scopeChanged || nearBottom) el.scrollTop = el.scrollHeight;
  }, [thread.length, scoped.id]);

  // LIVENESS: an active dispatcher run of the scoped item (outcome=running).
  const liveRun = (runsMap[scoped.id] || []).find((r) => r.outcome === 'running');

  // Status/assignee edits write a system entry on the task's thread server-side
  // — re-fetch it so the entry appears without closing the modal.
  const patchTracked = async (fields: Partial<Task>) => {
    await patch(task.id, fields);
    loadThread(task.id);
  };

  // prefix: '' = plain comment; '@M ' = Ask AI (board #134 item 4) — the @M
  // marker is what M's board-watcher scans for, so it answers in-thread.
  const postComment = async (prefix = '') => {
    if (!draftComment.trim()) return;
    try {
      await apiFetch(`/api/dev-tasks/${scoped.id}/comments`, {
        method: 'POST', body: JSON.stringify({ body: prefix + draftComment, author: commentAuthor }),
      });
      setDraftComment('');
      loadThread(scoped.id);
    } catch { /* surfaced by thread staying unchanged */ }
  };

  const saveDesc = () => {
    if (draftDesc !== (scoped.description || '')) patch(scoped.id, { description: draftDesc });
    setCtxEdit(false); setCtxPreview(false);
  };

  // Append an @role token to the draft (board #143). Adds a leading space so
  // it doesn't glue onto the previous word, and a trailing space to keep typing.
  const insertMention = (role: string) => {
    setDraftComment((d) => `${d}${d && !d.endsWith(' ') ? ' ' : ''}@${role} `);
    setMentionOpen(false);
  };
  // Registry roles for the picker (falls back to the static token set).
  //
  // Board #289 — per-goal lanes (`G281`) are ASSIGNABLE but NOT mentionable, and
  // the asymmetry is deliberate. `dev_tasks._write_mentions` only writes a
  // `task_mentions` row for a letter that EXISTS in the agents registry, and a
  // goal lane has no row there by design; an `@G281` would post, be dropped
  // silently, and never reach an inbox. Offering a dead token is worse than
  // offering none. Making goal mentions work is an API change, not a UI one.
  const pickableMentions = roles.filter((r) => r && !isGoalLane(r));
  const mentionRoles = pickableMentions.length ? pickableMentions : MENTION_ROLES;

  // Checklist ops operate on the SCOPED item (the selected pipeline row, or
  // the task itself) — ALWAYS send the whole array (JSONB replace).
  // Memoized because #184's deliverables useMemo depends on it: `|| []` mints a
  // fresh array every render, which would make that memo recompute every time.
  const steps: ChecklistStep[] = useMemo(() => scoped.checklist || [], [scoped]);
  const isChain = isProcessChain(steps);   // #182 rich chain vs legacy checklist
  const setSteps = (next: ChecklistStep[]) => patch(scoped.id, { checklist: next });
  const addStep = () => {
    if (!newStep.trim()) return;
    // On a #182 chain a quick-add is a real chain step (audible if the chain is
    // under way); legacy checklists keep their {text,done,role} shape (Step 6).
    setSteps(isChain
      ? [...steps, makeChainStep({ title: newStep.trim(), owner: null }, chainStarted(steps))]
      : [...steps, { text: newStep.trim(), done: false, role: null }]);
    setNewStep('');
  };
  const moveStep = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
  };

  // ── Step 6: structured insert / edit / remove of chain steps ────────────────
  // All route through setSteps → the whole-array checklist PATCH, which the API
  // accepts as a STRUCTURAL edit (add/edit/reorder of not-done steps) while
  // completion stays server-owned. Inserting while the chain is under way
  // auto-marks origin='audible' (makeChainStep) so course-corrections show.
  const started = chainStarted(steps);
  const currentIdx = steps.findIndex((s) => !s.done);   // -1 → chain complete
  const submitStepForm = (f: {
    owner: string | null; title: string; mode: 'execute' | 'discuss'; body: string;
    deliverables: ChecklistStep['deliverables'];
  }) => {
    if (!stepForm) return;
    // Board #184: `deliverables` rides the same STRUCTURAL PATCH the rest of the
    // step does — specs are author-owned. The form never emits a fill value (an
    // existing row keeps its server-owned fields verbatim; a new one is minted
    // unfilled), so this stays a structural edit and the API's fill guard is
    // never tripped. An empty list is omitted entirely, so a step that declares
    // no deliverables writes no `deliverables` key at all — the back-compat rail.
    const dels = f.deliverables && f.deliverables.length
      ? { deliverables: f.deliverables } : {};
    if (stepForm.mode === 'insert') {
      const step = { ...makeChainStep(f, started), ...dels };
      setSteps([...steps.slice(0, stepForm.index), step, ...steps.slice(stepForm.index)]);
    } else {
      // Edit touches only owner/title/mode/body/deliverables — id, origin, stamp,
      // done and completion provenance are preserved so the PATCH stays structural.
      setSteps(steps.map((s, i) => {
        if (i !== stepForm.index) return s;
        const next: ChecklistStep = {
          ...s, owner: f.owner, title: f.title, mode: f.mode, body: f.body, ...dels,
        };
        if (!f.deliverables?.length) delete next.deliverables;
        return next;
      }));
    }
    setStepForm(null);
  };
  const removeStep = (i: number) => {
    if (steps[i]?.done) return;   // completed steps are immutable (T4'; API 409s too)
    setSteps(steps.filter((_, j) => j !== i));
  };

  // Slash menu: detect a `/trigger` at the caret while editing the description,
  // filter the insert menu live, and act on a pick — content blocks drop text at
  // the caret; the process module opens the step form (insert after the current
  // step). Arrow/Enter/Escape are handled on the textarea while the menu is open.
  // The process module is offered only where inserting a chain step is SAFE and
  // intended: an existing chain, or an empty checklist (starting a fresh chain).
  // On a legacy checklist that already has steps the API would refuse the insert
  // (its done-but-idless legacy steps trip the "inserted step can't start done"
  // guard), and silently converting a legacy task is exactly what "legacy tasks
  // render unchanged" forbids — so it stays hidden there. Content inserts (text
  // only) are always available.
  const allowStepInsert = isChain || steps.length === 0;
  const slashItems = slash.open
    ? filterSlashItems(slash.filter).filter((it) => it.kind !== 'step' || allowStepInsert)
    : [];
  const onDescChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value;
    setDraftDesc(val);
    const hit = detectSlash(val, e.target.selectionStart ?? val.length);
    if (hit) setSlash({ open: true, filter: hit.filter, start: hit.start, end: hit.end, active: 0 });
    else if (slash.open) setSlash((s) => ({ ...s, open: false }));
  };
  const pickSlash = (item: SlashItem) => {
    if (item.kind === 'text') {
      const { text, caret } = applySlashInsert(draftDesc, slash.start, slash.end, item);
      setDraftDesc(text);
      setSlash((s) => ({ ...s, open: false }));
      requestAnimationFrame(() => {
        const el = descRef.current;
        if (el) { el.focus(); el.setSelectionRange(caret, caret); }
      });
    } else {
      // process module — strip the `/trigger`, then open the insert form after
      // the current step (a course-correction enters "as the following step").
      setDraftDesc(draftDesc.slice(0, slash.start) + draftDesc.slice(slash.end));
      setSlash((s) => ({ ...s, open: false }));
      setStepForm({ mode: 'insert', index: currentIdx < 0 ? steps.length : currentIdx + 1 });
    }
  };
  const onDescKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (!slash.open || slashItems.length === 0) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setSlash((s) => ({ ...s, active: (s.active + 1) % slashItems.length })); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSlash((s) => ({ ...s, active: (s.active - 1 + slashItems.length) % slashItems.length })); }
    else if (e.key === 'Enter') { e.preventDefault(); pickSlash(slashItems[Math.min(slash.active, slashItems.length - 1)]); }
    else if (e.key === 'Escape') { e.preventDefault(); setSlash((s) => ({ ...s, open: false })); }
  };

  const parseIds = (s: string) =>
    s.split(',').map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n));
  const parseTags = (s: string) => s.split(',').map((x) => x.trim()).filter(Boolean);

  // Header-level counts describe the MODAL task; the Process tab describes
  // the scoped item (a selected subtask is always a leaf — one-level rule).
  const doneCount = vision
    ? subtasks.filter((s) => isFinished(s.status)).length   // FINISHED, not just Done (#232)
    : (task.checklist || []).filter((s) => s.done).length;
  const totalCount = vision ? subtasks.length : (task.checklist || []).length;
  const stripRows = vision ? subtasks : [];

  const scopedIsLeaf = scoped.id === task.id ? !vision : true;
  const scopedDone = steps.filter((s) => s.done).length;
  // Board #262 — the Goal tab exists ONLY on a goal-typed row, and only when
  // the scope is that row (a selected subtask is an action task; it has no
  // goal params of its own). Same fall-back shape as the Process tab.
  const goalTabOk = isGoal && scoped.id === task.id;
  // Surfaced on the tab chip itself, so "this goal cannot start yet" is visible
  // without opening the tab.
  const goalMissing = useMemo(
    () => (isGoal ? missingGoalParams(task) : []), [isGoal, task]);
  // The type of whatever the Config tab is SHOWING (the modal task, or a
  // selected subtask) — the nesting guard is rendered against this, not the
  // header task's type.
  const scopedType = useMemo(
    () => taskTypeOf(scoped, allTasks.some((t) => t.parent_id === scoped.id)),
    [scoped, allTasks]);
  // The SCOPED item's parent — not the modal task's (`parent` above). When a
  // goal's subtask is selected, the container IS the modal task.
  const scopedParent = useMemo(
    () => (scoped.parent_id != null
      ? allTasks.find((t) => t.id === scoped.parent_id) || null : null),
    [scoped.parent_id, allTasks]);
  const parentIsGoal = !!scopedParent && taskTypeOf(scopedParent, true) === 'goal';
  // Board #184 §6 — the deliverables TAB (Half 2 Phase 1; the strip is a later
  // step, Kevin's order: tab before strip). It exists ONLY when the scoped
  // chain actually declares a deliverable, so a task with none gets no new
  // chrome at all. Scope FOLLOWS the context panel — selecting a subtask
  // re-scopes it exactly as it re-scopes the accordion; cross-subtask
  // aggregation on a vision is a deliberate non-goal (spec §8).
  const scopedDeliverables = useMemo(
    () => steps.flatMap((s, i) => stepDeliverables(s).map((d) => ({ d, step: s, i }))),
    [steps]);
  const deliverablesTabOk = scopedIsLeaf && scopedDeliverables.length > 0;
  const deliverablesFilled = scopedDeliverables.filter((r) => isDeliverableFilled(r.d)).length;
  // Process tab is invalid when the scope is the vision itself — fall back.
  const effTab: Tab = (tab === 'process' && !scopedIsLeaf)
    || (tab === 'goal' && !goalTabOk)
    || (tab === 'deliverables' && !deliverablesTabOk) ? 'summary' : tab;

  const selectRow = (id: number) => setSelected(id);

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4vh 16px',
    }}>
      <style>{`
        ${MD_CSS}
        @keyframes taskPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        @keyframes cmSlideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
        .cm-in { animation: cmSlideIn 0.35s ease; }
        .pulse-dot {
          display: inline-block; width: 8px; height: 8px; border-radius: 50%;
          background: var(--blue); animation: taskPulse 1.4s ease-in-out infinite;
        }
        /* Board #184 §7 (I1) — the deliverables strip scrolls SIDEWAYS and never
           wraps, so the scroll affordance has to be slim enough to fit inside the
           strip's fixed height instead of adding to it. Thin, not hidden: hiding
           it would leave no cue that there are more cards off the right edge. */
        .dlv-strip { scrollbar-width: thin; }
        .dlv-strip::-webkit-scrollbar { height: 4px; }
        .dlv-strip::-webkit-scrollbar-thumb {
          background: var(--border); border-radius: 4px;
        }
      `}</style>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, width: '90vw', maxWidth: 1200, height: '86vh',
        display: 'flex', flexDirection: 'column', padding: 16,
        boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        {/* ── Breadcrumb (subtask modals) + tier label ───────────── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 2 }}>
          {parent && (
            <button style={{ ...chipBtn(true), fontSize: 11.5 }}
              title="back to the parent vision (this subtask stays selected)"
              onClick={() => onOpenTask(parent.id, task.id)}>
              ← {parent.title.slice(0, 48)}
            </button>
          )}
          <span style={{ fontSize: 10.5, color: 'var(--text-tertiary)', letterSpacing: 0.6 }}>
            {/* board #261 — the type glyph, same one the board row wears */}
            <TaskTypeIcon type={taskType} />
            {TASK_TYPES[taskType].children
              ? `${TASK_TYPES[taskType].label.toUpperCase()} ITEM`
              : task.parent_id != null ? 'SUBTASK' : 'TASK'} · #{task.id}
          </span>
        </div>

        {/* ── Title ──────────────────────────────────────────────── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <input style={{ ...input, fontSize: 16, fontWeight: 600, flex: 1, padding: '5px 8px' }}
            value={task.title}
            onChange={(e) => patch(task.id, { title: e.target.value })} />
          <button style={{ ...input, cursor: 'pointer' }} onClick={onClose}>✕ close</button>
        </div>

        {/* ── Slim header row (amendments #5): status · assignee · Next ·
               progress · pri · set-flags · 🔍 · (vision) n/m ─────── */}
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', margin: '8px 0' }}>
          <select style={{ ...input, color: STATUS_COLOR[task.status] }} value={task.status}
            title={STATUS_DEF[task.status] || ''}
            onChange={(e) => patchTracked({ status: e.target.value })}>
            {statusOptionsFor(task, taskType).map((s) =>
              <option key={s} value={s} title={STATUS_DEF[s]}>{s}</option>)}
          </select>
          {/* impact chip lives next to status (board #136) */}
          <ImpactSelect t={task} onPick={(v) => patch(task.id, { impact: v })} />
          <RolePicker value={task.assignee} pickTitle="assignee" allowEmpty
            options={withLegacy(roles, task.assignee || '').filter(Boolean)}
            onPick={(r) => patchTracked({ assignee: r })} />
          {/* board #222 — the LABEL half of the session/agent cue. The avatar's
              shape already says it; here there is room for the word, and this is
              the header where "why has nothing run on this?" gets asked. */}
          <LaneKindChip role={task.assignee || ''} />
          <NextChip t={task} subtasks={subtasks} />
          {/* board #169 — handoff chain as owner avatars (mirrors the board card);
              a compact read of the same checklist the Process tab edits */}
          <HandoffChain task={task} />
          {/* board #151 tell — Todo + Kevin-next = queue-eligible but undispatchable */}
          <StuckChip t={task} subtasks={subtasks} />
          <ProgressBar done={doneCount} total={totalCount} />
          <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }} title="priority phase.seq (edit in Config)">
            {task.priority_phase}.{task.priority_seq}
          </span>
          {task.impacts_live &&
            <span title="touches live engine/trading code — deploy carefully" style={{ fontSize: 13 }}>🔴</span>}
          {task.needs_live_validation &&
            <span title="needs live-market data to confirm" style={{ fontSize: 13 }}>⏳</span>}
          {task.is_urgent &&
            <span title="urgent — jump the queue" style={{ fontSize: 13 }}>⚡</span>}
          {task.parent_id != null && task.origin === 'discovered' &&
            <span title="discovered mid-work (rabbit-hole fix)" style={{ fontSize: 13 }}>🔍</span>}
          <TwoTouchChip t={task} />
          {/* Approval-stage stamp buttons (board #136) — full labels here */}
          <StampButtons t={task} onStamp={stampTask} />
          {vision && totalCount > 0 &&
            <span style={tagChip} title="subtasks done / total">{doneCount}/{totalCount}</span>}
          <span style={{ flex: 1 }} />
          {/* board #198 — the three controls, deliberately distinct: standing
              permission (switch) · what the dispatcher is doing (pill) · run it
              once now (button). The task-level toggle sits here; the Config tab
              carries the same switch scoped to the selected subtask. */}
          <AiEligibleToggle task={task}
            onToggle={(v) => patch(task.id, { ai_eligible: v })} />
          <RunStatePill task={task} latestRun={(runsMap[task.id] || [])[0]} />
          <RunButton task={task} allTasks={allTasks} headless={headless}
            latestRun={(runsMap[task.id] || [])[0]} onRequest={requestRun} />
          {runErr && <span style={{ color: 'var(--red)', fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={runErr}>{runErr}</span>}
        </div>

        {/* ── Body: left (context [+ vision strip]) · right (activity) ──
            Board #278: BOTH children carry `minWidth`. The left one keeps a
            readable floor (CONTEXT_MIN_WIDTH); the right one is free to shrink
            (`minWidth: 0` on `panel`) so a long token in a comment can never
            make the Activity column the one that dictates the split. */}
        <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
          <div style={{
            flex: 2, display: 'flex', flexDirection: 'column', gap: 10,
            minWidth: CONTEXT_MIN_WIDTH,
          }}>

            {/* Context panel with tab chips */}
            <div style={{ ...panel, flex: 1 }}>
              <div style={panelHead}>
                <span>Context · {scoped.id === task.id ? 'what to do' : `subtask #${scoped.id}`}</span>
                <button style={chipBtn(effTab === 'summary')} onClick={() => setTab('summary')}>summary</button>
                {scopedIsLeaf && (
                  <button style={chipBtn(effTab === 'process')} onClick={() => setTab('process')}>
                    process {steps.length > 0 ? `${scopedDone}/${steps.length}` : ''}
                  </button>
                )}
                {goalTabOk && (
                  <button style={chipBtn(effTab === 'goal')} onClick={() => setTab('goal')}
                    title="goal parameters + the context block for Claude’s built-in /goal">
                    ⚑ goal {goalMissing.length ? `· ${goalMissing.length} missing` : ''}
                  </button>
                )}
                <button style={chipBtn(effTab === 'config')} onClick={() => setTab('config')}>config</button>
                {deliverablesTabOk && (
                  <button style={chipBtn(effTab === 'deliverables')} onClick={() => setTab('deliverables')}
                    title="everything delivered against this task — files, links and answers, in one place">
                    deliverables {deliverablesFilled}/{scopedDeliverables.length}
                  </button>
                )}
                <span style={{ flex: 1 }} />
                {effTab === 'summary' && !ctxEdit &&
                  <button style={chipBtn(true)} onClick={() => setCtxEdit(true)}>edit</button>}
                {effTab === 'summary' && ctxEdit && (
                  <>
                    <button style={chipBtn(!ctxPreview)} onClick={() => setCtxPreview(false)}>write</button>
                    <button style={chipBtn(ctxPreview)} onClick={() => setCtxPreview(true)}>preview</button>
                    <button style={{ ...chipBtn(true), background: 'var(--blue)', color: '#fff' }}
                      onClick={saveDesc}>save</button>
                  </>
                )}
              </div>
              <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
                {effTab === 'summary' && (
                  <>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{scoped.title}</div>
                    {!ctxEdit && <Md text={scoped.description} fallback="_no description yet — click Edit_" />}
                    {ctxEdit && !ctxPreview && (
                      <div style={{ position: 'relative' }}>
                        <textarea ref={descRef}
                          style={{ ...input, width: '100%', minHeight: 180, fontFamily: 'monospace', fontSize: 12.5 }}
                          value={draftDesc} onChange={onDescChange} onKeyDown={onDescKeyDown}
                          placeholder="markdown + sanitized inline HTML…   ·   type / to insert" />
                        {slash.open && (
                          <SlashMenu items={slashItems} filter={slash.filter} active={slash.active}
                            onPick={pickSlash} onClose={() => setSlash((s) => ({ ...s, open: false }))} />
                        )}
                      </div>
                    )}
                    {ctxEdit && ctxPreview && <Md text={draftDesc} fallback="_no description yet — click Edit_" />}
                    {/* Board #182 Step 4: the process chain IS the body of the
                        summary tab — summary text above, the ordered accordion
                        below. Only for #182 chains; legacy checklists keep the
                        flat Process tab untouched (feature detection). */}
                    {!ctxEdit && isChain && scopedIsLeaf && (
                      <ProcessChain steps={steps} viewer={commentAuthor}
                        openSteps={openSteps} onToggle={toggleStep}
                        busy={stepBusy} err={stepErr}
                        onComplete={completeStep} onStamp={stampStep} onRaiseIssue={raiseIssue}
                        onInsert={(i) => setStepForm({ mode: 'insert', index: i })}
                        onEdit={(i) => setStepForm({ mode: 'edit', index: i })}
                        onRemove={removeStep} onMove={moveStep}
                        onFillDeliverable={fillDeliverable}
                        onClearDeliverable={clearDeliverable}
                        onUploadDeliverable={uploadDeliverable}
                        onOpenDeliverable={openDeliverableFile} />
                    )}
                  </>
                )}

                {effTab === 'process' && scopedIsLeaf && (
                  <>
                    <div style={{ ...cfgHint, marginBottom: 8 }}>
                      {isChain
                        ? 'Structural editor — reorder, delete, or reassign steps here; the SOP body, stamps and completion live in the accordion on the Summary tab (Complete & hand off ticks a step, never this checkbox).'
                        : 'Handoff pipeline — each step’s chip is its owner; reassign the task at each handoff point. A step that needs its own thread gets promoted to a subtask.'}
                    </div>
                    {steps.map((s, i) => (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px',
                        borderRadius: 8, fontSize: 13, marginBottom: 2,
                      }}>
                        <input type="checkbox" checked={s.done} disabled={isChain}
                          title={isChain
                            ? 'completion is managed on the Summary tab (Complete & hand off) — the API refuses a checklist PATCH that ticks a chain step'
                            : undefined}
                          onChange={(e) => setSteps(steps.map((x, j) => j === i ? { ...x, done: e.target.checked } : x))} />
                        <span style={{
                          flex: 1, minWidth: 0,
                          textDecoration: s.done ? 'line-through' : 'none', opacity: s.done ? 0.6 : 1,
                        }}>{stepTitle(s)}</span>
                        <RolePicker value={stepOwner(s)} pickTitle="step owner" allowEmpty
                          options={roles.filter(Boolean)}
                          onPick={(r) => setSteps(steps.map((x, j) => j === i
                            ? (isChain ? { ...x, owner: r || null } : { ...x, role: r || null }) : x))} />
                        {/* board #222 — quiet unless the step is owned by a live
                            session, which is the case that never auto-dispatches;
                            this is the row where that gets decided. */}
                        <LaneKindChip role={stepOwner(s) || ''} sessionOnly />
                        <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move up" onClick={() => moveStep(i, -1)}>↑</span>
                        <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move down" onClick={() => moveStep(i, 1)}>↓</span>
                        <span style={{ cursor: 'pointer', color: 'var(--red)' }} title="delete step"
                          onClick={() => setSteps(steps.filter((_, j) => j !== i))}>✕</span>
                      </div>
                    ))}
                    {steps.length === 0 && (
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: '4px 8px' }}>
                        <span style={cfgHint}>No steps yet.</span>
                        <button style={{ ...input, cursor: 'pointer', fontSize: 12 }}
                          title="Build/investigate → M review → Kevin approval (where flagged) → Ship/close — editable after adding"
                          onClick={() => setSteps(defaultChain(scoped.assignee))}>⛓ add default chain</button>
                      </div>
                    )}
                    <div style={{ display: 'flex', gap: 6, padding: '4px 8px', marginTop: 2 }}>
                      <input style={{ ...input, flex: 1 }} placeholder="add a step…" value={newStep}
                        onChange={(e) => setNewStep(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && addStep()} />
                      <button style={{ ...input, cursor: 'pointer' }} onClick={addStep}>Add</button>
                    </div>
                  </>
                )}

                {effTab === 'goal' && goalTabOk && (
                  <GoalPanel task={task} type={taskType} patch={patch} />
                )}

                {/* Board #184 §6 — everything delivered against this task, in
                    one place. Grouped BY STEP in chain order, because the step
                    is the context that makes a bare URL mean something; and an
                    unfilled row shows a muted "— not provided", so the gaps are
                    exactly as visible as the fills. */}
                {effTab === 'deliverables' && deliverablesTabOk && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ ...cfgHint, marginBottom: 4 }}>
                      {deliverablesFilled} of {scopedDeliverables.length} delivered
                      {' · '}<span style={{ color: 'var(--red)' }}>*</span> = required
                      {scoped.id !== task.id && ' · scoped to the selected subtask'}
                    </div>
                    {deliverablesFilled === 0 && (
                      <div style={{ ...cfgHint, padding: '6px 0' }}>nothing delivered yet.</div>
                    )}
                    {steps.map((s, i) => {
                      const ds = stepDeliverables(s);
                      if (!ds.length) return null;
                      const owner = stepOwner(s);
                      return (
                        <div key={s.id || i} style={{ marginBottom: 10 }}>
                          <div style={{
                            display: 'flex', alignItems: 'center', gap: 6, fontSize: 11.5,
                            color: 'var(--text-tertiary)', marginBottom: 2,
                          }}>
                            <span>▸ Step {i + 1}</span>
                            {owner && <RoleChip role={owner} />}
                            <span style={{
                              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            }}>{stepTitle(s)}</span>
                            {s.done && <span style={{ color: 'var(--green)' }}>✓</span>}
                          </div>
                          {ds.map((d, di) => (
                            <div key={d.id || di} style={{
                              display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap',
                              padding: '4px 0 4px 14px', fontSize: 12.5,
                              borderTop: '1px dashed var(--border)',
                            }}>
                              <span style={{ flex: 'none' }} title={d.kind}>{deliverableIcon(d.kind)}</span>
                              <span style={{ flex: 'none', color: 'var(--text-secondary)' }}
                                title={d.hint || undefined}>
                                {d.label}
                                {d.required && <span style={{ color: 'var(--red)', fontWeight: 700 }}> *</span>}
                              </span>
                              <span style={{ flex: 1, minWidth: 120, wordBreak: 'break-word' }}>
                                {!isDeliverableFilled(d) ? <span style={cfgHint}>— not provided</span>
                                  : d.kind === 'file' ? (
                                    <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, padding: '1px 7px' }}
                                      onClick={() => d.id && openDeliverableFile(d.id)}
                                      title="open the stored file (signed link, 1h)">
                                      📎 {d.filename} <span style={cfgHint}>({fileSize(d.size_bytes)})</span>
                                    </button>
                                  ) : d.kind === 'link' ? (
                                    <a href={d.value || '#'} target="_blank" rel="noopener noreferrer"
                                      style={{ color: 'var(--blue)' }}>{d.value}</a>
                                  ) : d.value}
                              </span>
                              {isDeliverableFilled(d) && (
                                <span style={{ ...cfgHint, flex: 'none', whiteSpace: 'nowrap' }}>
                                  {d.filled_by}{d.filled_at ? ` · ${relTime(d.filled_at)}` : ''}
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      );
                    })}
                    {stepErr && <div style={{ color: 'var(--red)', fontSize: 11 }} title={stepErr}>{stepErr}</div>}
                  </div>
                )}

                {effTab === 'config' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14, maxWidth: 560 }}>
                    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                      <label style={cfgLabel}>priority (phase . seq)
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
                          <input style={{ ...input, width: 44 }} type="number" value={scoped.priority_phase}
                            onChange={(e) => patch(scoped.id, { priority_phase: +e.target.value })} />
                          <span>.</span>
                          <input style={{ ...input, width: 64 }} type="number" step="0.05" value={scoped.priority_seq}
                            onChange={(e) => patch(scoped.id, { priority_seq: +e.target.value })} />
                        </div>
                      </label>
                      <label style={cfgLabel}>area
                        <div style={{ marginTop: 2 }}>
                          <select style={input} value={scoped.area} onChange={(e) => patch(scoped.id, { area: e.target.value })}>
                            {AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
                          </select>
                        </div>
                      </label>
                      <label style={cfgLabel} title={IMPACT_DEF[scoped.impact || 'contained']}>impact
                        <div style={{ marginTop: 2 }}>
                          <ImpactSelect t={scoped} onPick={(v) => patch(scoped.id, { impact: v })} />
                        </div>
                      </label>
                      {/* Board #262 — THE NESTING GUARD, stated where the
                          nesting would be done. A goal is a top-level
                          container: no parent select at all, rather than a
                          select whose every choice the API will refuse. The
                          refusal text is the shared `goalNestError`, so the UI
                          and the API say the same thing. */}
                      {scopedType === 'goal' ? (
                        <label style={cfgLabel}>parent container
                          <div style={{ ...cfgHint, marginTop: 4, maxWidth: 340 }}>
                            {goalNestError('goal', scoped.parent_id ?? 1)}
                            {scoped.parent_id != null && (
                              <div style={{ color: 'var(--red)', marginTop: 4 }}>
                                ⚠ this goal HAS a parent (#{scoped.parent_id}) — clear it, or
                                make it a vision.{' '}
                                <button style={{ ...input, cursor: 'pointer', fontSize: 11 }}
                                  onClick={() => patch(scoped.id, { parent_id: null })}>clear parent</button>
                              </div>
                            )}
                          </div>
                        </label>
                      ) : (
                        <label style={cfgLabel}>parent container
                          <div style={{ marginTop: 2 }}>
                            <select style={input} value={scoped.parent_id ?? ''}
                              onChange={(e) => patch(scoped.id, { parent_id: e.target.value ? +e.target.value : null })}>
                              <option value="">— none (top level) —</option>
                              {visionOptions.filter((v) => v.id !== scoped.id).map((v) =>
                                <option key={v.id} value={v.id}>#{v.id} {v.title.slice(0, 32)}</option>)}
                            </select>
                          </div>
                        </label>
                      )}
                      {scoped.parent_id != null && (
                        <label style={cfgLabel}>origin
                          <div style={{ marginTop: 2 }}>
                            <select style={input} value={scoped.origin || 'planned'}
                              onChange={(e) => patch(scoped.id, { origin: e.target.value })}>
                              {ORIGINS.map((o) => <option key={o} value={o}>{o === 'discovered' ? '🔍 discovered' : o}</option>)}
                            </select>
                          </div>
                        </label>
                      )}
                    </div>
                    <label style={cfgLabel}>tags (comma-separated)
                      <input key={`tags-${scoped.id}`} style={{ ...input, width: '100%', marginTop: 2 }}
                        defaultValue={(scoped.tags || []).join(', ')}
                        onBlur={(e) => {
                          const tags = parseTags(e.target.value);
                          if (tags.join('|') !== (scoped.tags || []).join('|')) patch(scoped.id, { tags });
                        }} />
                    </label>
                    <label style={cfgLabel}>⛔ blocked by (task ids, comma-separated)
                      <input key={`blk-${scoped.id}`} style={{ ...input, width: '100%', marginTop: 2 }}
                        defaultValue={(scoped.blocked_by || []).join(', ')}
                        onBlur={(e) => {
                          const ids = parseIds(e.target.value);
                          if (ids.join('|') !== (scoped.blocked_by || []).join('|')) patch(scoped.id, { blocked_by: ids });
                        }} />
                    </label>
                    <div>
                      <label style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
                        <input type="checkbox" checked={scoped.impacts_live}
                          onChange={(e) => patch(scoped.id, { impacts_live: e.target.checked })} /> 🔴 live
                        <span style={cfgHint}> — touches the live engine/trading code, deploy carefully</span>
                      </label>
                      <label style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
                        <input type="checkbox" checked={scoped.needs_live_validation}
                          onChange={(e) => patch(scoped.id, { needs_live_validation: e.target.checked })} /> ⏳ validate
                        <span style={cfgHint}> — needs live-market data to confirm</span>
                      </label>
                      <label style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
                        <input type="checkbox" checked={scoped.is_urgent}
                          onChange={(e) => patch(scoped.id, { is_urgent: e.target.checked })} /> ⚡ urgent
                        <span style={cfgHint}> — jump the queue (a tag, not a priority)</span>
                      </label>
                      <label style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
                        <input type="checkbox" checked={!!scoped.kevin_final}
                          onChange={(e) => patch(scoped.id, { kevin_final: e.target.checked })} /> 🔏 two-touch
                        <span style={cfgHint}> — set by the Approval stamps (only Kevin signs off Review → Staged/Done); edit only to correct a mis-stamp</span>
                      </label>
                      <label style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
                        <input type="checkbox" checked={!!scoped.standing_approval}
                          onChange={(e) => patch(scoped.id, { standing_approval: e.target.checked })} /> 🪪 standing approval
                        {/* Board #262 wired this column: it IS a goal's
                            `autonomy`. Editing it here and editing autonomy on
                            the Goal tab are the SAME switch — which is the
                            point; a second field would be two switches for one
                            decision. */}
                        <span style={cfgHint}> — pre-approved class of work (07-25 agreement).
                          On a <b>goal</b> this is its <code>autonomy</code>: on = <code>standing</code>,
                          off = <code>approve_each</code> (board #262). Elsewhere still declarative.</span>
                      </label>
                      {/* Board #262 — a goal's child INHERITS its autonomy
                          unless it overrides it. Shown, not silently applied:
                          the child's own checkbox above says nothing about what
                          it inherited, and a session reading only that would get
                          it wrong. */}
                      {parentIsGoal && (
                        <div style={{ fontSize: 12.5, marginBottom: 6 }}>
                          ⚑ inherited autonomy:{' '}
                          <code>{inheritedAutonomy(scoped, scopedParent)}</code>
                          <span style={cfgHint}>
                            {' '}— from goal #{scopedParent?.id}
                            {String(goalParamsOf(scoped).autonomy_override || '').trim()
                              ? ' (OVERRIDDEN by this task’s goal_params.autonomy_override)'
                              : ' (inherited — set goal_params.autonomy_override to differ)'}
                          </span>
                        </div>
                      )}
                      {/* board #198 — the same switch as the header, scoped to
                          whichever item the Config tab is showing (so a subtask
                          can be armed without leaving the vision's modal).
                          Arming = launching: ON can dispatch within ~20s. */}
                      <div style={{ fontSize: 13, display: 'flex', alignItems: 'center', gap: 6 }}>
                        <AiEligibleToggle task={scoped}
                          onToggle={(v) => patch(scoped.id, { ai_eligible: v })} />
                        <span style={cfgHint}>
                          — standing permission for a headless agent to claim this task, independent
                          of the kanban status (armed by Kevin’s Approval stamp). ON can dispatch on
                          the dispatcher’s next poll (~20s).
                        </span>
                      </div>
                    </div>

                    {/* Run history (board #109) — dispatcher runs of the scoped item */}
                    <div>
                      <div style={{ ...cfgLabel, fontWeight: 600 }}>
                        run history
                        <span style={cfgHint}> — dispatcher runs of this task (newest first)</span>
                      </div>
                      {(runsMap[scoped.id] || []).length === 0 &&
                        <div style={cfgHint}>No dispatcher runs yet.</div>}
                      {(runsMap[scoped.id] || []).map((r) => (
                        <div key={r.id} style={{
                          border: '1px solid var(--border)', borderRadius: 8,
                          padding: '6px 8px', marginBottom: 6, fontSize: 12,
                        }}>
                          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                            <OutcomeChip outcome={r.outcome} />
                            <RoleChip role={r.agent_letter} title={`${r.agent_letter}·auto`} />
                            {r.run_id &&
                              <span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-tertiary)' }}>{r.run_id}</span>}
                            <span style={cfgHint}>
                              {r.requested_by
                                ? `requested by ${r.requested_by} ${relTime(r.requested_at)}`
                                : `queue dispatch ${relTime(r.requested_at)}`}
                              {r.started_at ? ` · started ${relTime(r.started_at)}` : ''}
                              {r.finished_at ? ` · finished ${relTime(r.finished_at)}` : ''}
                            </span>
                          </div>
                          {r.log_tail && (
                            <details style={{ marginTop: 4 }}>
                              <summary style={{ cursor: 'pointer', fontSize: 11, color: 'var(--text-secondary)' }}>log tail</summary>
                              {/* Board #278 — a run log is the densest source of
                                  unbreakable tokens on the page (paths, run ids,
                                  tracebacks). It scrolls in its own box and is
                                  capped at the panel width, so it can never be
                                  the thing that sets the column's floor. */}
                              <pre style={{
                                fontSize: 11, whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto',
                                overflowX: 'auto', maxWidth: '100%', overflowWrap: 'anywhere',
                                background: 'var(--bg-input)', padding: 6, borderRadius: 6, margin: '4px 0 0',
                              }}>{r.log_tail}</pre>
                            </details>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* ── Board #184 §7 — the deliverables STRIP (Half 2, Phase 2) ─────
                Kevin's two constraints are INVARIANTS, not preferences, so they
                are stated where the layout makes or breaks them:

                I1 · it must not eat real estate. `flex:'none'` + a fixed
                STRIP_H; one row that scrolls sideways and never wraps, so the
                height is independent of the card count; and — the load-bearing
                half — it renders NOTHING AT ALL when the scoped task has zero
                deliverables. A task that never asked for evidence gets no new
                chrome, which is why this reuses `deliverablesTabOk`: the strip
                and the tab appear and disappear together, never one alone.

                I2 · a vision must still show its subtask list. The strip is
                inserted BELOW the context panel and ABOVE the Pipeline, and its
                height comes out of the context panel's `flex: 1`. The Pipeline's
                `maxHeight: '45%'` resolves against the WHOLE left column — not
                the space left over — so it is untouched by construction, not by
                a number that happens to still fit. Do not "tidy" this by giving
                the strip a share of the Pipeline's box. */}
            {deliverablesTabOk && (
              <div style={{
                flex: 'none', height: STRIP_H, boxSizing: 'border-box',
                display: 'flex', flexDirection: 'column', gap: 2,
              }}>
                <div style={{
                  flex: 'none', height: STRIP_HEAD_H, lineHeight: `${STRIP_HEAD_H}px`,
                  fontSize: 10, letterSpacing: 0.6, textTransform: 'uppercase',
                  color: 'var(--text-tertiary)', display: 'flex', alignItems: 'center', gap: 6,
                }}>
                  <span>deliverables · {deliverablesFilled}/{scopedDeliverables.length}</span>
                  {scoped.id !== task.id &&
                    <span style={{ textTransform: 'none' }}>#{scoped.id}</span>}
                  <span style={{ flex: 1 }} />
                  <span style={{ textTransform: 'none', opacity: 0.8 }}>hover for detail</span>
                </div>
                <div className="dlv-strip" style={{
                  flex: 'none', height: STRIP_H - STRIP_HEAD_H - 2,
                  display: 'flex', flexWrap: 'nowrap', gap: 6,
                  overflowX: 'auto', overflowY: 'hidden', alignItems: 'flex-start',
                }}>
                  {scopedDeliverables.map(({ d, step, i }, n) => {
                    const filled = isDeliverableFilled(d);
                    return (
                      <button key={d.id || `${i}-${n}`}
                        title={deliverableTip(d, step, i)}
                        onClick={() => openDeliverableCard(d, step, i)}
                        style={{
                          flex: 'none', width: STRIP_CARD, height: STRIP_CARD,
                          boxSizing: 'border-box', position: 'relative', padding: 0,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          borderRadius: 8, cursor: 'pointer',
                          border: '1px solid var(--border)',
                          // The fill is what the card is FOR, so a filled one
                          // sits in a box and an unfilled one is an outline.
                          background: filled ? 'var(--bg-input)' : 'transparent',
                        }}>
                        {/* Three states, one glyph (spec §7). The opacity rides
                            the ICON, never the button — a muted `*` would defeat
                            the whole point of marking the required ones. */}
                        <span style={{
                          fontSize: 18, lineHeight: 1,
                          opacity: filled ? 1 : d.required ? 0.65 : 0.4,
                        }}>{deliverableIcon(d.kind)}</span>
                        {!filled && d.required && (
                          <span style={{
                            position: 'absolute', top: 1, right: 4, lineHeight: 1,
                            color: 'var(--red)', fontSize: 12, fontWeight: 700,
                          }}>*</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Subtask-selector strip — VISIONS ONLY (the vision's pipeline) */}
            {vision && (
              <div style={{ ...panel, maxHeight: '45%' }}>
                <div style={panelHead}>
                  <span>Pipeline · {doneCount}/{totalCount} done · your subtasks sign these off</span>
                </div>
                <div style={{ overflowY: 'auto', padding: '6px 8px' }}>
                  <div onClick={() => { setSelected('summary'); setTab('summary'); }} style={{
                    display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', borderRadius: 8,
                    cursor: 'pointer', marginBottom: 4, fontSize: 13, fontWeight: 600,
                    border: `1px dashed ${selected === 'summary' ? 'var(--blue)' : 'var(--border)'}`,
                    background: selected === 'summary' ? 'var(--bg-input)' : 'transparent',
                  }}>
                    <span>≡ Summary</span>
                    <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--text-tertiary)' }}>
                      the vision item itself
                    </span>
                  </div>
                  {stripRows.map((s) => (
                    <div key={s.id} onClick={() => selectRow(s.id)} style={{
                      display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', borderRadius: 8,
                      cursor: 'pointer', fontSize: 13, marginBottom: 2,
                      background: selected === s.id ? 'var(--bg-input)' : 'transparent',
                      border: `1px solid ${selected === s.id ? 'var(--blue)' : 'transparent'}`,
                    }}>
                      <span style={{ fontSize: 14 }}>{isFinished(s.status) ? '☑' : '☐'}</span>
                      <span style={{
                        flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        textDecoration: isFinished(s.status) ? 'line-through' : 'none',
                        opacity: isFinished(s.status) ? 0.6 : 1,
                      }}>
                        {s.title}
                        {s.origin === 'discovered' && <span style={{ ...tagChip, marginLeft: 6 }}>🔍</span>}
                      </span>
                      <ProgressBar done={(s.checklist || []).filter((x) => x.done).length}
                        total={(s.checklist || []).length} mini />
                      <span style={{ fontSize: 11, color: STATUS_COLOR[s.status] }}>{s.status}</span>
                      {s.assignee && <RoleChip role={s.assignee} title={`assigned to ${s.assignee}`} />}
                      {selected === s.id && (
                        <button style={{ ...chipBtn(true), whiteSpace: 'nowrap' }} title="open this subtask's full modal"
                          onClick={(e) => { e.stopPropagation(); onOpenTask(s.id); }}>open full task ↗</button>
                      )}
                    </div>
                  ))}
                  {stripRows.length === 0 &&
                    <div style={{ fontSize: 12, color: 'var(--text-tertiary)', padding: '4px 8px' }}>
                      No subtasks yet — add them from the board (+ subtask).</div>}
                </div>
              </div>
            )}
          </div>

          {/* Activity panel. `minWidth: 0` is inherited from `panel` and is the
              actual #278 fix — spelled out here too because this is the column
              that was refusing to shrink. */}
          <div style={{ ...panel, flex: 1, minWidth: 0 }}>
            <div style={panelHead}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Activity — {scoped.id === task.id ? 'this task' : `#${scoped.id} ${scoped.title.slice(0, 30)}`}
              </span>
              <span style={{ flex: 1 }} />
              {liveRun && (
                <span title={`dispatcher run ${liveRun.run_id || '?'} is live — thread updates every ~15s`}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 6, textTransform: 'none',
                    border: '1px solid var(--blue)', borderRadius: 10, padding: '1px 8px',
                    color: 'var(--blue)', fontSize: 11, fontWeight: 700, whiteSpace: 'nowrap',
                  }}>
                  <span className="pulse-dot" />
                  {liveRun.agent_letter}·auto working · {elapsedShort(liveRun.started_at || liveRun.requested_at)}
                </span>
              )}
            </div>
            <div ref={feedRef} style={{ flex: 1, overflowY: 'auto', padding: '8px 12px' }}>
              {thread.length === 0 &&
                <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>No activity yet.</div>}
              {thread.map((cm) => cm.author === 'system' ? (
                <div key={cm.id} className="cm-in" style={{ fontSize: 11.5, color: 'var(--text-tertiary)', fontStyle: 'italic', margin: '6px 0', display: 'flex', gap: 6, alignItems: 'baseline' }}>
                  <RoleChip role="system" title="system" />
                  {/* Board #278 — system lines are NOT markdown, so MD_CSS never
                      reached them, and they are the ones carrying branch names
                      and worktree paths ("dispatch skipped: …"). Own row, own
                      wrap rule. */}
                  <span style={{ minWidth: 0, overflowWrap: 'anywhere' }}>{cm.body} · {relTime(cm.created_at)}</span>
                </div>
              ) : (
                <div key={cm.id} className="cm-in" style={{ fontSize: 13, margin: '8px 0', paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
                  <RoleChip role={cm.author} title={cm.author} />
                  <span style={{ color: 'var(--text-tertiary)', fontSize: 11, marginLeft: 6 }}>{relTime(cm.created_at)}</span>
                  <div style={{ marginTop: 2 }}><Md text={cm.body} /></div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid var(--border)', alignItems: 'center' }}>
              <RolePicker value={commentAuthor} pickTitle="comment as" up
                options={withLegacy(roles.filter(Boolean), commentAuthor)}
                onPick={onPickAuthor} />
              {/* @-mention inserter (board #143): tag a teammate so it lands in
                  their Messages inbox. Simple insert picker — spec says no
                  fancy autocomplete needed. */}
              <span style={{ position: 'relative', display: 'inline-block' }}>
                <button style={{ ...input, cursor: 'pointer', fontWeight: 700, padding: '4px 9px' }}
                  title="mention a teammate (@) — it lands in their Messages inbox"
                  onClick={() => setMentionOpen((o) => !o)}>@</button>
                {mentionOpen && (
                  <>
                    <span onClick={() => setMentionOpen(false)} style={{ position: 'fixed', inset: 0, zIndex: 1500 }} />
                    <span style={{
                      position: 'absolute', bottom: '120%', left: 0, zIndex: 1600,
                      display: 'flex', gap: 5, padding: '6px 8px', borderRadius: 8,
                      border: '1px solid var(--border)', background: 'var(--bg-card, var(--bg-input))',
                      boxShadow: '0 6px 20px rgba(0,0,0,0.4)',
                    }}>
                      {mentionRoles.map((r) => (
                        <button key={r} title={`mention @${r}`}
                          style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', lineHeight: 1 }}
                          onClick={() => insertMention(r)}>
                          <RoleChip role={r} />
                        </button>
                      ))}
                    </span>
                  </>
                )}
              </span>
              <input style={{ ...input, flex: 1, minWidth: 0 }} placeholder="comment… (@ to mention)" value={draftComment}
                onChange={(e) => setDraftComment(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && postComment()} />
              <button style={{ ...input, cursor: 'pointer', background: 'var(--blue)', color: '#fff' }}
                onClick={() => postComment()}>Comment</button>
              <button style={{ ...input, cursor: 'pointer', borderColor: '#7c5cff', color: '#7c5cff', fontWeight: 600, whiteSpace: 'nowrap' }}
                title="post with an @M marker — M's board-watcher answers in this thread"
                onClick={() => postComment('@M ')}>🤖 Ask AI</button>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 10, textAlign: 'right' }}>
          <button style={{ ...input, cursor: 'pointer', color: 'var(--red)' }} onClick={() => del(task.id)}>Delete task</button>
        </div>
      </div>

      {/* Board #182 Step 6 — the structured step form (portals to body), used by
          both the slash "process module" and the accordion insert/edit. */}
      {stepForm && (
        <StepForm
          heading={stepForm.mode === 'insert' ? 'Insert process step' : 'Edit process step'}
          initial={stepForm.mode === 'edit' ? steps[stepForm.index] : undefined}
          roles={roles.filter(Boolean)}
          audible={stepForm.mode === 'insert' && started}
          submitLabel={stepForm.mode === 'insert' ? 'Insert step' : 'Save step'}
          onSubmit={submitStepForm}
          onCancel={() => setStepForm(null)} />
      )}
    </div>,
    document.body,
  );
}

/**
 * THE GOAL TAB (board #262) — a goal's parameters, and the context block that
 * starts its session.
 *
 * Two halves, in the order they are used: the editor above, the block below.
 *
 * THE BLOCK IS FOR CLAUDE'S BUILT-IN `/goal`. It is not a skill, a runner or a
 * scheduler, and this file must never grow into one (Kevin's ruling, 07-31).
 * The block is self-sufficient on purpose — the built-in's own prompt cannot be
 * introspected from here, so it states the objective, the end condition and the
 * rails outright instead of assuming it will be asked.
 *
 * THE REFUSAL IS THE GATE. Rendered from `renderGoalBlock`, which returns no
 * block at all when `done_when` is blank — so there is nothing to copy, and a
 * goal with no end condition cannot be started. That is the whole enforcement
 * mechanism for this task (M's step-1 decision: the render, not the API — a
 * half-written goal must still SAVE, because that is what a draft is).
 */
function GoalPanel({ task, type, patch }: {
  task: Task;
  type: ReturnType<typeof taskTypeOf>;
  patch: (id: number, body: Record<string, unknown>) => void;
}) {
  const [copied, setCopied] = useState(false);
  const params = goalParamsOf(task);
  const autonomy = autonomyOf(task);
  const rendered = renderGoalBlock(task, type, apiHostForBlock());

  const setParam = (key: string, value: unknown) =>
    patch(task.id, goalParamsPatch(task, key, value));

  const copy = async () => {
    if (!rendered.block) return;
    try {
      await navigator.clipboard.writeText(rendered.block);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);   // clipboard denied — the textarea below is the fallback
    }
  };

  const fieldInput = (f: GoalFieldDef) => {
    const missing = f.required && !f.column && !String(params[f.key] ?? '').trim();
    const border = missing ? { borderColor: 'var(--red)' } : {};
    if (f.column === 'standing_approval') {
      // THE WIRING: autonomy reads and writes the EXISTING column. There is no
      // `goal_params.autonomy` — see `autonomyOf`.
      return (
        <select style={{ ...input, width: '100%', marginTop: 2 }} value={autonomy}
          onChange={(e) => patch(task.id, autonomyPatch(e.target.value as Autonomy))}>
          {(f.options || []).map((o) => (
            <option key={o} value={o} title={AUTONOMY_DEF[o as Autonomy]}>{o}</option>
          ))}
        </select>
      );
    }
    if (f.kind === 'textarea') {
      return (
        <textarea key={`${f.key}-${task.id}`}
          style={{ ...input, ...border, width: '100%', marginTop: 2, minHeight: 56, fontSize: 12.5 }}
          defaultValue={String(params[f.key] ?? '')}
          onBlur={(e) => {
            if (e.target.value !== String(params[f.key] ?? '')) setParam(f.key, e.target.value.trim());
          }} />
      );
    }
    const shown = f.kind === 'list' && Array.isArray(params[f.key])
      ? (params[f.key] as string[]).join(', ')
      : String(params[f.key] ?? '');
    return (
      <input key={`${f.key}-${task.id}`}
        style={{ ...input, ...border, width: '100%', marginTop: 2 }}
        defaultValue={shown}
        onBlur={(e) => {
          const v = f.kind === 'list'
            ? e.target.value.split(',').map((s) => s.trim()).filter(Boolean)
            : e.target.value.trim();
          if (JSON.stringify(v) !== JSON.stringify(f.kind === 'list' ? (Array.isArray(params[f.key]) ? params[f.key] : []) : shown)) {
            setParam(f.key, f.kind === 'list' && !(v as string[]).length ? '' : v);
          }
        }} />
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 640 }}>
      <div style={cfgHint}>
        A <b>goal</b> is a container that carries its own session. These parameters ARE
        the session’s brief: they render the context block below, which is pasted into a
        session running Claude’s <b>built-in</b> <code>/goal</code>. Nothing here runs a
        goal — no skill, no runner, no scheduler.
      </div>

      {GOAL_PARAM_FIELDS.map((f) => (
        <label key={f.key} style={cfgLabel}>
          <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{f.key}</span>
          {f.required && <span style={{ color: 'var(--red)', marginLeft: 4 }}>*required</span>}
          {f.column && (
            <span style={{ ...cfgHint, marginLeft: 6 }}>
              → column <code>{f.column}</code>
            </span>
          )}
          <div style={cfgHint}>{f.hint}</div>
          {fieldInput(f)}
        </label>
      ))}

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10 }}>
        <div style={{ ...cfgLabel, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
          context block — paste into a session running <code>/goal</code>
          {rendered.ok && (
            <button style={{ ...input, cursor: 'pointer', fontSize: 11 }} onClick={copy}>
              {copied ? '✓ copied' : 'copy'}
            </button>
          )}
        </div>
        {!rendered.ok && (
          <div style={{
            border: '1px solid var(--red)', borderRadius: 8, padding: '8px 10px',
            fontSize: 12.5, color: 'var(--red)', background: 'rgba(220,60,60,0.08)',
          }}>
            <b>⛔ no block.</b> {rendered.refusal}
          </div>
        )}
        {rendered.ok && rendered.block && (
          <textarea readOnly value={rendered.block}
            style={{
              ...input, width: '100%', minHeight: 300, fontFamily: 'monospace',
              fontSize: 11.5, lineHeight: 1.45,
            }} />
        )}
      </div>
    </div>
  );
}

/** The API origin written INTO the block, so the goal session is told where the
 *  board actually is rather than guessing. Falls back to a placeholder the
 *  reader will notice rather than a wrong host (no silent defaults). */
function apiHostForBlock(): string {
  const env = process.env.NEXT_PUBLIC_API_URL;
  return (env && env.replace(/\/+$/, '')) || '<api-host>';
}

/** A step's approval-gate chip (board #182). Renders only for a stamp that is
 *  actually required; color + label track its state. */
function StampChip({ stamp }: { stamp?: StepStamp | null }) {
  if (!stamp?.required) return null;
  const state = stamp.state || 'pending';
  const map: Record<string, { c: string; label: string }> = {
    pending: { c: '#c9a227', label: '🔏 needs stamp' },
    approved: { c: 'var(--green)', label: '🔏 stamped' },
    rejected: { c: 'var(--red)', label: '🔏 rejected' },
  };
  const m = map[state] || map.pending;
  const tip = `${m.label}${stamp.by ? ` — by ${stamp.by}` : ''}${stamp.at ? ` · ${relTime(stamp.at)}` : ''}`;
  return (
    <span style={{ ...tagChip, borderColor: m.c, color: m.c, fontWeight: 700 }} title={tip}>
      {m.label}
    </span>
  );
}

/**
 * One deliverable row inside a step's accordion body (board #184, spec §5.1) —
 * the ask, its control, and its fill provenance, rendered where the ask was
 * made. Controls are live only on a NOT-yet-done step; a completed step renders
 * read-only, the same immutability the rest of a completed step already shows.
 */
function DeliverableRow({
  d, editable, busy, onFill, onClear, onUpload, onOpen,
}: {
  d: StepDeliverable;
  editable: boolean;
  busy: boolean;
  onFill: (id: string, value: string) => void;
  onClear: (id: string) => void;
  onUpload: (id: string, file: File) => void;
  onOpen: (id: string) => void;
}) {
  const [draft, setDraft] = useState(d.value || '');
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => { setDraft(d.value || ''); }, [d.value]);
  const filled = isDeliverableFilled(d);
  const id = d.id || '';
  const commit = () => {
    const v = draft.trim();
    if (v && v !== (d.value || '')) onFill(id, v);
  };
  const kindDef = DELIVERABLE_KINDS.find((k) => k.key === d.kind);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
      padding: '5px 0', borderTop: '1px dashed var(--border)',
    }}>
      <span style={{ flex: 'none', fontSize: 13 }} title={kindDef?.def || d.kind}>
        {deliverableIcon(d.kind)}
      </span>
      <span style={{ flex: 'none', fontSize: 12, color: 'var(--text-secondary)' }}
        title={d.hint || undefined}>
        {d.label}
        {d.required && <span style={{ color: 'var(--red)', fontWeight: 700 }} title="required — this step cannot be completed until it is filled (T10)"> *</span>}
      </span>
      {/* ── the control ── */}
      {!editable ? (
        // Completed step: read-only. A gap still SHOWS as a gap.
        <span style={{ flex: 1, minWidth: 120, fontSize: 12 }}>
          {!filled ? <span style={cfgHint}>— not provided</span>
            : d.kind === 'file' ? (
              <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, padding: '1px 7px' }}
                onClick={() => onOpen(id)} title="open the stored file (signed link, 1h)">
                📎 {d.filename} {d.size_bytes != null && <span style={cfgHint}>({fileSize(d.size_bytes)})</span>}
              </button>
            ) : d.kind === 'link' ? (
              <a href={d.value || '#'} target="_blank" rel="noopener noreferrer"
                style={{ color: 'var(--blue)' }}>{d.value}</a>
            ) : <span>{d.value}</span>}
        </span>
      ) : d.kind === 'file' ? (
        <span style={{ flex: 1, minWidth: 160, display: 'flex', gap: 6, alignItems: 'center' }}>
          {filled ? (
            <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, padding: '1px 7px' }}
              onClick={() => onOpen(id)} title="open the stored file (signed link, 1h)">
              📎 {d.filename} {d.size_bytes != null && <span style={cfgHint}>({fileSize(d.size_bytes)})</span>}
            </button>
          ) : (
            <>
              <input ref={fileRef} type="file" style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUpload(id, f);
                  e.target.value = '';
                }} />
              <button style={{ ...input, cursor: busy ? 'default' : 'pointer', fontSize: 11.5 }}
                disabled={busy} onClick={() => fileRef.current?.click()}
                title="upload a file (≤ 10 MB): png jpg gif webp pdf csv txt md json log zip">
                ⬆ upload…
              </button>
            </>
          )}
        </span>
      ) : (
        <input style={{ ...input, flex: 1, minWidth: 160, fontSize: 12 }}
          value={draft} disabled={busy}
          placeholder={d.hint || (d.kind === 'link' ? 'https://…' : 'your answer…')}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); commit(); } }} />
      )}
      {filled && (
        <span style={{ ...cfgHint, flex: 'none', whiteSpace: 'nowrap' }}
          title={`filled by ${d.filled_by}${d.filled_at ? ` ${relTime(d.filled_at)}` : ''}`}>
          ✓ {d.filled_by}
        </span>
      )}
      {editable && filled && (
        <button style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '0 6px', color: 'var(--red)', borderColor: 'var(--red)' }}
          disabled={busy} onClick={() => onClear(id)}
          title="clear this value (the stored file is kept — clearing must not be destructive)">✕</button>
      )}
    </div>
  );
}

/**
 * The process chain rendered as ordered accordion steps (board #182, Step 4) —
 * the body of the Summary tab. Each section's header carries one line of
 * substance even when collapsed (owner avatar · number · title · mode/audible/
 * stamp/completion chips); the body holds that step's SOP (markdown) and, on
 * the current step, the Step-5 actions (Complete & hand off, stamp approve/
 * reject, raise-issue → M). Expansion is caller-owned (default-expand the
 * viewer's step); this component is otherwise a pure read of the chain plus the
 * action callbacks — the transition rules themselves live server-side.
 *
 * Board #182 Step 6 adds STRUCTURED EDITING in the accordion itself: each
 * incomplete step carries edit / reorder / insert-after / remove controls, and
 * the chain ends with an "add step" — so revising the chain is as cheap AND as
 * safe as ticking it (no hand-edited JSON, the fragility that dropped a step
 * mid-flight). Completed steps stay immutable (T4'); the API refuses their
 * deletion/reorder-into anyway, so those controls are shown only on open steps.
 */
function ProcessChain({
  steps, viewer, openSteps, onToggle, onComplete, onStamp, onRaiseIssue, busy, err,
  onInsert, onEdit, onRemove, onMove,
  onFillDeliverable, onClearDeliverable, onUploadDeliverable, onOpenDeliverable,
}: {
  steps: ChecklistStep[];
  viewer: string;
  openSteps: Set<string>;
  onToggle: (key: string) => void;
  onComplete: () => void;
  onStamp: (decision: 'approve' | 'reject') => void;
  onRaiseIssue: (body: string) => void;
  busy: boolean;
  err: string | null;
  onInsert: (index: number) => void;
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
  onMove: (index: number, dir: number) => void;
  onFillDeliverable: (id: string, value: string) => void;
  onClearDeliverable: (id: string) => void;
  onUploadDeliverable: (id: string, file: File) => void;
  onOpenDeliverable: (id: string) => void;
}) {
  const [issueFor, setIssueFor] = useState<string | null>(null);
  const [issueDraft, setIssueDraft] = useState('');
  const currentIdx = steps.findIndex((s) => !s.done);   // -1 → chain complete
  const complete = currentIdx < 0;
  const iconBtn: React.CSSProperties = {
    ...input, cursor: 'pointer', fontSize: 11, padding: '1px 7px', lineHeight: 1.6,
  };

  return (
    <div style={{ marginTop: 14, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>
          Process
        </span>
        <span style={cfgHint}>
          {complete
            ? 'chain complete'
            : `step ${currentIdx + 1} of ${steps.length} · the owner acts, then hands off to the next`}
        </span>
      </div>
      {steps.map((s, i) => {
        const key = s.id || String(i);
        const owner = stepOwner(s);
        const isCurrent = i === currentIdx;
        const open = openSteps.has(key);
        const audible = s.origin === 'audible';
        const accent = owner ? roleColor(owner) : '#64748b';
        const mark = s.done ? '✓' : isCurrent ? '▶' : '·';
        const stampBlocks = !!s.stamp?.required && s.stamp?.state !== 'approved';
        // Board #184 T10 — the SAME visual language as the stamp gate, not a
        // second one: one "this step is blocked" treatment, two reasons.
        const missing = unfilledDeliverables(s, true);
        const ds = stepDeliverables(s);
        const blocks = stampBlocks || missing.length > 0;
        const blockWhy = stampBlocks
          ? 'this step needs an approved stamp first (T8)'
          : missing.length
            ? `needs its required deliverable "${missing[0].label}" before it can be `
              + 'completed (T10) — fill it, or ⚠ Raise issue → M if you cannot'
            : '';
        return (
          <div key={key} style={{
            border: '1px solid var(--border)',
            borderLeft: `3px solid ${audible ? '#a855f7' : isCurrent ? accent : 'var(--border)'}`,
            borderRadius: 8, marginBottom: 6, overflow: 'hidden',
            background: isCurrent ? 'var(--bg-input)' : 'transparent',
            opacity: !s.done && !isCurrent ? 0.9 : 1,
          }}>
            {/* Header — always one line of substance, even collapsed. */}
            <button onClick={() => onToggle(key)} style={{
              display: 'flex', alignItems: 'center', gap: 8, width: '100%',
              padding: '7px 9px', background: 'none', border: 'none', cursor: 'pointer',
              textAlign: 'left', color: 'inherit',
            }}>
              <span style={{ fontSize: 11, color: 'var(--text-tertiary)', width: 12, flex: 'none' }}>{open ? '▾' : '▸'}</span>
              <span style={{ flex: 'none' }} title={owner ? `owner: ${owner}` : 'no owner set'}>
                {owner ? <RoleChip role={owner} /> : <EmptyRoleCircle />}
              </span>
              <span style={{
                flex: 1, minWidth: 0, fontSize: 13, fontWeight: isCurrent ? 700 : 500,
                textDecoration: s.done ? 'line-through' : 'none',
                color: s.done ? 'var(--text-tertiary)' : 'var(--text-primary)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: open ? 'normal' : 'nowrap',
              }}>
                <span style={{ color: accent, marginRight: 6, fontWeight: 700 }}>{mark} {i + 1}.</span>
                {stepTitle(s)}
              </span>
              {s.mode === 'discuss' &&
                <span style={{ ...tagChip, borderColor: '#0ea5e9', color: '#0ea5e9', flex: 'none' }} title={MODE_DEF.discuss}>💬 discuss</span>}
              {audible &&
                <span style={{ ...tagChip, borderColor: '#a855f7', color: '#a855f7', fontWeight: 700, flex: 'none' }}
                  title="inserted mid-flight as an audible (origin=audible) — a course-correction recorded by the chain's own rules">🔀 audible</span>}
              <StampChip stamp={s.stamp} />
              {s.done && s.completed_by &&
                <span style={{ ...cfgHint, whiteSpace: 'nowrap', flex: 'none' }}
                  title={`completed by ${s.completed_by}${s.completed_at ? ` ${relTime(s.completed_at)}` : ''}`}>
                  ✓ {s.completed_by}
                </span>}
            </button>
            {/* Body — the SOP + (current step) the Step-5 actions. */}
            {open && (
              <div style={{ padding: '2px 12px 12px 34px', fontSize: 13 }}>
                {s.body
                  ? <Md text={s.body} />
                  : <span style={cfgHint}>_no SOP written for this step yet_</span>}
                {/* Board #184 §5.1 — the deliverables this step asks for, BELOW
                    the SOP and ABOVE the Step-6 controls: the ask and the place
                    you answer it are the same place, which is the whole point.
                    Renders nothing at all when the step declares none, so every
                    pre-#184 step is untouched. */}
                {ds.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ ...cfgHint, marginBottom: 2 }}>
                      Deliverables <span style={{ color: 'var(--red)' }}>*</span> = required
                    </div>
                    {ds.map((d, di) => (
                      <DeliverableRow key={d.id || di} d={d} editable={!s.done} busy={busy}
                        onFill={onFillDeliverable} onClear={onClearDeliverable}
                        onUpload={onUploadDeliverable} onOpen={onOpenDeliverable} />
                    ))}
                  </div>
                )}
                {/* Step 6 — structured editing on open (not-yet-done) steps.
                    Completed steps are immutable (T4'), so no controls there. */}
                {!s.done && (
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 10, alignItems: 'center' }}>
                    <button style={iconBtn} disabled={busy} onClick={() => onEdit(i)}
                      title="edit this step's owner / title / mode / SOP (structured — never hand-edit JSON)">✎ edit</button>
                    <button style={iconBtn} disabled={busy || i <= currentIdx} onClick={() => onMove(i, -1)}
                      title={i <= currentIdx ? 'already first among the open steps' : 'move earlier in the chain'}>↑</button>
                    <button style={iconBtn} disabled={busy || i >= steps.length - 1} onClick={() => onMove(i, 1)}
                      title={i >= steps.length - 1 ? 'already last' : 'move later in the chain'}>↓</button>
                    <button style={iconBtn} disabled={busy} onClick={() => onInsert(i + 1)}
                      title="insert a new step after this one (auto-marked audible if the chain is under way)">＋ insert after</button>
                    <button style={{ ...iconBtn, color: 'var(--red)', borderColor: 'var(--red)' }} disabled={busy}
                      onClick={() => onRemove(i)} title="remove this step (only not-yet-done steps can be removed)">🗑 remove</button>
                  </div>
                )}
                {isCurrent && owner && owner !== viewer && (
                  <div style={{ ...cfgHint, marginTop: 8 }}>
                    you’re acting as <b style={{ color: roleColor(viewer) }}>{viewer}</b>; this step is owned by{' '}
                    <b style={{ color: accent }}>{owner}</b>
                  </div>
                )}
                {isCurrent && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10, alignItems: 'center' }}>
                    {stampBlocks && (
                      <>
                        <button disabled={busy} onClick={() => onStamp('approve')}
                          style={{ ...input, cursor: busy ? 'default' : 'pointer', color: 'var(--green)', borderColor: 'var(--green)', fontWeight: 600 }}
                          title="approve this step's stamp — unblocks Complete & hand off">✅ Approve stamp</button>
                        <button disabled={busy} onClick={() => onStamp('reject')}
                          style={{ ...input, cursor: busy ? 'default' : 'pointer', color: 'var(--red)', borderColor: 'var(--red)', fontWeight: 600 }}
                          title="reject the stamp — records it and routes the task to M (T9)">⛔ Reject</button>
                      </>
                    )}
                    <button disabled={busy || blocks} onClick={onComplete}
                      style={{
                        ...input, cursor: busy || blocks ? 'default' : 'pointer', fontWeight: 700,
                        color: '#fff', background: 'var(--blue)', borderColor: 'var(--blue)',
                        opacity: blocks ? 0.4 : 1,
                      }}
                      title={blocks ? blockWhy
                        : 'mark this step complete and hand the task to the next owner — one action'}>
                      ✓ Complete &amp; hand off
                    </button>
                    {issueFor === key ? (
                      <span style={{ display: 'flex', gap: 6, alignItems: 'center', flex: 1, minWidth: 220 }}>
                        <input autoFocus style={{ ...input, flex: 1 }} placeholder="what's the problem? (routes the task to M)"
                          value={issueDraft} onChange={(e) => setIssueDraft(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && issueDraft.trim()) {
                              onRaiseIssue(issueDraft); setIssueDraft(''); setIssueFor(null);
                            }
                          }} />
                        <button disabled={busy || !issueDraft.trim()}
                          onClick={() => { onRaiseIssue(issueDraft); setIssueDraft(''); setIssueFor(null); }}
                          style={{ ...input, cursor: 'pointer' }}>send</button>
                        <button onClick={() => { setIssueFor(null); setIssueDraft(''); }}
                          style={{ ...input, cursor: 'pointer' }}>✕</button>
                      </span>
                    ) : (
                      <button disabled={busy} onClick={() => setIssueFor(key)}
                        style={{ ...input, cursor: busy ? 'default' : 'pointer', color: '#c9a227', borderColor: '#c9a227' }}
                        title="raise a blocking issue on this step — posts a comment and routes the task to M without ticking anything (T9)">
                        ⚠ Raise issue → M
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
      {/* Step 6 — append a step to the end of the chain (the last valid insert
          point; per-step "insert after" covers the rest). Audible once under way. */}
      <div style={{ marginTop: 4 }}>
        <button style={{ ...input, cursor: 'pointer', fontSize: 12, color: 'var(--blue)', borderColor: 'var(--blue)' }}
          onClick={() => onInsert(steps.length)}
          title={complete
            ? 'add a step to the end of the chain'
            : 'add a step to the end of the chain (auto-marked audible — the chain is under way)'}>
          ＋ Add step
        </button>
      </div>
      {err && <div style={{ color: 'var(--red)', fontSize: 11, marginTop: 6 }} title={err}>{err}</div>}
    </div>
  );
}

/**
 * The structured step form (board #182 Step 6) — the ClickUp-style "process
 * module". Opened by the slash menu ("/process module") and by the accordion's
 * insert / edit controls, so a chain step is authored through fields (owner ·
 * title · mode · SOP) rather than by hand-editing JSON. It only edits those four
 * fields; id / origin / stamp / completion are preserved by the caller so the
 * PATCH stays a structural edit. Portals to <body> so it works from either the
 * accordion view or the description editor. `audible` is a preview flag — the
 * actual origin is stamped by makeChainStep at insert time.
 */
function StepForm({ heading, initial, roles, audible, submitLabel, onSubmit, onCancel }: {
  heading: string;
  initial?: ChecklistStep;
  roles: string[];
  audible: boolean;
  submitLabel: string;
  onSubmit: (f: {
    owner: string | null; title: string; mode: 'execute' | 'discuss'; body: string;
    deliverables: StepDeliverable[];
  }) => void;
  onCancel: () => void;
}) {
  const [owner, setOwner] = useState<string | null>(initial?.owner ?? initial?.role ?? null);
  const [title, setTitle] = useState<string>(initial?.title ?? initial?.text ?? '');
  const [mode, setMode] = useState<'execute' | 'discuss'>(initial?.mode ?? 'execute');
  const [body, setBody] = useState<string>(initial?.body ?? '');
  // Board #184 §5.2 — the ONLY place a deliverable spec is authored. Fill state
  // is never touched here: existing rows keep their server-owned fields
  // (spread below), and a new row is minted unfilled by makeDeliverable.
  const [dels, setDels] = useState<StepDeliverable[]>(
    () => stepDeliverables(initial).map((d) => ({ ...d })));
  const setDel = (i: number, patchD: Partial<StepDeliverable>) =>
    setDels((prev) => prev.map((d, j) => (j === i ? { ...d, ...patchD } : d)));
  const canSave = title.trim().length > 0
    && dels.every((d) => d.label.trim().length > 0);
  const submit = () => {
    if (!canSave) return;
    onSubmit({
      owner, title: title.trim(), mode, body,
      deliverables: dels.map((d) => ({ ...d, label: d.label.trim() })),
    });
  };
  const modeBtn = (m: 'execute' | 'discuss'): React.CSSProperties => ({
    ...input, cursor: 'pointer', fontSize: 12, fontWeight: mode === m ? 700 : 400,
    borderColor: mode === m ? (m === 'discuss' ? '#0ea5e9' : 'var(--blue)') : 'var(--border)',
    color: mode === m ? (m === 'discuss' ? '#0ea5e9' : 'var(--blue)') : 'var(--text-secondary)',
  });
  return createPortal(
    <div onClick={onCancel} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 2000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        width: 'min(560px, 94vw)', maxHeight: '86vh', overflowY: 'auto',
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, padding: 16, boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>{heading}</span>
          {audible && (
            <span style={{ ...tagChip, borderColor: '#a855f7', color: '#a855f7', fontWeight: 700 }}
              title="the chain is under way — this insert is recorded as an audible (origin=audible) and renders distinctly">
              🔀 audible
            </span>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={cfgLabel}>owner</span>
              <RolePicker value={owner} pickTitle="step owner" allowEmpty options={roles}
                onPick={(r) => setOwner(r || null)} />
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={cfgLabel}>mode</span>
              <button type="button" style={modeBtn('execute')} title={MODE_DEF.execute}
                onClick={() => setMode('execute')}>execute</button>
              <button type="button" style={modeBtn('discuss')} title={MODE_DEF.discuss}
                onClick={() => setMode('discuss')}>💬 discuss</button>
            </span>
          </div>
          <label style={cfgLabel}>title <span style={cfgHint}>— one line of substance (shown even when collapsed)</span>
            <input autoFocus style={{ ...input, width: '100%', marginTop: 2 }} value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } }}
              placeholder="e.g. Wire the export button to the new endpoint" />
          </label>
          <label style={cfgLabel}>SOP body <span style={cfgHint}>— what this step requires (markdown)</span>
            <textarea style={{ ...input, width: '100%', minHeight: 120, marginTop: 2, fontFamily: 'monospace', fontSize: 12.5 }}
              value={body} onChange={(e) => setBody(e.target.value)}
              placeholder="constraint set · verification story · known traps…" />
          </label>
          {/* ── Deliverables (board #184 §5.2) — the authoring side ────────── */}
          <div>
            <span style={cfgLabel}>deliverables
              <span style={cfgHint}> — the evidence this step must produce. Optional by
                default; tick <b>required</b> only when the SHAPE of the answer is
                genuinely known (a required one blocks Complete).</span>
            </span>
            {dels.map((d, i) => (
              <div key={d.id || i} style={{
                display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap',
                marginTop: 6, paddingTop: 6, borderTop: '1px dashed var(--border)',
              }}>
                <select style={{ ...input, width: 92, fontSize: 12 }} value={d.kind}
                  title={DELIVERABLE_KINDS.find((k) => k.key === d.kind)?.def}
                  onChange={(e) => setDel(i, { kind: e.target.value as DeliverableKind })}>
                  {DELIVERABLE_KINDS.map((k) =>
                    <option key={k.key} value={k.key} title={k.def}>{k.icon} {k.label}</option>)}
                </select>
                <input style={{ ...input, flex: 1, minWidth: 150, fontSize: 12 }} value={d.label}
                  placeholder="the ask, in one line — e.g. link to the merged PR"
                  onChange={(e) => setDel(i, { label: e.target.value })} />
                <label style={{ ...cfgHint, display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}
                  title="required — the owner cannot complete this step until it is filled (T10)">
                  <input type="checkbox" checked={!!d.required}
                    onChange={(e) => setDel(i, { required: e.target.checked })} />
                  required
                </label>
                <input style={{ ...input, width: 130, fontSize: 12 }} value={d.hint || ''}
                  placeholder="hint (optional)"
                  onChange={(e) => setDel(i, { hint: e.target.value })} />
                <button type="button" style={{ ...input, cursor: 'pointer', fontSize: 11, padding: '0 6px', color: 'var(--red)', borderColor: 'var(--red)' }}
                  onClick={() => setDels((prev) => prev.filter((_, j) => j !== i))}
                  title={isDeliverableFilled(d)
                    ? 'remove this ask — it is already FILLED, so the thread will record what was dropped'
                    : 'remove this ask'}>
                  🗑
                </button>
                {isDeliverableFilled(d) && (
                  <span style={{ ...cfgHint, color: 'var(--green)' }}
                    title={`filled by ${d.filled_by} — removing it posts a system comment naming the value`}>
                    ✓ filled
                  </span>
                )}
              </div>
            ))}
            <button type="button" disabled={dels.length >= 8}
              style={{ ...input, cursor: dels.length >= 8 ? 'default' : 'pointer', fontSize: 12, marginTop: 8, opacity: dels.length >= 8 ? 0.4 : 1 }}
              onClick={() => setDels((prev) => [...prev, makeDeliverable()])}
              title={dels.length >= 8
                ? 'a step may declare at most 8 deliverables — if it needs more, it is really two steps'
                : 'add a deliverable this step must produce'}>
              ＋ add deliverable
            </button>
          </div>
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 14 }}>
          <button type="button" style={{ ...input, cursor: 'pointer' }} onClick={onCancel}>Cancel</button>
          <button type="button" disabled={!canSave}
            style={{
              ...input, cursor: canSave ? 'pointer' : 'default', fontWeight: 700,
              color: '#fff', background: 'var(--blue)', borderColor: 'var(--blue)', opacity: canSave ? 1 : 0.4,
            }}
            onClick={submit}>{submitLabel}</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
