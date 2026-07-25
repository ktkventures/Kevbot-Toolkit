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
 */
'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import { apiFetch } from '@/lib/api/client';
import {
  Task, Comment, ChecklistStep, RunRow, STATUSES, AREAS, ASSIGNEES, ORIGINS,
  STATUS_COLOR, STATUS_DEF, withLegacy, input, tagChip, relTime,
  RoleChip, NextChip, ProgressBar, RolePicker, RunButton, OutcomeChip,
} from './taskBoardShared';

// Default GitHub-style sanitize schema, plus data: image URIs (spec allows
// pasted data-URI images; scripts/handlers stay stripped).
const mdSchema = {
  ...defaultSchema,
  protocols: {
    ...defaultSchema.protocols,
    src: [...((defaultSchema.protocols as Record<string, string[]>)?.src || []), 'data'],
  },
};

const panel: React.CSSProperties = {
  border: '1px solid var(--border)', borderRadius: 10, background: 'var(--bg-card, var(--bg-input))',
  display: 'flex', flexDirection: 'column', minHeight: 0,
};
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

type Tab = 'summary' | 'process' | 'config';

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
}

export default function TaskDetailModal({
  task, allTasks, visionOptions, roles = ASSIGNEES, initialSelected, patch, del, onClose, onOpenTask,
  commentAuthor, onPickAuthor, headless = new Set(['M']), onRunRequested,
}: Props) {
  const subtasks = useMemo(
    () => allTasks.filter((t) => t.parent_id === task.id), [allTasks, task.id]);
  const vision = (task.tags || []).includes('vision') || subtasks.length > 0;
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
  const feedRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => { loadThread(scoped.id); loadRuns(scoped.id); setCtxEdit(false); setCtxPreview(false); }, [scoped.id, loadThread, loadRuns]);
  useEffect(() => { setDraftDesc(scoped.description || ''); }, [scoped.id, scoped.description]);
  const thread = threads[scoped.id] || [];
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [thread.length, scoped.id]);

  // Status/assignee edits write a system entry on the task's thread server-side
  // — re-fetch it so the entry appears without closing the modal.
  const patchTracked = async (fields: Partial<Task>) => {
    await patch(task.id, fields);
    loadThread(task.id);
  };

  const postComment = async () => {
    if (!draftComment.trim()) return;
    try {
      await apiFetch(`/api/dev-tasks/${scoped.id}/comments`, {
        method: 'POST', body: JSON.stringify({ body: draftComment, author: commentAuthor }),
      });
      setDraftComment('');
      loadThread(scoped.id);
    } catch { /* surfaced by thread staying unchanged */ }
  };

  const saveDesc = () => {
    if (draftDesc !== (scoped.description || '')) patch(scoped.id, { description: draftDesc });
    setCtxEdit(false); setCtxPreview(false);
  };

  // Checklist ops operate on the SCOPED item (the selected pipeline row, or
  // the task itself) — ALWAYS send the whole array (JSONB replace).
  const steps: ChecklistStep[] = scoped.checklist || [];
  const setSteps = (next: ChecklistStep[]) => patch(scoped.id, { checklist: next });
  const addStep = () => {
    if (!newStep.trim()) return;
    setSteps([...steps, { text: newStep.trim(), done: false, role: null }]);
    setNewStep('');
  };
  const moveStep = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    setSteps(next);
  };

  const parseIds = (s: string) =>
    s.split(',').map((x) => parseInt(x.trim(), 10)).filter((n) => Number.isFinite(n));
  const parseTags = (s: string) => s.split(',').map((x) => x.trim()).filter(Boolean);

  const Md = ({ text }: { text: string }) => (
    <div className="task-md">
      <ReactMarkdown remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, [rehypeSanitize, mdSchema]]}>
        {text || '_no description yet — click Edit_'}
      </ReactMarkdown>
    </div>
  );

  // Header-level counts describe the MODAL task; the Process tab describes
  // the scoped item (a selected subtask is always a leaf — one-level rule).
  const doneCount = vision
    ? subtasks.filter((s) => s.status === 'Done').length
    : (task.checklist || []).filter((s) => s.done).length;
  const totalCount = vision ? subtasks.length : (task.checklist || []).length;
  const stripRows = vision ? subtasks : [];

  const scopedIsLeaf = scoped.id === task.id ? !vision : true;
  const scopedDone = steps.filter((s) => s.done).length;
  // Process tab is invalid when the scope is the vision itself — fall back.
  const effTab: Tab = tab === 'process' && !scopedIsLeaf ? 'summary' : tab;

  const selectRow = (id: number) => setSelected(id);

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '4vh 16px',
    }}>
      <style>{`
        .task-md { font-size: 13px; line-height: 1.5; }
        .task-md h1, .task-md h2, .task-md h3 { margin: 10px 0 6px; }
        .task-md p { margin: 6px 0; }
        .task-md img { max-width: 100%; border-radius: 6px; }
        .task-md table { border-collapse: collapse; margin: 8px 0; }
        .task-md th, .task-md td { border: 1px solid var(--border); padding: 4px 8px; font-size: 12.5px; }
        .task-md code { background: var(--bg-input); padding: 1px 4px; border-radius: 4px; font-size: 12px; }
        .task-md pre { background: var(--bg-input); padding: 8px; border-radius: 6px; overflow-x: auto; }
        .task-md ul, .task-md ol { padding-left: 20px; margin: 6px 0; }
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
            {vision ? 'VISION ITEM' : task.parent_id != null ? 'SUBTASK' : 'TASK'} · #{task.id}
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
            {STATUSES.map((s) => <option key={s} value={s} title={STATUS_DEF[s]}>{s}</option>)}
          </select>
          <RolePicker value={task.assignee} pickTitle="assignee" allowEmpty
            options={withLegacy(roles, task.assignee || '').filter(Boolean)}
            onPick={(r) => patchTracked({ assignee: r })} />
          <NextChip t={task} subtasks={subtasks} />
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
          {vision && totalCount > 0 &&
            <span style={tagChip} title="subtasks done / total">{doneCount}/{totalCount}</span>}
          <span style={{ flex: 1 }} />
          <RunButton task={task} allTasks={allTasks} headless={headless}
            latestRun={(runsMap[task.id] || [])[0]} onRequest={requestRun} />
          {runErr && <span style={{ color: 'var(--red)', fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={runErr}>{runErr}</span>}
        </div>

        {/* ── Body: left (context [+ vision strip]) · right (activity) ── */}
        <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

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
                <button style={chipBtn(effTab === 'config')} onClick={() => setTab('config')}>config</button>
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
                    {!ctxEdit && <Md text={scoped.description} />}
                    {ctxEdit && !ctxPreview && (
                      <textarea style={{ ...input, width: '100%', minHeight: 180, fontFamily: 'monospace', fontSize: 12.5 }}
                        value={draftDesc} onChange={(e) => setDraftDesc(e.target.value)}
                        placeholder="markdown + sanitized inline HTML…" />
                    )}
                    {ctxEdit && ctxPreview && <Md text={draftDesc} />}
                  </>
                )}

                {effTab === 'process' && scopedIsLeaf && (
                  <>
                    <div style={{ ...cfgHint, marginBottom: 8 }}>
                      Handoff pipeline — each step&apos;s chip is its owner; reassign the task at each
                      handoff point. A step that needs its own thread gets promoted to a subtask.
                    </div>
                    {steps.map((s, i) => (
                      <div key={i} style={{
                        display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px',
                        borderRadius: 8, fontSize: 13, marginBottom: 2,
                      }}>
                        <input type="checkbox" checked={s.done}
                          onChange={(e) => setSteps(steps.map((x, j) => j === i ? { ...x, done: e.target.checked } : x))} />
                        <span style={{
                          flex: 1, minWidth: 0,
                          textDecoration: s.done ? 'line-through' : 'none', opacity: s.done ? 0.6 : 1,
                        }}>{s.text}</span>
                        <RolePicker value={s.role} pickTitle="step owner" allowEmpty
                          options={roles.filter(Boolean)}
                          onPick={(r) => setSteps(steps.map((x, j) => j === i ? { ...x, role: r || null } : x))} />
                        <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move up" onClick={() => moveStep(i, -1)}>↑</span>
                        <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move down" onClick={() => moveStep(i, 1)}>↓</span>
                        <span style={{ cursor: 'pointer', color: 'var(--red)' }} title="delete step"
                          onClick={() => setSteps(steps.filter((_, j) => j !== i))}>✕</span>
                      </div>
                    ))}
                    {steps.length === 0 &&
                      <div style={{ ...cfgHint, padding: '4px 8px' }}>No steps yet.</div>}
                    <div style={{ display: 'flex', gap: 6, padding: '4px 8px', marginTop: 2 }}>
                      <input style={{ ...input, flex: 1 }} placeholder="add a step…" value={newStep}
                        onChange={(e) => setNewStep(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && addStep()} />
                      <button style={{ ...input, cursor: 'pointer' }} onClick={addStep}>Add</button>
                    </div>
                  </>
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
                      <label style={cfgLabel}>parent vision
                        <div style={{ marginTop: 2 }}>
                          <select style={input} value={scoped.parent_id ?? ''}
                            onChange={(e) => patch(scoped.id, { parent_id: e.target.value ? +e.target.value : null })}>
                            <option value="">— none (vision) —</option>
                            {visionOptions.filter((v) => v.id !== scoped.id).map((v) =>
                              <option key={v.id} value={v.id}>#{v.id} {v.title.slice(0, 32)}</option>)}
                          </select>
                        </div>
                      </label>
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
                      <label style={{ fontSize: 13, display: 'block' }}>
                        <input type="checkbox" checked={scoped.is_urgent}
                          onChange={(e) => patch(scoped.id, { is_urgent: e.target.checked })} /> ⚡ urgent
                        <span style={cfgHint}> — jump the queue (a tag, not a priority)</span>
                      </label>
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
                              <pre style={{
                                fontSize: 11, whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto',
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
                      <span style={{ fontSize: 14 }}>{s.status === 'Done' ? '☑' : '☐'}</span>
                      <span style={{
                        flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                        textDecoration: s.status === 'Done' ? 'line-through' : 'none',
                        opacity: s.status === 'Done' ? 0.6 : 1,
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

          {/* Activity panel */}
          <div style={{ ...panel, flex: 1 }}>
            <div style={panelHead}>
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                Activity — {scoped.id === task.id ? 'this task' : `#${scoped.id} ${scoped.title.slice(0, 30)}`}
              </span>
            </div>
            <div ref={feedRef} style={{ flex: 1, overflowY: 'auto', padding: '8px 12px' }}>
              {thread.length === 0 &&
                <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>No activity yet.</div>}
              {thread.map((cm) => cm.author === 'system' ? (
                <div key={cm.id} style={{ fontSize: 11.5, color: 'var(--text-tertiary)', fontStyle: 'italic', margin: '6px 0', display: 'flex', gap: 6, alignItems: 'baseline' }}>
                  <RoleChip role="system" title="system" />
                  <span>{cm.body} · {relTime(cm.created_at)}</span>
                </div>
              ) : (
                <div key={cm.id} style={{ fontSize: 13, margin: '8px 0', paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
                  <RoleChip role={cm.author} title={cm.author} />
                  <span style={{ color: 'var(--text-tertiary)', fontSize: 11, marginLeft: 6 }}>{relTime(cm.created_at)}</span>
                  <div style={{ marginTop: 2, whiteSpace: 'pre-wrap' }}>{cm.body}</div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid var(--border)', alignItems: 'center' }}>
              <RolePicker value={commentAuthor} pickTitle="comment as" up
                options={withLegacy(roles.filter(Boolean), commentAuthor)}
                onPick={onPickAuthor} />
              <input style={{ ...input, flex: 1, minWidth: 0 }} placeholder="comment…" value={draftComment}
                onChange={(e) => setDraftComment(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && postComment()} />
              <button style={{ ...input, cursor: 'pointer', background: 'var(--blue)', color: '#fff' }}
                onClick={postComment}>Comment</button>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 10, textAlign: 'right' }}>
          <button style={{ ...input, cursor: 'pointer', color: 'var(--red)' }} onClick={() => del(task.id)}>Delete task</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
