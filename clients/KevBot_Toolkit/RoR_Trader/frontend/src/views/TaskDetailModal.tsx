/**
 * Three-panel task detail modal (Spec_Tasks_Team_Board.md Phase 3B).
 *
 * Layout: Context (markdown + sanitized HTML) top-left · Summary/Checklist
 * strip bottom-left · Activity thread right with the composer pinned at the
 * bottom. The checklist strip is a SELECTOR, not just a list:
 *  - Vision modal: rows are its live subtasks. Selecting one re-scopes BOTH
 *    the Context panel (that subtask's description) and the Activity panel
 *    (that subtask's thread) in place — a posted comment lands on the
 *    SELECTED subtask. "Summary" (first position) selects the vision itself.
 *  - Leaf modal: rows are the JSONB checklist steps — checkboxes with an
 *    optional role chip, deliberately NO per-step context/thread (a step that
 *    needs one gets promoted to a real subtask). Selection only highlights.
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
  Task, Comment, ChecklistStep, STATUSES, AREAS, ASSIGNEES, ORIGINS,
  STATUS_COLOR, withLegacy, input, tagChip, relTime,
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

interface Props {
  task: Task;
  allTasks: Task[];
  visionOptions: Task[];
  patch: (id: number, fields: Partial<Task>) => Promise<void> | void;
  del: (id: number) => void;
  onClose: () => void;
  onOpenTask: (id: number) => void;
  commentAuthor: string;
  onPickAuthor: (a: string) => void;
}

export default function TaskDetailModal({
  task, allTasks, visionOptions, patch, del, onClose, onOpenTask,
  commentAuthor, onPickAuthor,
}: Props) {
  const subtasks = useMemo(
    () => allTasks.filter((t) => t.parent_id === task.id), [allTasks, task.id]);
  const vision = (task.tags || []).includes('vision') || subtasks.length > 0;

  // Selector state: which item the Context + Activity panels are scoped to.
  const [selected, setSelected] = useState<'summary' | number>('summary');
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
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
  const [stripCollapsed, setStripCollapsed] = useState(false);
  const [newStep, setNewStep] = useState('');
  const feedRef = useRef<HTMLDivElement>(null);

  const loadThread = useCallback(async (id: number) => {
    try {
      const cs = await apiFetch<Comment[]>(`/api/dev-tasks/${id}/comments`);
      setThreads((m) => ({ ...m, [id]: cs || [] }));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadThread(scoped.id); setCtxEdit(false); setCtxPreview(false); }, [scoped.id, loadThread]);
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

  // Checklist ops (leaf only) — ALWAYS send the whole array (JSONB replace).
  const steps: ChecklistStep[] = task.checklist || [];
  const setSteps = (next: ChecklistStep[]) => patch(task.id, { checklist: next });
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

  const doneCount = vision
    ? subtasks.filter((s) => s.status === 'Done').length
    : steps.filter((s) => s.done).length;
  const totalCount = vision ? subtasks.length : steps.length;

  // Portal to <body>: the page content wrapper is its own stacking context
  // below the fixed sidebar, so an in-tree overlay renders UNDER the sidebar
  // once the modal is wide enough to overlap it.
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
        {/* ── Header: title + meta ─────────────────────────────── */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-tertiary)', letterSpacing: 0.6, marginBottom: 2 }}>
              {vision ? 'VISION ITEM' : task.parent_id != null ? `SUBTASK OF #${task.parent_id}` : 'TASK'} · #{task.id}
            </div>
            <input style={{ ...input, fontSize: 16, fontWeight: 600, width: '100%', padding: '5px 8px' }}
              value={task.title}
              onChange={(e) => patch(task.id, { title: e.target.value })} />
          </div>
          <button style={{ ...input, cursor: 'pointer' }} onClick={onClose}>✕ close</button>
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', margin: '8px 0' }}>
          <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>Pri</span>
          <input style={{ ...input, width: 34, padding: '3px' }} type="number" value={task.priority_phase}
            onChange={(e) => patch(task.id, { priority_phase: +e.target.value })} />
          <input style={{ ...input, width: 50, padding: '3px' }} type="number" step="0.05" value={task.priority_seq}
            onChange={(e) => patch(task.id, { priority_seq: +e.target.value })} />
          <select style={{ ...input, color: STATUS_COLOR[task.status] }} value={task.status}
            onChange={(e) => patchTracked({ status: e.target.value })}>
            {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select style={input} value={task.area} onChange={(e) => patch(task.id, { area: e.target.value })}>
            {AREAS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          <select style={input} value={task.assignee || ''} onChange={(e) => patchTracked({ assignee: e.target.value })}>
            {withLegacy(ASSIGNEES, task.assignee || '').map((a) => <option key={a} value={a}>@{a || '—'}</option>)}
          </select>
          <select style={input} title="parent vision item" value={task.parent_id ?? ''}
            onChange={(e) => patch(task.id, { parent_id: e.target.value ? +e.target.value : null })}>
            <option value="">— none (vision) —</option>
            {visionOptions.filter((v) => v.id !== task.id).map((v) =>
              <option key={v.id} value={v.id}>#{v.id} {v.title.slice(0, 32)}</option>)}
          </select>
          {task.parent_id != null && (
            <select style={input} title="origin" value={task.origin || 'planned'}
              onChange={(e) => patch(task.id, { origin: e.target.value })}>
              {ORIGINS.map((o) => <option key={o} value={o}>{o === 'discovered' ? '🔍 discovered' : o}</option>)}
            </select>
          )}
          <label style={{ fontSize: 12 }}><input type="checkbox" checked={task.impacts_live}
            onChange={(e) => patch(task.id, { impacts_live: e.target.checked })} /> 🔴 live</label>
          <label style={{ fontSize: 12 }}><input type="checkbox" checked={task.needs_live_validation}
            onChange={(e) => patch(task.id, { needs_live_validation: e.target.checked })} /> ⏳ validate</label>
          <label style={{ fontSize: 12 }}><input type="checkbox" checked={task.is_urgent}
            onChange={(e) => patch(task.id, { is_urgent: e.target.checked })} /> ⚡ urgent</label>
          <input key={`tags-${task.id}`} style={{ ...input, width: 150 }} title="tags (comma-separated)"
            placeholder="tags…" defaultValue={(task.tags || []).join(', ')}
            onBlur={(e) => {
              const tags = parseTags(e.target.value);
              if (tags.join('|') !== (task.tags || []).join('|')) patch(task.id, { tags });
            }} />
          <input key={`blk-${task.id}`} style={{ ...input, width: 110 }} title="⛔ blocked by (task ids)"
            placeholder="⛔ ids…" defaultValue={(task.blocked_by || []).join(', ')}
            onBlur={(e) => {
              const ids = parseIds(e.target.value);
              if (ids.join('|') !== (task.blocked_by || []).join('|')) patch(task.id, { blocked_by: ids });
            }} />
        </div>

        {/* ── Body: left (context + checklist) · right (activity) ── */}
        <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

            {/* Context panel */}
            <div style={{ ...panel, flex: 1 }}>
              <div style={panelHead}>
                <span>Context · {scoped.id === task.id ? 'what to do' : `subtask #${scoped.id}`}</span>
                <span style={{ flex: 1 }} />
                {!ctxEdit && <button style={chipBtn(true)} onClick={() => setCtxEdit(true)}>edit</button>}
                {ctxEdit && (
                  <>
                    <button style={chipBtn(!ctxPreview)} onClick={() => setCtxPreview(false)}>write</button>
                    <button style={chipBtn(ctxPreview)} onClick={() => setCtxPreview(true)}>preview</button>
                    <button style={{ ...chipBtn(true), background: 'var(--blue)', color: '#fff' }}
                      onClick={saveDesc}>save</button>
                  </>
                )}
              </div>
              <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6 }}>{scoped.title}</div>
                {!ctxEdit && <Md text={scoped.description} />}
                {ctxEdit && !ctxPreview && (
                  <textarea style={{ ...input, width: '100%', minHeight: 180, fontFamily: 'monospace', fontSize: 12.5 }}
                    value={draftDesc} onChange={(e) => setDraftDesc(e.target.value)}
                    placeholder="markdown + sanitized inline HTML…" />
                )}
                {ctxEdit && ctxPreview && <Md text={draftDesc} />}
              </div>
            </div>

            {/* Summary / checklist strip — the SELECTOR */}
            <div style={{ ...panel, maxHeight: stripCollapsed ? undefined : '45%' }}>
              <div style={{ ...panelHead, cursor: 'pointer' }} onClick={() => setStripCollapsed(!stripCollapsed)}>
                <span>{stripCollapsed ? '▸' : '▾'} Process checklist · {doneCount}/{totalCount} done
                  {vision ? ' · your subtasks sign these off' : ' · steps (promote a step to a subtask if it needs its own thread)'}</span>
              </div>
              {!stripCollapsed && (
                <div style={{ overflowY: 'auto', padding: '6px 8px' }}>
                  {/* Summary control — first position, selects the task itself */}
                  <div onClick={() => { setSelected('summary'); setSelectedStep(null); }} style={{
                    display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', borderRadius: 8,
                    cursor: 'pointer', marginBottom: 4, fontSize: 13, fontWeight: 600,
                    border: `1px dashed ${(!vision || selected === 'summary') ? 'var(--blue)' : 'var(--border)'}`,
                    background: (!vision || selected === 'summary') ? 'var(--bg-input)' : 'transparent',
                  }}>
                    <span>≡ Summary</span>
                    <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--text-tertiary)' }}>
                      {vision ? 'the vision item itself' : 'overview'}
                    </span>
                  </div>

                  {vision && subtasks.map((s) => (
                    <div key={s.id} onClick={() => setSelected(s.id)} style={{
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
                      <span style={{ fontSize: 11, color: STATUS_COLOR[s.status] }}>{s.status}</span>
                      {s.assignee && <span style={{ ...tagChip, marginLeft: 0 }}>@{s.assignee}</span>}
                      {selected === s.id && (
                        <button style={{ ...chipBtn(true), whiteSpace: 'nowrap' }} title="open this subtask's full modal"
                          onClick={(e) => { e.stopPropagation(); onOpenTask(s.id); }}>open full task ↗</button>
                      )}
                    </div>
                  ))}
                  {vision && subtasks.length === 0 &&
                    <div style={{ fontSize: 12, color: 'var(--text-tertiary)', padding: '4px 8px' }}>
                      No subtasks yet — add them from the board (+ subtask).</div>}

                  {!vision && steps.map((s, i) => (
                    <div key={i} onClick={() => setSelectedStep(i)} style={{
                      display: 'flex', alignItems: 'center', gap: 8, padding: '5px 8px', borderRadius: 8,
                      fontSize: 13, marginBottom: 2, cursor: 'default',
                      background: selectedStep === i ? 'var(--bg-input)' : 'transparent',
                    }}>
                      <input type="checkbox" checked={s.done}
                        onChange={(e) => setSteps(steps.map((x, j) => j === i ? { ...x, done: e.target.checked } : x))} />
                      <span style={{
                        flex: 1, minWidth: 0,
                        textDecoration: s.done ? 'line-through' : 'none', opacity: s.done ? 0.6 : 1,
                      }}>{s.text}</span>
                      <select style={{ ...input, fontSize: 11, padding: '1px 4px' }} title="owner role"
                        value={s.role || ''}
                        onChange={(e) => setSteps(steps.map((x, j) => j === i ? { ...x, role: e.target.value || null } : x))}>
                        {ASSIGNEES.map((a) => <option key={a} value={a}>{a || '—'}</option>)}
                      </select>
                      <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move up" onClick={() => moveStep(i, -1)}>↑</span>
                      <span style={{ cursor: 'pointer', opacity: 0.6 }} title="move down" onClick={() => moveStep(i, 1)}>↓</span>
                      <span style={{ cursor: 'pointer', color: 'var(--red)' }} title="delete step"
                        onClick={() => setSteps(steps.filter((_, j) => j !== i))}>✕</span>
                    </div>
                  ))}
                  {!vision && (
                    <div style={{ display: 'flex', gap: 6, padding: '4px 8px', marginTop: 2 }}>
                      <input style={{ ...input, flex: 1 }} placeholder="add a step…" value={newStep}
                        onChange={(e) => setNewStep(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && addStep()} />
                      <button style={{ ...input, cursor: 'pointer' }} onClick={addStep}>Add</button>
                    </div>
                  )}
                </div>
              )}
            </div>
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
                <div key={cm.id} style={{ fontSize: 11.5, color: 'var(--text-tertiary)', fontStyle: 'italic', margin: '6px 0' }}>
                  ⚙ {cm.body} · {relTime(cm.created_at)}
                </div>
              ) : (
                <div key={cm.id} style={{ fontSize: 13, margin: '8px 0', paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
                  <b>{cm.author}</b>
                  <span style={{ color: 'var(--text-tertiary)', fontSize: 11, marginLeft: 6 }}>{relTime(cm.created_at)}</span>
                  <div style={{ marginTop: 2, whiteSpace: 'pre-wrap' }}>{cm.body}</div>
                </div>
              ))}
            </div>
            <div style={{ display: 'flex', gap: 6, padding: 10, borderTop: '1px solid var(--border)' }}>
              <select style={input} title="comment as" value={commentAuthor}
                onChange={(e) => onPickAuthor(e.target.value)}>
                {withLegacy(ASSIGNEES.filter(Boolean), commentAuthor).map((a) => <option key={a} value={a}>@{a}</option>)}
              </select>
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
