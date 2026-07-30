/**
 * M-Session Dashboard — /admin/m-session (board #220, Phase 1).
 *
 * Kevin, 07-30: *"centralize everything that the MSession needs to do its job
 * well… My goal here is to have you be able to run pretty autonomously but do it
 * in a way that is transparent and clear to me. Otherwise it's hard for me to
 * help."* The purpose in one line: **make M's working state legible.**
 *
 * Five sections were scoped. THIS STEP BUILDS THREE, read-only:
 *
 *   5 ASK-AI INBOX  — top of the page. Unanswered `@M` comments, OLDEST first,
 *                     with AGE as the headline. Kevin's fast lane from the board
 *                     into the M session; the channel was live all along and the
 *                     session stopped listening (8 `@M` comments on 07-26, then
 *                     near-silence). Nothing is being plumbed — missing it is
 *                     simply being made visible.
 *   1 M'S QUEUE     — what is on M's plate. Phase 1 renders board order; the
 *                     order M INTENDS is a plan that lives nowhere today and is
 *                     step 4 (the comparator is already a parameter).
 *   2 ACTIVITY LOG  — events M did NOT generate, auto-classified by RULE.
 *
 * Sections 3 (the rules, RENDERED FROM SOURCE) and 4 (M's canvas) are step 4 of
 * the chain, gated behind M's review — they need the one genuinely new store
 * (M's ordering + seen marks + canvas text) and therefore a migration. The page
 * says so at the bottom rather than looking finished.
 *
 * TWO DESIGN PRINCIPLES this page is held to (Kevin, 07-30):
 *   A RENDER SOURCES, DO NOT RESTATE THEM — nothing here retypes a rule that
 *     lives in a file. Section 3 is deferred precisely because doing it right
 *     means reading the charter, not summarising it.
 *   B IT IS A VIEW, NOT A SECOND BOARD — every number is a read over
 *     `dev_tasks` / `dev_task_comments` / `run_history`. This step stores
 *     NOTHING. If it started holding task state it would drift from the board
 *     and Kevin would have two answers to one question.
 *
 * EVERY ROW DEEP-LINKS TO ITS TASK. That is load-bearing, not polish: the model
 * Kevin ruled for is chat-carries-pointers, detail-one-click-away. A row you
 * cannot click defeats the point of the page.
 *
 * Conventions follow AdminDispatchPage.tsx deliberately (same fetch shape, same
 * UTC stamps, same "print the gaps" footer) — this is a sibling surface, and two
 * idioms on one admin section is how a section stops being read.
 *
 * Production-bundle rails (CLAUDE.md): every hook before any early return, no
 * IIFE initialisation, declare before the useMemo that references it.
 *
 * RENDER BUDGET (board #240, Kevin's first real use): the activity log mapped
 * all ~435 feed rows into the DOM on first paint and stalled his browser. It now
 * scrolls inside itself and paints FEED_PAGE_ROWS at a time, extending on scroll
 * or on an explicit button. **Render less at a time, never keep less** — the log
 * is a coverage record, so every event stays fetched, classified, counted and
 * reachable; only the DOM is bounded. Section 5 stays unbounded on purpose.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import {
  ChecklistStep, HandoffChain, OutcomeChip, RoleChip, RunRow, STATUS_COLOR, Task,
  ageColor, ageShort, hoursSince, input, isProcessChain, relTime,
  stepOwner, stepTitle, tagChip, utcDay, utcStamp,
} from './taskBoardShared';
import { MD_CSS, Md } from './taskMarkdown';
import {
  FEED_MAX_HEIGHT, FEED_PAGE_ROWS, FEED_RULES, FeedClass, FeedComment, FeedEvent,
  InboxItem, MMark, MOrderRow, MSessionState, M_SESSION_AUTHOR, RulesPayload,
  UNCLASSIFIED, askAiInbox, buildFeed, feedWindow, groupByDay, intendedOrder,
  mQueue, nearBottom, waitingOnKevin,
} from './mSessionFeed';

/** Auto-refresh cadence. Slower than /admin/dispatch's 15s on purpose: this
 *  page pulls comment BODIES, which is the expensive payload on the board. */
const REFRESH_MS = 30000;
/** Board-wide comment window. 400 was the measured sample behind this task's
 *  design (131 system transitions, 89 agent reports, 42 R, 6 Kevin, 130 M). */
const COMMENT_LIMIT = 400;
/** Per-comment excerpt on the wire; the thread has the rest, one click away.
 *  Trimmed 1400 → 600 for board #240: at 300 comments a refresh was carrying up
 *  to ~420KB of body text every 30s to render 320-char excerpts. 600 still
 *  clears the feed excerpt with room to spare and leaves an Ask-AI question
 *  readable in full; anything longer already prints "excerpt of N chars". */
const COMMENT_CHARS = 600;
/** Run-log depth — matched to /admin/dispatch so the two agree about history. */
const RUN_LIMIT = 400;
/** How much of a body the feed shows before "open the task". */
const FEED_EXCERPT = 320;

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11.5 };
const dim: React.CSSProperties = { color: 'var(--text-tertiary)' };

const Panel = ({ n, title, sub, right, children }: {
  n: string; title: string; sub?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode;
}) => (
  <div style={{ marginBottom: 16 }}>
    <Card>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase',
            color: 'var(--text-tertiary)',
          }}>
            <span style={{ color: 'var(--text-secondary)' }}>{n} · </span>{title}
          </div>
          {sub && <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 3 }}>{sub}</div>}
        </div>
        {right}
      </div>
      {children}
    </Card>
  </div>
);

/** Deep link to a task thread — the one interaction the whole page rests on. */
const TaskLink = ({ id, title, bold = true }: { id: number | null; title?: string; bold?: boolean }) => {
  if (id == null) {
    return <span style={{ ...dim, fontSize: 12.5 }}>(no task — environment signal)</span>;
  }
  return (
    <Link href={`/admin/tasks?task=${id}`} title={`open task #${id} in the board`}
      style={{
        color: 'inherit', fontWeight: bold ? 600 : 400, fontSize: 13,
        minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>
      #{id}{title ? ` ${title}` : ''}
    </Link>
  );
};

/** Routine / attention chip, carrying the deciding rule in its tooltip. */
const ClassChip = ({ cls, rule, why }: { cls: FeedClass; rule: string; why: string }) => (
  <span title={`rule "${rule}" → ${cls}\n\n${why}`} style={{
    ...tagChip, marginLeft: 0, fontWeight: cls === 'attention' ? 700 : 500,
    borderColor: cls === 'attention' ? 'var(--amber, #d98c00)' : 'var(--border)',
    color: cls === 'attention' ? 'var(--amber, #d98c00)' : 'var(--text-tertiary)',
  }}>{cls === 'attention' ? '● ' : '· '}{rule}</span>
);

/** Which chain step a task is on — same first-un-done-step rule the dispatcher
 *  and /admin/dispatch use (shared helpers verbatim, not a re-implementation). */
function stepLabel(task: Task): string | null {
  const cl: ChecklistStep[] = task.checklist || [];
  if (cl.length === 0 || !isProcessChain(cl)) return null;
  const idx = cl.findIndex((s) => !s.done);
  if (idx < 0) return `all ${cl.length} steps complete`;
  const owner = stepOwner(cl[idx]);
  return `step ${idx + 1}/${cl.length} — ${stepTitle(cl[idx])}${owner ? ` (${owner})` : ''}`;
}

export default function AdminMSessionPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [comments, setComments] = useState<FeedComment[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [feedErr, setFeedErr] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  // Ages tick between fetches — an Ask AI that is "4m old" must not read 4m for
  // half a minute; age is the number this page is judged on. Every age here is
  // computed inline in the JSX, so the re-render IS the update and the counter's
  // value is never read — hence the discarded slot rather than a unused-var
  // suppression.
  const [, setTick] = useState(0);
  const [showRoutine, setShowRoutine] = useState(false);
  const [showAnswered, setShowAnswered] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [openBodies, setOpenBodies] = useState<Set<string>>(new Set());
  // How many pages of the activity log are PAINTED (board #240). Nothing is
  // dropped from `feed` — this bounds the DOM, not the record.
  const [feedPages, setFeedPages] = useState(1);
  const feedScroll = useRef<HTMLDivElement | null>(null);

  // ── Step-4 state: M's own store + the rules rendered from source ────────
  const [store, setStore] = useState<MSessionState | null>(null);
  const [storeErr, setStoreErr] = useState<string | null>(null);
  const [rules, setRules] = useState<RulesPayload | null>(null);
  const [rulesErr, setRulesErr] = useState<string | null>(null);
  const [openRule, setOpenRule] = useState<string | null>(null);
  // Local, uncommitted edits. Held separately from `store` so a 30s refresh
  // cannot silently overwrite what M is in the middle of typing — the canvas is
  // the one place on this page where losing a draft costs real work.
  const [plan, setPlan] = useState<MOrderRow[] | null>(null);
  const [canvasDraft, setCanvasDraft] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [saveErr, setSaveErr] = useState<string | null>(null);
  const [markNote, setMarkNote] = useState<{ key: string; text: string } | null>(null);

  const loadStore = useCallback(async () => {
    try {
      setStoreErr(null);
      setStore(await apiFetch<MSessionState>('/api/m-session/state'));
    } catch (e) { setStoreErr(String(e)); }
  }, []);

  const load = useCallback(async () => {
    try {
      setErr(null);
      // Three list GETs, joined client-side — never a per-row fetch (board #148).
      // include_done=true is required: the feed references closed tasks and
      // would otherwise render "#NNN (unknown)" across the history.
      const [ts, rh, cm] = await Promise.all([
        apiFetch<Task[]>('/api/dev-tasks?include_done=true'),
        apiFetch<RunRow[]>(`/api/run-history?limit=${RUN_LIMIT}`).catch(() => [] as RunRow[]),
        apiFetch<FeedComment[]>(
          `/api/dev-tasks/comments/recent?limit=${COMMENT_LIMIT}&max_chars=${COMMENT_CHARS}`)
          // A build without #220's endpoint must say so, not render an empty
          // feed that looks like a quiet board (the #157 dead-alarm lesson).
          .catch((e) => { setFeedErr(String(e)); return [] as FeedComment[]; }),
      ]);
      setTasks(ts || []);
      setRuns(rh || []);
      setComments(cm || []);
      if ((cm || []).length) setFeedErr(null);
      setFetchedAt(new Date().toISOString());
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { loadStore(); }, [loadStore]);
  useEffect(() => {
    // The rules are files on disk; they change on a deploy, not on a timer, so
    // this is a one-shot read rather than part of the 30s refresh.
    apiFetch<RulesPayload>('/api/m-session/rules')
      .then(setRules).catch((e) => setRulesErr(String(e)));
  }, []);
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) { load(); loadStore(); }
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [load, loadStore]);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  const taskById = useMemo(() => {
    const m = new Map<number, Task>();
    tasks.forEach((t) => m.set(t.id, t));
    return m;
  }, [tasks]);

  const inbox = useMemo(() => askAiInbox(comments), [comments]);
  const kevinQueue = useMemo(() => waitingOnKevin(comments), [comments]);
  const feed = useMemo(() => buildFeed(comments, runs), [comments, runs]);

  // M's stated plan, or the empty plan while the store is unprovisioned. The
  // comparator swap step 2 left a parameter for is exactly this one argument.
  const order = useMemo<MOrderRow[]>(
    () => plan ?? (store?.order || []), [plan, store]);
  const queue = useMemo(() => mQueue(tasks, intendedOrder(order)), [tasks, order]);
  const rankOf = useMemo(() => {
    const m = new Map<number, MOrderRow>();
    order.forEach((o) => m.set(o.task_id, o));
    return m;
  }, [order]);
  /** Coverage marks by feed event key — the seen/actioned record. */
  const markOf = useMemo(() => {
    const m = new Map<string, MMark>();
    (store?.marks || []).forEach((x) => m.set(x.event_key, x));
    return m;
  }, [store]);
  const storeReady = !!store?.available;
  const attentionRows = useMemo(
    () => feed.filter((e) => e.cls === 'attention'), [feed]);
  const unreviewed = useMemo(
    () => attentionRows.filter((e) => !markOf.has(e.key)).length,
    [attentionRows, markOf]);

  const feedRows = useMemo(
    () => (showRoutine ? feed : feed.filter((e) => e.cls === 'attention')),
    [feed, showRoutine]);
  const routineCount = useMemo(() => feed.filter((e) => e.cls === 'routine').length, [feed]);
  const unclassified = useMemo(
    () => feed.filter((e) => e.rule === UNCLASSIFIED.id).length, [feed]);
  // M's own comments never enter the feed; the count is the justification for
  // that exclusion, printed rather than asserted.
  const mOwnCount = useMemo(
    () => comments.filter((c) => (c.author || '') === M_SESSION_AUTHOR).length, [comments]);
  const oldestComment = useMemo(
    () => (comments.length ? comments[comments.length - 1].created_at : null), [comments]);

  // ── The render bound (board #240) ────────────────────────────────────────
  // `feedRows` is the full filtered record and stays that way; `win` is only
  // what gets painted. Day grouping runs over the WINDOW, so an unpainted day
  // costs nothing — the previous code grouped all ~435 rows and mapped every
  // one of them into the DOM on first paint.
  const win = useMemo(
    () => feedWindow(feedRows, feedPages), [feedRows, feedPages]);
  const feedDays = useMemo(() => groupByDay(win.rows, utcDay), [win.rows]);

  // Changing the filter changes what "page 1" means; keep the bound honest
  // rather than inheriting a page count grown against the other list.
  useEffect(() => { setFeedPages(1); }, [showRoutine]);

  const showMore = useCallback(() => setFeedPages((n) => n + 1), []);

  /** Lazy extend: painting the next page as the container nears its bottom is
   *  the "lazy-load stuff" half. The explicit button below does the same thing
   *  for anyone who would rather click than scroll — and it is what keeps this
   *  usable if a browser withholds the scroll event. */
  const onFeedScroll = useCallback(() => {
    const el = feedScroll.current;
    if (!el) return;
    if (nearBottom(el.scrollTop, el.clientHeight, el.scrollHeight)) {
      setFeedPages((n) => (n * FEED_PAGE_ROWS < feedRows.length ? n + 1 : n));
    }
  }, [feedRows.length]);

  const toggleBody = (key: string) => setOpenBodies((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const taskTitle = (id: number | null) =>
    (id != null && taskById.get(id)?.title) || '';

  /* ── Writes. Every one of them touches ONLY M's own store; not a single
       call here can change a task's status, assignee or priority. That is
       design principle B held at the call site as well as at the API. ── */

  const write = async (label: string, fn: () => Promise<unknown>) => {
    setSaving(label); setSaveErr(null);
    try { await fn(); await loadStore(); }
    catch (e) { setSaveErr(`${label}: ${String(e)}`); }
    finally { setSaving(null); }
  };

  /** Move a task within M's plan. Seeds the plan from the CURRENT queue order
   *  so the first nudge produces a complete, explicit plan rather than one
   *  ranked task floating above an implicit remainder. */
  const nudge = (taskId: number, dir: -1 | 1) => {
    const base = (plan && plan.length ? plan : queue.map((t, i) => ({
      task_id: t.id, rank: i + 1, why: rankOf.get(t.id)?.why ?? null,
    })));
    const idx = base.findIndex((o) => o.task_id === taskId);
    if (idx < 0) return;
    const j = idx + dir;
    if (j < 0 || j >= base.length) return;
    const next = base.slice();
    [next[idx], next[j]] = [next[j], next[idx]];
    setPlan(next.map((o, i) => ({ ...o, rank: i + 1 })));
  };

  const setWhy = (taskId: number, why: string) => {
    const base = (plan && plan.length ? plan : queue.map((t, i) => ({
      task_id: t.id, rank: i + 1, why: rankOf.get(t.id)?.why ?? null,
    })));
    setPlan(base.map((o) => (o.task_id === taskId ? { ...o, why } : o)));
  };

  const savePlan = () => write('save plan', async () => {
    await apiFetch('/api/m-session/order', {
      method: 'PUT',
      body: JSON.stringify({
        actor: M_SESSION_AUTHOR,
        // task_id + why ONLY — the API rejects anything else, and sending a
        // whole Task here is the mistake that guard exists to catch.
        items: (plan || []).map((o) => ({ task_id: o.task_id, why: o.why || null })),
      }),
    });
    setPlan(null);
  });

  const saveCanvas = () => write('save canvas', async () => {
    await apiFetch('/api/m-session/canvas', {
      method: 'PUT',
      body: JSON.stringify({ body: canvasDraft ?? '', actor: M_SESSION_AUTHOR }),
    });
    setCanvasDraft(null);
  });

  const mark = (e: FeedEvent, state: MMark['state'], note?: string) =>
    write(`mark ${e.key}`, () => apiFetch('/api/m-session/marks', {
      method: 'POST',
      body: JSON.stringify({
        event_key: e.key, state, note: note || null,
        task_id: e.taskId, actor: M_SESSION_AUTHOR,
      }),
    }));

  const unmark = (e: FeedEvent) =>
    write(`unmark ${e.key}`, () => apiFetch(
      `/api/m-session/marks/${encodeURIComponent(e.key)}`, { method: 'DELETE' }));

  const oldestOpenH = inbox.open.length ? hoursSince(inbox.open[0].comment.created_at) : -1;

  const askRow = (it: InboxItem, kind: 'ask-ai' | 'to-kevin') => {
    const c = it.comment;
    const h = hoursSince(c.created_at);
    const answered = !!it.answeredBy;
    return (
      <div key={`${kind}:${c.id}`} style={{
        display: 'flex', gap: 12, padding: '9px 11px', marginBottom: 7, borderRadius: 9,
        border: `1px solid ${answered ? 'var(--border)' : ageColor(h)}`,
        background: answered ? 'transparent' : 'rgba(217,140,0,0.06)',
        alignItems: 'flex-start', opacity: answered ? 0.7 : 1,
      }}>
        <div style={{ minWidth: 74, textAlign: 'right' }}>
          <div style={{ fontSize: 19, fontWeight: 700, color: answered ? 'var(--text-tertiary)' : ageColor(h) }}
            title={`asked ${utcStamp(c.created_at)} · ${relTime(c.created_at)}`}>
            {ageShort(c.created_at) || 'now'}
          </div>
          <div style={{ ...dim, fontSize: 10 }}>{answered ? 'answered' : 'unanswered'}</div>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', gap: 7, alignItems: 'center', flexWrap: 'wrap' }}>
            <RoleChip role={c.author} />
            <TaskLink id={c.task_id} title={taskTitle(c.task_id)} />
            <span style={{ ...dim, ...mono }}>{utcStamp(c.created_at)}</span>
            {answered && (
              <span style={{ ...tagChip, borderColor: 'var(--green)', color: 'var(--green)' }}
                title={`answered by ${it.answeredBy?.author} at ${utcStamp(it.answeredBy?.created_at)}`}>
                ✓ M replied {relTime(it.answeredBy?.created_at)}
              </span>
            )}
          </div>
          <div style={{
            fontSize: 13, marginTop: 4, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            color: 'var(--text-primary)',
          }}>{c.body}</div>
          {(c.body_chars || 0) > c.body.length && (
            <div style={{ ...dim, fontSize: 11, marginTop: 2 }}>
              … excerpt — {c.body_chars} chars in the thread
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <h1 style={{ fontSize: 22, marginBottom: 4 }}>M-Session Dashboard</h1>
        <Link href="/admin/tasks" style={{ fontSize: 12, color: 'var(--blue)' }}>→ task board</Link>
        <Link href="/admin/dispatch" style={{ fontSize: 12, color: 'var(--blue)' }}>→ dispatch (state)</Link>
        <span style={{ flex: 1 }} />
        <span style={{ ...dim, fontSize: 11.5 }}>
          as of {fetchedAt ? utcStamp(fetchedAt) : '—'} · auto-refresh 30s (paused when hidden)
        </span>
        <button style={{ ...input, cursor: 'pointer', fontSize: 12 }} onClick={load}>↻ refresh</button>
      </div>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 14, maxWidth: 980 }}>
        One surface for the M session, so the watch-list lives in a RULE TABLE rather than in M&apos;s
        head — and so Kevin can see what M is holding, in what order, and under what rules. Every
        number is a read over <code style={{ ...mono, margin: '0 4px' }}>dev_tasks</code>/
        <code style={{ ...mono, margin: '0 4px' }}>dev_task_comments</code>/
        <code style={{ ...mono, margin: '0 4px' }}>run_history</code>. The <b>only</b> thing this
        page stores is what the board cannot answer: M&apos;s intended ordering, M&apos;s
        seen/actioned marks, and the canvas text.
        {' '}<b>/admin/dispatch answers STATE</b> (what is running now); <b>this answers HISTORY</b>
        {' '}(what happened that M has to react to).
      </p>

      {err && <Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card>}
      {loading && <Card>Loading…</Card>}
      {feedErr && (
        <div style={{
          border: '1px solid var(--red)', borderRadius: 10, padding: '10px 14px', marginBottom: 14,
          background: 'rgba(200,60,60,0.08)',
        }}>
          <div style={{ color: 'var(--red)', fontWeight: 700, fontSize: 13 }}>
            ⚠ the comment feed did not load — the inbox and activity log below are EMPTY BECAUSE OF
            THIS ERROR, not because the board is quiet.
          </div>
          <div style={{ ...mono, ...dim, marginTop: 4 }}>{feedErr}</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
            Most likely cause: this API build predates board #220 and has no{' '}
            <code style={mono}>GET /api/dev-tasks/comments/recent</code>.
          </div>
        </div>
      )}

      {/* ── 5 · ASK-AI INBOX — first, because he is waiting ─────────────────── */}
      <Panel
        n="5"
        title={`ask-ai inbox — ${inbox.open.length} unanswered`}
        sub={<>
          Kevin&apos;s <b>🤖 Ask AI</b> button posts an <code style={mono}>@M </code>comment from the
          task modal — his fast lane into this session, and the <b>highest-priority thing on this
          page</b>: <i>&quot;usually when I ask the AI within the task board, I&apos;m looking for a
          response pretty quickly.&quot;</i> <b>Oldest first</b>, because this is the only item here
          that gets worse with time. <b>Answered</b> = a later comment on that task authored by the
          M SESSION (<code style={mono}>M</code>) — an <code style={mono}>F·auto</code> report on the
          same thread is <i>not</i> an answer to Kevin, and counting it as one would be exactly the
          false coverage this dashboard exists to prevent. <b>Deliberately UNBOUNDED</b>: the
          activity log below scrolls inside itself and paints incrementally (board #240), and this
          section pointedly does not — it is the top-priority queue and has to be visible at a
          glance. It is also small by construction; the log was the 435-row module.
        </>}
        right={inbox.open.length > 0 ? (
          <span style={{
            fontSize: 12, fontWeight: 700, color: ageColor(oldestOpenH),
            border: `1px solid ${ageColor(oldestOpenH)}`, borderRadius: 8, padding: '3px 9px',
          }}>oldest {ageShort(inbox.open[0].comment.created_at)}</span>
        ) : null}>
        {inbox.open.length === 0 && (
          <div style={{ fontSize: 13, color: 'var(--green)' }}>
            ✓ nothing unanswered in the last {COMMENT_LIMIT} board comments
            {oldestComment ? ` (back to ${utcStamp(oldestComment)})` : ''}.
          </div>
        )}
        {inbox.open.map((it) => askRow(it, 'ask-ai'))}

        {inbox.answered.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }}
              onClick={() => setShowAnswered((v) => !v)}>
              {showAnswered ? '▾' : '▸'} {inbox.answered.length} already answered
            </button>
            {showAnswered && (
              <div style={{ marginTop: 7 }}>{inbox.answered.map((it) => askRow(it, 'ask-ai'))}</div>
            )}
          </div>
        )}

        {/* The reverse direction — both halves of the conversation, one place. */}
        <div style={{ marginTop: 12, paddingTop: 9, borderTop: '1px solid var(--border)' }}>
          <div style={{ ...dim, fontSize: 10.5, letterSpacing: 0.6, marginBottom: 4 }}>
            WAITING ON KEVIN — {kevinQueue.length} open question{kevinQueue.length === 1 ? '' : 's'} M
            {' '}has put to him (a comment by <code style={mono}>M</code> opening with{' '}
            <code style={mono}>@kevin</code>, with no reply from him since)
          </div>
          {kevinQueue.length === 0
            ? <div style={{ ...dim, fontSize: 12.5 }}>nothing outstanding in this window.</div>
            : kevinQueue.map((it) => askRow(it, 'to-kevin'))}
        </div>
      </Panel>

      {/* ── 1 · M's QUEUE ──────────────────────────────────────────────────── */}
      <Panel
        n="1"
        title={`m’s queue — ${queue.length} on M’s plate`}
        sub={<>
          Everything the board says is waiting on the M session (<code style={mono}>assignee=M</code>,
          excluding {'Done'} and {'Blocked'} — assignee = whoever the task waits on, board #171),
          {' '}<b>in the order M INTENDS</b>. That order is a <i>plan</i>, not board metadata, and it
          is the only thing in this section that is stored: <code style={mono}>task_id</code>,{' '}
          <code style={mono}>rank</code> and a one-line <code style={mono}>why</code>. Everything
          else on each row — title, status, chain step, age — is read live from the board.
          {' '}<b>Below the divider M has not decided</b>; those fall back to board priority. The{' '}
          <b>why</b> is what answers <i>&quot;why is that third?&quot;</i> without Kevin having to ask.
        </>}
        right={plan ? (
          <span style={{ display: 'flex', gap: 6 }}>
            <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, borderColor: 'var(--green)', color: 'var(--green)' }}
              disabled={!!saving} onClick={savePlan}>
              {saving === 'save plan' ? 'saving…' : '✓ save plan'}
            </button>
            <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }}
              onClick={() => setPlan(null)}>discard</button>
          </span>
        ) : null}>
        {!storeReady && (
          <div style={{ ...dim, fontSize: 12, marginBottom: 8 }}>
            Ordering is board priority until the store is provisioned — see the red banner below.
          </div>
        )}
        {queue.length === 0 && <div style={{ ...dim, fontSize: 13 }}>nothing assigned to M.</div>}
        {queue.map((t, i) => {
          const step = stepLabel(t);
          const o = rankOf.get(t.id);
          const prev = i > 0 ? rankOf.get(queue[i - 1].id) : undefined;
          const divider = !o && (i === 0 || !!prev);
          return (
            <React.Fragment key={t.id}>
              {divider && (
                <div style={{
                  ...dim, fontSize: 10.5, letterSpacing: 0.6, marginTop: 8, paddingTop: 5,
                  borderTop: '1px dashed var(--border)',
                }}>
                  ↓ BELOW HERE M HAS NOT DECIDED — board priority order, no stated intention
                </div>
              )}
              <div style={{
                display: 'flex', gap: 9, alignItems: 'center', flexWrap: 'wrap',
                padding: '6px 0', borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ ...dim, ...mono, minWidth: 22, textAlign: 'right' }}>{i + 1}</span>
                {storeReady && (
                  <span style={{ display: 'flex', gap: 2 }}>
                    <button title="earlier in M's plan" disabled={i === 0}
                      style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 4px' }}
                      onClick={() => nudge(t.id, -1)}>▲</button>
                    <button title="later in M's plan" disabled={i === queue.length - 1}
                      style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 4px' }}
                      onClick={() => nudge(t.id, 1)}>▼</button>
                  </span>
                )}
                <span style={{ ...mono, ...dim, minWidth: 42 }}
                  title="board priority (phase.seq) — the fallback sort key, owned by dev_tasks">
                  {t.priority_phase}.{t.priority_seq}
                </span>
                <TaskLink id={t.id} title={t.title} />
                <span style={{ flex: 1 }} />
                {step && (
                  <span style={{ ...dim, fontSize: 11.5, maxWidth: 340, overflow: 'hidden',
                    textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={step}>{step}</span>
                )}
                <HandoffChain task={t} />
                <span style={{ ...tagChip, borderColor: STATUS_COLOR[t.status], color: STATUS_COLOR[t.status] }}>
                  {t.status}
                </span>
                <span style={{ fontSize: 11.5, color: ageColor(hoursSince(t.updated_at)) }}
                  title={`last board edit ${utcStamp(t.updated_at)}`}>
                  {ageShort(t.updated_at)}
                </span>
              </div>
              {storeReady && (
                <input
                  style={{ ...input, fontSize: 11.5, width: '100%', marginBottom: 3, opacity: 0.9 }}
                  placeholder={`why is #${t.id} here? (M's reason — stored, one line)`}
                  value={(plan?.find((p) => p.task_id === t.id)?.why ?? o?.why) || ''}
                  onChange={(ev) => setWhy(t.id, ev.target.value)} />
              )}
            </React.Fragment>
          );
        })}
      </Panel>

      {/* ── 2 · ACTIVITY LOG ───────────────────────────────────────────────── */}
      <Panel
        n="2"
        title={`activity log — ${win.shown} rendered of ${feedRows.length} shown, ${feed.length} kept`}
        sub={<>
          What happened that <b>M did not do</b>: system transitions, agent reports, Kevin&apos;s
          comments, and run outcomes. <b>Comments authored by the M SESSION are excluded</b> — they
          are not news to M ({mOwnCount} of the last {comments.length} board comments in this window
          were M&apos;s own). <b><code style={mono}>M·auto</code> is NOT the M session</b>: it is a
          dispatched agent, and its reports are precisely what M must react to, so they stay in.
          {' '}Every row carries the RULE that classified it — hover the chip for the reason.
          {' '}<b>This module scrolls inside itself and paints {FEED_PAGE_ROWS} rows at a time</b>
          {' '}(board #240 — ~435 rows at once stalled the browser). <b>Render less at a time, not
          keep less</b>: nothing is dropped, and scrolling to the bottom — or the button there —
          reaches every older event.
        </>}
        right={
          <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, whiteSpace: 'nowrap' }}
            onClick={() => setShowRoutine((v) => !v)}>
            {showRoutine ? '▾ hiding nothing' : `▸ show ${routineCount} routine`}
          </button>
        }>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginBottom: 9 }}>
          <span style={{ fontSize: 12, color: 'var(--amber, #d98c00)', fontWeight: 700 }}>
            ● {feed.length - routineCount} attention
          </span>
          <span style={{ ...dim, fontSize: 12 }}>· {routineCount} routine (rule-classified)</span>
          {storeReady && (
            <span style={{
              ...tagChip, fontWeight: 700,
              borderColor: unreviewed ? 'var(--amber, #d98c00)' : 'var(--green)',
              color: unreviewed ? 'var(--amber, #d98c00)' : 'var(--green)',
            }} title="attention events with no seen/actioned mark. This number going DOWN is the only evidence of coverage on this page — and an un-reviewed row keeps a live left edge so it cannot age quietly.">
              {unreviewed
                ? `${unreviewed} un-reviewed of ${attentionRows.length}`
                : `all ${attentionRows.length} reviewed`}
            </span>
          )}
          {unclassified > 0 && (
            <span style={{ ...tagChip, borderColor: 'var(--red)', color: 'var(--red)', fontWeight: 700 }}
              title="events no rule recognises. They are shown as ATTENTION on purpose — unrecognised is never routine — and each one is a rule waiting to be written.">
              {unclassified} unclassified
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }}
            onClick={() => setShowRules((v) => !v)}>
            {showRules ? '▾' : '▸'} the {FEED_RULES.length} classifier rules
          </button>
        </div>

        {/* The rule table, rendered from the module that DOES the classifying —
            not a description of it. Kevin's principle A, applied at the scale
            this step can honestly reach (section 3 renders the charter itself). */}
        {showRules && (
          <div style={{
            border: '1px solid var(--border)', borderRadius: 9, padding: '8px 11px',
            marginBottom: 10, background: 'var(--bg-input)',
          }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>
              First match wins. This list is generated from <code style={mono}>mSessionFeed.ts</code>&apos;s
              {' '}own <code style={mono}>FEED_RULES</code> — the array the classifier actually runs, so
              it cannot drift from what the page did. <b>Every future miss adds a rule here</b> instead
              of a resolution to remember harder; that is what makes this the M-session watch-list
              rather than a viewer.
            </div>
            {[...FEED_RULES, { ...UNCLASSIFIED, test: null }].map((r, i) => (
              <div key={r.id} style={{ display: 'flex', gap: 8, padding: '3px 0', fontSize: 12 }}>
                <span style={{ ...dim, ...mono, minWidth: 20, textAlign: 'right' }}>
                  {i < FEED_RULES.length ? i + 1 : '—'}
                </span>
                <ClassChip cls={r.cls} rule={r.id} why={r.why} />
                <span style={{ color: 'var(--text-secondary)', flex: 1 }}>{r.why}</span>
              </div>
            ))}
          </div>
        )}

        {feedDays.length === 0 && (
          <div style={{ ...dim, fontSize: 13 }}>
            {showRoutine ? 'no events in this window.' : 'nothing needing attention in this window.'}
          </div>
        )}
        {/* THE BOUND (board #240). maxHeight + overflowY is the whole "scroll
            bar inside of the module" ask: the page stops growing with the
            board, and sections 3/4 and the known-gaps footer stay one scroll
            away instead of ~435 rows away. The controls above sit OUTSIDE this
            container on purpose — filtering a list you have scrolled past is
            how a bounded module becomes annoying. */}
        <div ref={feedScroll} onScroll={onFeedScroll} style={{
          maxHeight: FEED_MAX_HEIGHT, overflowY: 'auto', overscrollBehavior: 'contain',
          paddingRight: 4,
        }}>
          {feedDays.map((d) => (
            <div key={d.day} style={{ marginBottom: 10 }}>
              <div style={{
                display: 'flex', gap: 8, alignItems: 'baseline', padding: '4px 0',
                borderBottom: '1px solid var(--border)', marginBottom: 2,
              }}>
                <span style={{ fontSize: 12, fontWeight: 700 }}>{d.day || 'unknown day'}</span>
                <span style={{ ...dim, fontSize: 11.5 }}>{d.rows.length} events</span>
              </div>
              {d.rows.map((e) => {
                const open = openBodies.has(e.key);
                const long = e.body.length > FEED_EXCERPT;
                const mk = markOf.get(e.key);
                return (
                  <div key={e.key} style={{
                    padding: '5px 6px', borderRadius: 6, marginBottom: 2,
                    background: mk ? 'transparent'
                      : e.cls === 'attention' ? 'rgba(217,140,0,0.05)' : 'transparent',
                    opacity: mk ? 0.6 : 1,
                    // An UN-reviewed attention row keeps a live left edge. Visible
                    // gaps beat false coverage: nothing here ages quietly into
                    // looking handled.
                    borderLeft: e.cls === 'attention' && !mk
                      ? '2px solid var(--amber, #d98c00)' : '2px solid transparent',
                  }}>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', fontSize: 12 }}>
                      <RoleChip role={e.actor.replace('·auto', '')} title={e.actor} />
                      <span style={{ ...dim, ...mono, minWidth: 68 }}>{utcStamp(e.at, false)}</span>
                      <TaskLink id={e.taskId} title={taskTitle(e.taskId)} bold={false} />
                      <ClassChip cls={e.cls} rule={e.rule} why={e.why} />
                      {e.source === 'run' && <OutcomeChip outcome={e.kind.replace('run-', '')} />}
                      <span style={{ ...dim, fontSize: 11 }}>{relTime(e.at)}</span>
                    </div>
                    <div style={{
                      fontSize: 12.5, marginLeft: 28, marginTop: 2, whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      color: e.cls === 'attention' ? 'var(--text-primary)' : 'var(--text-secondary)',
                    }}>
                      {open || !long ? e.body || e.headline
                        : `${(e.body || e.headline).slice(0, FEED_EXCERPT)}…`}
                    </div>
                    {(long || (e.bodyChars || 0) > e.body.length) && (
                      <div style={{ marginLeft: 28, marginTop: 2, display: 'flex', gap: 8, alignItems: 'center' }}>
                        {long && (
                          <button style={{ ...input, cursor: 'pointer', fontSize: 10.5, padding: '0 6px' }}
                            onClick={() => toggleBody(e.key)}>{open ? '▾ less' : '▸ more'}</button>
                        )}
                        {(e.bodyChars || 0) > e.body.length && (
                          <span style={{ ...dim, fontSize: 10.5 }}>
                            excerpt of {e.bodyChars} chars — open the task for the rest
                          </span>
                        )}
                      </div>
                    )}
                    {/* The coverage record: event → M saw it → M did X, or judged
                        nothing was needed. Confirming the rule is one click;
                        an EXCEPTION gets a real note. */}
                    {storeReady && (
                      <div style={{ marginLeft: 28, marginTop: 3, display: 'flex', gap: 6,
                        alignItems: 'center', flexWrap: 'wrap' }}>
                        {mk ? (
                          <>
                            <span style={{ ...tagChip, borderColor: 'var(--green)', color: 'var(--green)' }}
                              title={`${mk.state} by ${mk.marked_by || 'M'} at ${utcStamp(mk.marked_at)}`}>
                              ✓ {mk.state}
                            </span>
                            {mk.note && (
                              <span style={{ fontSize: 11.5, color: 'var(--text-secondary)' }}>
                                — {mk.note}
                              </span>
                            )}
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 5px' }}
                              title="un-mark. A wrong 'I looked at this' is false coverage."
                              onClick={() => unmark(e)}>undo</button>
                          </>
                        ) : (
                          <>
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 6px' }}
                              title="the rule's verdict stands and M has read it"
                              onClick={() => mark(e, 'seen')}>seen</button>
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 6px' }}
                              title="M looked and judged nothing was needed — say why"
                              onClick={() => setMarkNote({ key: e.key, text: '' })}>no action…</button>
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10, padding: '0 6px' }}
                              title="M did something — say what"
                              onClick={() => setMarkNote({ key: e.key, text: '' })}>actioned…</button>
                          </>
                        )}
                        {markNote?.key === e.key && (
                          <span style={{ display: 'flex', gap: 5, flex: 1, minWidth: 260 }}>
                            <input autoFocus style={{ ...input, fontSize: 11.5, flex: 1 }}
                              placeholder="what M did, or why nothing was needed"
                              value={markNote.text}
                              onChange={(ev) => setMarkNote({ key: e.key, text: ev.target.value })} />
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10.5 }}
                              onClick={() => { mark(e, 'actioned', markNote.text); setMarkNote(null); }}>
                              actioned
                            </button>
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10.5 }}
                              onClick={() => { mark(e, 'no-action', markNote.text); setMarkNote(null); }}>
                              no action
                            </button>
                            <button style={{ ...input, cursor: 'pointer', fontSize: 10.5 }}
                              onClick={() => setMarkNote(null)}>✕</button>
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
          {/* What is KEPT but not yet PAINTED — printed, never implied. An
              unpainted event that reads as an absent one is the #157 dead-alarm
              failure at page scale. */}
          {win.more && (
            <div style={{
              display: 'flex', gap: 9, alignItems: 'center', flexWrap: 'wrap',
              padding: '8px 2px 2px', borderTop: '1px dashed var(--border)',
            }}>
              <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }} onClick={showMore}>
                ▾ show {Math.min(FEED_PAGE_ROWS, win.hidden)} more
              </button>
              <span style={{ ...dim, fontSize: 11.5 }}>
                {win.hidden} older event{win.hidden === 1 ? '' : 's'} kept and classified but not yet
                rendered — keep scrolling and they paint {FEED_PAGE_ROWS} at a time. Nothing here is
                discarded.
              </span>
            </div>
          )}
          {!win.more && win.total > FEED_PAGE_ROWS && (
            <div style={{ ...dim, fontSize: 11.5, padding: '8px 2px 2px', borderTop: '1px dashed var(--border)' }}>
              end of the window — all {win.total} events rendered.
            </div>
          )}
        </div>
      </Panel>

      {/* ── 3 · THE RULES — rendered FROM SOURCE, never restated ───────────── */}
      <Panel
        n="3"
        title={`the rules that govern how M works${rules ? ` — ${rules.sources.length} sources` : ''}`}
        sub={<>
          <b>Rendered from the source files, verbatim.</b> Kevin&apos;s design principle A: <i>&quot;a
          hand-maintained copy of the rules on a page is a second source of truth that will silently
          diverge&quot;</i> — the exact failure mode of the untracked-copy problem that bit us three
          times on 07-30. Nothing below is retyped: each block is a byte-for-byte slice of a file on
          disk, cut at a heading anchor. <b>Edit the file and this page changes.</b> If an anchor
          moves, this section says so <b>in red</b> — it never renders quietly empty, because
          &quot;no rules&quot; and &quot;the rules moved&quot; look identical and debug differently.
        </>}
        right={rules && rules.missing > 0 ? (
          <span style={{ ...tagChip, borderColor: 'var(--red)', color: 'var(--red)', fontWeight: 700 }}>
            {rules.missing} SOURCE{rules.missing === 1 ? '' : 'S'} MISSING
          </span>
        ) : null}>
        {rulesErr && (
          <div style={{ color: 'var(--red)', fontSize: 13, fontWeight: 700 }}>
            ⚠ the rules endpoint did not load — {rulesErr}
          </div>
        )}
        {!rules && !rulesErr && <div style={{ ...dim, fontSize: 13 }}>reading the source files…</div>}
        {rules?.sources.map((s) => (
          <div key={s.id} style={{
            border: `1px solid ${s.found ? 'var(--border)' : 'var(--red)'}`,
            background: s.found ? 'transparent' : 'rgba(200,60,60,0.08)',
            borderRadius: 9, padding: '8px 11px', marginBottom: 7,
          }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', flexWrap: 'wrap' }}>
              <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }}
                disabled={!s.found}
                onClick={() => setOpenRule(openRule === s.id ? null : s.id)}>
                {openRule === s.id ? '▾' : '▸'} {s.label}
              </button>
              <code style={{ ...mono, ...dim }}>{s.path} · {s.anchor}</code>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>{s.why}</div>
            {!s.found && (
              <div style={{ marginTop: 5 }}>
                <div style={{ color: 'var(--red)', fontWeight: 700, fontSize: 12.5 }}>
                  ⚠ NOT RENDERED — {s.error}
                </div>
                <div style={{ ...mono, ...dim, marginTop: 2 }}>tried: {s.resolved_path}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                  Either the heading moved (fix the anchor in{' '}
                  <code style={mono}>api/routers/m_session.py</code>&apos;s{' '}
                  <code style={mono}>RULE_SOURCES</code>) or this deploy&apos;s working copy does not
                  carry that file (set <code style={mono}>RORT_RULES_ROOT</code>/
                  <code style={mono}>RORT_REPO_ROOT</code>).
                </div>
              </div>
            )}
            {s.found && openRule === s.id && (
              <div style={{ marginTop: 7, borderTop: '1px solid var(--border)', paddingTop: 7 }}>
                <style>{MD_CSS}</style>
                <div style={{ fontSize: 13 }}><Md text={s.text} /></div>
              </div>
            )}
          </div>
        ))}
        {rules && (
          <div style={{ ...dim, fontSize: 11, marginTop: 4 }}>
            read {relTime(rules.generated_at)} from <code style={mono}>{rules.ror_root}</code> and{' '}
            <code style={mono}>{rules.repo_root}</code>
          </div>
        )}
      </Panel>

      {/* ── 4 · M's CANVAS ─────────────────────────────────────────────────── */}
      <Panel
        n="4"
        title="m’s canvas"
        sub={<>
          What M is <b>doing</b>, <b>worried about</b>, and <b>waiting on</b> — the cross-task
          picture. Kevin: <i>&quot;a good canvas for you to publish the details and I&apos;ll take a
          look at it.&quot;</i> This is the fix for the read-it-twice problem, and his ruling scopes
          it precisely: <b>task-specific detail stays on the task thread</b> (its audit trail), the
          canvas carries what spans tasks, and chat collapses to TLDRs and pointers —{' '}
          <i>&quot;left a comment on #217&quot;</i> — because the detail is one click away. Markdown;
          <code style={mono}> #123 </code>links are just task links, so write pointers, not copies.
        </>}
        right={store?.canvas?.updated_at ? (
          <span style={{ ...dim, fontSize: 11.5 }}>
            updated {relTime(store.canvas.updated_at)} by {store.canvas.updated_by || 'M'}
          </span>
        ) : null}>
        {!storeReady ? (
          <div style={{ ...dim, fontSize: 13 }}>
            the canvas store is not provisioned — see the red banner below.
          </div>
        ) : canvasDraft === null ? (
          <>
            <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, marginBottom: 8 }}
              onClick={() => setCanvasDraft(store?.canvas?.body || '')}>✎ edit</button>
            <style>{MD_CSS}</style>
            <div style={{ fontSize: 13.5 }}>
              <Md text={store?.canvas?.body || ''}
                fallback="The canvas is empty. M writes here; Kevin reads here." />
            </div>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', gap: 6, marginBottom: 7 }}>
              <button style={{ ...input, cursor: 'pointer', fontSize: 11.5, borderColor: 'var(--green)', color: 'var(--green)' }}
                disabled={!!saving} onClick={saveCanvas}>
                {saving === 'save canvas' ? 'saving…' : '✓ save'}
              </button>
              <button style={{ ...input, cursor: 'pointer', fontSize: 11.5 }}
                onClick={() => setCanvasDraft(null)}>cancel</button>
              <span style={{ ...dim, fontSize: 11.5, alignSelf: 'center' }}>
                {canvasDraft.length} chars · markdown · a 30s refresh will not overwrite this draft
              </span>
            </div>
            <textarea value={canvasDraft} onChange={(ev) => setCanvasDraft(ev.target.value)}
              rows={18} spellCheck={false}
              placeholder={'## Doing\n\n## Worried about\n\n## Waiting on'}
              style={{ ...input, width: '100%', ...mono, fontSize: 12.5, lineHeight: 1.5,
                resize: 'vertical' }} />
          </>
        )}
      </Panel>

      {/* ── The store's own state — loud when the migration is unapplied ───── */}
      {(!storeReady || storeErr || saveErr) && (
        <div style={{
          border: '1px solid var(--red)', borderRadius: 10, padding: '10px 14px', marginBottom: 14,
          background: 'rgba(200,60,60,0.08)',
        }}>
          <div style={{ color: 'var(--red)', fontWeight: 700, fontSize: 13 }}>
            ⚠ M&apos;s store is not available — sections 1 (ordering), 2 (seen/actioned) and 4
            (canvas) are READ-ONLY and EMPTY BECAUSE OF THIS, not because M has written nothing.
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 4 }}>
            {store?.reason || storeErr || saveErr}
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 4 }}>
            <code style={mono}>src/migrations/m_session_state.sql</code> is authored and
            {' '}<b>deliberately unapplied</b>: never DDL against prod. Authorization path — author
            the file → hand to M → <b>Kevin authorizes by name</b> → M or Kevin applies → only then
            may the code merge.
          </div>
        </div>
      )}
      {saveErr && storeReady && (
        <div style={{ color: 'var(--red)', fontSize: 12.5, marginBottom: 10 }}>⚠ {saveErr}</div>
      )}

      {/* ── Known gaps — printed, not hidden (the /admin/dispatch posture) ─── */}
      <div style={{ marginTop: 14 }}>
        <Card>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, color: 'var(--text-tertiary)', marginBottom: 6 }}>
            KNOWN GAPS — what this page CANNOT see
          </div>
          <ol style={{ fontSize: 12.5, color: 'var(--text-secondary)', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <li>
              <b>The window is the last {COMMENT_LIMIT} board comments</b>
              {oldestComment && <> (back to <code style={mono}>{utcStamp(oldestComment)}</code>)</>}.
              An Ask AI older than that is invisible here — a silence in this window is not proof of
              an empty inbox.
            </li>
            <li>
              <b>Roughly half of what M actually watches is not in the database at all</b>:
              stranded/unpushed branches, the dispatcher loop&apos;s version drifting from dev,
              preflight reds, and untracked-vs-tracked collisions that make{' '}
              <code style={mono}>git merge --ff-only</code> refuse silently. Every one of those bit
              us in the 24h before this task was filed. Phase 2 posts them into this same feed as{' '}
              <code style={mono}>source:&nbsp;env</code> events; the event shape already allows a
              null task id so that writer needs no rework here.
            </li>
            <li>
              <b>Coverage is only as good as M&apos;s honesty about what it actually looked at.</b>
              {' '}The seen/actioned marks record that M looked; they cannot prove M read. The
              un-reviewed counter and the live left edge on unmarked attention rows exist so a gap
              stays visible rather than aging quietly — <b>visible gaps beat false coverage</b>.
            </li>
            <li>
              <b>Section 3 renders only the anchors it is told about.</b> A rule that lives in a
              file nobody listed is not on this page. Adding one is a row in{' '}
              <code style={mono}>RULE_SOURCES</code>, not a paragraph typed here — and a missing
              anchor is shown in red rather than skipped.
            </li>
            <li>
              <b>The activity log renders {FEED_PAGE_ROWS} rows at a time inside a{' '}
              {FEED_MAX_HEIGHT}px scroller</b> (board #240). That is a RENDER bound, not a retention
              one — the full window is still fetched, classified and counted, and the footer prints
              how many are kept but unpainted. It does mean <b>ctrl-F finds only what is painted</b>:
              scroll to the end (or click through) before concluding an event is not there.
            </li>
            <li>
              <b>Self-review caveat, stated not buried:</b> once M both classifies and confirms, that
              is self-review. It is meaningful only because the classifier is a mechanical RULE and
              M&apos;s job is catching what the rule got wrong. Visible gaps beat false coverage.
            </li>
          </ol>
        </Card>
      </div>
    </div>
  );
}
