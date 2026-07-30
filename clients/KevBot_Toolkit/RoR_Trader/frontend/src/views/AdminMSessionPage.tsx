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
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import {
  ChecklistStep, HandoffChain, OutcomeChip, RoleChip, RunRow, STATUS_COLOR, Task,
  ageColor, ageShort, hoursSince, input, isProcessChain, relTime,
  stepOwner, stepTitle, tagChip, utcDay, utcStamp,
} from './taskBoardShared';
import {
  FEED_RULES, FeedClass, FeedComment, FeedEvent, InboxItem, M_SESSION_AUTHOR,
  UNCLASSIFIED, askAiInbox, buildFeed, mQueue, waitingOnKevin,
} from './mSessionFeed';

/** Auto-refresh cadence. Slower than /admin/dispatch's 15s on purpose: this
 *  page pulls comment BODIES, which is the expensive payload on the board. */
const REFRESH_MS = 30000;
/** Board-wide comment window. 400 was the measured sample behind this task's
 *  design (131 system transitions, 89 agent reports, 42 R, 6 Kevin, 130 M). */
const COMMENT_LIMIT = 400;
/** Per-comment excerpt on the wire; the thread has the rest, one click away. */
const COMMENT_CHARS = 1400;
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
  useEffect(() => {
    const id = setInterval(() => { if (!document.hidden) load(); }, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);
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
  const queue = useMemo(() => mQueue(tasks), [tasks]);
  const feed = useMemo(() => buildFeed(comments, runs), [comments, runs]);

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

  const feedDays = useMemo(() => {
    const days: { day: string; rows: FeedEvent[] }[] = [];
    feedRows.forEach((e) => {
      const d = utcDay(e.at);
      const last = days[days.length - 1];
      if (last && last.day === d) last.rows.push(e);
      else days.push({ day: d, rows: [e] });
    });
    return days;
  }, [feedRows]);

  const toggleBody = (key: string) => setOpenBodies((prev) => {
    const next = new Set(prev);
    if (next.has(key)) next.delete(key); else next.add(key);
    return next;
  });

  const taskTitle = (id: number | null) =>
    (id != null && taskById.get(id)?.title) || '';

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
        <code style={{ ...mono, margin: '0 4px' }}>run_history</code>; this page stores nothing.
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
          false coverage this dashboard exists to prevent.
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
          excluding {'Done'} and {'Blocked'} — assignee = whoever the task waits on, board #171).
          {' '}<b>This is BOARD ORDER, not the order M intends.</b> Kevin asked for the order M plans
          to take these in — a plan that today lives only in M&apos;s head and a session-local todo
          list he cannot see. That store is <b>step 4</b> of this chain and needs a migration; the
          sort is already a swappable comparator so step 4 changes one argument, not this page.
        </>}>
        {queue.length === 0 && <div style={{ ...dim, fontSize: 13 }}>nothing assigned to M.</div>}
        {queue.map((t, i) => {
          const step = stepLabel(t);
          return (
            <div key={t.id} style={{
              display: 'flex', gap: 9, alignItems: 'center', flexWrap: 'wrap',
              padding: '6px 0', borderBottom: '1px solid var(--border)',
            }}>
              <span style={{ ...dim, ...mono, minWidth: 22, textAlign: 'right' }}>{i + 1}</span>
              <span style={{ ...mono, ...dim, minWidth: 42 }}
                title="board priority (phase.seq) — the Phase-1 sort key">
                {t.priority_phase}.{t.priority_seq}
              </span>
              <TaskLink id={t.id} title={t.title} />
              <span style={{ flex: 1 }} />
              {step && (
                <span style={{ ...dim, fontSize: 11.5, maxWidth: 420, overflow: 'hidden',
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
          );
        })}
      </Panel>

      {/* ── 2 · ACTIVITY LOG ───────────────────────────────────────────────── */}
      <Panel
        n="2"
        title={`activity log — ${feedRows.length} shown of ${feed.length}`}
        sub={<>
          What happened that <b>M did not do</b>: system transitions, agent reports, Kevin&apos;s
          comments, and run outcomes. <b>Comments authored by the M SESSION are excluded</b> — they
          are not news to M ({mOwnCount} of the last {comments.length} board comments in this window
          were M&apos;s own). <b><code style={mono}>M·auto</code> is NOT the M session</b>: it is a
          dispatched agent, and its reports are precisely what M must react to, so they stay in.
          {' '}Every row carries the RULE that classified it — hover the chip for the reason.
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
              return (
                <div key={e.key} style={{
                  padding: '5px 6px', borderRadius: 6, marginBottom: 2,
                  background: e.cls === 'attention' ? 'rgba(217,140,0,0.05)' : 'transparent',
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
                </div>
              );
            })}
          </div>
        ))}
      </Panel>

      {/* ── Not built yet — said out loud, not implied by absence ──────────── */}
      <Card>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.8, color: 'var(--text-tertiary)', marginBottom: 6 }}>
          SECTIONS 3 &amp; 4 — NOT BUILT YET (step 4 of this chain)
        </div>
        <ul style={{ fontSize: 12.5, color: 'var(--text-secondary)', paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4 }}>
          <li>
            <b>3 · The rules that govern how M works</b> — charter §4 (assignee), §7 (chains +
            retrofit), §8 (cadence), the required rails, the sizing law, and the Ask-AI-first
            priority order. It must <b>render those files&apos; actual text</b>, never a retyped
            summary: a hand-maintained copy of the rules is a second source of truth that diverges
            silently. The classifier table above is the part of that principle this step could
            honestly reach — it is generated from the code that runs.
          </li>
          <li>
            <b>4 · M&apos;s canvas</b> — what M is doing, worried about, and waiting on. Kevin&apos;s
            fix for the read-it-twice problem: task-specific detail stays on the task thread, the
            canvas carries the cross-task picture, and chat collapses to TLDRs and pointers.
          </li>
          <li>
            Both need the one genuinely new store — M&apos;s intended ordering, M&apos;s
            seen/actioned marks, and the canvas text — which is a schema change, and therefore
            gated: author the <code style={mono}>.sql</code>, hand it to M, <b>Kevin authorizes by
            name</b>, apply, and only then may the code merge.
          </li>
        </ul>
      </Card>

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
              <b>There is no seen/actioned state yet.</b> The classifier sorts routine from
              attention, but nothing records that M looked. Until step 4 lands, this page shows what
              happened — it does not yet prove coverage, and it must never be read as if it did.
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
