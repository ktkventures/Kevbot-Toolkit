/**
 * Roadmap — /admin/roadmap (Spec_Admin_Roadmap.md, board #139).
 *
 * The WHY leg of the hub (Tasks = what · Roadmap = why · Agents = who):
 * Kevin's read-only big picture. Awaiting-Kevin strip on top, vision cards
 * with progress rollups grouped by priority phase, ungrouped tasks footer.
 * This page is a PROJECTION of the board — GET requests only, no mutation
 * code paths; every interaction deep-links to /admin/tasks?task=<id>.
 *
 * Strip membership note: the spec's needs-review-tag arm predates board #136
 * retiring that tag (Review is a STATUS now, kevin_final gates who closes).
 * The shared nextActor() derivation is the current form of the same rule —
 * it puts Approval items, two-touch (kevin_final) Reviews, and kevin-owned
 * checklist steps/assignees in the strip.
 */
'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Card from '@/components/Card';
import { apiFetch } from '@/lib/api/client';
import {
  Task, STATUS_COLOR, roleColor, tagChip, badge, input,
  RoleChip, NextChip, ProgressBar, nextActor, groupBoard, isFinished,
} from './taskBoardShared';

// Cosmetic phase labels — M-maintained, one line to extend, no schema (§4.3).
const PHASE_LABELS: Record<number, string> = { 0: 'infra' };
const phaseTitle = (n: number) =>
  `PHASE ${n}${PHASE_LABELS[n] ? ` — ${PHASE_LABELS[n].toUpperCase()}` : ''}`;

const KEVIN_GOLD = roleColor('kevin');
const sectionHeader: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, letterSpacing: 0.8, textTransform: 'uppercase',
  color: 'var(--text-tertiary)', margin: '0 0 8px 2px',
};
const oneLiner: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0',
  fontSize: 12.5, color: 'var(--text-secondary)', textDecoration: 'none',
};
const truncate: React.CSSProperties = {
  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
};

const StatusChip = ({ status }: { status: string }) => (
  <span style={{
    ...tagChip, marginLeft: 0, fontWeight: 600,
    borderColor: STATUS_COLOR[status] || 'var(--border)',
    color: STATUS_COLOR[status] || 'var(--text-secondary)',
  }}>{status}</span>
);

const LiveDot = ({ title = 'touches live engine/trading code' }: { title?: string }) => (
  <span title={title} style={{ fontSize: 10 }}>🔴</span>
);

/** Why this row is in Kevin's strip — his stamp, his sign-off, or his step. */
const ReasonChip = ({ t }: { t: Task }) => {
  if (t.status === 'Approval') return (
    <span style={{ ...tagChip, marginLeft: 0, borderColor: KEVIN_GOLD, color: KEVIN_GOLD, fontWeight: 700 }}
      title="in Approval — awaiting Kevin's stamp">✋ stamp</span>);
  if (t.status === 'Review') return (
    <span style={{ ...tagChip, marginLeft: 0, borderColor: KEVIN_GOLD, color: KEVIN_GOLD, fontWeight: 700 }}
      title="in Review, two-touch — only Kevin signs it off">👀 review</span>);
  return (
    <span style={{ ...tagChip, marginLeft: 0, borderColor: KEVIN_GOLD, color: KEVIN_GOLD, fontWeight: 700 }}
      title="first open step / assignee is kevin">→ K</span>);
};

export default function AdminRoadmapPage() {
  const router = useRouter();
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [showDone, setShowDone] = useState(false);

  // The page's ONE fetch — include_done defaults true on the API; Done
  // subtasks are needed for the rollups (§3).
  const load = useCallback(async () => {
    try {
      setErr(null);
      const data = await apiFetch<Task[]>('/api/dev-tasks');
      setTasks(data || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

  // Same grouping implementation as /admin/tasks — never disagrees (§2).
  const { byParent, visionItems, looseTasks } = useMemo(() => groupBoard(tasks), [tasks]);

  // Strip membership (§4.2): open LEAF tasks only (visions WITH subtasks are
  // excluded on purpose — their next-actor derives from their first open
  // subtask, so listing both would double-count). API priority order kept.
  const awaitingKevin = useMemo(() =>
    tasks.filter((t) => !isFinished(t.status) && !byParent.has(t.id) &&
      nextActor(t, []).next === 'kevin'),
  [tasks, byParent]);

  // Phase buckets ascend from 0 (§4.3); within a phase the API order
  // (priority_seq) is preserved, matching the grouped order on /admin/tasks.
  const phases = useMemo(() => {
    const m = new Map<number, Task[]>();
    visionItems.forEach((v) => m.set(v.priority_phase, [...(m.get(v.priority_phase) || []), v]));
    return Array.from(m.entries()).sort((a, b) => a[0] - b[0]);
  }, [visionItems]);

  const titleById = useMemo(() => {
    const m = new Map<number, string>();
    tasks.forEach((t) => m.set(t.id, t.title));
    return m;
  }, [tasks]);

  // Rollup (§3): subtasks done/total; a subtask-less vision falls back to
  // its checklist — same rule as the modal header. Neither → no bar
  // (ProgressBar returns null at total 0).
  const rollup = (v: Task): { done: number; total: number } => {
    const kids = byParent.get(v.id) || [];
    // FINISHED, not literally Done (board #232) — a Closed subtask counts
    // toward the roll-up, or every vision loses progress as Kevin closes work.
    if (kids.length > 0) return { done: kids.filter((k) => isFinished(k.status)).length, total: kids.length };
    const cl = v.checklist || [];
    return { done: cl.filter((s) => s.done).length, total: cl.length };
  };

  const openVisions = visionItems.filter((v) => !isFinished(v.status));
  const phaseCount = new Set(openVisions.map((v) => v.priority_phase)).size;
  const shownLoose = looseTasks.filter((t) => showDone || !isFinished(t.status));

  const SubtaskLine = ({ k }: { k: Task }) => (
    <Link href={`/admin/tasks?task=${k.id}`} onClick={(e) => e.stopPropagation()}
      title={`${k.title} — ${k.status}${k.assignee ? ` · ${k.assignee}` : ''}`} style={oneLiner}>
      <span style={{ color: STATUS_COLOR[k.status] || 'var(--text-tertiary)', fontSize: 9 }}
        title={k.status}>●</span>
      <span style={{ color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>#{k.id}</span>
      <span style={{ ...truncate, flex: 1, minWidth: 0 }}>{k.title}</span>
      {k.origin === 'discovered' &&
        <span title="discovered mid-work (rabbit-hole fix)" style={{ fontSize: 11 }}>🔍</span>}
      {k.impacts_live && <LiveDot />}
      {k.assignee && <RoleChip role={k.assignee} />}
    </Link>
  );

  const VisionCard = ({ v }: { v: Task }) => {
    const kids = byParent.get(v.id) || [];
    const openKids = kids.filter((k) => !isFinished(k.status));
    const r = rollup(v);
    const crew = Array.from(new Set(
      openKids.map((k) => k.assignee).filter((a): a is string => !!a && a !== v.assignee)));
    const liveKids = openKids.filter((k) => k.impacts_live).length;
    return (
      <div onClick={() => router.push(`/admin/tasks?task=${v.id}`)}
        title={`open #${v.id} on Tasks`}
        style={{
          border: '1px solid var(--border)', borderRadius: 10, padding: 14,
          background: 'var(--bg-card, var(--bg-input))', cursor: 'pointer',
          display: 'flex', flexDirection: 'column', gap: 8,
          opacity: isFinished(v.status) ? 0.55 : 1,
        }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'var(--text-tertiary)', fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>#{v.id}</span>
          <span style={{ ...truncate, fontWeight: 600, fontSize: 14, flex: 1, minWidth: 0 }}>{v.title}</span>
          {v.is_urgent && <span title="urgent">⚡</span>}
          <StatusChip status={v.status} />
        </div>
        <div><ProgressBar done={r.done} total={r.total} width={140} /></div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, flexWrap: 'wrap' }}>
          {v.assignee && <RoleChip role={v.assignee} title={`owner: ${v.assignee}`} />}
          {crew.map((a) => <RoleChip key={a} role={a} title={`crew: ${a} (open subtask assignee)`} />)}
          <NextChip t={v} subtasks={kids} />
          {v.impacts_live && <span style={badge('var(--red)')} title="this vision touches live engine/trading code">🔴 live</span>}
          {liveKids > 0 && !v.impacts_live &&
            <span style={{ ...tagChip, marginLeft: 0, borderColor: 'var(--red)', color: 'var(--red)' }}
              title="open subtasks flagged impacts_live">🔴 {liveKids} subtask{liveKids > 1 ? 's' : ''} touch{liveKids > 1 ? '' : 'es'} live</span>}
        </div>
        {openKids.length > 0 && (
          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 6, marginTop: 'auto' }}>
            {openKids.slice(0, 3).map((k) => <SubtaskLine key={k.id} k={k} />)}
            {openKids.length > 3 && (
              <Link href="/admin/tasks" onClick={(e) => e.stopPropagation()}
                style={{ ...oneLiner, color: 'var(--text-tertiary)', fontSize: 11.5 }}>
                +{openKids.length - 3} more →</Link>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ padding: 20, maxWidth: 1500, margin: '0 auto' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Roadmap</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 16 }}>
        Read-only big picture — click anything to open it on Tasks; edits happen there.
        {' '}{phaseCount} phase{phaseCount === 1 ? '' : 's'} · {openVisions.length} vision{openVisions.length === 1 ? '' : 's'} open
        · {awaitingKevin.length} awaiting Kevin
      </p>

      {err && <div style={{ marginBottom: 12 }}><Card><div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div></Card></div>}

      <div style={{ marginBottom: 12 }}>
        <Card>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, alignItems: 'center' }}>
            <button style={{ ...input, cursor: 'pointer' }} onClick={load}>↻ refresh</button>
            <label title="reveal Done visions dimmed in place">
              <input type="checkbox" checked={showDone} onChange={(e) => setShowDone(e.target.checked)} /> show completed</label>
          </div>
        </Card>
      </div>

      {loading ? <Card>Loading…</Card> : (
        <>
          {/* Awaiting-Kevin strip (§4.2) — his inbox, pinned above all phases */}
          <div style={{
            border: `1px solid ${KEVIN_GOLD}`, borderLeft: `4px solid ${KEVIN_GOLD}`,
            borderRadius: 10, padding: '10px 14px', marginBottom: 20,
            background: 'var(--bg-card, var(--bg-input))',
          }}>
            <div style={{ ...sectionHeader, color: KEVIN_GOLD }}>✋ awaiting kevin</div>
            {awaitingKevin.length === 0
              ? <div style={{ fontSize: 12.5, color: 'var(--text-tertiary)' }}>Nothing waiting on Kevin</div>
              : awaitingKevin.map((t) => (
                <Link key={t.id} href={`/admin/tasks?task=${t.id}`} style={oneLiner}>
                  <ReasonChip t={t} />
                  <span style={{ color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>#{t.id}</span>
                  <span style={{ ...truncate, color: 'var(--text-primary)', fontWeight: 500 }}>{t.title}</span>
                  {t.parent_id != null && (
                    <span style={{ ...truncate, color: 'var(--text-tertiary)', fontSize: 11.5, maxWidth: 260 }}>
                      under #{t.parent_id} {titleById.get(t.parent_id) || ''}</span>
                  )}
                  <StatusChip status={t.status} />
                  {t.impacts_live && <LiveDot />}
                </Link>
              ))}
          </div>

          {/* Phase sections (§4.3) — ascending, 0 first; M-managed order */}
          {phases.map(([n, visions]) => {
            const shown = visions.filter((v) => showDone || !isFinished(v.status));
            if (shown.length === 0) return null;
            return (
              <div key={n} style={{ marginBottom: 20 }}>
                <div style={sectionHeader}>{phaseTitle(n)}</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
                  {shown.map((v) => <VisionCard key={v.id} v={v} />)}
                </div>
              </div>
            );
          })}

          {/* Ungrouped footer (§4.5) — nothing on the board is invisible here */}
          {shownLoose.length > 0 && (
            <Card>
              <div style={sectionHeader}>ungrouped — tasks without a vision parent</div>
              {shownLoose.map((t) => (
                <Link key={t.id} href={`/admin/tasks?task=${t.id}`}
                  style={{ ...oneLiner, opacity: isFinished(t.status) ? 0.55 : 1 }}>
                  <span style={{ color: 'var(--text-tertiary)', fontVariantNumeric: 'tabular-nums' }}>#{t.id}</span>
                  <span style={{ ...truncate, flex: 1, minWidth: 0, color: 'var(--text-primary)' }}>{t.title}</span>
                  {t.impacts_live && <LiveDot />}
                  {t.assignee && <RoleChip role={t.assignee} />}
                  <StatusChip status={t.status} />
                </Link>
              ))}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
