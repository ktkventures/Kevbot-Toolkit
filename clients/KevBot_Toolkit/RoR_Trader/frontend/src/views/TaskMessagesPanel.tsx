/**
 * Messages tab (board #143, V2.12) — the unread @-mention inbox that closes
 * the last gap where a comment one teammate leaves the other never circles
 * back to. Shows mentions addressed to the current viewer role (`forRole` =
 * whoever you're commenting as); clicking a row opens that task modal AND
 * marks the mention seen. "Mark all read" clears the inbox; a toggle reveals
 * recently-seen history (last 7d).
 *
 * The symmetric agent side (SessionStart hook + dispatcher inject unseen
 * mentions) is M's half — it consumes the SAME GET/seen-all endpoints.
 */
'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '@/lib/api/client';
import {
  Mention, RoleChip, STATUS_COLOR, input, tagChip, relTime,
} from './taskBoardShared';

export default function TaskMessagesPanel({
  forRole, onOpenTask, onSeenChange, refreshSignal = 0,
}: {
  forRole: string;
  onOpenTask: (id: number) => void;
  onSeenChange: () => void;
  /** bump to force a reload (e.g. board poll tick / view switch). */
  refreshSignal?: number;
}) {
  const [rows, setRows] = useState<Mention[]>([]);
  const [showSeen, setShowSeen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!forRole) { setRows([]); setLoading(false); return; }
    try {
      setErr(null);
      const data = await apiFetch<Mention[]>(
        `/api/mentions?for=${encodeURIComponent(forRole)}&unseen=${!showSeen}`);
      setRows(data || []);
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, [forRole, showSeen]);
  useEffect(() => { load(); }, [load, refreshSignal]);

  const markSeen = async (id: number) => {
    try { await apiFetch(`/api/mentions/${id}/seen`, { method: 'POST' }); }
    catch { /* the row just stays until the next load */ }
    onSeenChange();
  };
  const openRow = async (m: Mention) => {
    // Optimistic: drop it from the unseen list immediately, then persist +
    // open the task (which marks it seen server-side anyway).
    if (!m.seen_at) setRows((rs) => rs.filter((r) => r.id !== m.id));
    await markSeen(m.id);
    onOpenTask(m.task_id);
  };
  const markAll = async () => {
    try { await apiFetch(`/api/mentions/seen-all?for=${encodeURIComponent(forRole)}`, { method: 'POST' }); }
    catch (e) { setErr(String(e)); }
    onSeenChange();
    load();
  };

  const unseenCount = rows.filter((r) => !r.seen_at).length;

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>
          Messages for <RoleChip role={forRole} title={`mentions addressed to ${forRole}`} />{' '}
          <span style={{ color: 'var(--text-secondary)' }}>
            — {unseenCount} unread{showSeen ? ' · showing recently-seen too' : ''}
          </span>
        </span>
        <span style={{ flex: 1 }} />
        <label style={{ fontSize: 12 }} title="also show mentions you've already read from the last 7 days">
          <input type="checkbox" checked={showSeen} onChange={(e) => setShowSeen(e.target.checked)} /> show already-seen (7d)
        </label>
        <button style={{ ...input, cursor: unseenCount ? 'pointer' : 'default', opacity: unseenCount ? 1 : 0.4 }}
          disabled={!unseenCount}
          title="mark every unread mention as read" onClick={markAll}>✓ mark all read</button>
        <button style={{ ...input, cursor: 'pointer' }} onClick={load}>↻ refresh</button>
      </div>

      {err && <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 8 }}>⚠ {err}</div>}

      {loading ? (
        <div style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>Loading…</div>
      ) : rows.length === 0 ? (
        <div style={{ color: 'var(--text-tertiary)', fontSize: 13, padding: '18px 4px' }}>
          {showSeen
            ? `No mentions for ${forRole} in the last 7 days.`
            : `📭 Nothing unread — you're all caught up, ${forRole}.`}
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {rows.map((m) => {
            const seen = !!m.seen_at;
            return (
              <div key={m.id} onClick={() => openRow(m)} title="open this task and mark the mention read"
                style={{
                  display: 'flex', gap: 10, alignItems: 'flex-start', cursor: 'pointer',
                  padding: '9px 11px', borderRadius: 9, opacity: seen ? 0.62 : 1,
                  border: `1px solid ${seen ? 'var(--border)' : 'var(--blue)'}`,
                  background: seen ? 'transparent' : 'var(--bg-input)',
                }}>
                {!seen && <span title="unread" style={{
                  width: 8, height: 8, borderRadius: '50%', background: 'var(--blue)',
                  flex: 'none', marginTop: 5,
                }} />}
                <RoleChip role={m.author} title={`${m.author} tagged you`} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                    <b style={{ color: 'var(--text-secondary)' }}>{m.author}</b>
                    <span style={{ color: 'var(--text-tertiary)' }}>tagged @{m.mentioned} on</span>
                    <span style={{ fontWeight: 600 }}>#{m.task_id} {m.task_title || '(task)'}</span>
                    {m.task_status && (
                      <span style={{ ...tagChip, color: STATUS_COLOR[m.task_status] || 'var(--text-secondary)' }}>
                        {m.task_status}</span>
                    )}
                    <span style={{ color: 'var(--text-tertiary)' }}>· {relTime(m.created_at)}</span>
                  </div>
                  <div style={{
                    fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 3,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>{(m.excerpt || '').replace(/\s+/g, ' ').trim() || '(no text)'}</div>
                </div>
                {!seen && (
                  <button style={{ ...tagChip, cursor: 'pointer', marginLeft: 4 }}
                    title="mark read without opening"
                    onClick={(e) => { e.stopPropagation(); setRows((rs) => rs.filter((r) => r.id !== m.id)); markSeen(m.id); }}>
                    mark read</button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
