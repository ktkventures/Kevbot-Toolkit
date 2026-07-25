/**
 * Strategy notes canvas (board #70 Phase B) — per-sid explanations on the
 * Strategy Health page ("why is this sid red"), written by humans and the
 * nightly/AI writers. List newest first + add with the role-avatar author
 * picker; backed by /api/strategy-notes (mirrors dev_task_comments).
 */
'use client';

import React, { useCallback, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { apiFetch } from '@/lib/api/client';
import {
  ASSIGNEES, AUTHOR_LS_KEY, RoleChip, RolePicker, withLegacy, relTime,
} from './taskBoardShared';

interface Note { id: number; sid: number; author: string; body: string; created_at: string; }

const panelHead: React.CSSProperties = {
  fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: 'var(--text-tertiary)',
  padding: '8px 12px', borderBottom: '1px solid var(--border)', textTransform: 'uppercase',
  display: 'flex', alignItems: 'center', gap: 8,
};

export default function StrategyNotesModal({ sid, name, onClose, onChanged }: {
  sid: number; name: string; onClose: () => void; onChanged?: () => void;
}) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [draft, setDraft] = useState('');
  const [author, setAuthor] = useState('kevin');
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem(AUTHOR_LS_KEY);
    if (saved) setAuthor(saved);
  }, []);

  const load = useCallback(() => {
    apiFetch<Note[]>(`/api/strategy-notes/${sid}`).then(setNotes)
      .catch((e) => setErr(String(e)));
  }, [sid]);
  useEffect(() => { load(); }, [load]);

  const add = async () => {
    if (!draft.trim()) return;
    try {
      await apiFetch(`/api/strategy-notes/${sid}`, {
        method: 'POST', body: JSON.stringify({ body: draft, author }),
      });
      setDraft(''); load(); onChanged?.();
    } catch (e) { setErr(String(e)); }
  };
  const pickAuthor = (a: string) => {
    setAuthor(a); localStorage.setItem(AUTHOR_LS_KEY, a);
  };

  return createPortal(
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '6vh 16px',
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: 'var(--bg-card, var(--bg-input))', border: '1px solid var(--border)',
        borderRadius: 12, width: '90vw', maxWidth: 640, maxHeight: '78vh',
        display: 'flex', flexDirection: 'column', boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
      }}>
        <div style={panelHead}>
          <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            Notes · sid {sid} — {name}
          </span>
          <button onClick={onClose} style={{
            background: 'var(--bg-input)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '2px 8px',
            cursor: 'pointer', fontSize: 12,
          }}>✕ close</button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 14px' }}>
          {err && <div style={{ color: 'var(--red)', fontSize: 13 }}>⚠ {err}</div>}
          {notes.length === 0 && !err &&
            <div style={{ fontSize: 12.5, color: 'var(--text-tertiary)' }}>No notes yet.</div>}
          {notes.map((n) => (
            <div key={n.id} style={{ fontSize: 13, margin: '8px 0', paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
              <RoleChip role={n.author} title={n.author} />
              <span style={{ color: 'var(--text-tertiary)', fontSize: 11, marginLeft: 6 }}>{relTime(n.created_at)}</span>
              <div style={{ marginTop: 2, whiteSpace: 'pre-wrap' }}>{n.body}</div>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, padding: 10, borderTop: '1px solid var(--border)', alignItems: 'center' }}>
          <RolePicker value={author} pickTitle="note as" up
            options={withLegacy(ASSIGNEES.filter(Boolean), author)}
            onPick={pickAuthor} />
          <input style={{
            flex: 1, minWidth: 0, background: 'var(--bg-input)', color: 'var(--text-primary)',
            border: '1px solid var(--border)', borderRadius: 6, padding: '4px 6px', fontSize: 13,
          }} placeholder="add a note…" value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && add()} />
          <button onClick={add} style={{
            background: 'var(--blue)', color: '#fff', border: '1px solid var(--border)',
            borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontSize: 13,
          }}>Add</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
