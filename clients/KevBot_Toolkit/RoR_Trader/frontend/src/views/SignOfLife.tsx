/**
 * SIGN OF LIFE — is anyone actually running? (board #251)
 *
 * Rendered on BOTH session dashboards (/admin/m-session and /admin/r-session)
 * from THIS component and the ONE endpoint, so the two surfaces can never
 * disagree about whether R is alive. It lives in its own file rather than in
 * either page for exactly that reason — and so that importing it does not drag
 * a whole sibling dashboard into the other's bundle.
 *
 * WHY IT EXISTS. A dashboard fixes *visibility*; it does not fix *attention*.
 * On 07-30 nothing was broken — #246 sat `Staged` for four hours because no
 * session was watching. Every other module answers "what is owed"; this one
 * answers "is anybody there", which is the question that made the rest moot.
 *
 * NOT RUNNING IS LOUD, AND `never` IS NOT `down`. The #157 dead-alarm lesson is
 * that an absent signal rendered as a confident zero. Here, a session that has
 * never written a heartbeat and a session that stopped writing one get
 * different labels, because "never started" and "stopped" want different
 * reactions — while both read NOT RUNNING, because the consequence is the same.
 *
 * The page NEVER writes a heartbeat on a session's behalf. A session writes its
 * own `system_settings.session_heartbeat_<letter>`; a stale pill is therefore
 * the truth, and faking it would make the one honest signal on the page a lie.
 */
'use client';

import React from 'react';
import { HeartbeatPayload, agoShort, heartbeatDisplay } from './rSessionSignals';

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11.5 };
const dim: React.CSSProperties = { color: 'var(--text-tertiary)' };

export default function SignOfLife({ hb, error }: {
  hb: HeartbeatPayload | null; error: string | null;
}) {
  // An unreadable endpoint is NOT "no session is running" — nothing was
  // measured, and saying otherwise is the false red this whole page is built
  // against.
  if (error) {
    return (
      <div style={{ ...dim, fontSize: 12, marginBottom: 12 }}>
        ⚠️ sign-of-life unreadable — <span style={mono}>{error}</span>. That is not
        the same as “no session is running”: nothing was measured.
      </div>
    );
  }
  if (!hb) {
    return <div style={{ ...dim, fontSize: 12, marginBottom: 12 }}>sign of life — loading…</div>;
  }
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ ...dim, fontSize: 11, fontWeight: 700, letterSpacing: 0.6 }}>
          SIGN OF LIFE
        </span>
        {hb.sessions.map((s) => {
          const d = heartbeatDisplay(s);
          return (
            <span
              key={s.key}
              title={`${s.why}${s.at_source ? ` (freshness from ${s.at_source})` : ''}`}
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 6,
                padding: '3px 9px', borderRadius: 5, background: 'var(--bg-tertiary)',
                border: `1px solid ${d.color}`, fontSize: 12,
              }}
            >
              <b style={{ color: d.color }}>{s.letter}</b>
              <span style={{ color: d.color, fontWeight: 700, fontSize: 11 }}>{d.label}</span>
              {!d.notRunning && <span style={dim}>last seen {agoShort(s.age_s)} ago</span>}
            </span>
          );
        })}
      </div>
      {!hb.readable && (
        <div style={{ fontSize: 11.5, color: 'var(--red)', marginTop: 5 }}>
          ⚠️ the <span style={mono}>system_settings</span> read FAILED
          {hb.read_error ? ` — ${hb.read_error}` : ''}. Every pill above would read
          “never” regardless, so treat them as UNMEASURED, not as down.
        </div>
      )}
      <div style={{ ...dim, fontSize: 10.5, marginTop: 4 }}>
        fresh ≤ {Math.round(hb.thresholds.fresh_max_s / 60)}m · NOT RUNNING past{' '}
        {Math.round(hb.thresholds.stale_max_s / 60)}m — thresholds come from the API,
        never restated here. A session writes{' '}
        <span style={mono}>session_heartbeat_&lt;letter&gt;</span> itself each cycle; these
        pages never write one on its behalf, so a stale pill is the truth.
      </div>
    </div>
  );
}
