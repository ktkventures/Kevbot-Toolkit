/**
 * R-Session Dashboard — /admin/r-session (board #251).
 *
 * Kevin, 07-30: *"get the R session fully prepped… so they can take [releases]
 * off your plate."* The measured motivation: #246 was approved, built, tested
 * and PR'd inside an hour, then sat in `Staged` for four hours because M never
 * created the train that ships it. The dispatch page correctly read
 * `dispatchable=0`. **Nothing was broken; nobody was watching.**
 *
 * Deliberately SMALLER than /admin/m-session. Five modules, in this order:
 *
 *   1 CARS WAITING FOR A TRAIN — the exact thing that failed on 07-30, and the
 *     reason this page exists. Non-zero with an idle dispatcher = a train is
 *     OWED. Everything else on the page is context for this number.
 *   2 RECENT TRAINS — wave, cars, chain progress, run outcome, and whether the
 *     deploy log records it.
 *   3 DEPLOY LOG — as the SERVING container sees it, which is what makes "is
 *     the Wave-N log PR still open?" answerable without a GitHub credential.
 *   4 SERVICE — `/health` on the commit this api is serving. Deploy status is
 *     not health: on Wave 24 deploy-watch reported 7/7 `success` while the api
 *     served 502 for twenty minutes (#238).
 *   5 BLOCKED, RELEASE-RELATED — the short list R should actually read.
 *
 * Plus SIGN OF LIFE across the top — which live sessions are running at all.
 * A dashboard fixes visibility, not attention; the heartbeat strip is the part
 * that says whether anyone is looking.
 *
 * THE DERIVATIONS ARE NOT IN THIS FILE. They live in `rSessionSignals.ts`
 * because `src/test_r_session_dashboard_251.py` lifts and EXECUTES them in
 * node. Clause 2 of "cars waiting" — has this task got unmerged code? — is not
 * implemented at all: it is `shippingRows`, the #245 resolver, imported. Two
 * definitions of "has code" would drift, and that one already has tests.
 *
 * Conventions follow AdminDispatchPage/AdminMSessionPage deliberately: same
 * fetch shape, same UTC stamps, same PRINT-THE-GAPS footer. Two idioms on one
 * admin section is how a section stops being read.
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
  OutcomeChip, RunRow, STATUS_COLOR, Task, ageColor, ageShort,
  relTime, utcStamp,
} from './taskBoardShared';
import { shippingRows } from './AdminDispatchPage';
import SignOfLife from './SignOfLife';
import {
  CarsPanel, HeartbeatPayload, R_PAGE_ROWS, TrainRow,
  carsWaitingForATrain, recentTrains, releaseBlocked,
} from './rSessionSignals';

/** Auto-refresh cadence — matched to /admin/dispatch. R watches for a train
 *  being owed, and a stale-by-a-minute answer to that is fine; a stale-by-ten
 *  one is the four-hour stall again. */
const REFRESH_MS = 15000;
/** Run-log depth — matched to the sibling pages so the three agree about
 *  history rather than each holding its own window. */
const RUN_LIMIT = 400;

interface DeployEntry {
  day: string | null; time: string; sha: string; wave: number | null; excerpt: string;
}
interface DeployLogPayload {
  entries: DeployEntry[]; waves: number[]; found: boolean;
  resolved_path: string; error: string | null; note?: string; generated_at: string;
}
interface ServicePayload {
  status: string; commit: string | null; branch: string | null;
  service: string | null; deployment_id: string | null;
  scope: string; known_gaps: string[]; generated_at: string;
}

const mono: React.CSSProperties = { fontFamily: 'monospace', fontSize: 11.5 };
const dim: React.CSSProperties = { color: 'var(--text-tertiary)' };

const Panel = ({ n, title, sub, right, children }: {
  n: string; title: string; sub?: React.ReactNode; right?: React.ReactNode;
  children: React.ReactNode;
}) => (
  <div style={{ marginBottom: 16 }}>
    <Card>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
        {/* board #278 — `minWidth: 0` so a long token in `sub` cannot push
            `right` (the headline number) off the card. */}
        <div style={{ flex: 1, minWidth: 0, overflowWrap: 'anywhere' }}>
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

/** Deep link to a task thread — the interaction the whole page rests on. */
const TaskLink = ({ id, title }: { id: number; title?: string }) => (
  <Link href={`/admin/tasks?task=${id}`} style={{
    color: 'var(--accent)', textDecoration: 'none', fontWeight: 600, fontSize: 12.5,
  }}>
    #{id}{title ? ` · ${title}` : ''}
  </Link>
);

const StatusChip = ({ status }: { status: string }) => (
  <span style={{
    fontSize: 10.5, fontWeight: 700, padding: '1px 6px', borderRadius: 4,
    background: 'var(--bg-tertiary)', color: STATUS_COLOR[status] || 'var(--text-secondary)',
  }}>{status}</span>
);

export default function AdminRSessionPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [hb, setHb] = useState<HeartbeatPayload | null>(null);
  const [hbErr, setHbErr] = useState<string | null>(null);
  const [dlog, setDlog] = useState<DeployLogPayload | null>(null);
  const [svc, setSvc] = useState<ServicePayload | null>(null);
  const [svcErr, setSvcErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    try {
      setErr(null);
      // include_done=true is REQUIRED: past trains are Done, and module 2 is
      // about them. Two list GETs joined in memory, never a per-row fetch (the
      // board-#148 poll-efficiency rail).
      const [ts, rh] = await Promise.all([
        apiFetch<Task[]>('/api/dev-tasks?include_done=true'),
        apiFetch<RunRow[]>(`/api/run-history?limit=${RUN_LIMIT}`).catch(() => [] as RunRow[]),
      ]);
      setTasks(ts || []);
      setRuns(rh || []);
      setFetchedAt(new Date().toISOString());
    } catch (e) { setErr(String(e)); }
    finally { setLoading(false); }
  }, []);

  /** The three r-session reads. Each keeps its own failure: an unreadable
   *  heartbeat endpoint and an empty one are different facts, and folding them
   *  together is the false-green shape this page is built against. */
  const loadSession = useCallback(async () => {
    try {
      const p = await apiFetch<HeartbeatPayload>('/api/r-session/heartbeats');
      setHb(p); setHbErr(null);
    } catch (e) { setHb(null); setHbErr(String(e)); }
    try {
      setDlog(await apiFetch<DeployLogPayload>('/api/r-session/deploy-log'));
    } catch { setDlog(null); }
    try {
      const s = await apiFetch<ServicePayload>('/api/r-session/service');
      setSvc(s); setSvcErr(null);
    } catch (e) { setSvc(null); setSvcErr(String(e)); }
  }, []);

  useEffect(() => { load(); loadSession(); }, [load, loadSession]);
  useEffect(() => {
    const id = setInterval(() => {
      if (!document.hidden) { load(); loadSession(); }
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, [load, loadSession]);
  useEffect(() => {
    const id = setInterval(() => setTick((n) => n + 1), 5000);
    return () => clearInterval(id);
  }, []);

  const runsByTask = useMemo(() => {
    const m = new Map<number, RunRow[]>();
    runs.forEach((r) => { const a = m.get(r.task_id) || []; a.push(r); m.set(r.task_id, a); });
    return m;
  }, [runs]);

  /** Is `pushed_branch` present in the payload AT ALL? Absent for everyone
   *  means the column is unreadable, and the resolver degrades rather than
   *  rendering an empty (falsely reassuring) lane. Same test the dispatch page
   *  uses — deliberately, so the two pages cannot disagree. */
  const pushedKnown = useMemo(
    () => runs.some((r) => Object.prototype.hasOwnProperty.call(r, 'pushed_branch')),
    [runs]);

  const lane = useMemo(
    () => shippingRows(tasks, runsByTask, pushedKnown),
    [tasks, runsByTask, pushedKnown, tick]);      // eslint-disable-line react-hooks/exhaustive-deps

  const cars = useMemo<CarsPanel>(
    () => carsWaitingForATrain(lane, tasks, R_PAGE_ROWS), [lane, tasks]);

  const trains = useMemo<TrainRow[]>(
    () => recentTrains(tasks, runsByTask, dlog?.found ? dlog.waves : null, R_PAGE_ROWS),
    [tasks, runsByTask, dlog, tick]);             // eslint-disable-line react-hooks/exhaustive-deps

  const blocked = useMemo(() => releaseBlocked(tasks, R_PAGE_ROWS), [tasks]);

  return (
    <div style={{ padding: '18px 20px 60px', maxWidth: 1180, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 6 }}>
        <h1 style={{ fontSize: 19, fontWeight: 700, margin: 0 }}>R Session</h1>
        <span style={{ ...dim, fontSize: 12 }}>
          the release lane · R merges to <span style={mono}>dev</span>, and is the only lane that does
        </span>
        <span style={{ flex: 1 }} />
        <Link href="/admin/dispatch" style={{ fontSize: 12, color: 'var(--accent)' }}>dispatch →</Link>
      </div>
      <div style={{ ...dim, fontSize: 11.5, marginBottom: 12 }}>
        {loading ? 'loading…' : `fetched ${fetchedAt ? utcStamp(fetchedAt) : '—'} · refreshes every ${REFRESH_MS / 1000}s`}
        {err && <span style={{ color: 'var(--red)' }}> · {err}</span>}
      </div>

      <SignOfLife hb={hb} error={hbErr} />

      {/* ── 1 · CARS WAITING FOR A TRAIN ─────────────────────────────────── */}
      <Panel
        n="1"
        title="Cars waiting for a train"
        sub={<>
          <b>Staged</b> · an unmerged branch · <b>no open train carrying it</b>.
          Non-zero here with an idle dispatcher means a train is <b>OWED</b> —
          this is the exact shape of the 07-30 stall (#246 sat 4h).
        </>}
        right={
          <div style={{ textAlign: 'right' }}>
            <div style={{
              fontSize: 26, fontWeight: 800, lineHeight: 1,
              color: cars.trainOwed ? 'var(--red)' : 'var(--green)',
            }}>{cars.count}</div>
            <div style={{ ...dim, fontSize: 10.5 }}>
              {cars.trainOwed ? `oldest ${Math.round(cars.oldestH)}h` : 'nothing waiting'}
            </div>
          </div>
        }
      >
        {cars.count === 0 ? (
          <div style={{ fontSize: 12.5, color: 'var(--green)' }}>
            ✓ No car is waiting. Every `Staged` task with unmerged code is on an
            open train{cars.openTrainIds.length
              ? ` (${cars.openTrainIds.map((i) => `#${i}`).join(', ')})`
              : ', and there is none to assemble'}.
          </div>
        ) : (
          <div>
            <div style={{
              fontSize: 12.5, fontWeight: 700, color: 'var(--red)', marginBottom: 8,
            }}>
              🚂 A TRAIN IS OWED — {cars.count} car{cars.count === 1 ? '' : 's'} parked with
              nobody coming for {cars.count === 1 ? 'it' : 'them'}.
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
              <thead>
                <tr style={{ ...dim, fontSize: 10.5, textAlign: 'left' }}>
                  <th style={{ padding: '3px 6px 3px 0' }}>TASK</th>
                  <th style={{ padding: '3px 6px' }}>BRANCH</th>
                  <th style={{ padding: '3px 6px' }}>WAITING</th>
                </tr>
              </thead>
              <tbody>
                {cars.rows.map((r) => (
                  <tr key={r.task.id} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '5px 6px 5px 0' }}>
                      <TaskLink id={r.task.id} title={r.task.title} />
                      {r.mergeUnknown && (
                        <span title={r.mergeWhy} style={{ ...dim, fontSize: 10.5, marginLeft: 6 }}>
                          ? merge state unreadable
                        </span>
                      )}
                    </td>
                    <td style={{ ...mono, padding: '5px 6px', color: 'var(--text-secondary)' }}>
                      {r.branch || <span style={dim}>— no branch recorded</span>}
                    </td>
                    <td style={{ padding: '5px 6px', color: ageColor(r.ageH), fontWeight: 700 }}
                      title={`age measured from the ${r.ageFrom}`}>
                      {Math.round(r.ageH)}h <span style={{ ...dim, fontWeight: 400 }}>({r.ageFrom})</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {cars.count > cars.rows.length && (
              <div style={{ ...dim, fontSize: 11, marginTop: 6 }}>
                showing {cars.rows.length} of {cars.count} — the rest are counted, not painted (#240).
              </div>
            )}
          </div>
        )}
        {cars.covered.length > 0 && (
          <div style={{ ...dim, fontSize: 11.5, marginTop: 8 }}>
            {cars.covered.length} Staged car(s) EXCLUDED — already on an open train:{' '}
            {cars.covered.map((c) => `#${c.id}→#${c.trainId}`).join(', ')}
          </div>
        )}
        {cars.warning && (
          <div style={{ fontSize: 11.5, color: 'var(--yellow, #d69e2e)', marginTop: 8 }}>
            ⚠️ inherited from the shipping-lane resolver: {cars.warning}
          </div>
        )}
      </Panel>

      {/* ── 2 · RECENT TRAINS ────────────────────────────────────────────── */}
      <Panel n="2" title="Recent trains and how they went"
        sub="Newest wave first. LOG is INFERRED, not queried: ✓ means the deploy-log entry for that wave is present in the commit this api serves. Some entries do not name their wave in the headline, so `owed` can be a false alarm — deliberately the safe direction.">
        {trains.length === 0 ? (
          <div style={dim}>No train task on the board.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12.5 }}>
            <thead>
              <tr style={{ ...dim, fontSize: 10.5, textAlign: 'left' }}>
                <th style={{ padding: '3px 6px 3px 0' }}>WAVE</th>
                <th style={{ padding: '3px 6px' }}>TRAIN</th>
                <th style={{ padding: '3px 6px' }}>CARS</th>
                <th style={{ padding: '3px 6px' }}>CHAIN</th>
                <th style={{ padding: '3px 6px' }}>LAST RUN</th>
                <th style={{ padding: '3px 6px' }}>LOG</th>
              </tr>
            </thead>
            <tbody>
              {trains.map((t) => (
                <tr key={t.task.id} style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{ padding: '5px 6px 5px 0', fontWeight: 700 }}>
                    {t.wave != null ? t.wave : <span style={dim}>—</span>}
                  </td>
                  <td style={{ padding: '5px 6px' }}>
                    <TaskLink id={t.task.id} />{' '}
                    <StatusChip status={t.task.status} />
                  </td>
                  <td style={{ ...mono, padding: '5px 6px', color: 'var(--text-secondary)' }}>
                    {t.cars.length ? t.cars.map((c) => `#${c}`).join(' ') : <span style={dim}>none named</span>}
                  </td>
                  <td style={{ padding: '5px 6px', ...dim }}>{t.done}/{t.total}</td>
                  <td style={{ padding: '5px 6px' }}>
                    {t.outcome ? <OutcomeChip outcome={t.outcome} /> : <span style={dim}>never ran</span>}
                    {t.finishedAt && <span style={{ ...dim, marginLeft: 6 }}>{relTime(t.finishedAt)}</span>}
                  </td>
                  <td style={{ padding: '5px 6px' }}>
                    {t.logged === true && <span style={{ color: 'var(--green)' }}>✓ logged</span>}
                    {t.logged === false && <span style={{ color: 'var(--yellow, #d69e2e)' }}>owed / PR open</span>}
                    {t.logged == null && <span style={dim}>? unreadable</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>

      {/* ── 3 · DEPLOY LOG ───────────────────────────────────────────────── */}
      <Panel n="3" title="Deploy log — as the serving container sees it"
        sub={dlog?.note || 'the log read out of the commit this api is running.'}>
        {!dlog && <div style={dim}>loading…</div>}
        {dlog && !dlog.found && (
          <div style={{ fontSize: 12.5, color: 'var(--red)' }}>
            ⚠️ {dlog.error} — tried <span style={mono}>{dlog.resolved_path}</span>.
            This is a loud failure on purpose: an empty log would read as “no
            deploys have happened”.
          </div>
        )}
        {dlog && dlog.found && dlog.entries.length === 0 && (
          <div style={{ fontSize: 12.5, color: 'var(--red)' }}>
            ⚠️ the file was READ but no entry matched the expected format at{' '}
            <span style={mono}>{dlog.resolved_path}</span> — the parser and the
            log have drifted. Not “no deploys”.
          </div>
        )}
        {dlog && dlog.entries.map((e, i) => (
          <div key={`${e.sha}-${i}`} style={{
            padding: '5px 0', borderTop: i ? '1px solid var(--border)' : 'none', fontSize: 12,
          }}>
            <span style={{ ...mono, color: 'var(--accent)' }}>{e.sha}</span>{' '}
            <span style={dim}>{e.day} {e.time}</span>
            {e.wave != null && <b style={{ marginLeft: 6 }}>Wave {e.wave}</b>}
            <div style={{ ...dim, fontSize: 11.5 }}>{e.excerpt}</div>
          </div>
        ))}
      </Panel>

      {/* ── 4 · SERVICE ──────────────────────────────────────────────────── */}
      <Panel n="4" title="/health and the head this api is serving"
        sub="A train is not finished until /health returns 200 (#238). Deploy status is NOT health — on Wave 24 deploy-watch read 7/7 success while the api served 502 for 20 minutes.">
        {svcErr && (
          <div style={{ fontSize: 12.5, color: 'var(--red)' }}>
            ⚠️ the api did not answer: {svcErr}. That is itself the signal.
          </div>
        )}
        {svc && (
          <div style={{ fontSize: 12.5 }}>
            <div>
              <b style={{ color: svc.status === 'ok' ? 'var(--green)' : 'var(--red)' }}>
                {svc.status === 'ok' ? '200 · ok' : svc.status}
              </b>{' '}
              <span style={dim}>({svc.scope})</span>
            </div>
            <div style={{ ...mono, marginTop: 4, color: 'var(--text-secondary)' }}>
              commit {svc.commit || '(not reported)'} · branch {svc.branch || '—'} ·
              service {svc.service || '—'}
            </div>
            <ul style={{ ...dim, fontSize: 11.5, marginTop: 8, paddingLeft: 18 }}>
              {svc.known_gaps.map((g) => <li key={g}>{g}</li>)}
            </ul>
          </div>
        )}
      </Panel>

      {/* ── 5 · BLOCKED, RELEASE-RELATED ─────────────────────────────────── */}
      <Panel n="5" title="Blocked, and it names a release"
        sub="Matched on TITLE only — descriptions mention trains in passing, and a panel full of false positives is a panel nobody reads.">
        {blocked.count === 0 ? (
          <div style={{ fontSize: 12.5, color: 'var(--green)' }}>✓ nothing blocked names a release.</div>
        ) : blocked.rows.map((t) => (
          <div key={t.id} style={{ padding: '4px 0', fontSize: 12.5 }}>
            <TaskLink id={t.id} title={t.title} />{' '}
            <StatusChip status={t.status} />
            <span style={{ ...dim, marginLeft: 6 }}>{ageShort(t.updated_at)} since last touch</span>
          </div>
        ))}
      </Panel>

      <div style={{ ...dim, fontSize: 11, marginTop: 20, lineHeight: 1.6 }}>
        <b>WHAT THIS PAGE CANNOT SEE.</b> The app runs on Railway with no
        checkout, no GitHub credential and no `railway` CLI. Merged-ness is read
        from #242’s git-gated ship-step tick, never computed here. PR state is
        inferred, never queried. The full 7-service deploy rollup is E’s to read.
        A dashboard whose blind spots are invisible manufactures false
        confidence — that is the #157 dead-alarm lesson, and this page is not
        repeating it. Sign-of-life is written BY each session; this page never
        writes a heartbeat on another session’s behalf.
      </div>
    </div>
  );
}
