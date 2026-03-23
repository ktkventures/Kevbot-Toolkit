'use client';

import Card from '@/components/Card';

/* ========================================================================= */
/* DATA                                                                       */
/* ========================================================================= */

const services = [
  { name: 'Polygon.io', ok: true },
  { name: 'Alpaca', ok: false },
  { name: 'Supabase', ok: true },
  { name: 'Railway', ok: true },
];

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsConnectionsV3() {
  const allOk = services.every((s) => s.ok);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Connections</h1>

      <div className="max-w-sm">
        <Card>
          {/* Summary line */}
          <div className="flex items-center gap-2 mb-4 pb-4 border-b" style={{ borderColor: 'var(--border)' }}>
            <span
              className="w-3 h-3 rounded-full"
              style={{
                background: allOk ? 'var(--green)' : 'var(--yellow, #e5a813)',
                boxShadow: allOk ? '0 0 6px var(--green)' : '0 0 6px var(--yellow, #e5a813)',
              }}
            />
            <span className="text-sm font-medium">
              {allOk ? 'All systems operational' : 'Some services disconnected'}
            </span>
          </div>

          {/* Status dots list */}
          <div className="space-y-3">
            {services.map((svc) => (
              <div key={svc.name} className="flex items-center justify-between">
                <span className="text-sm" style={{ color: 'var(--text-primary)' }}>
                  {svc.name}
                </span>
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ background: svc.ok ? 'var(--green)' : 'var(--text-muted)' }}
                  />
                  <span
                    className="text-xs"
                    style={{ color: svc.ok ? 'var(--green)' : 'var(--text-muted)' }}
                  >
                    {svc.ok ? 'OK' : 'Off'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
