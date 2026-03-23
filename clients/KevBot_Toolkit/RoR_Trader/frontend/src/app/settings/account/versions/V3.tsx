'use client';

import Card from '@/components/Card';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsAccountV3() {
  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Account</h1>

      <div className="max-w-sm">
        <Card>
          <div className="space-y-4">
            <div>
              <p className="text-xs mb-0.5" style={{ color: 'var(--text-muted)' }}>Email</p>
              <p className="text-sm font-medium">kevin@example.com</p>
            </div>

            <div className="border-t" style={{ borderColor: 'var(--border)' }} />

            <button
              className="w-full px-4 py-2.5 rounded-lg text-sm font-medium"
              style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
            >
              Sign Out
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}
