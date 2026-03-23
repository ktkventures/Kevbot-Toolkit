'use client';

import { useState } from 'react';
import Card from '@/components/Card';

/* ========================================================================= */
/* COMPONENT                                                                  */
/* ========================================================================= */

export default function SettingsDisplayV3() {
  const [timezone, setTimezone] = useState('US/Mountain');
  const [dateFormat, setDateFormat] = useState('YYYY-MM-DD');

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Display</h1>

      <div className="max-w-sm">
        <Card>
          <div className="space-y-5">
            {/* Timezone — inline selector */}
            <div className="flex items-center justify-between">
              <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Timezone</span>
              <select
                value={timezone}
                onChange={(e) => setTimezone(e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>US/Mountain</option>
                <option>US/Eastern</option>
                <option>US/Central</option>
                <option>US/Pacific</option>
                <option>UTC</option>
              </select>
            </div>

            <div className="border-t" style={{ borderColor: 'var(--border)' }} />

            {/* Date Format — inline selector */}
            <div className="flex items-center justify-between">
              <span className="text-sm" style={{ color: 'var(--text-primary)' }}>Date Format</span>
              <select
                value={dateFormat}
                onChange={(e) => setDateFormat(e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm"
                style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
              >
                <option>YYYY-MM-DD</option>
                <option>MM/DD/YYYY</option>
                <option>DD/MM/YYYY</option>
              </select>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
