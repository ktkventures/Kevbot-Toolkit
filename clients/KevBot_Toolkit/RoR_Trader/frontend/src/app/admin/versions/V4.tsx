'use client';

import { useState, useEffect } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import ChartPlaceholder from '@/components/ChartPlaceholder';

const mockPlatformStats = {
  totalUsers: 2847, newThisMonth: 342, dau: 456,
  mrr: 18420, arr: 221040, mrrGrowth: 12.3,
  totalPortfolios: 67, totalStrategies: 234, totalPacks: 45,
  dailyAlerts: 12847, alertSuccessRate: 99.2, uptime: 99.97,
};

const geoDistribution = [
  { region: 'North America', users: 1284, pct: 45.1 },
  { region: 'Europe', users: 712, pct: 25.0 },
  { region: 'Asia Pacific', users: 456, pct: 16.0 },
  { region: 'Latin America', users: 228, pct: 8.0 },
  { region: 'Other', users: 167, pct: 5.9 },
];

const regionColors = ['var(--accent)', 'var(--blue)', 'var(--green)', 'var(--orange)', 'var(--text-muted)'];

const alertHeatmapData = [
  { hour: '00', mon: 12, tue: 8, wed: 15, thu: 10, fri: 7, sat: 3, sun: 2 },
  { hour: '04', mon: 5, tue: 4, wed: 6, thu: 5, fri: 3, sat: 1, sun: 1 },
  { hour: '08', mon: 45, tue: 52, wed: 48, thu: 50, fri: 42, sat: 8, sun: 5 },
  { hour: '09', mon: 120, tue: 135, wed: 128, thu: 130, fri: 115, sat: 12, sun: 8 },
  { hour: '10', mon: 180, tue: 195, wed: 175, thu: 185, fri: 160, sat: 15, sun: 10 },
  { hour: '12', mon: 95, tue: 88, wed: 92, thu: 90, fri: 85, sat: 10, sun: 7 },
  { hour: '14', mon: 150, tue: 162, wed: 148, thu: 155, fri: 140, sat: 12, sun: 9 },
  { hour: '16', mon: 85, tue: 78, wed: 82, thu: 80, fri: 72, sat: 8, sun: 5 },
  { hour: '20', mon: 30, tue: 25, wed: 28, thu: 27, fri: 22, sat: 5, sun: 3 },
];

function AnimatedCounter({ target, prefix = '', suffix = '' }: { target: number; prefix?: string; suffix?: string }) {
  const [current, setCurrent] = useState(0);

  useEffect(() => {
    const duration = 1500;
    const steps = 40;
    const increment = target / steps;
    let step = 0;
    const timer = setInterval(() => {
      step++;
      if (step >= steps) {
        setCurrent(target);
        clearInterval(timer);
      } else {
        setCurrent(Math.round(increment * step));
      }
    }, duration / steps);
    return () => clearInterval(timer);
  }, [target]);

  return <span>{prefix}{current.toLocaleString()}{suffix}</span>;
}

function getHeatColor(value: number, max: number): string {
  const ratio = value / max;
  if (ratio > 0.7) return 'var(--accent)';
  if (ratio > 0.4) return 'var(--blue)';
  if (ratio > 0.15) return 'var(--blue-muted)';
  return 'var(--bg-input)';
}

export default function AdminV4() {
  const [pulsePhase, setPulsePhase] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setPulsePhase((p) => (p + 1) % 3), 2000);
    return () => clearInterval(timer);
  }, []);

  const pulseColors = ['var(--green)', 'var(--accent)', 'var(--blue)'];

  return (
    <div>
      <PageHeader
        title="Admin Dashboard"
        subtitle="Real-time platform pulse"
        actions={
          <div className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: pulseColors[pulsePhase], boxShadow: `0 0 8px ${pulseColors[pulsePhase]}`, transition: 'all 0.5s' }}
            />
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Live</span>
          </div>
        }
      />

      {/* Animated KPI Counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: 'Total Users', value: mockPlatformStats.totalUsers, color: 'var(--accent)' },
          { label: 'MRR', value: mockPlatformStats.mrr, color: 'var(--green)', prefix: '$' },
          { label: 'Daily Alerts', value: mockPlatformStats.dailyAlerts, color: 'var(--blue)' },
          { label: 'Uptime', value: mockPlatformStats.uptime, color: 'var(--green)', suffix: '%' },
        ].map((kpi) => (
          <div
            key={kpi.label}
            className="rounded-xl border p-4"
            style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}
          >
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{kpi.label}</p>
            <p className="text-2xl font-bold" style={{ color: kpi.color }}>
              <AnimatedCounter target={kpi.value} prefix={kpi.prefix} suffix={kpi.suffix} />
            </p>
          </div>
        ))}
      </div>

      {/* Geographic Distribution + Revenue Waterfall */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
        <Card>
          <h3 className="text-sm font-semibold mb-4">User Distribution</h3>
          <div className="space-y-3">
            {geoDistribution.map((g, i) => (
              <div key={g.region}>
                <div className="flex justify-between text-sm mb-1">
                  <span style={{ color: 'var(--text-secondary)' }}>{g.region}</span>
                  <span className="font-medium">{g.users.toLocaleString()} ({g.pct}%)</span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{ width: `${g.pct}%`, background: regionColors[i] }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-semibold mb-4">Revenue Waterfall</h3>
          <ChartPlaceholder label="Animated revenue waterfall visualization" height={220} />
          <div className="grid grid-cols-3 gap-3 mt-4">
            <div className="text-center">
              <p className="text-lg font-bold" style={{ color: 'var(--green)' }}>
                <AnimatedCounter target={18420} prefix="$" />
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Gross MRR</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold" style={{ color: 'var(--orange)' }}>
                <AnimatedCounter target={2763} prefix="$" />
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Platform Fee</p>
            </div>
            <div className="text-center">
              <p className="text-lg font-bold" style={{ color: 'var(--accent)' }}>
                <AnimatedCounter target={15657} prefix="$" />
              </p>
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Creator Payouts</p>
            </div>
          </div>
        </Card>
      </div>

      {/* Alert Volume Heatmap */}
      <Card className="mb-6">
        <h3 className="text-sm font-semibold mb-4">Alert Volume Heatmap</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="text-xs py-2 pr-3 text-left" style={{ color: 'var(--text-muted)' }}>Hour</th>
                {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((d) => (
                  <th key={d} className="text-xs py-2 px-1 text-center" style={{ color: 'var(--text-muted)' }}>{d}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {alertHeatmapData.map((row) => (
                <tr key={row.hour}>
                  <td className="text-xs py-1 pr-3 font-mono" style={{ color: 'var(--text-muted)' }}>{row.hour}:00</td>
                  {[row.mon, row.tue, row.wed, row.thu, row.fri, row.sat, row.sun].map((val, i) => (
                    <td key={i} className="py-1 px-1">
                      <div
                        className="w-full h-6 rounded flex items-center justify-center text-xs font-mono"
                        style={{
                          background: getHeatColor(val, 195),
                          color: val > 100 ? 'var(--bg-primary)' : 'var(--text-muted)',
                        }}
                      >
                        {val}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Platform Pulse */}
      <Card>
        <h3 className="text-sm font-semibold mb-3">Platform Pulse</h3>
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Webhook Delivery', value: '99.8%', status: 'healthy' },
            { label: 'Alert Success', value: '99.2%', status: 'healthy' },
            { label: 'Error Rate', value: '0.03%', status: 'healthy' },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-input)' }}>
              <span
                className="w-3 h-3 rounded-full"
                style={{
                  background: s.status === 'healthy' ? 'var(--green)' : 'var(--red)',
                  boxShadow: s.status === 'healthy' ? '0 0 6px var(--green)' : '0 0 6px var(--red)',
                }}
              />
              <div>
                <p className="text-sm font-semibold">{s.value}</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{s.label}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
