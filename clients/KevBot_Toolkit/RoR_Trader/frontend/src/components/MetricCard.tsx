interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  positive?: boolean;
}

export default function MetricCard({ label, value, delta, positive }: MetricCardProps) {
  return (
    <div
      className="rounded-lg border p-4"
      style={{
        background: 'var(--bg-card)',
        borderColor: 'var(--border)',
      }}
    >
      <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{label}</p>
      <p className="text-xl font-semibold">{value}</p>
      {delta && (
        <p
          className="text-xs mt-1"
          style={{ color: positive ? 'var(--green)' : 'var(--red)' }}
        >
          {delta}
        </p>
      )}
    </div>
  );
}
