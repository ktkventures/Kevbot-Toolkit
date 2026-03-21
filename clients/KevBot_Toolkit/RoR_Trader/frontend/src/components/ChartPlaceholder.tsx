interface ChartPlaceholderProps {
  label: string;
  height?: number;
}

export default function ChartPlaceholder({ label, height = 300 }: ChartPlaceholderProps) {
  return (
    <div
      className="rounded-lg flex items-center justify-center"
      style={{ background: 'var(--bg-input)', height }}
    >
      <span className="text-sm" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  );
}
