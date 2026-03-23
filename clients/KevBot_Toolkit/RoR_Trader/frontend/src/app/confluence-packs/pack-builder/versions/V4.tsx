'use client';

import { useState, useCallback, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';

/* ─────────────────────────── Types ─────────────────────────── */

interface ParamDef {
  id: string;
  name: string;
  type: 'int' | 'float';
  default: number;
  min: number;
  max: number;
}

interface OutputState {
  id: string;
  code: string;
  description: string;
  color: string;
}

interface TriggerDef {
  id: string;
  name: string;
  direction: 'LONG' | 'SHORT' | 'BOTH';
  type: 'ENTRY' | 'EXIT';
  execution: string;
  fromState?: string;
  toState?: string;
}

interface LogicNode {
  id: string;
  type: 'indicator' | 'condition' | 'output' | 'trigger';
  label: string;
  x: number;
  y: number;
  color: string;
  connections: string[];
}

interface ValidationItem {
  label: string;
  status: 'pass' | 'fail' | 'warn';
  message: string;
}

/* ─────────────────────────── Constants ─────────────────────────── */

const execTypeOptions = ['[C]', '[L0]', '[L1]', '[HM]', '[HL]'];
const directionOptions: Array<'LONG' | 'SHORT' | 'BOTH'> = ['LONG', 'SHORT', 'BOTH'];
const triggerTypeOptions: Array<'ENTRY' | 'EXIT'> = ['ENTRY', 'EXIT'];
const categoryOptions = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend', 'Custom'];

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[L1]': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--red)', bg: 'var(--red-muted)' },
};

const directionColors: Record<string, { color: string; bg: string; icon: string }> = {
  LONG: { color: 'var(--green)', bg: 'var(--green-muted)', icon: '\u2191' },
  SHORT: { color: 'var(--red)', bg: 'var(--red-muted)', icon: '\u2193' },
  BOTH: { color: 'var(--blue)', bg: 'var(--blue-muted)', icon: '\u2195' },
};

const triggerTypeColors: Record<string, { color: string; bg: string }> = {
  ENTRY: { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  EXIT: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
};

const validationStatusStyles: Record<string, { color: string; icon: string }> = {
  pass: { color: 'var(--green)', icon: '\u2713' },
  fail: { color: 'var(--red)', icon: '\u2717' },
  warn: { color: 'var(--orange)', icon: '!' },
};

const stateColors = ['var(--green)', 'var(--blue)', 'var(--orange)', 'var(--red)', 'var(--accent)', 'var(--text-muted)'];

/* ─────────────────────────── Initial Data ─────────────────────────── */

const initialParams: ParamDef[] = [
  { id: 'p1', name: 'period', type: 'int', default: 14, min: 2, max: 100 },
  { id: 'p2', name: 'overbought', type: 'int', default: 70, min: 50, max: 95 },
  { id: 'p3', name: 'oversold', type: 'int', default: 30, min: 5, max: 50 },
];

const initialOutputs: OutputState[] = [
  { id: 'o1', code: 'OVERBOUGHT', description: 'RSI above overbought threshold', color: 'var(--red)' },
  { id: 'o2', code: 'NEUTRAL', description: 'RSI between oversold and overbought', color: 'var(--blue)' },
  { id: 'o3', code: 'OVERSOLD', description: 'RSI below oversold threshold', color: 'var(--green)' },
];

const initialTriggers: TriggerDef[] = [
  { id: 't1', name: 'RSI Cross Above OB', direction: 'SHORT', type: 'ENTRY', execution: '[C]', fromState: 'NEUTRAL', toState: 'OVERBOUGHT' },
  { id: 't2', name: 'RSI Cross Below OS', direction: 'LONG', type: 'ENTRY', execution: '[C]', fromState: 'NEUTRAL', toState: 'OVERSOLD' },
];

const initialNodes: LogicNode[] = [
  { id: 'n1', type: 'indicator', label: 'RSI(14)', x: 50, y: 120, color: 'var(--accent)', connections: ['n2', 'n3'] },
  { id: 'n2', type: 'condition', label: 'RSI > 70', x: 250, y: 60, color: 'var(--red)', connections: ['n4'] },
  { id: 'n3', type: 'condition', label: 'RSI < 30', x: 250, y: 180, color: 'var(--green)', connections: ['n5'] },
  { id: 'n4', type: 'output', label: 'OVERBOUGHT', x: 450, y: 60, color: 'var(--red)', connections: ['n6'] },
  { id: 'n5', type: 'output', label: 'OVERSOLD', x: 450, y: 180, color: 'var(--green)', connections: ['n7'] },
  { id: 'n6', type: 'trigger', label: 'Short Entry', x: 620, y: 60, color: 'var(--red)', connections: [] },
  { id: 'n7', type: 'trigger', label: 'Long Entry', x: 620, y: 180, color: 'var(--green)', connections: [] },
];

/* ─────────────────────────── Node-based Logic Builder ─────────────────────────── */

function LogicFlowDiagram({ nodes }: { nodes: LogicNode[] }) {
  const width = 780;
  const height = 260;

  const nodeTypeStyles: Record<string, { shape: string; radius: number }> = {
    indicator: { shape: 'circle', radius: 30 },
    condition: { shape: 'diamond', radius: 28 },
    output: { shape: 'rect', radius: 0 },
    trigger: { shape: 'hexagon', radius: 26 },
  };

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} fill="none" className="w-full">
      {/* Grid background */}
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--border)" strokeWidth="0.3" />
        </pattern>
      </defs>
      <rect width={width} height={height} fill="url(#grid)" rx="8" />

      {/* Connection lines */}
      {nodes.map((node) =>
        node.connections.map((targetId) => {
          const target = nodes.find((n) => n.id === targetId);
          if (!target) return null;
          const startX = node.x + 35;
          const endX = target.x - 5;
          const midX = (startX + endX) / 2;
          return (
            <g key={`${node.id}-${targetId}`}>
              <path
                d={`M${startX},${node.y} C${midX},${node.y} ${midX},${target.y} ${endX},${target.y}`}
                stroke={node.color}
                strokeWidth="2"
                opacity="0.4"
                fill="none"
              />
              {/* Animated flow dot */}
              <circle r="3" fill={node.color} opacity="0.8">
                <animateMotion
                  dur={`${2 + Math.random()}s`}
                  repeatCount="indefinite"
                  path={`M${startX},${node.y} C${midX},${node.y} ${midX},${target.y} ${endX},${target.y}`}
                />
              </circle>
            </g>
          );
        }),
      )}

      {/* Nodes */}
      {nodes.map((node) => {
        const style = nodeTypeStyles[node.type];
        return (
          <g key={node.id}>
            {/* Glow */}
            <circle cx={node.x} cy={node.y} r={style.radius + 8} fill={node.color} opacity="0.08" />

            {node.type === 'indicator' && (
              <circle cx={node.x} cy={node.y} r={style.radius} fill={`${node.color}15`} stroke={node.color} strokeWidth="2" />
            )}
            {node.type === 'condition' && (
              <polygon
                points={`${node.x},${node.y - style.radius} ${node.x + style.radius},${node.y} ${node.x},${node.y + style.radius} ${node.x - style.radius},${node.y}`}
                fill={`${node.color}15`}
                stroke={node.color}
                strokeWidth="2"
              />
            )}
            {node.type === 'output' && (
              <rect
                x={node.x - 36}
                y={node.y - 16}
                width="72"
                height="32"
                rx="6"
                fill={`${node.color}15`}
                stroke={node.color}
                strokeWidth="2"
              />
            )}
            {node.type === 'trigger' && (
              <rect
                x={node.x - 36}
                y={node.y - 16}
                width="72"
                height="32"
                rx="16"
                fill={`${node.color}15`}
                stroke={node.color}
                strokeWidth="2"
                strokeDasharray="4 2"
              />
            )}

            {/* Label */}
            <text
              x={node.x}
              y={node.y + 4}
              textAnchor="middle"
              fontSize="9"
              fill={node.color}
              fontWeight="600"
              fontFamily="monospace"
            >
              {node.label}
            </text>

            {/* Type label below */}
            <text
              x={node.x}
              y={node.y + (style.radius || 20) + 14}
              textAnchor="middle"
              fontSize="7"
              fill="var(--text-muted)"
            >
              {node.type.toUpperCase()}
            </text>
          </g>
        );
      })}

      {/* Flow labels */}
      <text x={150} y={20} fontSize="8" fill="var(--text-muted)" fontStyle="italic">Data flows left to right</text>
    </svg>
  );
}

/* ─────────────────────────── Code Preview ─────────────────────────── */

function CodePreview({ params, outputs, triggers, packName }: { params: ParamDef[]; outputs: OutputState[]; triggers: TriggerDef[]; packName: string }) {
  const lines = [
    { text: `# ${packName} Pack - Auto-generated`, color: 'var(--text-muted)' },
    { text: '', color: '' },
    { text: 'def compute_indicator(df, params):', color: 'var(--blue)' },
    ...params.map((p) => ({ text: `    ${p.name} = params.get("${p.name}", ${p.default})`, color: 'var(--text-secondary)' })),
    { text: `    df["indicator"] = ta.rsi(df["close"], ${params[0]?.default || 14})`, color: 'var(--text-secondary)' },
    { text: '    return df', color: 'var(--text-secondary)' },
    { text: '', color: '' },
    { text: 'def interpret(row, params):', color: 'var(--blue)' },
    ...outputs.map((o) => ({ text: `    # ${o.code}: ${o.description}`, color: 'var(--text-muted)' })),
    { text: `    if row["indicator"] > params["overbought"]:`, color: 'var(--text-secondary)' },
    { text: `        return "${outputs[0]?.code || 'STATE_A'}"`, color: 'var(--green)' },
    { text: `    elif row["indicator"] < params["oversold"]:`, color: 'var(--text-secondary)' },
    { text: `        return "${outputs[2]?.code || 'STATE_C'}"`, color: 'var(--green)' },
    { text: `    return "${outputs[1]?.code || 'STATE_B'}"`, color: 'var(--green)' },
    { text: '', color: '' },
    { text: '# Triggers', color: 'var(--text-muted)' },
    ...triggers.map((t) => ({ text: `# ${t.name}: ${t.direction} ${t.type} ${t.execution}`, color: 'var(--text-muted)' })),
  ];

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ background: '#0d1117', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center gap-2 px-4 py-2 border-b" style={{ borderColor: '#21262d' }}>
        <div className="w-3 h-3 rounded-full" style={{ background: '#ff5f56' }} />
        <div className="w-3 h-3 rounded-full" style={{ background: '#ffbd2e' }} />
        <div className="w-3 h-3 rounded-full" style={{ background: '#27c93f' }} />
        <span className="text-[10px] font-mono ml-2" style={{ color: '#8b949e' }}>
          {packName.toLowerCase().replace(/\s+/g, '_')}_pack.py
        </span>
      </div>
      <div className="p-4 overflow-x-auto">
        {lines.map((line, i) => (
          <div key={i} className="flex items-start gap-3">
            <span className="text-[10px] font-mono w-6 text-right flex-shrink-0 select-none" style={{ color: '#484f58' }}>
              {line.text ? i + 1 : ''}
            </span>
            <pre className="text-xs font-mono" style={{ color: line.color || '#c9d1d9' }}>
              {line.text}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─────────────────────────── State Transition Diagram ─────────────────────────── */

function StateTransitionDiagram({ outputs, triggers }: { outputs: OutputState[]; triggers: TriggerDef[] }) {
  const width = 400;
  const height = 200;
  const cx = width / 2;
  const cy = height / 2;
  const r = 70;

  // Position states around a circle
  const statePositions = outputs.map((o, i) => {
    const angle = (i / outputs.length) * 2 * Math.PI - Math.PI / 2;
    return {
      ...o,
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    };
  });

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} fill="none" className="w-full">
      {/* Transition arrows from triggers */}
      {triggers.map((trigger, i) => {
        const from = statePositions.find((s) => s.code === trigger.fromState);
        const to = statePositions.find((s) => s.code === trigger.toState);
        if (!from || !to) return null;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = dx / len;
        const ny = dy / len;
        const startX = from.x + nx * 28;
        const startY = from.y + ny * 28;
        const endX = to.x - nx * 28;
        const endY = to.y - ny * 28;
        const midX = (startX + endX) / 2 + ny * 20;
        const midY = (startY + endY) / 2 - nx * 20;
        const dir = directionColors[trigger.direction];

        return (
          <g key={trigger.id}>
            <path
              d={`M${startX},${startY} Q${midX},${midY} ${endX},${endY}`}
              stroke={dir?.color || 'var(--text-muted)'}
              strokeWidth="1.5"
              opacity="0.6"
              fill="none"
              markerEnd={`url(#arrow-${i})`}
            />
            <defs>
              <marker id={`arrow-${i}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={dir?.color || 'var(--text-muted)'} opacity="0.6" />
              </marker>
            </defs>
            <text x={midX} y={midY - 4} textAnchor="middle" fontSize="7" fill={dir?.color || 'var(--text-muted)'} fontWeight="600">
              {trigger.name.substring(0, 12)}
            </text>
          </g>
        );
      })}

      {/* State nodes */}
      {statePositions.map((state) => (
        <g key={state.id}>
          <circle cx={state.x} cy={state.y} r="24" fill={`${state.color}15`} stroke={state.color} strokeWidth="2" />
          <text x={state.x} y={state.y + 3} textAnchor="middle" fontSize="8" fill={state.color} fontWeight="bold" fontFamily="monospace">
            {state.code.length > 8 ? state.code.substring(0, 7) + '..' : state.code}
          </text>
        </g>
      ))}
    </svg>
  );
}

/* ─────────────────────────── Mock Chart Preview ─────────────────────────── */

function IndicatorPreviewChart({ params }: { params: ParamDef[] }) {
  const period = params.find((p) => p.name === 'period')?.default || 14;
  const ob = params.find((p) => p.name === 'overbought')?.default || 70;
  const os = params.find((p) => p.name === 'oversold')?.default || 30;

  // Generate mock RSI-like line
  const points = useMemo(() => {
    const pts: number[] = [];
    let val = 50;
    for (let i = 0; i < 60; i++) {
      val += (Math.random() - 0.5) * 12;
      val = Math.max(5, Math.min(95, val));
      pts.push(val);
    }
    return pts;
  }, [period]);

  const width = 700;
  const height = 120;
  const toX = (i: number) => 10 + (i / (points.length - 1)) * (width - 20);
  const toY = (v: number) => height - 10 - ((v / 100) * (height - 20));

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(p).toFixed(1)}`).join(' ');
  const obY = toY(ob);
  const osY = toY(os);

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} fill="none" className="w-full">
      {/* OB zone */}
      <rect x="10" y="10" width={width - 20} height={obY - 10} rx="2" fill="var(--red)" opacity="0.06" />
      {/* OS zone */}
      <rect x="10" y={osY} width={width - 20} height={height - 10 - osY} rx="2" fill="var(--green)" opacity="0.06" />

      {/* OB/OS lines */}
      <line x1="10" y1={obY} x2={width - 10} y2={obY} stroke="var(--red)" strokeWidth="0.5" strokeDasharray="4 2" />
      <line x1="10" y1={osY} x2={width - 10} y2={osY} stroke="var(--green)" strokeWidth="0.5" strokeDasharray="4 2" />
      {/* Center line */}
      <line x1="10" y1={toY(50)} x2={width - 10} y2={toY(50)} stroke="var(--text-muted)" strokeWidth="0.5" strokeDasharray="2 4" opacity="0.4" />

      {/* Labels */}
      <text x={width - 8} y={obY + 3} textAnchor="end" fontSize="8" fill="var(--red)">{ob}</text>
      <text x={width - 8} y={osY + 3} textAnchor="end" fontSize="8" fill="var(--green)">{os}</text>
      <text x={width - 8} y={toY(50) + 3} textAnchor="end" fontSize="8" fill="var(--text-muted)">50</text>

      {/* Indicator line */}
      <path d={linePath} stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" />

      {/* Signal markers at zone crosses */}
      {points.map((p, i) => {
        if (i === 0) return null;
        const prev = points[i - 1];
        if (prev < ob && p >= ob) {
          return <circle key={`ob-${i}`} cx={toX(i)} cy={toY(p)} r="3" fill="var(--red)" />;
        }
        if (prev > os && p <= os) {
          return <circle key={`os-${i}`} cx={toX(i)} cy={toY(p)} r="3" fill="var(--green)" />;
        }
        return null;
      })}

      {/* Period annotation */}
      <text x="12" y="18" fontSize="8" fill="var(--text-muted)">Period: {period}</text>
    </svg>
  );
}

/* ─────────────────────────── Test Results ─────────────────────────── */

function TestResultsPanel({ triggers }: { triggers: TriggerDef[] }) {
  const mockResults = triggers.map((t) => ({
    trigger: t.name,
    fires: Math.floor(Math.random() * 50 + 10),
    accuracy: Math.floor(Math.random() * 40 + 40),
    avgBars: Math.floor(Math.random() * 15 + 3),
  }));

  const totalFires = mockResults.reduce((sum, r) => sum + r.fires, 0);
  const avgAccuracy = mockResults.length > 0 ? mockResults.reduce((sum, r) => sum + r.accuracy, 0) / mockResults.length : 0;

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-secondary)' }}>Test Results (Mock Data)</h3>
        <span className="text-[10px] px-2 py-1 rounded-full" style={{ background: 'var(--green-muted)', color: 'var(--green)' }}>
          60 bars
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-lg font-bold font-mono" style={{ color: 'var(--accent)' }}>{totalFires}</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Total Fires</p>
        </div>
        <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-lg font-bold font-mono" style={{ color: avgAccuracy >= 60 ? 'var(--green)' : 'var(--orange)' }}>{avgAccuracy.toFixed(0)}%</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Avg Accuracy</p>
        </div>
        <div className="rounded-lg p-2 text-center" style={{ background: 'var(--bg-input)' }}>
          <span className="text-lg font-bold font-mono" style={{ color: 'var(--text-primary)' }}>{triggers.length}</span>
          <p className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Triggers</p>
        </div>
      </div>
      <div className="space-y-1.5">
        {mockResults.map((r, i) => (
          <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
            <span className="text-xs font-medium flex-1" style={{ color: 'var(--text-primary)' }}>{r.trigger}</span>
            <span className="text-[10px] font-mono" style={{ color: 'var(--accent)' }}>{r.fires} fires</span>
            <span className="text-[10px] font-mono" style={{ color: r.accuracy >= 60 ? 'var(--green)' : 'var(--orange)' }}>{r.accuracy}%</span>
            <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{r.avgBars} bars</span>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ═══════════════════════════ Main Component ═══════════════════════════ */

export default function PackBuilderV4() {
  const [packName, setPackName] = useState('My RSI Pack');
  const [packCategory, setPackCategory] = useState('Momentum');
  const [packDescription, setPackDescription] = useState('Classic RSI with customizable overbought/oversold thresholds');
  const [params, setParams] = useState<ParamDef[]>(initialParams);
  const [outputs, setOutputs] = useState<OutputState[]>(initialOutputs);
  const [triggers, setTriggers] = useState<TriggerDef[]>(initialTriggers);
  const [nodes] = useState<LogicNode[]>(initialNodes);
  const [activePanel, setActivePanel] = useState<'flow' | 'code' | 'preview' | 'test'>('flow');
  const [builderTab, setBuilderTab] = useState<'info' | 'params' | 'outputs' | 'triggers'>('info');

  // Validation
  const validation: ValidationItem[] = useMemo(() => [
    { label: 'Pack Name', status: packName.length > 0 ? 'pass' : 'fail', message: packName.length > 0 ? 'Pack name set' : 'Pack name required' },
    { label: 'Parameters', status: params.length > 0 ? 'pass' : 'warn', message: `${params.length} parameter${params.length !== 1 ? 's' : ''} defined` },
    { label: 'Output States', status: outputs.length >= 2 ? 'pass' : 'fail', message: outputs.length >= 2 ? `${outputs.length} states defined` : 'Need at least 2 output states' },
    { label: 'Triggers', status: triggers.length > 0 ? 'pass' : 'warn', message: triggers.length > 0 ? `${triggers.length} trigger${triggers.length !== 1 ? 's' : ''} defined` : 'No triggers (condition-only pack)' },
    { label: 'State Coverage', status: outputs.length > 0 && triggers.every((t) => !t.fromState || outputs.some((o) => o.code === t.fromState)) ? 'pass' : 'warn', message: 'Trigger states map to valid outputs' },
    { label: 'Naming Conventions', status: outputs.every((o) => o.code === o.code.toUpperCase()) ? 'pass' : 'fail', message: 'Output codes should be UPPERCASE' },
  ], [packName, params, outputs, triggers]);

  const passCount = validation.filter((v) => v.status === 'pass').length;
  const validationPct = (passCount / validation.length) * 100;

  // Add/remove param
  const addParam = useCallback(() => {
    const id = `p${Date.now()}`;
    setParams((prev) => [...prev, { id, name: `param_${prev.length + 1}`, type: 'int', default: 0, min: 0, max: 100 }]);
  }, []);
  const removeParam = useCallback((id: string) => {
    setParams((prev) => prev.filter((p) => p.id !== id));
  }, []);

  // Add/remove output
  const addOutput = useCallback(() => {
    const id = `o${Date.now()}`;
    const colorIdx = outputs.length % stateColors.length;
    setOutputs((prev) => [...prev, { id, code: `STATE_${String.fromCharCode(65 + prev.length)}`, description: 'New state', color: stateColors[colorIdx] }]);
  }, [outputs.length]);
  const removeOutput = useCallback((id: string) => {
    setOutputs((prev) => prev.filter((o) => o.id !== id));
  }, []);

  // Add/remove trigger
  const addTrigger = useCallback(() => {
    const id = `t${Date.now()}`;
    setTriggers((prev) => [...prev, { id, name: 'New Trigger', direction: 'BOTH', type: 'ENTRY', execution: '[C]' }]);
  }, []);
  const removeTrigger = useCallback((id: string) => {
    setTriggers((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <div>
      <style>{`
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(8px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .builder-section { animation: fade-in-up 0.4s ease forwards; }
      `}</style>

      <PageHeader
        title="Pack Builder"
        subtitle="Visual node-based logic builder with live preview and integrated testing"
        actions={
          <div className="flex gap-2">
            <button
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Import JSON
            </button>
            <button
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={{ background: 'var(--accent)', color: 'white' }}
            >
              Save Pack
            </button>
          </div>
        }
      />

      {/* Validation header */}
      <div className="flex items-center gap-4 mb-6 px-4 py-3 rounded-xl" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Validation</span>
            <span className="text-xs font-mono" style={{ color: passCount === validation.length ? 'var(--green)' : 'var(--orange)' }}>
              {passCount}/{validation.length} checks passing
            </span>
          </div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
            <div
              className="h-full rounded-full transition-all"
              style={{
                width: `${validationPct}%`,
                background: validationPct === 100 ? 'var(--green)' : validationPct >= 60 ? 'var(--orange)' : 'var(--red)',
              }}
            />
          </div>
        </div>
        <div className="flex gap-1">
          {validation.map((v, i) => {
            const s = validationStatusStyles[v.status];
            return (
              <div
                key={i}
                className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold"
                style={{ background: `${s.color}15`, color: s.color }}
                title={`${v.label}: ${v.message}`}
              >
                {s.icon}
              </div>
            );
          })}
        </div>
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left: Builder panels */}
        <div className="col-span-5 space-y-4">
          {/* Builder tabs */}
          <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
            {(['info', 'params', 'outputs', 'triggers'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setBuilderTab(tab)}
                className="flex-1 py-2 text-xs font-medium rounded-lg transition-all"
                style={{
                  background: builderTab === tab ? 'var(--bg-card)' : 'transparent',
                  color: builderTab === tab ? 'var(--text-primary)' : 'var(--text-muted)',
                  boxShadow: builderTab === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
                }}
              >
                {tab === 'info' ? 'Info' : tab === 'params' ? `Params (${params.length})` : tab === 'outputs' ? `States (${outputs.length})` : `Triggers (${triggers.length})`}
              </button>
            ))}
          </div>

          {/* Info panel */}
          {builderTab === 'info' && (
            <Card className="builder-section">
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Pack Name</label>
                  <input
                    className="w-full px-3 py-2 rounded-lg text-sm"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    value={packName}
                    onChange={(e) => setPackName(e.target.value)}
                  />
                </div>
                <div>
                  <label className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-secondary)' }}>Category</label>
                  <div className="flex flex-wrap gap-2">
                    {categoryOptions.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => setPackCategory(cat)}
                        className="px-3 py-1.5 rounded-lg text-[10px] font-medium transition-all"
                        style={{
                          background: packCategory === cat ? 'var(--accent)' : 'var(--bg-input)',
                          color: packCategory === cat ? 'white' : 'var(--text-secondary)',
                          border: packCategory === cat ? 'none' : '1px solid var(--border)',
                        }}
                      >
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-xs font-medium mb-1 block" style={{ color: 'var(--text-secondary)' }}>Description</label>
                  <textarea
                    className="w-full px-3 py-2 rounded-lg text-sm resize-none"
                    style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                    rows={3}
                    value={packDescription}
                    onChange={(e) => setPackDescription(e.target.value)}
                  />
                </div>
                <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)' }}>
                  <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                    Pack ID: {packName.toLowerCase().replace(/\s+/g, '_')}_pack
                  </span>
                </div>
              </div>
            </Card>
          )}

          {/* Parameters panel */}
          {builderTab === 'params' && (
            <Card className="builder-section">
              <div className="space-y-3">
                {params.map((param) => (
                  <div key={param.id} className="flex items-center gap-2 px-3 py-2 rounded-xl" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <input
                      className="text-xs font-mono flex-1 px-2 py-1 rounded"
                      style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)', border: 'none' }}
                      value={param.name}
                      onChange={(e) => setParams((prev) => prev.map((p) => p.id === param.id ? { ...p, name: e.target.value } : p))}
                    />
                    <select
                      className="text-[10px] px-1.5 py-1 rounded"
                      style={{ background: 'var(--bg-primary)', color: 'var(--text-secondary)', border: 'none' }}
                      value={param.type}
                      onChange={(e) => setParams((prev) => prev.map((p) => p.id === param.id ? { ...p, type: e.target.value as 'int' | 'float' } : p))}
                    >
                      <option value="int">int</option>
                      <option value="float">float</option>
                    </select>
                    <input
                      type="number"
                      className="text-xs font-mono w-14 text-center px-1 py-1 rounded"
                      style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)', border: 'none' }}
                      value={param.default}
                      onChange={(e) => setParams((prev) => prev.map((p) => p.id === param.id ? { ...p, default: Number(e.target.value) } : p))}
                    />
                    <button
                      onClick={() => removeParam(param.id)}
                      className="w-6 h-6 rounded flex items-center justify-center text-xs"
                      style={{ color: 'var(--red)' }}
                    >
                      x
                    </button>
                  </div>
                ))}
                <button
                  onClick={addParam}
                  className="w-full py-2 rounded-lg text-xs font-medium"
                  style={{ background: 'var(--bg-input)', border: '1px dashed var(--border)', color: 'var(--text-muted)' }}
                >
                  + Add Parameter
                </button>
              </div>
            </Card>
          )}

          {/* Outputs panel */}
          {builderTab === 'outputs' && (
            <Card className="builder-section">
              <div className="space-y-3">
                {outputs.map((output) => (
                  <div key={output.id} className="px-3 py-3 rounded-xl" style={{ background: 'var(--bg-input)', border: `1px solid ${output.color}40` }}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 rounded-full" style={{ background: output.color }} />
                      <input
                        className="text-xs font-mono font-bold flex-1 px-2 py-1 rounded"
                        style={{ background: 'var(--bg-primary)', color: output.color, border: 'none' }}
                        value={output.code}
                        onChange={(e) => setOutputs((prev) => prev.map((o) => o.id === output.id ? { ...o, code: e.target.value } : o))}
                      />
                      <button onClick={() => removeOutput(output.id)} className="w-6 h-6 rounded flex items-center justify-center text-xs" style={{ color: 'var(--red)' }}>
                        x
                      </button>
                    </div>
                    <input
                      className="text-[11px] w-full px-2 py-1 rounded"
                      style={{ background: 'var(--bg-primary)', color: 'var(--text-muted)', border: 'none' }}
                      value={output.description}
                      onChange={(e) => setOutputs((prev) => prev.map((o) => o.id === output.id ? { ...o, description: e.target.value } : o))}
                      placeholder="Description..."
                    />
                  </div>
                ))}
                <button onClick={addOutput} className="w-full py-2 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px dashed var(--border)', color: 'var(--text-muted)' }}>
                  + Add Output State
                </button>
                {/* State transition diagram */}
                {outputs.length >= 2 && triggers.length > 0 && (
                  <div className="mt-3 rounded-xl overflow-hidden" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <div className="px-3 py-2">
                      <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>State Transitions</span>
                    </div>
                    <StateTransitionDiagram outputs={outputs} triggers={triggers} />
                  </div>
                )}
              </div>
            </Card>
          )}

          {/* Triggers panel */}
          {builderTab === 'triggers' && (
            <Card className="builder-section">
              <div className="space-y-3">
                {triggers.map((trigger) => {
                  const dir = directionColors[trigger.direction];
                  const tType = triggerTypeColors[trigger.type];
                  const exec = execTypeBadgeStyles[trigger.execution];
                  return (
                    <div key={trigger.id} className="px-3 py-3 rounded-xl" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-6 h-6 rounded flex items-center justify-center text-sm" style={{ background: dir?.bg, color: dir?.color }}>
                          {dir?.icon}
                        </div>
                        <input
                          className="text-xs font-medium flex-1 px-2 py-1 rounded"
                          style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)', border: 'none' }}
                          value={trigger.name}
                          onChange={(e) => setTriggers((prev) => prev.map((t) => t.id === trigger.id ? { ...t, name: e.target.value } : t))}
                        />
                        <button onClick={() => removeTrigger(trigger.id)} className="w-6 h-6 rounded flex items-center justify-center text-xs" style={{ color: 'var(--red)' }}>
                          x
                        </button>
                      </div>
                      <div className="flex items-center gap-2">
                        <select
                          className="text-[10px] px-1.5 py-1 rounded"
                          style={{ background: dir?.bg, color: dir?.color, border: 'none' }}
                          value={trigger.direction}
                          onChange={(e) => setTriggers((prev) => prev.map((t) => t.id === trigger.id ? { ...t, direction: e.target.value as 'LONG' | 'SHORT' | 'BOTH' } : t))}
                        >
                          {directionOptions.map((d) => <option key={d} value={d}>{d}</option>)}
                        </select>
                        <select
                          className="text-[10px] px-1.5 py-1 rounded"
                          style={{ background: tType?.bg, color: tType?.color, border: 'none' }}
                          value={trigger.type}
                          onChange={(e) => setTriggers((prev) => prev.map((t) => t.id === trigger.id ? { ...t, type: e.target.value as 'ENTRY' | 'EXIT' } : t))}
                        >
                          {triggerTypeOptions.map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                        <select
                          className="text-[10px] px-1.5 py-1 rounded font-mono"
                          style={{ background: exec?.bg, color: exec?.color, border: 'none' }}
                          value={trigger.execution}
                          onChange={(e) => setTriggers((prev) => prev.map((t) => t.id === trigger.id ? { ...t, execution: e.target.value } : t))}
                        >
                          {execTypeOptions.map((e) => <option key={e} value={e}>{e}</option>)}
                        </select>
                      </div>
                    </div>
                  );
                })}
                <button onClick={addTrigger} className="w-full py-2 rounded-lg text-xs font-medium" style={{ background: 'var(--bg-input)', border: '1px dashed var(--border)', color: 'var(--text-muted)' }}>
                  + Add Trigger
                </button>
              </div>
            </Card>
          )}
        </div>

        {/* Right: Preview panels */}
        <div className="col-span-7 space-y-4">
          {/* Preview tabs */}
          <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-input)' }}>
            {(['flow', 'code', 'preview', 'test'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActivePanel(tab)}
                className="flex-1 py-2 text-xs font-medium rounded-lg transition-all"
                style={{
                  background: activePanel === tab ? 'var(--bg-card)' : 'transparent',
                  color: activePanel === tab ? 'var(--text-primary)' : 'var(--text-muted)',
                  boxShadow: activePanel === tab ? '0 1px 3px rgba(0,0,0,0.2)' : 'none',
                }}
              >
                {tab === 'flow' ? 'Logic Flow' : tab === 'code' ? 'Code Preview' : tab === 'preview' ? 'Indicator Chart' : 'Test Results'}
              </button>
            ))}
          </div>

          {/* Logic Flow */}
          {activePanel === 'flow' && (
            <Card className="builder-section">
              <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Logic Flow Diagram</h3>
              <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
                <LogicFlowDiagram nodes={nodes} />
              </div>
              <div className="flex items-center gap-4 mt-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span className="flex items-center gap-1">
                  <div className="w-3 h-3 rounded-full border" style={{ borderColor: 'var(--accent)' }} /> Indicator
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-3 h-3 rotate-45 border" style={{ borderColor: 'var(--orange)' }} /> Condition
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-4 h-3 rounded border" style={{ borderColor: 'var(--blue)' }} /> Output
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-4 h-3 rounded-full border border-dashed" style={{ borderColor: 'var(--green)' }} /> Trigger
                </span>
              </div>
            </Card>
          )}

          {/* Code Preview */}
          {activePanel === 'code' && (
            <div className="builder-section">
              <CodePreview params={params} outputs={outputs} triggers={triggers} packName={packName} />
            </div>
          )}

          {/* Indicator Preview Chart */}
          {activePanel === 'preview' && (
            <Card className="builder-section">
              <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Live Indicator Preview</h3>
              <div className="rounded-xl overflow-hidden" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                <IndicatorPreviewChart params={params} />
              </div>
              <div className="flex items-center gap-4 mt-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ background: 'var(--red)' }} /> OB Zone
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ background: 'var(--green)' }} /> OS Zone
                </span>
                <span className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} /> Signal
                </span>
              </div>
            </Card>
          )}

          {/* Test Results */}
          {activePanel === 'test' && (
            <div className="builder-section">
              <TestResultsPanel triggers={triggers} />
            </div>
          )}

          {/* Validation details */}
          <Card>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-secondary)' }}>Validation Checklist</h3>
            <div className="space-y-1.5">
              {validation.map((v, i) => {
                const s = validationStatusStyles[v.status];
                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 px-3 py-2 rounded-lg"
                    style={{ background: v.status === 'fail' ? `${s.color}08` : 'var(--bg-input)' }}
                  >
                    <div
                      className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                      style={{ background: `${s.color}20`, color: s.color }}
                    >
                      {s.icon}
                    </div>
                    <span className="text-xs font-medium flex-1" style={{ color: 'var(--text-primary)' }}>{v.label}</span>
                    <span className="text-[10px]" style={{ color: s.color }}>{v.message}</span>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
