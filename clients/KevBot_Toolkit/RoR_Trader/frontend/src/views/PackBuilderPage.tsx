'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import ChartPlaceholder from '@/components/ChartPlaceholder';
import Modal from '@/components/Modal';
import MetricCard from '@/components/MetricCard';
import { useGenerateStructure, useGenerateCode, useRequestFix, useValidatePack, useInstallPack } from '@/hooks/mutations/useAiBuilder';
import { useRunBacktest, useBacktestTradeZoom } from '@/hooks/queries/useBacktest';
import { useConfluenceTriggers, useStopLossPacks, useTakeProfitPacks } from '@/hooks/queries/usePacks';
import { useChartPrefs } from '@/hooks/useChartPrefs';
import { buildStrategyChartPanes } from '@/charts/buildStrategyChartPanes';
import EquityCurve from '@/charts/EquityCurve';

const SyncedChartPane = dynamic(() => import('@/charts/SyncedChartPane'), { ssr: false });
const TradeZoomModal = dynamic(() => import('@/components/TradeZoomModal'), { ssr: false });

/* ========================================================================= */
/* CONSTANTS                                                                   */
/* ========================================================================= */

const EXEC_BADGE_COLOR = '#2196F3';
const FIDELITY_BADGE_COLOR = '#26C6DA';

const TF_CATEGORIES = ['Momentum', 'Trend', 'Volume', 'Volatility', 'Mean Reversion', 'Custom'];
const GEN_CATEGORIES = ['Time', 'Session', 'Calendar', 'Custom'];
const DISPLAY_TYPES = ['overlay', 'oscillator', 'hidden'] as const;
const EXEC_TYPES = ['[C]', '[L]', '[LC]', '[CC]'] as const;
const SENTIMENTS = ['bullish', 'bearish', 'neutral'] as const;
const SB_TIMEFRAMES = ['1Min', '5Min', '15Min', '1H', '1D'] as const;

/* ========================================================================= */
/* TYPES                                                                       */
/* ========================================================================= */

type PackType = 'tf_confluence' | 'general';

interface ParamDef { id: string; name: string; label: string; type: string; defaultVal: string; min: string; max: string; }
interface OutputDef { id: string; code: string; description: string; }
interface TriggerDef { id: string; name: string; base: string; sentiment: 'bullish' | 'bearish' | 'neutral'; fromState: string; toState: string; }
interface ExecTypeConfig { enabled: boolean; referenceBar: 0 | -1; orderType: 'market' | 'limit'; holdSeconds?: number; confirmBarOffset?: 0 | 1; bailAction?: 'exit_market' | 'exit_limit'; }
interface ValidationItem { id: string; label: string; category: string; status: 'pass' | 'fail' | 'warn' | 'pending'; message: string; }

/* ========================================================================= */
/* HELPERS                                                                     */
/* ========================================================================= */

function generateId(): string { return 'id-' + Math.random().toString(36).substring(2, 9); }
function toSlug(name: string): string { return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').substring(0, 40); }

function ExecBadge({ tag }: { tag: string }) {
  return <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full" style={{ color: EXEC_BADGE_COLOR, background: EXEC_BADGE_COLOR + '20' }}>{tag}</span>;
}
function FidelityBadge({ tag }: { tag: string }) {
  return <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full" style={{ color: FIDELITY_BADGE_COLOR, background: FIDELITY_BADGE_COLOR + '20' }}>{tag}</span>;
}
function SentimentBadge({ sentiment }: { sentiment: string }) {
  const c = sentiment === 'bullish' ? { color: 'var(--green)', bg: 'var(--green-muted)' } : sentiment === 'bearish' ? { color: 'var(--red)', bg: 'var(--red-muted)' } : { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full capitalize" style={{ color: c.color, background: c.bg }}>{sentiment}</span>;
}
function StatusIcon({ status }: { status: string }) {
  if (status === 'pass') return <span style={{ color: 'var(--green)' }}>✓</span>;
  if (status === 'fail') return <span style={{ color: 'var(--red)' }}>✗</span>;
  if (status === 'warn') return <span style={{ color: 'var(--orange)' }}>⚠</span>;
  return <span style={{ color: 'var(--text-muted)' }}>○</span>;
}

/* ========================================================================= */
/* STYLES                                                                      */
/* ========================================================================= */

const inputStyle: React.CSSProperties = { background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '6px 10px', borderRadius: '8px', fontSize: '0.8rem', width: '100%' };
const btnPrimary: React.CSSProperties = { background: 'var(--accent)', color: 'white', border: 'none', padding: '8px 16px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer', fontWeight: 600 };
const btnSecondary: React.CSSProperties = { background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)', padding: '6px 14px', borderRadius: '8px', fontSize: '0.875rem', cursor: 'pointer' };

/* ========================================================================= */
/* MOCK DATA                                                                   */
/* ========================================================================= */

/* {{ai_response}} — These will be populated by the AI wizard API when connected */
const PLACEHOLDER_PARAMS: ParamDef[] = [];
const PLACEHOLDER_OUTPUTS: OutputDef[] = [];
const PLACEHOLDER_TRIGGERS: TriggerDef[] = [];
const PLACEHOLDER_VALIDATION: ValidationItem[] = [];

/* ========================================================================= */
/* COMPONENT                                                                   */
/* ========================================================================= */

export default function PackBuilderPage() {
  const [step, setStep] = useState(1);

  // Step 1: Info
  const [packType, setPackType] = useState<PackType>('tf_confluence');
  const [packName, setPackName] = useState('');
  const [category, setCategory] = useState('');
  const [displayType, setDisplayType] = useState('oscillator');
  const [description, setDescription] = useState('');
  const [pineScript, setPineScript] = useState('');

  // Step 2: AI-generated structure
  const [structureGenerated, setStructureGenerated] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [params, setParams] = useState<ParamDef[]>([]);
  const [outputs, setOutputs] = useState<OutputDef[]>([]);
  const [triggers, setTriggers] = useState<TriggerDef[]>([]);

  // Step 4: AI code generation + validation
  const [codeGenerated, setCodeGenerated] = useState(false);
  const [isGeneratingCode, setIsGeneratingCode] = useState(false);
  const [validation, setValidation] = useState<ValidationItem[]>([]);
  const [autoFixAttempt, setAutoFixAttempt] = useState(0);
  const [isAutoFixing, setIsAutoFixing] = useState(false);

  // AI conversation
  const [conversation, setConversation] = useState<{ role: string; content: string; time: string }[]>([]);
  const [aiModel, setAiModel] = useState('claude-sonnet');

  // Pack-level execution config (applies to all triggers)
  const [execConfig, setExecConfig] = useState<Record<string, ExecTypeConfig>>({
    C: { enabled: true, referenceBar: 0, orderType: 'market' },
    L: { enabled: true, referenceBar: -1, orderType: 'market', holdSeconds: 0 },
    LC: { enabled: false, referenceBar: -1, orderType: 'market', confirmBarOffset: 0, bailAction: 'exit_market' },
    CC: { enabled: false, referenceBar: 0, orderType: 'market', confirmBarOffset: 1, bailAction: 'exit_market' },
  });

  // Request Fix modal (Step 5)
  const [showFixModal, setShowFixModal] = useState(false);
  const [fixDescription, setFixDescription] = useState('');
  const [fixContext, setFixContext] = useState('');
  const [fixCount, setFixCount] = useState(0);
  const [activeReviewTab, setActiveReviewTab] = useState('Overview');

  // Generated code state (populated by AI in Step 4)
  const [generatedManifest, setGeneratedManifest] = useState<Record<string, unknown> | null>(null);
  const [generatedIndicator, setGeneratedIndicator] = useState('');
  const [generatedInterpreter, setGeneratedInterpreter] = useState('');
  const [generatedPine, setGeneratedPine] = useState<string | null>(null);
  const [isInstalling, setIsInstalling] = useState(false);
  const [installResult, setInstallResult] = useState<{ success: boolean; slug: string; errors: string[] } | null>(null);

  // Sandbox backtest config
  const [sbSymbol, setSbSymbol] = useState('NVDA');
  const [sbTimeframe, setSbTimeframe] = useState('5Min');
  const [sbDirection, setSbDirection] = useState<'LONG' | 'SHORT'>('LONG');
  const [sbDays, setSbDays] = useState(30);
  const [sbEntryTrigger, setSbEntryTrigger] = useState('');
  const [sbExitTrigger, setSbExitTrigger] = useState('');
  const [sbStopPack, setSbStopPack] = useState('');
  const [sbTargetPack, setSbTargetPack] = useState('');
  const [sbHifi, setSbHifi] = useState(false);
  const [sbEqXAxis, setSbEqXAxis] = useState<'trade' | 'time'>('trade');
  const [sbZoomTrade, setSbZoomTrade] = useState<{ idx: number; side: 'entry' | 'exit'; trade: any } | null>(null);
  const [sbLastConfig, setSbLastConfig] = useState<any>(null);

  // AI mutation hooks (must be before any early returns per React rules)
  const generateStructureMutation = useGenerateStructure();
  const generateCodeMutation = useGenerateCode();
  const requestFixMutation = useRequestFix();
  const validatePackMutation = useValidatePack();
  const installPackMutation = useInstallPack();

  // Sandbox hooks
  const sbBacktestMut = useRunBacktest();
  const sbTradeZoomMut = useBacktestTradeZoom();
  const chartPrefs = useChartPrefs();
  const { data: sbEntryTriggers } = useConfluenceTriggers(sbDirection);
  const { data: sbExitTriggers } = useConfluenceTriggers('EXIT');
  const { data: sbStopPacks } = useStopLossPacks();
  const { data: sbTargetPacks } = useTakeProfitPacks();

  // Derived
  const slug = useMemo(() => toSlug(packName), [packName]);
  const categories = packType === 'tf_confluence' ? TF_CATEGORIES : GEN_CATEGORIES;
  const canProceed1 = packName.trim().length > 0 && category.length > 0 && description.trim().length > 0;

  // Sandbox derived state
  const sbPackTriggers = useMemo(() => {
    if (!sbEntryTriggers || !installResult?.slug) return [];
    return Object.entries(sbEntryTriggers)
      .filter(([id]) => id.includes(installResult.slug))
      .map(([id, name]) => ({ id, name: String(name) }));
  }, [sbEntryTriggers, installResult]);

  const sbExitTriggerList = useMemo(() => {
    if (!sbExitTriggers) return [];
    return Object.entries(sbExitTriggers).map(([id, name]) => ({ id, name: String(name) }));
  }, [sbExitTriggers]);

  const sbStopPackList = useMemo(() => {
    if (!sbStopPacks) return [];
    return (sbStopPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [sbStopPacks]);

  const sbTargetPackList = useMemo(() => {
    if (!sbTargetPacks) return [];
    return (sbTargetPacks as any[]).map((p) => ({ id: p.id, label: `${p.base_template} (${p.version})` }));
  }, [sbTargetPacks]);

  const sbResult = useMemo(() => {
    if (!sbBacktestMut.data) return null;
    const d = sbBacktestMut.data;
    return {
      trades: (d.trades || []).map((t: any, i: number) => ({
        id: i + 1,
        entryTime: t.entry_time ? new Date(t.entry_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        exitTime: t.exit_time ? new Date(t.exit_time).toLocaleString('en-US', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
        direction: (t.direction || 'LONG') as 'LONG' | 'SHORT',
        entryPrice: t.entry_price || 0, exitPrice: t.exit_price || 0,
        rMultiple: t.r_multiple || 0,
        execType: ({ HM: 'LC', HL: 'LC', L0: 'L', L1: 'L' } as Record<string, string>)[t.exec_type] || t.exec_type || 'C',
        exitReason: t.exit_reason || '',
      })),
      kpis: {
        winRate: d.kpis?.win_rate ?? 0, profitFactor: d.kpis?.profit_factor ?? 0,
        dailyR: d.kpis?.daily_r ?? 0, totalTrades: d.kpis?.total_trades ?? 0,
        totalR: d.kpis?.total_r ?? 0, avgR: d.kpis?.avg_r ?? 0,
        rSquared: d.kpis?.r_squared ?? 0, maxRDrawdown: d.kpis?.max_r_drawdown ?? 0,
      },
      equityCurve: d.equity_curve || [],
      chartData: d.chart_data,
      rawTrades: d.trades || [],
      overlay_indicators: (d as any).overlay_indicators || [],
      oscillator_indicators: (d as any).oscillator_indicators || [],
      heatmap_conditions: (d as any).heatmap_conditions || [],
    };
  }, [sbBacktestMut.data]);

  const sbChartPanes = useMemo(() => {
    if (!sbResult?.chartData || sbResult.chartData.length === 0) return [];
    const tfMs = (() => {
      const tf = sbTimeframe;
      if (tf.includes('Min')) return parseInt(tf) * 60 * 1000;
      if (tf.includes('H')) return 3600 * 1000;
      if (tf.includes('D')) return 86400 * 1000;
      return 60000;
    })();
    return buildStrategyChartPanes({
      bars: sbResult.chartData,
      trades: sbResult.rawTrades,
      direction: sbDirection,
      overlayNames: sbResult.overlay_indicators,
      oscNames: sbResult.oscillator_indicators,
      heatmapConds: sbResult.heatmap_conditions,
      tfMs,
      chartPrefs,
    });
  }, [sbResult, sbDirection, sbTimeframe, chartPrefs]);

  const STEPS = ['Pack Info', 'Generate Structure', 'Refine Structure', 'Generate & Validate Code', 'Review & Install'];

  function handleGenerateStructure() {
    setIsGenerating(true);
    setConversation([{ role: 'system', content: 'Generating pack structure from description...', time: new Date().toLocaleTimeString() }]);

    generateStructureMutation.mutate({
      pack_name: packName,
      pack_type: packType,
      category,
      display_type: displayType,
      description,
      pine_script: pineScript,
      ai_model: aiModel,
    }, {
      onSuccess: (data) => {
        setParams(data.parameters.map((p) => ({
          id: generateId(),
          name: p.name,
          label: p.label || p.name,
          type: p.type || 'int',
          defaultVal: String(p.default ?? ''),
          min: p.min != null ? String(p.min) : '',
          max: p.max != null ? String(p.max) : '',
        })));
        setOutputs(data.outputs.map((o) => ({
          id: generateId(),
          code: o.code,
          description: o.description || '',
        })));
        setTriggers(data.triggers.map((t) => ({
          id: generateId(),
          name: t.name,
          base: t.base,
          sentiment: (t.sentiment === 'bullish' || t.sentiment === 'bearish' || t.sentiment === 'neutral' ? t.sentiment : 'neutral') as 'bullish' | 'bearish' | 'neutral',
          fromState: '',
          toState: '',
        })));
        setStructureGenerated(true);
        setIsGenerating(false);
        setConversation((prev) => [...prev,
          { role: 'assistant', content: data.ai_message, time: new Date().toLocaleTimeString() },
        ]);
      },
      onError: (error) => {
        setIsGenerating(false);
        setConversation((prev) => [...prev,
          { role: 'error', content: `Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`, time: new Date().toLocaleTimeString() },
        ]);
      },
    });
  }

  function handleGenerateCode() {
    setIsGeneratingCode(true);
    setConversation((prev) => [...prev,
      { role: 'system', content: 'Generating code from refined structure...', time: new Date().toLocaleTimeString() },
    ]);

    generateCodeMutation.mutate({
      pack_name: packName,
      slug,
      pack_type: packType,
      category,
      display_type: displayType,
      description,
      parameters: params.map((p) => ({ name: p.name, type: p.type, default: p.defaultVal, min: p.min, max: p.max, label: p.label })),
      outputs: outputs.map((o) => ({ code: o.code, description: o.description })),
      triggers: triggers.map((t) => ({ name: t.name, base: t.base, sentiment: t.sentiment, direction: t.sentiment === 'bullish' ? 'LONG' : t.sentiment === 'bearish' ? 'SHORT' : 'BOTH', type: 'ENTRY', execution: 'bar_close' })),
      pine_script: pineScript,
      ai_model: aiModel,
    }, {
      onSuccess: (data) => {
        setGeneratedManifest(data.manifest);
        setGeneratedIndicator(data.indicator_code);
        setGeneratedInterpreter(data.interpreter_code);
        setGeneratedPine(data.pine_script_code);
        setCodeGenerated(true);
        setIsGeneratingCode(false);
        setValidation(data.validation.map((v) => ({ ...v, status: v.status as 'pass' | 'fail' | 'warn' | 'pending' })));
        setConversation((prev) => [...prev,
          { role: 'assistant', content: data.ai_message, time: new Date().toLocaleTimeString() },
        ]);
      },
      onError: (error) => {
        setIsGeneratingCode(false);
        setConversation((prev) => [...prev,
          { role: 'error', content: `Code generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`, time: new Date().toLocaleTimeString() },
        ]);
      },
    });
  }

  function handleAutoFix() {
    if (!generatedManifest || !generatedIndicator || !generatedInterpreter) return;

    const failedChecks = validation.filter((v) => v.status === 'fail' || v.status === 'warn');
    setIsAutoFixing(true);
    setConversation((prev) => [...prev,
      { role: 'error', content: `Issues found: ${failedChecks.map((v) => v.label).join(', ')}`, time: new Date().toLocaleTimeString() },
      { role: 'system', content: `Auto-fix attempt ${autoFixAttempt + 1}/3 — sending corrections to AI...`, time: new Date().toLocaleTimeString() },
    ]);

    requestFixMutation.mutate({
      manifest: generatedManifest,
      indicator_code: generatedIndicator,
      interpreter_code: generatedInterpreter,
      validation_errors: failedChecks.map((v) => `${v.label}: ${v.message}`),
      ai_model: aiModel,
    }, {
      onSuccess: (data) => {
        setGeneratedManifest(data.manifest);
        setGeneratedIndicator(data.indicator_code);
        setGeneratedInterpreter(data.interpreter_code);
        if (data.pine_script_code) setGeneratedPine(data.pine_script_code);
        setAutoFixAttempt((prev) => prev + 1);
        setIsAutoFixing(false);
        setValidation(data.validation.map((v) => ({ ...v, status: v.status as 'pass' | 'fail' | 'warn' | 'pending' })));
        setConversation((prev) => [...prev,
          { role: 'assistant', content: data.ai_message, time: new Date().toLocaleTimeString() },
        ]);
      },
      onError: (error) => {
        setAutoFixAttempt((prev) => prev + 1);
        setIsAutoFixing(false);
        setConversation((prev) => [...prev,
          { role: 'error', content: `Auto-fix failed: ${error instanceof Error ? error.message : 'Unknown error'}`, time: new Date().toLocaleTimeString() },
        ]);
      },
    });
  }

  function handleSbRunBacktest() {
    if (!sbEntryTrigger) return;
    const req: any = {
      symbol: sbSymbol,
      timeframe: sbTimeframe,
      direction: sbDirection,
      days: sbDays,
      entry_trigger_confluence_id: sbEntryTrigger,
      exit_trigger_confluence_ids: sbExitTrigger ? [sbExitTrigger] : [],
      stop_loss_pack_id: sbStopPack || undefined,
      take_profit_pack_id: sbTargetPack || undefined,
      hifi_mode: sbHifi,
      include_chart_data: true,
    };
    setSbLastConfig(req);
    sbBacktestMut.mutate(req);
  }

  return (
    <div>
      <PageHeader title="Pack Builder" subtitle="Create custom confluence packs with AI assistance" />

      {/* ---- Wizard Stepper ---- */}
      <div className="flex items-center mb-6">
        {STEPS.map((s, i) => {
          const stepNum = i + 1;
          const isComplete = step > stepNum;
          const isCurrent = step === stepNum;
          return (
            <div key={i} className="flex items-center flex-1">
              <button className="flex items-center gap-2" style={{ cursor: isComplete ? 'pointer' : 'default', background: 'none', border: 'none', padding: 0 }}
                onClick={() => isComplete && setStep(stepNum)}>
                <span className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0" style={{
                  background: isComplete ? 'var(--green)' : isCurrent ? 'var(--accent)' : 'var(--bg-input)',
                  color: isComplete || isCurrent ? 'white' : 'var(--text-muted)',
                }}>{isComplete ? '✓' : stepNum}</span>
                <span className="text-xs font-medium hidden lg:inline" style={{ color: isCurrent ? 'var(--accent)' : isComplete ? 'var(--text-primary)' : 'var(--text-muted)' }}>{s}</span>
              </button>
              {i < STEPS.length - 1 && <div className="flex-1 h-0.5 mx-2" style={{ background: isComplete ? 'var(--green)' : 'var(--border)' }} />}
            </div>
          );
        })}
      </div>

      {/* ================================================================= */}
      {/* STEP 1: PACK INFO (easy fields only)                              */}
      {/* ================================================================= */}
      {step === 1 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card>
              {/* Pack type */}
              <p className="text-sm font-medium mb-3">Pack Type</p>
              <div className="grid grid-cols-2 gap-3 mb-5">
                {([
                  { type: 'tf_confluence' as PackType, name: 'TF Confluence', desc: 'Indicator-based with state classifiers and trade triggers' },
                  { type: 'general' as PackType, name: 'General', desc: 'Scalar time/session-based conditions' },
                ]).map((opt) => (
                  <button key={opt.type} className="text-left p-4 rounded-lg" style={{
                    background: packType === opt.type ? 'var(--accent-muted)' : 'var(--bg-input)',
                    border: packType === opt.type ? '2px solid var(--accent)' : '2px solid var(--border)', cursor: 'pointer',
                  }} onClick={() => { setPackType(opt.type); setCategory(''); }}>
                    <p className="text-sm font-semibold" style={{ color: packType === opt.type ? 'var(--accent)' : 'var(--text-primary)' }}>{opt.name}</p>
                    <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{opt.desc}</p>
                  </button>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Pack Name</label>
                  <input type="text" value={packName} onChange={(e) => setPackName(e.target.value)} placeholder="RSI Zones" style={inputStyle} />
                  {slug && <p className="text-[10px] font-mono mt-1" style={{ color: 'var(--text-muted)' }}>Slug: {slug}</p>}
                </div>
                <div>
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Category</label>
                  <div className="flex flex-wrap gap-1">
                    {categories.map((c) => (
                      <button key={c} className="text-xs px-2.5 py-1 rounded-full" style={{
                        background: category === c ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: category === c ? 'var(--accent)' : 'var(--text-muted)',
                        border: category === c ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer',
                      }} onClick={() => setCategory(c)}>{c}</button>
                    ))}
                  </div>
                </div>
              </div>

              {packType === 'tf_confluence' && (
                <div className="mb-4">
                  <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Display Type</label>
                  <div className="flex gap-2">
                    {DISPLAY_TYPES.map((dt) => (
                      <button key={dt} className="text-xs px-3 py-1.5 rounded-lg capitalize" style={{
                        background: displayType === dt ? 'var(--accent-muted)' : 'var(--bg-input)',
                        color: displayType === dt ? 'var(--accent)' : 'var(--text-muted)',
                        border: displayType === dt ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer',
                      }} onClick={() => setDisplayType(dt)}>{dt}</button>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <label className="text-xs block mb-1" style={{ color: 'var(--text-muted)' }}>Describe what this pack should do</label>
                <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={4}
                  placeholder="I want an RSI indicator that classifies bars into overbought, neutral, and oversold zones. It should fire triggers when RSI crosses above the overbought level or below the oversold level..."
                  style={{ ...inputStyle, resize: 'vertical' }} />
                <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                  Be specific about what states you want, what triggers should fire, and any parameters the user should be able to customize. The AI will propose the full structure from your description.
                </p>
              </div>

              {/* Optional Pine Script */}
              <details className="mt-4">
                <summary className="text-xs font-medium cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                  Optional: Paste TradingView Pine Script for translation reference
                </summary>
                <textarea value={pineScript} onChange={(e) => setPineScript(e.target.value)} rows={5} placeholder="// Paste your Pine Script v5 code here..."
                  className="w-full mt-2 font-mono" style={{ ...inputStyle, fontSize: '0.75rem' }} />
              </details>
            </Card>
          </div>

          {/* Info panel */}
          <Card>
            <h4 className="text-sm font-medium mb-3">How it works</h4>
            <div className="space-y-3 text-xs" style={{ color: 'var(--text-muted)' }}>
              <div className="flex gap-2">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>1</span>
                <p><strong>Describe</strong> — Tell us what you want in plain language</p>
              </div>
              <div className="flex gap-2">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>2</span>
                <p><strong>Structure</strong> — AI proposes parameters, outputs, and triggers</p>
              </div>
              <div className="flex gap-2">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>3</span>
                <p><strong>Refine</strong> — Tweak the proposed structure as needed</p>
              </div>
              <div className="flex gap-2">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>4</span>
                <p><strong>Generate</strong> — AI writes the code, system validates it</p>
              </div>
              <div className="flex gap-2">
                <span className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0" style={{ background: 'var(--accent)', color: 'white' }}>5</span>
                <p><strong>Install</strong> — Preview on real data, verify parity, install</p>
              </div>
              <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--border)' }}>
                {packType === 'tf_confluence' ? (
                  <>
                    <p><strong>Exec types:</strong> <ExecBadge tag="[C]" /> <ExecBadge tag="[L]" /> <ExecBadge tag="[LC]" /> <ExecBadge tag="[CC]" /></p>
                    <p className="mt-1"><strong>Fidelity:</strong> <FidelityBadge tag="[PB]" /> <FidelityBadge tag="[CB]" /></p>
                    <p className="mt-1"><strong>Files:</strong> manifest.json + indicator.py + interpreter.py</p>
                  </>
                ) : (
                  <>
                    <p><strong>Exec:</strong> <ExecBadge tag="[C]" /> only</p>
                    <p className="mt-1"><strong>States:</strong> Binary states (IN/OUT)</p>
                    <p className="mt-1"><strong>Files:</strong> manifest.json + evaluator.py</p>
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 2: GENERATE STRUCTURE                                        */}
      {/* ================================================================= */}
      {step === 2 && (
        <div className="grid grid-cols-12 gap-4">
          {/* Left: AI Conversation */}
          <div className="col-span-4">
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>AI Conversation</h4>
                <select value={aiModel} onChange={(e) => setAiModel(e.target.value)}
                  className="text-[10px] px-2 py-1 rounded" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                  <option value="claude-sonnet">Claude Sonnet</option>
                  <option value="claude-opus">Claude Opus</option>
                  <option value="gpt-4">GPT-4</option>
                  <option value="gpt-4o">GPT-4o</option>
                </select>
              </div>
              <div className="space-y-2" style={{ maxHeight: 350, overflowY: 'auto' }}>
                {conversation.length === 0 && (
                  <p className="text-xs py-4 text-center" style={{ color: 'var(--text-muted)' }}>Click Generate to start</p>
                )}
                {conversation.map((msg, i) => (
                  <div key={i} className="rounded-lg px-3 py-2" style={{
                    background: msg.role === 'assistant' ? 'var(--accent-muted)' : msg.role === 'error' ? 'var(--red-muted)' : 'var(--bg-input)',
                    borderLeft: msg.role === 'assistant' ? '3px solid var(--accent)' : msg.role === 'error' ? '3px solid var(--red)' : '3px solid var(--border)',
                  }}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[10px] font-medium capitalize" style={{ color: msg.role === 'error' ? 'var(--red)' : 'var(--text-muted)' }}>{msg.role}</span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
                    </div>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{msg.content}</p>
                  </div>
                ))}
                {isGenerating && (
                  <div className="flex items-center gap-2 px-3 py-2">
                    <div className="w-3 h-3 rounded-full border border-t-transparent animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
                    <span className="text-xs" style={{ color: 'var(--accent)' }}>Generating...</span>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Right: Summary + Generated structure */}
          <div className="col-span-8">
            <Card className="mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold">{packName}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{packType === 'tf_confluence' ? 'TF Confluence' : 'General'} &middot; {category} &middot; {displayType}</p>
                </div>
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{slug}</span>
              </div>
              <p className="text-xs mt-2" style={{ color: 'var(--text-secondary)' }}>{description}</p>
            </Card>

            {!structureGenerated ? (
              <Card>
                <div className="text-center py-8">
                  <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Ready to generate pack structure via AI</p>
                  <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                    The AI will analyze your description and propose parameters, states, and triggers. You can refine everything in the next step.
                  </p>
                  <button style={btnPrimary} onClick={handleGenerateStructure} disabled={isGenerating}>
                    {isGenerating ? 'Generating...' : 'Generate Structure'}
                  </button>
                </div>
              </Card>
            ) : (
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-sm font-medium" style={{ color: 'var(--green)' }}>Structure Generated</h4>
                  <button style={{ ...btnSecondary, fontSize: '0.75rem' }} onClick={handleGenerateStructure}>Regenerate</button>
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <p className="text-xs font-medium mb-2">{params.length} Parameters</p>
                    {params.map((p) => <p key={p.id} className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{p.name}: {p.type} = {p.defaultVal}</p>)}
                  </div>
                  <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <p className="text-xs font-medium mb-2">{outputs.length} States</p>
                    {outputs.map((o) => <p key={o.id} className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{o.code}</p>)}
                  </div>
                  <div className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                    <p className="text-xs font-medium mb-2">{triggers.length} Triggers</p>
                    {triggers.map((t) => (
                      <div key={t.id} className="flex items-center gap-1 mb-1">
                        <SentimentBadge sentiment={t.sentiment} />
                        <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{t.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 3: REFINE STRUCTURE                                          */}
      {/* ================================================================= */}
      {step === 3 && (
        <div className="space-y-4">
          {/* Parameters */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Parameters <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({params.length})</span></h4>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }}
                onClick={() => setParams([...params, { id: generateId(), name: '', label: '', type: 'int', defaultVal: '', min: '', max: '' }])}>+ Add</button>
            </div>
            <div className="space-y-2">
              {params.map((p, i) => (
                <div key={p.id} className="grid grid-cols-6 gap-2 items-end">
                  <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Name</label><input type="text" value={p.name} onChange={(e) => { const n = [...params]; n[i].name = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} /></div>
                  <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Label</label><input type="text" value={p.label} onChange={(e) => { const n = [...params]; n[i].label = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} /></div>
                  <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Type</label><select value={p.type} onChange={(e) => { const n = [...params]; n[i].type = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }}><option value="int">int</option><option value="float">float</option>{packType === 'general' && <><option value="bool">bool</option><option value="select">select</option></>}</select></div>
                  <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Default</label><input type="text" value={p.defaultVal} onChange={(e) => { const n = [...params]; n[i].defaultVal = e.target.value; setParams(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} /></div>
                  <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Min / Max</label><div className="flex gap-1"><input type="text" value={p.min} onChange={(e) => { const n = [...params]; n[i].min = e.target.value; setParams(n); }} placeholder="min" style={{ ...inputStyle, fontSize: '0.75rem', width: '50%' }} /><input type="text" value={p.max} onChange={(e) => { const n = [...params]; n[i].max = e.target.value; setParams(n); }} placeholder="max" style={{ ...inputStyle, fontSize: '0.75rem', width: '50%' }} /></div></div>
                  <button className="text-xs" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }} onClick={() => setParams(params.filter((_, j) => j !== i))}>Remove</button>
                </div>
              ))}
            </div>
          </Card>

          {/* Outputs */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">States <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({outputs.length})</span></h4>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }} onClick={() => setOutputs([...outputs, { id: generateId(), code: '', description: '' }])}>+ Add</button>
            </div>
            <div className="space-y-2">
              {outputs.map((o, i) => (
                <div key={o.id} className="flex items-center gap-3">
                  <input type="text" value={o.code} onChange={(e) => { const n = [...outputs]; n[i].code = e.target.value.toUpperCase(); setOutputs(n); }} placeholder="STATE_NAME" className="font-mono" style={{ ...inputStyle, fontSize: '0.75rem', width: 160 }} />
                  <input type="text" value={o.description} onChange={(e) => { const n = [...outputs]; n[i].description = e.target.value; setOutputs(n); }} placeholder="Description..." style={{ ...inputStyle, fontSize: '0.75rem', flex: 1 }} />
                  <button className="text-xs" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }} onClick={() => setOutputs(outputs.filter((_, j) => j !== i))}>Remove</button>
                </div>
              ))}
            </div>
          </Card>

          {/* Triggers */}
          <Card>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Triggers <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({triggers.length})</span></h4>
              <button style={{ ...btnSecondary, fontSize: '0.75rem', padding: '4px 10px' }} onClick={() => setTriggers([...triggers, { id: generateId(), name: '', base: '', sentiment: 'neutral', fromState: '', toState: '' }])}>+ Add</button>
            </div>
            <div className="space-y-3">
              {triggers.map((t, i) => (
                <div key={t.id} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
                  <div className="grid grid-cols-3 gap-3 mb-2">
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Name</label>
                      <input type="text" value={t.name} onChange={(e) => { const n = [...triggers]; n[i].name = e.target.value; n[i].base = toSlug(e.target.value); setTriggers(n); }} style={{ ...inputStyle, fontSize: '0.75rem' }} />
                      {t.base && <p className="text-[10px] font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>Base: {t.base}</p>}
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Sentiment</label>
                      <div className="flex gap-1">
                        {SENTIMENTS.map((s) => (
                          <button key={s} className="text-[10px] px-2 py-1 rounded capitalize" style={{
                            background: t.sentiment === s ? (s === 'bullish' ? 'var(--green-muted)' : s === 'bearish' ? 'var(--red-muted)' : 'var(--bg-card)') : 'transparent',
                            color: t.sentiment === s ? (s === 'bullish' ? 'var(--green)' : s === 'bearish' ? 'var(--red)' : 'var(--text-muted)') : 'var(--text-muted)',
                            cursor: 'pointer', border: t.sentiment === s ? 'none' : '1px solid var(--border)',
                          }} onClick={() => { const n = [...triggers]; n[i].sentiment = s; setTriggers(n); }}>{s}</button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>State Transition</label>
                      <div className="flex items-center gap-1">
                        <select value={t.fromState} onChange={(e) => { const n = [...triggers]; n[i].fromState = e.target.value; setTriggers(n); }} style={{ ...inputStyle, fontSize: '0.7rem', width: '45%' }}>
                          <option value="">From...</option>{outputs.map((o) => <option key={o.id} value={o.code}>{o.code}</option>)}
                        </select>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>→</span>
                        <select value={t.toState} onChange={(e) => { const n = [...triggers]; n[i].toState = e.target.value; setTriggers(n); }} style={{ ...inputStyle, fontSize: '0.7rem', width: '45%' }}>
                          <option value="">To...</option>{outputs.map((o) => <option key={o.id} value={o.code}>{o.code}</option>)}
                        </select>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center justify-end mt-1">
                    <button className="text-xs" style={{ color: 'var(--red)', cursor: 'pointer', background: 'transparent', border: 'none' }}
                      onClick={() => setTriggers(triggers.filter((_, j) => j !== i))}>Remove</button>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Execution Defaults (pack-level — applies to all triggers) */}
          {packType === 'tf_confluence' && (
            <Card>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="text-sm font-medium">Execution Defaults</h4>
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                    These settings apply to all triggers in this pack. Users can override per-variation on the TF Confluence page.
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {(['C', 'L', 'LC', 'CC'] as const).map((et) => {
                  const cfg = execConfig[et] as ExecTypeConfig;
                  const label = et === 'C' ? 'Bar Close' : et === 'L' ? 'Level' : et === 'LC' ? 'Level-Close' : 'Close-Close';
                  const updateField = (field: string, value: unknown) => {
                    setExecConfig((prev) => ({ ...prev, [et]: { ...prev[et], [field]: value } }));
                  };
                  return (
                    <div key={et} className="rounded-lg p-3" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', opacity: cfg.enabled ? 1 : 0.5 }}>
                      <div className="flex items-center gap-2 mb-2">
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <input type="checkbox" checked={cfg.enabled} onChange={(e) => updateField('enabled', e.target.checked)}
                            className="w-3.5 h-3.5" style={{ accentColor: EXEC_BADGE_COLOR }} />
                          <ExecBadge tag={`[${et}]`} />
                          <span className="text-xs font-medium">{label}</span>
                        </label>
                      </div>
                      {cfg.enabled && (
                        <div className="grid grid-cols-2 gap-2 mt-2">
                          <div>
                            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Reference Bar</label>
                            <select value={cfg.referenceBar} onChange={(e) => updateField('referenceBar', Number(e.target.value))}
                              style={{ ...inputStyle, fontSize: '0.7rem' }}>
                              <option value={0}>Current (0)</option>
                              <option value={-1}>Previous (-1)</option>
                            </select>
                          </div>
                          <div>
                            <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Order Type</label>
                            <select value={cfg.orderType} onChange={(e) => updateField('orderType', e.target.value)}
                              style={{ ...inputStyle, fontSize: '0.7rem' }}>
                              <option value="market">Market</option>
                              {et !== 'CC' && <option value="limit">Limit</option>}
                            </select>
                          </div>
                          {(et === 'L' || et === 'LC') && (
                            <div>
                              <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Hold Seconds</label>
                              <input type="number" value={cfg.holdSeconds ?? 0} min={0}
                                onChange={(e) => updateField('holdSeconds', Number(e.target.value))}
                                style={{ ...inputStyle, fontSize: '0.7rem' }} />
                            </div>
                          )}
                          {(et === 'LC' || et === 'CC') && (
                            <>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Confirm Offset</label>
                                <select value={cfg.confirmBarOffset ?? (et === 'CC' ? 1 : 0)}
                                  onChange={(e) => updateField('confirmBarOffset', Number(e.target.value))}
                                  style={{ ...inputStyle, fontSize: '0.7rem' }}>
                                  <option value={0}>Same bar</option>
                                  <option value={1}>Next bar</option>
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Bail Action</label>
                                <select value={cfg.bailAction ?? 'exit_market'}
                                  onChange={(e) => updateField('bailAction', e.target.value)}
                                  style={{ ...inputStyle, fontSize: '0.7rem' }}>
                                  <option value="exit_market">Exit at market</option>
                                  {et === 'LC' && <option value="exit_limit">Exit at limit</option>}
                                </select>
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 4: GENERATE & VALIDATE CODE                                  */}
      {/* ================================================================= */}
      {step === 4 && (
        <div className="grid grid-cols-12 gap-4">
          {/* Left: AI Conversation */}
          <div className="col-span-3">
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>AI Conversation</h4>
                {autoFixAttempt > 0 && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: 'var(--orange)' + '20', color: 'var(--orange)' }}>
                    Fix {autoFixAttempt}/3
                  </span>
                )}
              </div>
              <div className="space-y-2" style={{ maxHeight: 450, overflowY: 'auto' }}>
                {conversation.map((msg, i) => (
                  <div key={i} className="rounded-lg px-3 py-2" style={{
                    background: msg.role === 'assistant' ? 'var(--accent-muted)' : msg.role === 'error' ? 'var(--red-muted)' : 'var(--bg-input)',
                    borderLeft: msg.role === 'assistant' ? '3px solid var(--accent)' : msg.role === 'error' ? '3px solid var(--red)' : '3px solid var(--border)',
                  }}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[10px] font-medium capitalize" style={{ color: msg.role === 'error' ? 'var(--red)' : 'var(--text-muted)' }}>{msg.role}</span>
                      <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{msg.time}</span>
                    </div>
                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{msg.content}</p>
                  </div>
                ))}
                {(isGeneratingCode || isAutoFixing) && (
                  <div className="flex items-center gap-2 px-3 py-2">
                    <div className="w-3 h-3 rounded-full border border-t-transparent animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
                    <span className="text-xs" style={{ color: 'var(--accent)' }}>{isAutoFixing ? 'Applying fix...' : 'Generating code...'}</span>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Center: Code preview */}
          <div className="col-span-5">
            {!codeGenerated && !isGeneratingCode ? (
              <Card>
                <div className="text-center py-12">
                  <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Ready to generate code</p>
                  <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                    The AI will generate manifest.json, indicator.py, and interpreter.py based on your refined structure.
                  </p>
                  <button style={btnPrimary} onClick={handleGenerateCode}>Generate Code</button>
                </div>
              </Card>
            ) : (
              <Card>
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium">Generated Code</h4>
                  <button style={{ ...btnSecondary, fontSize: '0.75rem' }} onClick={handleGenerateCode}>Regenerate</button>
                </div>
                <TabBar tabs={packType === 'tf_confluence' ? ['manifest.json', 'indicator.py', 'interpreter.py'] : ['manifest.json', 'evaluator.py']}>
                  {(tab) => (
                    <div className="rounded-lg p-4 font-mono text-xs" style={{ background: '#0d1117', color: '#c9d1d9', maxHeight: 350, overflowY: 'auto', lineHeight: 1.6, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {tab === 'manifest.json' && <pre>{generatedManifest ? JSON.stringify(generatedManifest, null, 2) : '// Generating...'}</pre>}
                      {tab === 'indicator.py' && <pre>{generatedIndicator || '# Generating...'}</pre>}
                      {tab === 'interpreter.py' && <pre>{generatedInterpreter || '# Generating...'}</pre>}
                      {tab === 'evaluator.py' && <pre>{generatedIndicator || '# Generating...'}</pre>}
                    </div>
                  )}
                </TabBar>
              </Card>
            )}
          </div>

          {/* Right: Validation */}
          <div className="col-span-4">
            <Card>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium">Validation</h4>
                {validation.length > 0 && (
                  <span className="text-xs font-bold" style={{ color: validation.filter(v => v.status === 'fail').length === 0 ? 'var(--green)' : 'var(--red)' }}>
                    {validation.filter(v => v.status === 'pass').length}/{validation.length}
                  </span>
                )}
              </div>
              {validation.length === 0 ? (
                <p className="text-xs py-8 text-center" style={{ color: 'var(--text-muted)' }}>Generate code to run validation</p>
              ) : (
                <div className="space-y-3">
                  {['schema', 'safety', 'functions', 'execution', 'backtest'].map((cat) => {
                    const items = validation.filter(v => v.category === cat);
                    if (items.length === 0) return null;
                    return (
                      <div key={cat}>
                        <p className="text-xs font-medium mb-1 capitalize" style={{ color: 'var(--text-secondary)' }}>{cat}</p>
                        <div className="space-y-1">
                          {items.map((v) => (
                            <div key={v.id} className="flex items-start gap-2 text-xs">
                              <StatusIcon status={v.status} />
                              <div>
                                <span style={{ color: 'var(--text-secondary)' }}>{v.label}</span>
                                <p className="text-[10px]" style={{ color: v.status === 'fail' ? 'var(--red)' : v.status === 'warn' ? 'var(--orange)' : 'var(--text-muted)' }}>{v.message}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                  {/* Auto-fix button */}
                  {validation.some(v => v.status === 'fail' || v.status === 'warn') && autoFixAttempt < 3 && (
                    <button className="w-full mt-2" style={{ ...btnPrimary, background: 'var(--orange)', fontSize: '0.75rem' }}
                      onClick={handleAutoFix} disabled={isAutoFixing}>
                      {isAutoFixing ? 'Fixing...' : `Auto-Fix Issues (${autoFixAttempt}/3)`}
                    </button>
                  )}
                  {autoFixAttempt >= 3 && validation.some(v => v.status === 'fail') && (
                    <p className="text-xs mt-2 text-center" style={{ color: 'var(--red)' }}>
                      Max auto-fix attempts reached. Use Request Fix in Step 5 for manual corrections.
                    </p>
                  )}
                </div>
              )}
            </Card>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* STEP 5: REVIEW & INSTALL                                          */}
      {/* ================================================================= */}
      {step === 5 && (
        <div>
          {/* Request Fix banner */}
          {fixCount > 0 && (
            <div className="flex items-center gap-2 mb-3 px-3 py-2 rounded-lg" style={{ background: 'var(--accent-muted)', border: '1px solid var(--accent)30' }}>
              <span className="text-xs" style={{ color: 'var(--accent)' }}>Fix iteration {fixCount} applied</span>
            </div>
          )}

          <TabBar tabs={['Overview', 'Chart Preview', 'Signal Validation', 'Parity Simulator', 'Sandbox', 'Code']}>
            {(tab) => {
              // Track active tab for Request Fix context
              if (tab !== activeReviewTab) setActiveReviewTab(tab);
              return (
              <div>
                {tab === 'Overview' && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card>
                      <h4 className="text-sm font-medium mb-3">Pack Summary</h4>
                      <div className="space-y-2 text-xs">
                        {[
                          { label: 'Name', value: packName || 'RSI Zones' },
                          { label: 'Slug', value: slug || 'rsi_zones' },
                          { label: 'Type', value: packType === 'tf_confluence' ? 'TF Confluence' : 'General' },
                          { label: 'Category', value: category || 'Momentum' },
                          { label: 'Display', value: packType === 'tf_confluence' ? displayType : 'hidden' },
                        ].map((r) => (
                          <div key={r.label} className="flex justify-between"><span style={{ color: 'var(--text-muted)' }}>{r.label}</span><span className="font-medium">{r.value}</span></div>
                        ))}
                      </div>
                    </Card>
                    <Card>
                      <h4 className="text-sm font-medium mb-3">Triggers</h4>
                      <div className="space-y-2">
                        {triggers.map((t) => (
                          <div key={t.id} className="flex items-center gap-2 flex-wrap">
                            <SentimentBadge sentiment={t.sentiment} />
                            <span className="text-xs">{t.name}</span>
                            {t.fromState && t.toState && <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>{t.fromState} → {t.toState}</span>}
                          </div>
                        ))}
                      </div>
                    </Card>
                  </div>
                )}

                {tab === 'Chart Preview' && (
                  <Card>
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-sm font-medium">Confluence States & Trigger Visualization</h4>
                      <div className="flex gap-2">
                        <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                          <input type="checkbox" defaultChecked style={{ accentColor: 'var(--accent)' }} /> Show Confluence
                        </label>
                        <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                          <input type="checkbox" defaultChecked style={{ accentColor: 'var(--accent)' }} /> Show Triggers
                        </label>
                      </div>
                    </div>
                    <ChartPlaceholder label={`Price chart with ${packName || 'pack'} applied: background shading shows state changes (${outputs.map(o => o.code).join('/')}), arrow markers show trigger fires (${triggers.map(t => t.name).join(', ')}). Heatmap pane below showing state transitions.`} height={400} />
                    <div className="flex gap-4 mt-3 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {outputs.map((o, i) => (
                        <span key={o.id} className="flex items-center gap-1">
                          <span className="w-3 h-3 rounded" style={{ background: ['var(--red)', 'var(--text-muted)', 'var(--green)'][i % 3] + '30' }} />
                          {o.code}
                        </span>
                      ))}
                      {triggers.map((t) => (
                        <span key={t.id} className="flex items-center gap-1">
                          <span style={{ color: t.sentiment === 'bullish' ? 'var(--green)' : 'var(--red)' }}>{t.sentiment === 'bullish' ? '▲' : '▼'}</span>
                          {t.name}
                        </span>
                      ))}
                    </div>
                  </Card>
                )}

                {tab === 'Signal Validation' && (
                  <Card>
                    <h4 className="text-sm font-medium mb-2">Signal Validation</h4>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                      Runs the pack on 90 days of sample data to verify signals fire correctly, all states are reached, and trigger frequency is reasonable.
                    </p>
                    {/* {{ai_response}} — Signal validation KPIs will be populated after AI generates and validates code */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
                      {[
                        { label: 'Total Signals', value: '--' },
                        { label: 'Avg Bars Between', value: '--' },
                        { label: 'State Coverage', value: '--' },
                        { label: 'All States Reached', value: '--' },
                      ].map((m) => (
                        <div key={m.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                          <p className="text-lg font-bold">{m.value}</p>
                        </div>
                      ))}
                    </div>
                    <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Per-Trigger Breakdown</h5>
                    <div style={{ overflowX: 'auto' }}>
                      <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Trigger', 'Sentiment', 'Fires', 'Avg Bars Between', 'State Match'].map((h) => (
                              <th key={h} className="text-left py-2 px-3 text-[10px] font-medium" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {triggers.length === 0 ? (
                            <tr><td colSpan={5} className="py-4 px-3 text-center" style={{ color: 'var(--text-muted)' }}>No triggers defined yet</td></tr>
                          ) : triggers.map((t) => (
                            <tr key={t.id} style={{ borderBottom: '1px solid var(--border)' }}>
                              <td className="py-2 px-3">{t.name}</td>
                              <td className="py-2 px-3"><SentimentBadge sentiment={t.sentiment} /></td>
                              <td className="py-2 px-3">--</td>
                              <td className="py-2 px-3">--</td>
                              <td className="py-2 px-3"><span style={{ color: 'var(--text-muted)' }}>--</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <ChartPlaceholder label="Signal timeline: dots showing when each trigger fired over 90 days, colored by sentiment" height={120} />
                  </Card>
                )}

                {tab === 'Parity Simulator' && (
                  <Card>
                    <h4 className="text-sm font-medium mb-2">Backtest ↔ Live Parity Simulator</h4>
                    <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
                      Replays historical data bar-by-bar through both backtest and live engine paths. Compares trigger timing to verify they match.
                    </p>
                    <div className="flex items-center gap-4 mb-4">
                      <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Ticker</label><select style={{ ...inputStyle, width: 120, fontSize: '0.75rem' }}><option>NVDA</option><option>SPY</option><option>AAPL</option><option>TSLA</option></select></div>
                      <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Timeframe</label><select style={{ ...inputStyle, width: 100, fontSize: '0.75rem' }}><option>1Min</option><option>5Min</option><option>15Min</option></select></div>
                      <div><label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Bars</label><select style={{ ...inputStyle, width: 80, fontSize: '0.75rem' }}><option>200</option><option>500</option><option>1000</option></select></div>
                      <div className="flex items-end"><button style={btnPrimary}>Run Parity Test</button></div>
                    </div>

                    <ChartPlaceholder label="Bar-by-bar replay: candles build progressively. Backtest triggers (blue above) vs Live triggers (green below). Matched = green line. Mismatched = red highlight." height={300} />

                    {/* {{ai_response}} — Parity KPIs will be populated after running parity test */}
                    <div className="grid grid-cols-4 gap-4 mt-4 mb-4">
                      {[
                        { label: 'Total Triggers', value: '--' },
                        { label: 'Matched', value: '--' },
                        { label: 'Mismatched', value: '--' },
                        { label: 'Parity Score', value: '--' },
                      ].map((m) => (
                        <div key={m.label}>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.label}</p>
                          <p className="text-lg font-bold">{m.value}</p>
                        </div>
                      ))}
                    </div>

                    {/* {{ai_response}} — Timing table will be populated after parity test */}
                    <h5 className="text-xs font-medium mb-2" style={{ color: 'var(--text-muted)' }}>Trigger Timing Detail</h5>
                    <div style={{ overflowX: 'auto', maxHeight: 250, overflowY: 'auto' }}>
                      <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {['Bar #', 'Timestamp', 'Trigger', 'Backtest', 'Live', 'Match', 'Delta'].map((h) => (
                              <th key={h} className="text-left py-1.5 px-2 text-[10px] font-medium sticky top-0" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          <tr><td colSpan={7} className="py-4 px-2 text-center" style={{ color: 'var(--text-muted)' }}>Run parity test to see results</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </Card>
                )}

                {tab === 'Sandbox' && (
                  <div>
                    {!installResult?.success ? (
                      <Card>
                        <div className="text-center py-12">
                          <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Install the pack first to test it in the Sandbox.</p>
                          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>The backtest engine needs registered triggers to run. Click &quot;Install Pack&quot; below, then return to this tab.</p>
                        </div>
                      </Card>
                    ) : (
                      <div className="grid grid-cols-12 gap-4">
                        {/* Config Panel */}
                        <div className="col-span-12 lg:col-span-3">
                          <Card>
                            <h4 className="text-sm font-medium mb-3">Backtest Config</h4>
                            <div className="space-y-3">
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Symbol</label>
                                <input type="text" value={sbSymbol} onChange={(e) => setSbSymbol(e.target.value.toUpperCase())} style={inputStyle} />
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Timeframe</label>
                                <select value={sbTimeframe} onChange={(e) => setSbTimeframe(e.target.value)} style={inputStyle}>
                                  {SB_TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Direction</label>
                                <select value={sbDirection} onChange={(e) => setSbDirection(e.target.value as 'LONG' | 'SHORT')} style={inputStyle}>
                                  <option value="LONG">LONG</option>
                                  <option value="SHORT">SHORT</option>
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Lookback (days)</label>
                                <input type="number" value={sbDays} min={1} max={365} onChange={(e) => setSbDays(parseInt(e.target.value) || 30)} style={inputStyle} />
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Entry Trigger</label>
                                <select value={sbEntryTrigger} onChange={(e) => setSbEntryTrigger(e.target.value)} style={inputStyle}>
                                  <option value="">Select...</option>
                                  {sbPackTriggers.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                                </select>
                                {sbPackTriggers.length === 0 && <p className="text-[10px] mt-0.5" style={{ color: 'var(--orange)' }}>No triggers found. Try refreshing the page.</p>}
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Exit Trigger</label>
                                <select value={sbExitTrigger} onChange={(e) => setSbExitTrigger(e.target.value)} style={inputStyle}>
                                  <option value="">None (stop/target only)</option>
                                  {sbExitTriggerList.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Stop Loss Pack</label>
                                <select value={sbStopPack} onChange={(e) => setSbStopPack(e.target.value)} style={inputStyle}>
                                  <option value="">None</option>
                                  {sbStopPackList.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                                </select>
                              </div>
                              <div>
                                <label className="text-[10px] block mb-0.5" style={{ color: 'var(--text-muted)' }}>Take Profit Pack</label>
                                <select value={sbTargetPack} onChange={(e) => setSbTargetPack(e.target.value)} style={inputStyle}>
                                  <option value="">None</option>
                                  {sbTargetPackList.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                                </select>
                              </div>
                              <label className="flex items-center gap-2 text-xs cursor-pointer" style={{ color: 'var(--text-muted)' }}>
                                <input type="checkbox" checked={sbHifi} onChange={(e) => setSbHifi(e.target.checked)} style={{ accentColor: 'var(--accent)' }} />
                                Hi-Fi Mode
                              </label>
                              <button style={{ ...btnPrimary, width: '100%', opacity: (!sbEntryTrigger || sbBacktestMut.isPending) ? 0.5 : 1 }}
                                disabled={!sbEntryTrigger || sbBacktestMut.isPending}
                                onClick={handleSbRunBacktest}>
                                {sbBacktestMut.isPending ? 'Running...' : 'Run Backtest'}
                              </button>
                            </div>
                          </Card>
                        </div>

                        {/* Results Panel */}
                        <div className="col-span-12 lg:col-span-9 space-y-4">
                          {sbBacktestMut.isPending && (
                            <Card>
                              <div className="flex items-center gap-3 py-4">
                                <div className="w-4 h-4 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
                                <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Running backtest on {sbSymbol}...</span>
                              </div>
                            </Card>
                          )}

                          {sbBacktestMut.isError && (
                            <Card>
                              <p className="text-sm" style={{ color: 'var(--red)' }}>
                                Backtest failed: {sbBacktestMut.error instanceof Error ? sbBacktestMut.error.message : 'Unknown error'}
                              </p>
                            </Card>
                          )}

                          {!sbResult && !sbBacktestMut.isPending && !sbBacktestMut.isError && (
                            <Card>
                              <div className="text-center py-8">
                                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Configure your backtest and click <strong>Run Backtest</strong> to test the pack.</p>
                                <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{sbSymbol} | {sbDirection} | {sbTimeframe} | {sbDays} days</p>
                              </div>
                            </Card>
                          )}

                          {sbResult && (
                            <>
                              {/* KPI Strip */}
                              <div className="grid grid-cols-4 sm:grid-cols-4 xl:grid-cols-8 gap-2">
                                <MetricCard label="Trades" value={String(sbResult.kpis.totalTrades)} />
                                <MetricCard label="Win Rate" value={`${sbResult.kpis.winRate.toFixed(1)}%`} positive={sbResult.kpis.winRate > 50} />
                                <MetricCard label="PF" value={sbResult.kpis.profitFactor.toFixed(2)} positive={sbResult.kpis.profitFactor > 1} />
                                <MetricCard label="Avg R" value={`${sbResult.kpis.avgR >= 0 ? '+' : ''}${sbResult.kpis.avgR.toFixed(2)}`} positive={sbResult.kpis.avgR > 0} />
                                <MetricCard label="Total R" value={`${sbResult.kpis.totalR >= 0 ? '+' : ''}${sbResult.kpis.totalR.toFixed(1)}`} positive={sbResult.kpis.totalR > 0} />
                                <MetricCard label="Daily R" value={`${sbResult.kpis.dailyR >= 0 ? '+' : ''}${sbResult.kpis.dailyR.toFixed(2)}`} positive={sbResult.kpis.dailyR > 0} />
                                <MetricCard label="R²" value={sbResult.kpis.rSquared.toFixed(2)} positive={sbResult.kpis.rSquared > 0.7} />
                                <MetricCard label="Max DD" value={`${sbResult.kpis.maxRDrawdown.toFixed(1)}R`} positive={false} />
                              </div>

                              {/* Price Chart */}
                              <Card>
                                <div style={{ minHeight: 400 }}>
                                  {sbChartPanes.length > 0 ? (
                                    <SyncedChartPane
                                      panes={sbChartPanes}
                                      upColor={chartPrefs?.candleUp}
                                      downColor={chartPrefs?.candleDown}
                                      upBorderColor={chartPrefs?.candleUpBorder}
                                      gridLines={chartPrefs?.gridLines}
                                      rightOffset={chartPrefs?.rightOffset}
                                    />
                                  ) : (
                                    <ChartPlaceholder label="No chart data available" height={400} />
                                  )}
                                </div>
                              </Card>

                              {/* Equity Curve */}
                              <Card>
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="text-sm font-medium">Equity Curve</h4>
                                  <div className="flex gap-1">
                                    {(['trade', 'time'] as const).map((mode) => (
                                      <button key={mode} className="px-2 py-0.5 rounded text-[10px]"
                                        style={{ background: sbEqXAxis === mode ? 'var(--accent-muted)' : 'transparent', color: sbEqXAxis === mode ? 'var(--accent)' : 'var(--text-muted)', border: sbEqXAxis === mode ? '1px solid var(--accent)' : '1px solid var(--border)', cursor: 'pointer' }}
                                        onClick={() => setSbEqXAxis(mode)}>
                                        {mode === 'trade' ? 'Per Trade' : 'Per Day'}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                                {sbResult.equityCurve.length > 0 ? (
                                  <EquityCurve
                                    data={sbResult.equityCurve.map((pt: any, i: number) => ({
                                      trade_number: i + 1,
                                      cumulative_r: pt.cumulative_r ?? 0,
                                      timestamp: pt.timestamp,
                                    }))}
                                    height={250}
                                    showZeroLine
                                    xAxis={sbEqXAxis}
                                  />
                                ) : (
                                  <ChartPlaceholder label="No equity curve data" height={200} />
                                )}
                              </Card>

                              {/* Trade History */}
                              <Card>
                                <h4 className="text-sm font-medium mb-2">Trade History <span className="text-xs font-normal" style={{ color: 'var(--text-muted)' }}>({sbResult.trades.length} trades — click to drill down)</span></h4>
                                <div style={{ overflowX: 'auto', maxHeight: 300, overflowY: 'auto' }}>
                                  <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
                                    <thead>
                                      <tr>
                                        {['#', 'Entry', 'Exit', 'Dir', 'Entry $', 'Exit $', 'P&L (R)', 'Exec', 'Exit Reason'].map((h) => (
                                          <th key={h} className="text-left py-1.5 px-2 text-[10px] font-medium sticky top-0" style={{ color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>{h}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {sbResult.trades.length === 0 ? (
                                        <tr><td colSpan={9} className="py-4 px-2 text-center" style={{ color: 'var(--text-muted)' }}>No trades</td></tr>
                                      ) : sbResult.trades.map((t: any) => (
                                        <tr key={t.id} style={{ borderBottom: '1px solid var(--border)', cursor: sbLastConfig ? 'pointer' : undefined }}
                                          onClick={() => {
                                            if (!sbLastConfig) return;
                                            setSbZoomTrade({ idx: t.id - 1, side: 'entry', trade: t });
                                            sbTradeZoomMut.mutate({ ...sbLastConfig, trade_idx: t.id - 1, side: 'entry' });
                                          }}>
                                          <td className="px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>{t.id}</td>
                                          <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)', fontSize: '0.65rem' }}>{t.entryTime}</td>
                                          <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-secondary)', fontSize: '0.65rem' }}>{t.exitTime}</td>
                                          <td className="px-2 py-1.5"><span style={{ color: t.direction === 'LONG' ? 'var(--green)' : 'var(--red)' }}>{t.direction}</span></td>
                                          <td className="px-2 py-1.5 text-right font-mono">${t.entryPrice.toFixed(2)}</td>
                                          <td className="px-2 py-1.5 text-right font-mono">${t.exitPrice.toFixed(2)}</td>
                                          <td className="px-2 py-1.5 text-right font-mono" style={{ color: t.rMultiple >= 0 ? 'var(--green)' : 'var(--red)' }}>{t.rMultiple >= 0 ? '+' : ''}{t.rMultiple.toFixed(2)}R</td>
                                          <td className="px-2 py-1.5"><ExecBadge tag={`[${t.execType}]`} /></td>
                                          <td className="px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>{t.exitReason.replace(/_/g, ' ')}</td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </Card>
                            </>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Trade Zoom Modal */}
                    {sbZoomTrade && (
                      <TradeZoomModal
                        isOpen={!!sbZoomTrade}
                        onClose={() => { setSbZoomTrade(null); sbTradeZoomMut.reset(); }}
                        tradeIdx={sbZoomTrade.idx}
                        side={sbZoomTrade.side}
                        trade={{
                          entry_price: sbZoomTrade.trade.entryPrice,
                          exit_price: sbZoomTrade.trade.exitPrice,
                          r_multiple: sbZoomTrade.trade.rMultiple,
                          exec_type: sbZoomTrade.trade.execType,
                          exit_reason: sbZoomTrade.trade.exitReason,
                        }}
                        zoomData={sbTradeZoomMut.data ?? null}
                        isLoading={sbTradeZoomMut.isPending}
                        error={sbTradeZoomMut.error ? String(sbTradeZoomMut.error) : null}
                      />
                    )}
                  </div>
                )}

                {tab === 'Code' && (
                  <Card>
                    <div className="space-y-3">
                      {(packType === 'tf_confluence' ? ['manifest.json', 'indicator.py', 'interpreter.py'] : ['manifest.json', 'evaluator.py']).map((file) => (
                        <details key={file}>
                          <summary className="text-xs font-mono font-medium cursor-pointer py-2" style={{ color: 'var(--accent)' }}>{file}</summary>
                          <div className="rounded-lg p-3 font-mono text-xs mt-1" style={{ background: '#0d1117', color: '#c9d1d9', maxHeight: 250, overflowY: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {file === 'manifest.json' && <pre>{generatedManifest ? JSON.stringify(generatedManifest, null, 2) : '// No code generated yet'}</pre>}
                            {file === 'indicator.py' && <pre>{generatedIndicator || '# No code generated yet'}</pre>}
                            {file === 'interpreter.py' && <pre>{generatedInterpreter || '# No code generated yet'}</pre>}
                            {file === 'evaluator.py' && <pre>{generatedIndicator || '# No code generated yet'}</pre>}
                          </div>
                        </details>
                      ))}
                    </div>
                  </Card>
                )}
              </div>
              );
            }}
          </TabBar>

          {/* Action row */}
          <div className="flex items-center gap-3 mt-6">
            <button style={{ ...btnPrimary, opacity: (isInstalling || !generatedManifest || validation.some((v) => v.status === 'fail')) ? 0.5 : 1 }}
              disabled={isInstalling || !generatedManifest || validation.some((v) => v.status === 'fail')}
              onClick={() => {
                if (!generatedManifest || !generatedIndicator || !generatedInterpreter) return;
                setIsInstalling(true);
                installPackMutation.mutate({
                  manifest: generatedManifest,
                  indicator_code: generatedIndicator,
                  interpreter_code: generatedInterpreter,
                  pine_script_code: generatedPine,
                }, {
                  onSuccess: (data) => {
                    setIsInstalling(false);
                    setInstallResult(data);
                    if (data.success) {
                      setConversation((prev) => [...prev,
                        { role: 'assistant', content: `Pack "${data.slug}" installed successfully! It is now available in your confluence groups.`, time: new Date().toLocaleTimeString() },
                      ]);
                    } else {
                      setConversation((prev) => [...prev,
                        { role: 'error', content: `Install failed: ${data.errors.join(', ')}`, time: new Date().toLocaleTimeString() },
                      ]);
                    }
                  },
                  onError: (error) => {
                    setIsInstalling(false);
                    setConversation((prev) => [...prev,
                      { role: 'error', content: `Install failed: ${error instanceof Error ? error.message : 'Unknown error'}`, time: new Date().toLocaleTimeString() },
                    ]);
                  },
                });
              }}>
              {isInstalling ? 'Installing...' : installResult?.success ? 'Installed' : 'Install Pack'}
            </button>
            <button style={btnSecondary}>Save Draft</button>
            <button style={btnSecondary}>Export JSON</button>
            <span className="flex-1" />
            <button style={{ ...btnSecondary, color: 'var(--orange)', borderColor: 'var(--orange)' }}
              onClick={() => { setFixContext(`Issue found on ${activeReviewTab} tab`); setShowFixModal(true); }}>
              Request Fix {fixCount > 0 && `(${fixCount})`}
            </button>
          </div>

          {/* Request Fix Modal */}
          <Modal title="Request Fix" isOpen={showFixModal} onClose={() => setShowFixModal(false)} width="600px">
            <div className="mb-3 px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)' }}>
              <p className="text-xs font-medium" style={{ color: 'var(--text-muted)' }}>Context (auto-captured)</p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-secondary)' }}>{fixContext}</p>
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                Pack: {packName} ({slug}) &middot; {params.length} params &middot; {outputs.length} states &middot; {triggers.length} triggers
              </p>
            </div>
            <div className="mb-4">
              <label className="text-xs font-medium block mb-1" style={{ color: 'var(--text-muted)' }}>Describe what&apos;s wrong</label>
              <textarea value={fixDescription} onChange={(e) => setFixDescription(e.target.value)} rows={4}
                placeholder="The trigger fires one bar late on the live path... / The OVERBOUGHT state never appears even though RSI goes above 70... / The interpreter returns NaN for the first 14 bars..."
                style={{ ...inputStyle, resize: 'vertical' }} />
              <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
                The fix prompt will include: your description, the current code, the pack spec, and the specific test/parity results. The AI will be asked to make a targeted fix without changing working code.
              </p>
            </div>
            <div className="flex justify-end gap-2">
              <button style={btnSecondary} onClick={() => setShowFixModal(false)}>Cancel</button>
              <button style={{ ...btnPrimary, background: 'var(--orange)' }}
                onClick={() => {
                  if (!generatedManifest || !generatedIndicator || !generatedInterpreter) return;

                  setFixCount(fixCount + 1);
                  setShowFixModal(false);
                  setConversation((prev) => [
                    ...prev,
                    { role: 'error', content: `Fix requested (${activeReviewTab}): ${fixDescription}`, time: new Date().toLocaleTimeString() },
                    { role: 'system', content: `Sending targeted fix to AI (attempt ${fixCount + 1})...`, time: new Date().toLocaleTimeString() },
                  ]);

                  requestFixMutation.mutate({
                    manifest: generatedManifest,
                    indicator_code: generatedIndicator,
                    interpreter_code: generatedInterpreter,
                    validation_errors: validation.filter((v) => v.status === 'fail' || v.status === 'warn').map((v) => `${v.label}: ${v.message}`),
                    user_description: fixDescription,
                    ai_model: aiModel,
                  }, {
                    onSuccess: (data) => {
                      setGeneratedManifest(data.manifest);
                      setGeneratedIndicator(data.indicator_code);
                      setGeneratedInterpreter(data.interpreter_code);
                      if (data.pine_script_code) setGeneratedPine(data.pine_script_code);
                      setValidation(data.validation.map((v) => ({ ...v, status: v.status as 'pass' | 'fail' | 'warn' | 'pending' })));
                      setConversation((prev) => [...prev,
                        { role: 'assistant', content: data.ai_message, time: new Date().toLocaleTimeString() },
                      ]);
                    },
                    onError: (error) => {
                      setConversation((prev) => [...prev,
                        { role: 'error', content: `Fix failed: ${error instanceof Error ? error.message : 'Unknown error'}`, time: new Date().toLocaleTimeString() },
                      ]);
                    },
                  });
                  setFixDescription('');
                }}>
                Send Fix to AI
              </button>
            </div>
          </Modal>
        </div>
      )}

      {/* ---- Navigation ---- */}
      <div className="flex justify-between mt-6">
        {step > 1 ? <button style={btnSecondary} onClick={() => setStep(step - 1)}>← Back</button> : <div />}
        {step < 5 && (
          <button style={btnPrimary}
            disabled={(step === 1 && !canProceed1) || (step === 2 && !structureGenerated)}
            onClick={() => setStep(step + 1)}>
            Next →
          </button>
        )}
      </div>
    </div>
  );
}
