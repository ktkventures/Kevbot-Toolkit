'use client';

/**
 * WorkflowTrace — renders execution workflow steps with replay-aware states.
 *
 * Extracted from ExecutionTypesPage inline JSX. Supports three visual states
 * per step: completed, active (pulsing), pending (dimmed).
 *
 * Also renders a pre-entry "Confluence Monitoring" section when monitoring
 * data is provided.
 */

import type { StepState, ConfluenceCondition } from '@/hooks/useScenarioReplay';

const EXEC_BADGE_COLOR = '#2196F3';

interface WorkflowTraceProps {
  steps: any[];
  stepStates?: StepState[];
  // Pre-entry monitoring
  confluenceConditions?: ConfluenceCondition[];
  confluenceAllMet?: boolean;
  triggerStatus?: 'waiting' | 'fired';
  triggerName?: string;
}

// Pulse animation via inline keyframes
const pulseKeyframes = `
@keyframes wf-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(33,150,243,0.4); }
  50% { box-shadow: 0 0 0 4px rgba(33,150,243,0.1); }
}
`;

export default function WorkflowTrace({
  steps, stepStates,
  confluenceConditions, confluenceAllMet, triggerStatus, triggerName,
}: WorkflowTraceProps) {
  const hasMonitoring = confluenceConditions && confluenceConditions.length > 0;

  return (
    <div>
      {/* Inject pulse animation */}
      <style>{pulseKeyframes}</style>

      {/* ---- Trade Execution Workflow ---- */}
      <div className="space-y-0">
        {steps.map((step: any, i: number) => {
          const state: StepState = stepStates?.[i] || 'completed';
          const isPending = state === 'pending';
          const isActive = state === 'active';

          // Circle styling
          const circleStyle: React.CSSProperties = isPending ? {
            background: 'var(--bg-input)',
            color: 'var(--text-muted)',
            border: '1px solid var(--border)',
            opacity: 0.35,
          } : {
            background: step.isWebhook ? 'var(--accent)'
              : step.action === 'exit' ? (step.color === 'var(--green)' ? 'var(--green)' : 'var(--red)')
              : 'var(--bg-input)',
            color: step.isWebhook || step.action === 'exit' ? 'white' : 'var(--text-muted)',
            border: step.isWebhook || step.action === 'exit' ? 'none' : '1px solid var(--border)',
            ...(isActive ? { animation: 'wf-pulse 1.5s ease-in-out infinite' } : {}),
          };

          return (
            <div key={i} className="flex items-start gap-2">
              <div className="flex flex-col items-center">
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center text-[8px] font-bold flex-shrink-0"
                  style={circleStyle}
                >
                  {i + 1}
                </div>
                {i < steps.length - 1 && (
                  <div
                    className="w-0.5 h-3"
                    style={{
                      background: 'var(--border)',
                      ...(isPending ? { opacity: 0.3, borderLeft: '1px dashed var(--border)', width: 0 } : {}),
                    }}
                  />
                )}
              </div>
              <div className="pb-1.5" style={isPending ? { opacity: 0.35 } : {}}>
                <p className="text-[10px]" style={{
                  color: isPending ? 'var(--text-muted)'
                    : step.color || (step.isWebhook ? 'var(--accent)' : 'var(--text-primary)'),
                }}>
                  {step.label}
                </p>
                {step.badge && (
                  <span className="text-[8px] font-mono font-bold px-1 py-0.5 rounded-full"
                    style={{
                      color: EXEC_BADGE_COLOR,
                      background: EXEC_BADGE_COLOR + '20',
                      ...(isPending ? { opacity: 0.5 } : {}),
                    }}>
                    [{step.badge}]
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
