'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';

const roleCards = [
  { id: 'subscriber', title: 'Copy Trade', description: 'I want to subscribe to proven portfolios', price: 'Free', color: 'var(--green)', icon: '$' },
  { id: 'strategy', title: 'Build Strategies', description: 'I want to create and test my own', price: '$29/mo', color: 'var(--blue)', icon: '\u{1F4CA}' },
  { id: 'portfolio', title: 'Build Portfolios', description: 'I want to curate portfolios and earn revenue', price: '$49/mo', color: 'var(--accent)', icon: '\u{1F4BC}' },
  { id: 'pack', title: 'Create Packs', description: 'I want to build indicators and tools', price: '$79/mo', color: 'var(--purple)', icon: '\u{1F9E9}' },
];

const experienceLevels = ['Beginner', 'Intermediate', 'Advanced'];

const steps = ['Welcome', 'Profile', 'Broker', 'First Action', 'Done'];

export default function OnboardingV2() {
  const [step, setStep] = useState(0);
  const [selectedRole, setSelectedRole] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState('');
  const [experience, setExperience] = useState('Beginner');
  const [webhookUrl, setWebhookUrl] = useState('');
  const [webhookTested, setWebhookTested] = useState(false);

  function nextStep() {
    if (step < steps.length - 1) setStep(step + 1);
  }
  function prevStep() {
    if (step > 0) setStep(step - 1);
  }

  function testWebhook() {
    // Mock test
    setWebhookTested(true);
  }

  return (
    <div className="max-w-2xl mx-auto">
      {/* Progress Bar */}
      <div className="flex items-center gap-2 mb-8">
        {steps.map((s, i) => (
          <div key={s} className="flex items-center flex-1">
            <div className="flex items-center gap-2 flex-1">
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold shrink-0"
                style={{
                  background: i <= step ? 'var(--accent)' : 'var(--bg-input)',
                  color: i <= step ? 'var(--bg-primary)' : 'var(--text-muted)',
                }}
              >
                {i < step ? '\u2713' : i + 1}
              </div>
              <span className="text-xs hidden md:inline" style={{ color: i <= step ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {s}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div className="h-px flex-1 mx-2" style={{ background: i < step ? 'var(--accent)' : 'var(--border)' }} />
            )}
          </div>
        ))}
      </div>

      {/* Step 1: Welcome */}
      {step === 0 && (
        <div>
          <PageHeader
            title="What brings you to RoR Trader?"
            subtitle="Choose the path that fits your goals"
          />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {roleCards.map((role) => (
              <Card
                key={role.id}
                className={`cursor-pointer transition-all ${selectedRole === role.id ? 'ring-2' : ''}`}
              >
                <div onClick={() => setSelectedRole(role.id)} className="p-2">
                  <div className="flex items-center gap-3 mb-2">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-lg"
                      style={{
                        background: selectedRole === role.id ? role.color + '33' : 'var(--bg-input)',
                        border: `1px solid ${selectedRole === role.id ? role.color : 'var(--border)'}`,
                      }}
                    >
                      {role.icon}
                    </div>
                    <div className="flex-1">
                      <p className="font-semibold">{role.title}</p>
                      <p className="text-xs" style={{ color: role.color }}>{role.price}</p>
                    </div>
                  </div>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{role.description}</p>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Step 2: Profile Setup */}
      {step === 1 && (
        <div>
          <PageHeader
            title="Set up your profile"
            subtitle="Tell us a bit about yourself"
          />
          <Card>
            <div className="space-y-5">
              <div className="flex items-center gap-4 mb-4">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold cursor-pointer"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)', border: '2px dashed var(--accent)' }}
                >
                  {displayName ? displayName.charAt(0).toUpperCase() : '+'}
                </div>
                <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Click to upload avatar</p>
              </div>

              <div>
                <label className="text-sm block mb-1" style={{ color: 'var(--text-secondary)' }}>Display Name</label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="How should we call you?"
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>

              <div>
                <label className="text-sm block mb-2" style={{ color: 'var(--text-secondary)' }}>Experience Level</label>
                <div className="flex gap-2">
                  {experienceLevels.map((level) => (
                    <button
                      key={level}
                      onClick={() => setExperience(level)}
                      className="flex-1 py-2 rounded-lg text-sm font-medium transition-colors"
                      style={{
                        background: experience === level ? 'var(--accent)' : 'var(--bg-input)',
                        color: experience === level ? 'var(--bg-primary)' : 'var(--text-secondary)',
                        border: `1px solid ${experience === level ? 'var(--accent)' : 'var(--border)'}`,
                      }}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Step 3: Broker Connection */}
      {step === 2 && (
        <div>
          <PageHeader
            title="Connect your broker"
            subtitle="Set up your webhook URL to receive trade alerts"
          />
          <Card>
            <div className="space-y-5">
              <div>
                <label className="text-sm block mb-1" style={{ color: 'var(--text-secondary)' }}>Broker</label>
                <select
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                >
                  <option>Select your broker...</option>
                  <option>Alpaca</option>
                  <option>TradingView</option>
                  <option>Interactive Brokers</option>
                  <option>TD Ameritrade</option>
                  <option>Other</option>
                </select>
              </div>

              <div>
                <label className="text-sm block mb-1" style={{ color: 'var(--text-secondary)' }}>Webhook URL</label>
                <input
                  type="url"
                  value={webhookUrl}
                  onChange={(e) => { setWebhookUrl(e.target.value); setWebhookTested(false); }}
                  placeholder="https://your-broker-webhook-url.com/..."
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                />
              </div>

              <div className="flex gap-3">
                <button
                  onClick={testWebhook}
                  className="px-4 py-2 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                  disabled={!webhookUrl}
                >
                  Test Connection
                </button>
                {webhookTested && (
                  <span className="flex items-center gap-1 text-sm" style={{ color: 'var(--green)' }}>
                    &#10003; Connection successful
                  </span>
                )}
              </div>

              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                You can skip this step and set it up later in Settings &gt; Connections.
              </p>
            </div>
          </Card>
        </div>
      )}

      {/* Step 4: First Action */}
      {step === 3 && (
        <div>
          <PageHeader
            title="Your first step"
            subtitle={selectedRole === 'subscriber' ? 'Browse the marketplace and subscribe to a portfolio' : 'Create your first strategy or import an existing one'}
          />
          <Card>
            {selectedRole === 'subscriber' || !selectedRole ? (
              <div className="text-center py-6">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
                  style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
                >
                  $
                </div>
                <h3 className="text-lg font-semibold mb-2">Browse the Marketplace</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                  Check out our curated free portfolios — no subscription required.
                </p>
                <button
                  className="px-5 py-2.5 rounded-lg text-sm font-medium"
                  style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                >
                  Browse Portfolios
                </button>
              </div>
            ) : (
              <div className="text-center py-6">
                <div
                  className="w-16 h-16 rounded-full flex items-center justify-center text-3xl mx-auto mb-4"
                  style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
                >
                  +
                </div>
                <h3 className="text-lg font-semibold mb-2">Create Your First Strategy</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
                  Open the Strategy Builder and configure your first trading strategy.
                </p>
                <div className="flex gap-3 justify-center">
                  <button
                    className="px-5 py-2.5 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
                  >
                    Open Strategy Builder
                  </button>
                  <button
                    className="px-5 py-2.5 rounded-lg text-sm font-medium"
                    style={{ background: 'var(--bg-input)', color: 'var(--text-primary)', border: '1px solid var(--border)' }}
                  >
                    Import Existing
                  </button>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Step 5: Done */}
      {step === 4 && (
        <div className="text-center py-12">
          <div
            className="w-20 h-20 rounded-full flex items-center justify-center text-4xl mx-auto mb-6"
            style={{ background: 'var(--green-muted)', color: 'var(--green)' }}
          >
            &#10003;
          </div>
          <h2 className="text-2xl font-bold mb-2">You&apos;re all set!</h2>
          <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>
            Your account is ready. Here are your next steps:
          </p>

          <div className="max-w-md mx-auto space-y-3 mb-8 text-left">
            {[
              selectedRole === 'subscriber' ? 'Browse the marketplace for portfolios' : 'Create your first strategy',
              'Set up your webhook URL for trade alerts',
              'Configure your risk settings',
              'Join the community forum',
            ].map((step, i) => (
              <div key={i} className="flex items-center gap-3 p-3 rounded-lg" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
                <span className="text-xs font-mono w-5 text-center" style={{ color: 'var(--text-muted)' }}>{i + 1}</span>
                <span className="text-sm">{step}</span>
              </div>
            ))}
          </div>

          <button
            className="px-8 py-3 rounded-lg text-sm font-medium"
            style={{ background: 'var(--accent)', color: 'var(--bg-primary)' }}
          >
            Go to Dashboard
          </button>
        </div>
      )}

      {/* Navigation */}
      {step < 4 && (
        <div className="flex justify-between mt-6">
          <button
            onClick={prevStep}
            className="px-4 py-2 rounded-lg text-sm"
            style={{
              background: 'var(--bg-input)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border)',
              visibility: step === 0 ? 'hidden' : 'visible',
            }}
          >
            Back
          </button>
          <div className="flex gap-2">
            {step === 2 && (
              <button
                onClick={nextStep}
                className="px-4 py-2 rounded-lg text-sm"
                style={{ color: 'var(--text-muted)' }}
              >
                Skip
              </button>
            )}
            <button
              onClick={nextStep}
              className="px-6 py-2 rounded-lg text-sm font-medium"
              style={{
                background: (step === 0 && !selectedRole) ? 'var(--bg-input)' : 'var(--accent)',
                color: (step === 0 && !selectedRole) ? 'var(--text-muted)' : 'var(--bg-primary)',
              }}
              disabled={step === 0 && !selectedRole}
            >
              {step === 3 ? 'Finish Setup' : 'Continue'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
