'use client';

import { useState } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import TabBar from '@/components/TabBar';
import Modal from '@/components/Modal';
import ChartPlaceholder from '@/components/ChartPlaceholder';

interface TfPack {
  id: string;
  name: string;
  template: string;
  enabled: boolean;
  isDefault: boolean;
  params: string;
  outputs: string[];
  triggers: number;
}

const mockPacks: TfPack[] = [
  { id: 'ema-stack-default', name: 'EMA Stack (Default)', template: 'Moving Averages', enabled: true, isDefault: true, params: 'S:9, M:21, L:200', outputs: ['SML','SLM','MSL','MLS','LSM','LMS'], triggers: 4 },
  { id: 'macd-line-default', name: 'MACD Line (Default)', template: 'Momentum', enabled: true, isDefault: true, params: 'Fast:12, Slow:26, Sig:9', outputs: ['M>S+','M>S-','M<S-','M<S+'], triggers: 4 },
  { id: 'macd-hist-default', name: 'MACD Histogram (Default)', template: 'Momentum', enabled: false, isDefault: true, params: 'Fast:12, Slow:26, Sig:9', outputs: ['H+up','H+dn','H-dn','H-up'], triggers: 4 },
  { id: 'vwap-default', name: 'VWAP (Default)', template: 'Volume', enabled: true, isDefault: true, params: 'SD1:1.0, SD2:2.0', outputs: ['>+2\u03c3','>+1\u03c3','>V','@V','<V','<-1\u03c3','<-2\u03c3'], triggers: 17 },
  { id: 'rvol-default', name: 'Relative Volume (Default)', template: 'Volume', enabled: false, isDefault: true, params: 'SMA:20, High:1.5, Ext:3.0', outputs: ['EXTREME','HIGH','NORMAL','LOW','MINIMAL'], triggers: 3 },
  { id: 'ema-stack-scalping', name: 'EMA Stack (Scalping)', template: 'Moving Averages', enabled: true, isDefault: false, params: 'S:5, M:13, L:50', outputs: ['SML','SLM','MSL','MLS','LSM','LMS'], triggers: 4 },
];

const templateCategories = ['Moving Averages', 'Momentum', 'Volume', 'Volatility', 'Trend'];

const categoryColors: Record<string, { color: string; bg: string }> = {
  'Moving Averages': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  'Momentum': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  'Volume': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
  'Volatility': { color: 'var(--red)', bg: 'var(--red-muted)' },
  'Trend': { color: 'var(--green)', bg: 'var(--green-muted)' },
};

const execTypeBadgeStyles: Record<string, { color: string; bg: string }> = {
  '[C]': { color: 'var(--green)', bg: 'var(--green-muted)' },
  '[L0]': { color: 'var(--blue)', bg: 'var(--blue-muted)' },
  '[HM]': { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  '[HL]': { color: 'var(--accent)', bg: 'var(--accent-muted)' },
};

const mockTriggers: Record<string, { name: string; execType: string }[]> = {
  'Moving Averages': [
    { name: 'EMA Bull Cross', execType: '[C]' },
    { name: 'EMA Bear Cross', execType: '[C]' },
    { name: 'EMA Expansion Long', execType: '[L0]' },
    { name: 'EMA Expansion Short', execType: '[HM]' },
  ],
  'Momentum': [
    { name: 'MACD Bull Cross', execType: '[C]' },
    { name: 'MACD Bear Cross', execType: '[C]' },
    { name: 'Zero Line Cross Up', execType: '[L0]' },
    { name: 'Zero Line Cross Down', execType: '[HL]' },
  ],
  'Volume': [
    { name: 'VWAP Reclaim', execType: '[C]' },
    { name: 'VWAP Rejection', execType: '[C]' },
    { name: 'Band Squeeze', execType: '[L0]' },
    { name: 'Band Breakout Up', execType: '[HM]' },
    { name: 'Band Breakout Down', execType: '[HM]' },
  ],
};

export default function TfConfluenceV1() {
  const [packs, setPacks] = useState<TfPack[]>(mockPacks);
  const [detailPack, setDetailPack] = useState<TfPack | null>(null);
  const [showNewModal, setShowNewModal] = useState(false);
  const [newTemplate, setNewTemplate] = useState(templateCategories[0]);
  const [newName, setNewName] = useState('');

  const enabledCount = packs.filter((p) => p.enabled).length;

  function togglePack(id: string) {
    setPacks((prev) => prev.map((p) => p.id === id ? { ...p, enabled: !p.enabled } : p));
  }

  if (detailPack) {
    return (
      <div>
        <PageHeader
          title={detailPack.name}
          subtitle={`${detailPack.template} pack`}
          backHref="#"
          actions={
            <button
              onClick={() => setDetailPack(null)}
              className="px-4 py-2 rounded-lg text-sm"
              style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
            >
              Back to Packs
            </button>
          }
        />

        <TabBar tabs={['Parameters', 'Plot Settings', 'Outputs & Triggers', 'Preview', 'Danger Zone']}>
          {(tab) => (
            <div>
              {tab === 'Parameters' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Pack Parameters</h3>
                  <div className="space-y-4">
                    {detailPack.params.split(', ').map((param) => {
                      const [label, value] = param.split(':');
                      return (
                        <div key={label} className="flex items-center gap-4">
                          <label className="text-sm w-40" style={{ color: 'var(--text-secondary)' }}>{label.trim()}</label>
                          <input
                            className="px-3 py-2 rounded-lg text-sm flex-1"
                            style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}
                            defaultValue={value.trim()}
                          />
                        </div>
                      );
                    })}
                  </div>
                  <div className="flex gap-3 mt-6">
                    <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Save Changes</button>
                    <button className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Reset to Default</button>
                  </div>
                </Card>
              )}
              {tab === 'Plot Settings' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Visual Settings</h3>
                  <div className="space-y-4">
                    {[
                      { label: 'Primary Line Color', value: '#3b82f6' },
                      { label: 'Secondary Line Color', value: '#f59e0b' },
                      { label: 'Fill Color', value: '#3b82f620' },
                    ].map((setting) => (
                      <div key={setting.label} className="flex items-center gap-4">
                        <label className="text-sm w-48" style={{ color: 'var(--text-secondary)' }}>{setting.label}</label>
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded border" style={{ background: setting.value, borderColor: 'var(--border)' }} />
                          <input className="px-3 py-2 rounded-lg text-sm font-mono w-28" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} defaultValue={setting.value} />
                        </div>
                      </div>
                    ))}
                    <div className="flex items-center gap-4">
                      <label className="text-sm w-48" style={{ color: 'var(--text-secondary)' }}>Line Width</label>
                      <input type="range" min="1" max="5" defaultValue="2" className="flex-1" />
                      <span className="text-sm w-8 text-center" style={{ color: 'var(--text-muted)' }}>2px</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <label className="text-sm w-48" style={{ color: 'var(--text-secondary)' }}>Fill Opacity</label>
                      <input type="range" min="0" max="100" defaultValue="20" className="flex-1" />
                      <span className="text-sm w-8 text-center" style={{ color: 'var(--text-muted)' }}>20%</span>
                    </div>
                  </div>
                  <button className="mt-6 px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>Save Plot Settings</button>
                </Card>
              )}
              {tab === 'Outputs & Triggers' && (
                <div className="grid grid-cols-2 gap-6">
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Interpreter Outputs</h3>
                    <div className="space-y-2">
                      {detailPack.outputs.map((output) => (
                        <div key={output} className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm font-mono" style={{ color: 'var(--text-primary)' }}>{output}</span>
                          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>State</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                  <Card>
                    <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--text-secondary)' }}>Available Triggers</h3>
                    <div className="space-y-2">
                      {(mockTriggers[detailPack.template] || []).map((trigger) => (
                        <div key={trigger.name} className="flex items-center justify-between px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)' }}>
                          <span className="text-sm" style={{ color: 'var(--text-primary)' }}>{trigger.name}</span>
                          <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ color: execTypeBadgeStyles[trigger.execType]?.color || 'var(--text-muted)', background: execTypeBadgeStyles[trigger.execType]?.bg || 'var(--bg-card)' }}>{trigger.execType}</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}
              {tab === 'Preview' && (
                <div>
                  <div className="flex gap-3 mb-4">
                    {[['NVDA','SPY','AAPL','TSLA'], ['1Min','5Min','15Min','1H'], ['RTH Only','Extended Hours']].map((opts, i) => (
                      <select key={i} className="px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                        {opts.map((o) => <option key={o}>{o}</option>)}
                      </select>
                    ))}
                  </div>
                  <Card><ChartPlaceholder label={`Price chart with ${detailPack.name} overlay`} height={400} /></Card>
                </div>
              )}
              {tab === 'Danger Zone' && (
                <Card>
                  <h3 className="text-sm font-medium mb-4" style={{ color: 'var(--red)' }}>Danger Zone</h3>
                  <div className="space-y-6">
                    <div>
                      <label className="text-sm mb-2 block" style={{ color: 'var(--text-secondary)' }}>Rename Version</label>
                      <div className="flex gap-3">
                        <input className="px-3 py-2 rounded-lg text-sm flex-1" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} defaultValue={detailPack.name} disabled={detailPack.isDefault} />
                        <button className="px-4 py-2 rounded-lg text-sm" style={{ background: detailPack.isDefault ? 'var(--bg-input)' : 'var(--bg-card)', border: '1px solid var(--border)', color: detailPack.isDefault ? 'var(--text-muted)' : 'var(--text-secondary)', cursor: detailPack.isDefault ? 'not-allowed' : 'pointer' }} disabled={detailPack.isDefault}>Rename</button>
                      </div>
                      {detailPack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be renamed.</p>}
                    </div>
                    <div className="pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
                      <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>Delete this pack permanently.</p>
                      <button className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: detailPack.isDefault ? 'var(--bg-input)' : 'var(--red)', color: detailPack.isDefault ? 'var(--text-muted)' : 'white', cursor: detailPack.isDefault ? 'not-allowed' : 'pointer' }} disabled={detailPack.isDefault}>Delete Pack</button>
                      {detailPack.isDefault && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Default packs cannot be deleted.</p>}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          )}
        </TabBar>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="TF Confluence Packs" subtitle="Timeframe-specific indicators that compute on each bar" actions={<button onClick={() => setShowNewModal(true)} className="px-4 py-2 rounded-lg text-sm font-medium" style={{ background: 'var(--accent)', color: 'white' }}>+ New Pack</button>} />
      <p className="text-sm mb-6" style={{ color: 'var(--text-muted)' }}>{packs.length} packs, {enabledCount} enabled</p>
      <div className="space-y-3">
        {packs.map((pack) => (
          <Card key={pack.id}>
            <div className="flex items-center gap-4">
              <button onClick={() => togglePack(pack.id)} className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors" style={{ background: pack.enabled ? 'var(--accent)' : 'var(--bg-input)', border: pack.enabled ? 'none' : '1px solid var(--border)' }}>
                <div className="w-4 h-4 rounded-full absolute top-1 transition-all" style={{ background: pack.enabled ? 'white' : 'var(--text-muted)', left: pack.enabled ? '22px' : '4px' }} />
              </button>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-semibold text-sm">{pack.name}</span>
                  {pack.isDefault && <span className="text-xs px-1.5 py-0.5 rounded" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>default</span>}
                  <span className="text-xs px-2 py-0.5 rounded" style={{ color: categoryColors[pack.template]?.color || 'var(--text-muted)', background: categoryColors[pack.template]?.bg || 'var(--bg-input)' }}>{pack.template}</span>
                </div>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{pack.params} &middot; {pack.triggers} triggers</p>
              </div>
              <div className="flex items-center gap-1 flex-shrink-0">
                {pack.outputs.slice(0, 4).map((output) => (
                  <span key={output} className="text-xs font-mono px-1.5 py-0.5 rounded" style={{ background: 'var(--bg-input)', color: 'var(--text-secondary)' }}>{output}</span>
                ))}
                {pack.outputs.length > 4 && <span className="text-xs" style={{ color: 'var(--text-muted)' }}>+{pack.outputs.length - 4} more</span>}
              </div>
              <div className="flex gap-2 flex-shrink-0">
                <button onClick={() => setDetailPack(pack)} className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>Details</button>
                <button className="px-3 py-1.5 rounded text-xs" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}>Copy</button>
              </div>
            </div>
          </Card>
        ))}
      </div>
      <Modal title="Create New Pack" isOpen={showNewModal} onClose={() => setShowNewModal(false)} width="500px">
        <div className="space-y-4">
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Template</label>
            <select className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} value={newTemplate} onChange={(e) => setNewTemplate(e.target.value)}>
              {templateCategories.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div>
            <label className="text-sm mb-1 block" style={{ color: 'var(--text-secondary)' }}>Version Name</label>
            <input className="w-full px-3 py-2 rounded-lg text-sm" style={{ background: 'var(--bg-input)', border: '1px solid var(--border)', color: 'var(--text-primary)' }} placeholder="e.g. Scalping, Swing, Conservative..." value={newName} onChange={(e) => setNewName(e.target.value)} />
          </div>
          <div>
            <label className="text-xs mb-1 block" style={{ color: 'var(--text-muted)' }}>Pack ID Preview</label>
            <p className="text-sm font-mono px-3 py-2 rounded-lg" style={{ background: 'var(--bg-input)', color: 'var(--text-muted)' }}>{newTemplate.toLowerCase().replace(/\s+/g, '-')}-{newName ? newName.toLowerCase().replace(/\s+/g, '-') : 'custom'}</p>
          </div>
          <div className="flex gap-3 pt-2">
            <button className="px-4 py-2 rounded-lg text-sm font-medium flex-1" style={{ background: 'var(--accent)', color: 'white' }} onClick={() => setShowNewModal(false)}>Create Pack</button>
            <button className="px-4 py-2 rounded-lg text-sm flex-1" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }} onClick={() => setShowNewModal(false)}>Cancel</button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
