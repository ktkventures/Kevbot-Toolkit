'use client';

/**
 * Time Exit Packs — V5-style management page.
 *
 * Layout follows User Packs page pattern:
 *   - PageHeader with "+ Create Variation" button top-right
 *   - Full-width single-column list grouped by template
 *   - Clicking "Edit" navigates to an inline detail view
 *   - Variations are nested under their parent template
 */

import { useState, useMemo } from 'react';
import Card from '@/components/Card';
import PageHeader from '@/components/PageHeader';
import Modal from '@/components/Modal';
import { useTimeExitPacks, useTimeExitTemplates } from '@/hooks/queries/usePacks';
import type { TimeExitPackDTO, TimeExitTemplateDTO } from '@/hooks/queries/usePacks';
import { useSaveTimeExitPacks } from '@/hooks/mutations/usePackMutations';

/* ========================================================================
   Types
   ======================================================================== */

interface PackParam {
  key: string;
  label: string;
  type: 'int' | 'float';
  value: number;
  default: number;
  min?: number;
  max?: number;
}

interface TimeExitPackLocal {
  id: string;
  templateKey: string;
  templateName: string;
  version: string;
  category: string;
  enabled: boolean;
  isDefault: boolean;
  description: string;
  exitSummary: string;
  params: PackParam[];
}

/* ========================================================================
   Style Constants
   ======================================================================== */

const categoryColors: Record<string, { color: string; bg: string }> = {
  Time: { color: 'var(--orange)', bg: 'var(--orange-muted)' },
  Duration: { color: 'var(--blue)', bg: 'var(--blue-muted)' },
};

const inputStyle: React.CSSProperties = {
  background: 'var(--bg-input)',
  border: '1px solid var(--border)',
  color: 'var(--text-primary)',
};

const btnPrimary: React.CSSProperties = {
  background: 'var(--accent)',
  color: 'white',
};

const btnSecondary: React.CSSProperties = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border)',
  color: 'var(--text-secondary)',
};

/* ========================================================================
   API → Local Mapping
   ======================================================================== */

function apiToPack(dto: TimeExitPackDTO, template?: TimeExitTemplateDTO): TimeExitPackLocal {
  const schema = template?.parameters_schema ?? {};
  const params: PackParam[] = Object.entries(schema).map(([key, s]) => ({
    key,
    label: s.label || key,
    type: (s.type === 'int' ? 'int' : 'float') as 'int' | 'float',
    value: (dto.parameters?.[key] ?? s.default ?? 0) as number,
    default: (s.default ?? 0) as number,
    min: s.min,
    max: s.max,
  }));

  return {
    id: dto.id,
    templateKey: dto.base_template,
    templateName: template?.name ?? dto.base_template,
    version: dto.version,
    category: template?.category ?? 'Time',
    enabled: dto.enabled,
    isDefault: dto.is_default,
    description: template?.description ?? dto.description,
    exitSummary: dto.exit_summary || dto.version || '--',
    params,
  };
}

function packToDto(pack: TimeExitPackLocal): TimeExitPackDTO {
  const parameters: Record<string, unknown> = {};
  for (const p of pack.params) {
    parameters[p.key] = p.value;
  }
  return {
    id: pack.id,
    base_template: pack.templateKey,
    version: pack.version,
    description: pack.description,
    enabled: pack.enabled,
    is_default: pack.isDefault,
    parameters,
  };
}

/* ========================================================================
   Sub-components
   ======================================================================== */

function CategoryBadge({ category }: { category: string }) {
  const s = categoryColors[category] || { color: 'var(--text-muted)', bg: 'var(--bg-input)' };
  return (
    <span className="text-[10px] font-medium px-1.5 py-0.5 rounded" style={{ color: s.color, background: s.bg }}>
      {category}
    </span>
  );
}

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onToggle(); }}
      className="w-10 h-6 rounded-full relative flex-shrink-0 transition-colors"
      style={{
        background: enabled ? 'var(--accent)' : 'var(--bg-input)',
        border: enabled ? 'none' : '1px solid var(--border)',
      }}
    >
      <div
        className="w-4 h-4 rounded-full absolute top-1 transition-all"
        style={{
          background: enabled ? 'white' : 'var(--text-muted)',
          left: enabled ? '22px' : '4px',
        }}
      />
    </button>
  );
}

/* ========================================================================
   Pack Card (list view — full-width, matches User Packs pattern)
   ======================================================================== */

function PackCard({
  pack,
  onToggle,
  onEdit,
  onClone,
  onDelete,
}: {
  pack: TimeExitPackLocal;
  onToggle: () => void;
  onEdit: () => void;
  onClone: () => void;
  onDelete: () => void;
}) {
  return (
    <Card>
      <div className="flex items-start gap-4">
        {/* Toggle */}
        <div className="pt-0.5">
          <Toggle enabled={pack.enabled} onToggle={onToggle} />
        </div>

        {/* Main info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>
              {pack.version}
            </span>
            <span
              className="text-xs px-1.5 py-0.5 rounded font-mono"
              style={{ color: 'var(--accent)', background: 'var(--accent-muted)' }}
            >
              {pack.exitSummary}
            </span>
            {pack.isDefault && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ color: 'var(--text-muted)', background: 'var(--bg-input)' }}>
                Default
              </span>
            )}
          </div>

          {/* Parameter summary */}
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {pack.params.map((p) => `${p.label}: ${p.value}`).join(', ')}
          </p>
        </div>

        {/* Actions */}
        <div className="flex gap-2 flex-shrink-0">
          <button
            onClick={(e) => { e.stopPropagation(); onEdit(); }}
            className="px-3 py-1.5 rounded text-xs font-medium"
            style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}
          >
            Edit
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onClone(); }}
            className="px-3 py-1.5 rounded text-xs"
            style={btnSecondary}
          >
            Clone
          </button>
          {!pack.isDefault && (
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              className="px-3 py-1.5 rounded text-xs"
              style={{ background: 'var(--red-muted)', color: 'var(--red)' }}
            >
              Delete
            </button>
          )}
        </div>
      </div>
    </Card>
  );
}

/* ========================================================================
   Detail View (replaces list when editing — matches User Packs pattern)
   ======================================================================== */

function DetailView({
  pack,
  apiPacks,
  onBack,
  onSave,
  onDelete,
}: {
  pack: TimeExitPackLocal;
  apiPacks: TimeExitPackDTO[];
  onBack: () => void;
  onSave: (updated: TimeExitPackDTO[]) => void;
  onDelete: () => void;
}) {
  const [editingParams, setEditingParams] = useState<Record<string, number>>(() => {
    const map: Record<string, number> = {};
    for (const p of pack.params) { map[p.key] = p.value; }
    return map;
  });
  const [editingVersion, setEditingVersion] = useState(pack.version);

  function handleSave() {
    const updated = apiPacks.map((p) => {
      if (p.id !== pack.id) return p;
      return {
        ...p,
        version: editingVersion || p.version,
        parameters: { ...p.parameters, ...editingParams },
      };
    });
    onSave(updated);
    onBack();
  }

  return (
    <div>
      <PageHeader
        title={`${pack.templateName} — ${editingVersion}`}
        subtitle={pack.description}
        backHref="#"
        actions={
          <div className="flex items-center gap-3">
            <CategoryBadge category={pack.category} />
            <button
              onClick={handleSave}
              className="px-4 py-2 rounded-lg text-sm font-medium"
              style={btnPrimary}
            >
              Save Changes
            </button>
            <button onClick={onBack} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
              Back to Packs
            </button>
          </div>
        }
      />

      <div className="space-y-5 mt-2">
        {/* Parameters */}
        <Card>
          <h3 className="text-sm font-medium mb-5" style={{ color: 'var(--text-secondary)' }}>
            Parameters
          </h3>

          <div className="space-y-4">
            <div>
              <div className="flex items-center gap-4">
                <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                  Version Name
                </label>
                <input
                  type="text"
                  className="px-3 py-2 rounded-lg text-sm flex-1"
                  style={inputStyle}
                  value={editingVersion}
                  onChange={(e) => setEditingVersion(e.target.value)}
                />
              </div>
              <div className="flex gap-4 mt-1">
                <span className="w-52 flex-shrink-0" />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  Identifies this variation (e.g. &quot;15min&quot;, &quot;Aggressive&quot;)
                </span>
              </div>
            </div>

            {pack.params.map((param) => (
              <div key={param.key}>
                <div className="flex items-center gap-4">
                  <label className="text-sm w-52 flex-shrink-0" style={{ color: 'var(--text-secondary)' }}>
                    {param.label}
                  </label>
                  <input
                    type="number"
                    className="px-3 py-2 rounded-lg text-sm flex-1 font-mono"
                    style={inputStyle}
                    value={editingParams[param.key] ?? param.value}
                    min={param.min}
                    max={param.max}
                    step={param.type === 'int' ? 1 : 0.1}
                    onChange={(e) => {
                      const val = param.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value);
                      if (!isNaN(val)) {
                        setEditingParams((prev) => ({ ...prev, [param.key]: val }));
                      }
                    }}
                  />
                </div>
                <div className="flex gap-4 mt-1">
                  <span className="w-52 flex-shrink-0" />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    {param.type === 'int' ? 'Integer' : 'Decimal'}
                    {param.min !== undefined && param.max !== undefined && ` · Range: ${param.min} – ${param.max}`}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="flex gap-3 mt-6 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <button onClick={handleSave} className="px-4 py-2 rounded-lg text-sm font-medium" style={btnPrimary}>
              Save Changes
            </button>
            <button onClick={onBack} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>
              Cancel
            </button>
          </div>
        </Card>

        {/* Info */}
        <Card>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>About This Exit Type</h3>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            Time exit packs are mechanical risk controls. They force position closure based on time, regardless of price signals.
            They fire at <strong>Priority 2</strong> in the exit chain — after stop loss, but before targets, signal exits, and bar count exits.
            This means &quot;be flat by X&quot; always wins over chart-based exit logic.
          </p>
        </Card>

        {/* Danger zone */}
        {!pack.isDefault && (
          <Card>
            <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--red)' }}>Danger Zone</h3>
            <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
              Deleting this variation is permanent. Active strategies using it will fall back to no time exit.
            </p>
            <button onClick={onDelete} className="px-4 py-2 rounded-lg text-sm" style={{ background: 'var(--red-muted)', color: 'var(--red)' }}>
              Delete Variation
            </button>
          </Card>
        )}
      </div>
    </div>
  );
}

/* ========================================================================
   Create Variation Modal
   ======================================================================== */

function CreateVariationModal({
  isOpen,
  onClose,
  templates,
  onSubmit,
}: {
  isOpen: boolean;
  onClose: () => void;
  templates: Record<string, TimeExitTemplateDTO>;
  onSubmit: (templateKey: string, versionName: string) => void;
}) {
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [versionName, setVersionName] = useState('');

  const templateEntries = Object.entries(templates);

  return (
    <Modal title="Create Variation" isOpen={isOpen} onClose={onClose} width="480px">
      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium block mb-1.5" style={{ color: 'var(--text-secondary)' }}>
            Template
          </label>
          <select
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
          >
            <option value="">Select a template...</option>
            {templateEntries.map(([key, t]) => (
              <option key={key} value={key}>{t.name} — {t.description}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-sm font-medium block mb-1.5" style={{ color: 'var(--text-secondary)' }}>
            Version Name
          </label>
          <input
            type="text"
            placeholder="e.g. Aggressive, Conservative, 5min..."
            className="w-full px-3 py-2 rounded-lg text-sm"
            style={inputStyle}
            value={versionName}
            onChange={(e) => setVersionName(e.target.value)}
          />
        </div>
        <div className="flex justify-end gap-2 pt-2">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-sm" style={btnSecondary}>Cancel</button>
          <button
            onClick={() => {
              if (selectedTemplate && versionName.trim()) {
                onSubmit(selectedTemplate, versionName.trim());
                setSelectedTemplate('');
                setVersionName('');
                onClose();
              }
            }}
            disabled={!selectedTemplate || !versionName.trim()}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={{ ...btnPrimary, opacity: (!selectedTemplate || !versionName.trim()) ? 0.5 : 1 }}
          >
            Create
          </button>
        </div>
      </div>
    </Modal>
  );
}

/* ========================================================================
   Main Page
   ======================================================================== */

export default function TimeExitPage() {
  const { data: apiPacks, isLoading: packsLoading } = useTimeExitPacks();
  const { data: apiTemplates } = useTimeExitTemplates();
  const saveMutation = useSaveTimeExitPacks();

  const [detailPackId, setDetailPackId] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Map API packs to local format
  const packs: TimeExitPackLocal[] = useMemo(() => {
    if (!apiPacks) return [];
    return apiPacks.map((dto: any) => {
      const template = apiTemplates?.[dto.base_template];
      return apiToPack(dto, template);
    });
  }, [apiPacks, apiTemplates]);

  // Group packs by template
  const groupedPacks = useMemo(() => {
    const groups: Record<string, { templateName: string; category: string; description: string; packs: TimeExitPackLocal[] }> = {};
    for (const pack of packs) {
      if (!groups[pack.templateKey]) {
        groups[pack.templateKey] = {
          templateName: pack.templateName,
          category: pack.category,
          description: pack.description,
          packs: [],
        };
      }
      groups[pack.templateKey].packs.push(pack);
    }
    return groups;
  }, [packs]);

  const enabledCount = packs.filter((p) => p.enabled).length;

  // Actions
  function handleToggle(id: string) {
    if (!apiPacks) return;
    const updated = apiPacks.map((p: any) => p.id === id ? { ...p, enabled: !p.enabled } : p);
    saveMutation.mutate(updated);
  }

  function handleDelete(id: string) {
    if (!apiPacks) return;
    saveMutation.mutate(apiPacks.filter((p: any) => p.id !== id));
    if (detailPackId === id) setDetailPackId(null);
  }

  function handleClone(pack: TimeExitPackLocal) {
    if (!apiPacks) return;
    const newId = `${pack.templateKey}_custom_${Date.now()}`;
    const newDto: TimeExitPackDTO = {
      ...packToDto(pack),
      id: newId,
      version: `${pack.version} (Copy)`,
      is_default: false,
      enabled: true,
    };
    saveMutation.mutate([...apiPacks, newDto]);
  }

  function handleCreate(templateKey: string, versionName: string) {
    if (!apiPacks || !apiTemplates) return;
    const template = apiTemplates[templateKey];
    if (!template) return;
    const newId = `${templateKey}_custom_${Date.now()}`;
    const params: Record<string, unknown> = {};
    for (const [key, s] of Object.entries(template.parameters_schema)) {
      params[key] = s.default;
    }
    const newDto: TimeExitPackDTO = {
      id: newId,
      base_template: templateKey,
      version: versionName,
      description: template.description,
      enabled: true,
      is_default: false,
      parameters: params,
    };
    saveMutation.mutate([...apiPacks, newDto]);
    setDetailPackId(newId);
  }

  function handleSave(updated: TimeExitPackDTO[]) {
    saveMutation.mutate(updated);
  }

  // Loading
  if (packsLoading) {
    return (
      <div>
        <PageHeader title="Time Exit Packs" subtitle="Loading..." />
      </div>
    );
  }

  // Detail view — replaces the list (User Packs pattern)
  const detailPack = packs.find((p) => p.id === detailPackId);
  if (detailPack && apiPacks) {
    return (
      <DetailView
        pack={detailPack}
        apiPacks={apiPacks}
        onBack={() => setDetailPackId(null)}
        onSave={handleSave}
        onDelete={() => handleDelete(detailPack.id)}
      />
    );
  }

  // List view
  return (
    <div>
      <PageHeader
        title="Time Exit Packs"
        subtitle="Mechanical time-based exit rules — force position closure independent of price signals"
        actions={
          <button
            onClick={() => setShowCreateModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium"
            style={btnPrimary}
          >
            + Create Variation
          </button>
        }
      />

      {/* Summary stats */}
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {packs.length} {packs.length === 1 ? 'variation' : 'variations'}, {enabledCount} enabled
        </span>
        <div className="flex items-center gap-1.5">
          {Array.from(new Set(packs.map((p) => p.category))).map((cat) => (
            <CategoryBadge key={cat} category={cat} />
          ))}
        </div>
      </div>

      {packs.length === 0 ? (
        <Card>
          <div className="text-center py-16">
            <p className="text-lg font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              No time exit packs configured
            </p>
            <p className="text-sm mb-8 max-w-md mx-auto" style={{ color: 'var(--text-muted)' }}>
              Create variations of time-based exit rules. Each variation becomes
              an optimizable variable you can compare in the Strategy Builder.
            </p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-5 py-2.5 rounded-lg text-sm font-medium"
              style={btnPrimary}
            >
              + Create Variation
            </button>
          </div>
        </Card>
      ) : (
        <div className="space-y-6">
          {Object.entries(groupedPacks).map(([templateKey, group]) => (
            <div key={templateKey}>
              {/* Template heading */}
              <div className="flex items-center gap-3 mb-3">
                <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {group.templateName}
                </h2>
                <CategoryBadge category={group.category} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {group.packs.length} {group.packs.length === 1 ? 'variation' : 'variations'}
                </span>
              </div>
              <p className="text-xs mb-3" style={{ color: 'var(--text-muted)' }}>
                {group.description}
              </p>

              {/* Nested variation cards */}
              <div className="space-y-2">
                {group.packs.map((pack) => (
                  <PackCard
                    key={pack.id}
                    pack={pack}
                    onToggle={() => handleToggle(pack.id)}
                    onEdit={() => setDetailPackId(pack.id)}
                    onClone={() => handleClone(pack)}
                    onDelete={() => handleDelete(pack.id)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Variation Modal */}
      {apiTemplates && (
        <CreateVariationModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          templates={apiTemplates}
          onSubmit={handleCreate}
        />
      )}
    </div>
  );
}
