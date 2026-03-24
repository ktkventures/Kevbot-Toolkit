'use client';

import { useState, useEffect, ComponentType } from 'react';
import VersionPanel, { VersionMeta } from './VersionPanel';

interface VersionConfig {
  meta: VersionMeta;
  component: ComponentType;
}

interface VersionedPageProps {
  /** Unique key for localStorage, e.g. "tf-confluence" */
  pageKey: string;
  versions: VersionConfig[];
}

export default function VersionedPage({ pageKey, versions }: VersionedPageProps) {
  const storageKey = `ror-version-${pageKey}`;
  const defaultId = versions[0]?.meta.id ?? '';

  const [activeVersion, setActiveVersion] = useState<string>(defaultId);

  // Load saved version on mount
  useEffect(() => {
    const saved = localStorage.getItem(storageKey);
    if (saved && versions.some((v) => v.meta.id === saved)) {
      setActiveVersion(saved);
    }
  }, [storageKey, versions]);

  const handleVersionChange = (id: string) => {
    setActiveVersion(id);
    localStorage.setItem(storageKey, id);
  };

  const activeConfig = versions.find((v) => v.meta.id === activeVersion) || versions[0];
  const ActiveComponent = activeConfig?.component;

  if (!ActiveComponent) return null;

  return (
    <>
      <ActiveComponent />
      <VersionPanel
        versions={versions.map((v) => v.meta)}
        activeVersion={activeVersion}
        onVersionChange={handleVersionChange}
      />
    </>
  );
}
