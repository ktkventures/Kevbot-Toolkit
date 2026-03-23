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

  const [activeVersion, setActiveVersion] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(storageKey);
      if (saved && versions.some((v) => v.meta.id === saved)) return saved;
    }
    return versions[0]?.meta.id ?? '';
  });

  useEffect(() => {
    if (activeVersion) {
      localStorage.setItem(storageKey, activeVersion);
    }
  }, [activeVersion, storageKey]);

  const activeConfig = versions.find((v) => v.meta.id === activeVersion) || versions[0];
  const ActiveComponent = activeConfig?.component;

  if (!ActiveComponent) return null;

  return (
    <>
      <ActiveComponent />
      <VersionPanel
        versions={versions.map((v) => v.meta)}
        activeVersion={activeVersion}
        onVersionChange={setActiveVersion}
      />
    </>
  );
}
