'use client';

import { useViewMode } from '@/providers/StoreProvider';
import VersionedPage from '@/components/VersionedPage';
import { ComponentType } from 'react';

interface VersionMeta {
  id: string;
  name: string;
  description: string;
  rationale: string;
}

interface VersionEntry {
  meta: VersionMeta;
  component: ComponentType;
}

interface PageSwitchProps {
  /** The clean API-first component (shown in Live mode) */
  live: ComponentType;
  /** Design reference versions (shown in Design Reference mode) */
  versions: VersionEntry[];
  /** Key for VersionedPage localStorage persistence */
  pageKey: string;
}

/**
 * PageSwitch — renders either the clean Live component or the
 * VersionedPage design reference based on the sidebar toggle.
 */
export default function PageSwitch({ live: LiveComponent, versions, pageKey }: PageSwitchProps) {
  const { mode } = useViewMode();

  if (mode === 'design') {
    return <VersionedPage pageKey={pageKey} versions={versions} />;
  }

  return <LiveComponent />;
}
