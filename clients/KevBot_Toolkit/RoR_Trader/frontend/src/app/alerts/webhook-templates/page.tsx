'use client';

import VersionedPage from '@/components/VersionedPage';
import V1 from './versions/V1';
import V2 from './versions/V2';
import V3 from './versions/V3';
import V4 from './versions/V4';

const versions = [
  {
    meta: {
      id: 'v1-initial',
      name: 'Initial Scaffold',
      description: 'Simple card list of webhook templates with Edit and Test action buttons.',
      rationale: 'First pass — minimal layout showing template names with inline actions. Establishes the structure for payload template management.',
    },
    component: V1,
  },
  {
    meta: {
      id: 'v2-full-parity',
      name: 'Full Parity',
      description: 'Complete template management: rich cards with URL/type/usage, full editor modal with payload textarea, placeholder picker, custom headers, rendered preview, test delivery with response display, and clone/delete with confirmation.',
      rationale: 'Matches Streamlit webhook template system — CRUD, placeholder-based payload editing, live preview, and test delivery workflow.',
    },
    component: V2,
  },
  {
    meta: {
      id: 'v3-streamlined',
      name: 'Streamlined',
      description: 'Compact table-like rows with name, masked URL, usage count, and inline Edit/Test/Delete icon buttons. Simple modal for edit/create with name, URL, and payload textarea. Inline test feedback, inline delete confirmation. No placeholder picker, no headers editor, no response display.',
      rationale: 'Maximum density — all templates visible at once in a scannable list. Test sends inline with status feedback. Editing uses a minimal modal without advanced features.',
    },
    component: V3,
  },
  {
    meta: {
      id: 'v4-creative',
      name: 'Creative',
      description: 'Webhook Studio with rich template cards showing status indicators, delivery stats, and expandable detail panels. Features payload diff view (template vs rendered), delivery timeline with status dots, response inspector with headers/body, placeholder palette organized by category (signal/strategy/position/meta), live payload preview in editor, and aggregate delivery metrics strip.',
      rationale: 'Elevates webhook management into a full development studio. The payload diff view shows template vs rendered side-by-side. The delivery timeline visualizes webhook health at a glance. The response inspector shows exactly what came back from test sends. The placeholder palette makes it easy to build payloads without memorizing variable names.',
    },
    component: V4,
  },
];

export default function WebhookTemplatesPage() {
  return <VersionedPage pageKey="webhook-templates" versions={versions} />;
}
