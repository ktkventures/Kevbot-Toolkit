/**
 * Markdown-lite body renderer shared by the team-board Activity thread,
 * Context/Summary descriptions, and strategy notes (board #134 item 1).
 *
 * remark-breaks keeps SINGLE newlines as line breaks so agent reports and
 * phone-typed comments don't render bunched into one paragraph; gfm adds
 * tables/task-lists; rehype-raw + sanitize allow pasted inline HTML minus
 * scripts/handlers. data: image URIs stay allowed (pasted screenshots, per
 * the tasks-page spec).
 */
'use client';

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';

const mdSchema = {
  ...defaultSchema,
  protocols: {
    ...defaultSchema.protocols,
    src: [...((defaultSchema.protocols as Record<string, string[]>)?.src || []), 'data'],
  },
};

/** One <style> block per modal — pair with <div className="task-md">. */
export const MD_CSS = `
  .task-md { font-size: 13px; line-height: 1.5; overflow-wrap: break-word; }
  .task-md > :first-child { margin-top: 0; }
  .task-md > :last-child { margin-bottom: 0; }
  .task-md h1, .task-md h2, .task-md h3 { margin: 10px 0 6px; }
  .task-md p { margin: 6px 0; }
  .task-md img { max-width: 100%; border-radius: 6px; }
  .task-md table { border-collapse: collapse; margin: 8px 0; }
  .task-md th, .task-md td { border: 1px solid var(--border); padding: 4px 8px; font-size: 12.5px; }
  .task-md code { background: var(--bg-input); padding: 1px 4px; border-radius: 4px; font-size: 12px; }
  .task-md pre { background: var(--bg-input); padding: 8px; border-radius: 6px; overflow-x: auto; }
  .task-md pre code { background: none; padding: 0; white-space: pre; }
  .task-md ul, .task-md ol { padding-left: 20px; margin: 6px 0; }
  .task-md blockquote { border-left: 3px solid var(--border); margin: 6px 0; padding: 2px 10px; color: var(--text-secondary); }
`;

export const Md = ({ text, fallback }: { text?: string | null; fallback?: string }) => (
  <div className="task-md">
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}
      rehypePlugins={[rehypeRaw, [rehypeSanitize, mdSchema]]}>
      {text || fallback || ''}
    </ReactMarkdown>
  </div>
);
