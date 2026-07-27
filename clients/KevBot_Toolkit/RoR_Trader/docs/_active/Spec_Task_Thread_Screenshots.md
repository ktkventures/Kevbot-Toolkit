# Spec — Inline screenshots in task threads (proposed board V4.17)

**Author:** M (M·auto, 2026-07-25, discovered during #107) · **Implements:** F
**Status:** Backlog — awaiting M scoping polish + Kevin approval per lifecycle.

## Problem

Task-thread comments render as escaped plain text (`AdminTasksPage.tsx` comment
map, `<div>{cm.body}</div>` — verified on dev 07-25): no markdown, no images, no
attachment column, no upload path anywhere in the board. M's visual-verification
workflow (SOP_Local_Preview_Visual_Verification.md Part B) therefore references
screenshots by repo file path — works, but Kevin must open the file himself.
Goal: the screenshot renders IN the thread.

## V1 (small, recommended)

1. **Render (frontend, `AdminTasksPage.tsx` comment map):** when a comment line
   matches exactly `SCREENSHOT: <https-url>` or markdown `![alt](https://...)`,
   render an `<img src={url} style={{maxWidth:'100%'}}>` (click opens full-size in
   a new tab). Everything else stays escaped plain text exactly as today.
   - XSS guard: NO `dangerouslySetInnerHTML`, no markdown engine. Build the `<img>`
     element with the URL as a React prop, only after it passes a strict
     `^https://` + URL-parse validation. Whitelist the Supabase storage host.
2. **Host (backend, `src/api/routers/dev_tasks.py`):** new admin-gated
   `POST /api/dev-tasks/{task_id}/screenshots` — accepts a PNG (base64 JSON or
   multipart), stores to a new Supabase Storage bucket `task-screenshots` at
   `task-{id}/{filename}`, returns the URL. Public-read bucket is acceptable
   (admin-dashboard screenshots); signed long-expiry URLs if Kevin prefers.
3. **M helper (tools/visual-verify):** add `--post <task_id>` to `screenshot.mjs`
   → capture, upload via the new endpoint, and add the comment
   (`SCREENSHOT: <url>` + verification sentence) in one command.

## Non-goals

Full markdown rendering, arbitrary HTML, non-image attachments, editing/deleting
uploads from the UI.

## Lane & shipping

Tasks-page frontend + `dev_tasks.py` router are org tooling (PR #71 precedent —
F-implemented per M spec); no engine/data files. Default-safe (pure addition; no
flag needed — feature is inert until a comment contains the marker). Ship via
normal release train; smoke = post a screenshot comment on a scratch task on dev
and see it render.

## Acceptance

- A comment created via the new `--post` flow renders the image inline on
  `/admin/tasks` (dev), plain-text comments render byte-identical to today.
- A comment containing `<img>`/`<script>` text still renders as escaped text.
- Upload endpoint rejects non-admin users and non-PNG payloads.
