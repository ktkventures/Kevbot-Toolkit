# SOP — Reliable Local Preview + M Visual Verification (board #107 · V4.11)

**Owner:** M · **Created:** 2026-07-25 (M·auto, task #107)
**Why this exists:** 07-23 a local `:3000` tab went zombie — an hours-old dev tab
with a stale HMR socket and dead chunk hashes showed an eternal spinner and burned
review time. This SOP makes pre-release preview dependable (Part A) and gives Kevin
visual confirmation in task threads without clicking through the app (Part B).

---

## Part A — Local preview that can't go zombie (Kevin + all roles)

**The zombie mechanism:** a long-lived `localhost:3000` tab holds an HMR websocket
and a chunk manifest from the dev-server process that built it. When that server
restarts (or recompiles heavily), the old tab's chunk hashes 404 and the HMR socket
is dead → eternal spinner. The TAB is stale, not the app.

**Rules:**

1. **Fresh tab per review round.** Open a NEW tab (or incognito window — the
   guarantee) every time you sit down to review. Never trust a `:3000` tab that is
   older than the dev server behind it. Minimum bar if reusing: hard reload
   (Ctrl+Shift+R).
2. **One dev server, in the RIGHT checkout.** Before starting one:
   `ps aux | grep "next dev"` and kill strays. Then start from the checkout whose
   branch you're reviewing — for F's UI work that is F's worktree
   (`../Kevbot-frontend/frontend`), NOT the main checkout (which is usually on an
   engine branch): `npm run dev` from that `frontend/` dir.
3. **Weirdness → clean rebuild.** Kill the server, `rm -rf frontend/.next`,
   `npm run dev`. Ten extra seconds, zero mystery.
4. **Env is read at startup only.** `frontend/.env.local` points
   `NEXT_PUBLIC_API_URL` at the dev API (verified 07-25). Inline env on the command
   line does NOT override the file — edit `.env.local` and restart the server.
5. **Ready check before reviewing:**
   `curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/login` → `200`.
   First hit on a route triggers on-demand compile — a brief spinner there is
   normal; an eternal one is Rule 1.

**Zombie signature (recognize, don't debug):** eternal spinner + DevTools console
404s on `/_next/static/...` chunks. Close the tab, open a fresh one. If a fresh
tab also spins → the server is actually down/stuck → Rule 3.

---

## Part B — M visual verification: Playwright screenshots into task threads

**Tool:** `tools/visual-verify/screenshot.mjs` (built + proven 07-25 against dev
`/admin/tasks`). Headless Chromium, logs in with the test account
(`ROR_TEST_EMAIL`/`ROR_TEST_PASSWORD` from `src/.env`), navigates, screenshots.

```bash
cd tools/visual-verify
# dev site (default base):
node screenshot.mjs /admin/tasks ../../docs/_active/task_screenshots/task-107_admin-tasks_2026-07-25.png --wait 6000
# local preview:
node screenshot.mjs /admin/tasks <out.png> --base http://localhost:3000
# chart-heavy pages need a longer settle:
node screenshot.mjs /strategies/303 <out.png> --wait 12000 --full
```

**Conventions:**

- **Where screenshots live:** `docs/_active/task_screenshots/` — gitignored via its
  own `.gitignore` (only the `.gitignore` is committable), so no binary bloat and
  no git-status noise. Names: `task-<id>_<slug>_<YYYY-MM-DD>.png`.
- **How they reach a thread (v1, today):** comments render PLAIN TEXT (no images —
  see `Spec_Task_Thread_Screenshots.md` for the upgrade), so the comment carries:
  `SCREENSHOT: docs/_active/task_screenshots/<file>` plus **one sentence of what M
  visually verified in it**. Kevin opens the path from the VS Code explorer —
  visual confirmation without clicking through the app.
- **M ALWAYS views the PNG before referencing it** (Read tool renders it). Never
  post an unviewed screenshot; the narrated verification is the point, the file is
  the evidence.

**Ops notes (learned building it):**

- WSL box runs Node 18 → playwright is pinned `^1.49.x` in
  `tools/visual-verify/package.json` (1.50+ requires Node 20). Its Chromium
  (build v1148) is cached at `~/.cache/ms-playwright/`; if a fresh machine errors
  "Executable doesn't exist": `npx playwright install chromium` from that dir.
- Login is per-run (stateless browser; token expires ~1h anyway). Login clicks the
  button by accessible name `Sign In` — NOT `button[type=submit]` (sidebar nav
  buttons are also type=submit).
- A `git push` to dev redeploys the dev API → transient down window; wait for the
  API to answer before screenshotting dev.
- The Playwright MCP is an alternative in interactive sessions, but this script is
  the SOP tool: it works headless (dispatcher runs), is deterministic, and needs no
  MCP connection.

---

## Part C — Upgrade path

Inline screenshot rendering + upload in task threads (so images show IN the thread
instead of by path reference): spec'd in `docs/_active/Spec_Task_Thread_Screenshots.md`,
proposed as board **V4.17** (origin=discovered, under V4 — M to file; headless run
couldn't create board rows). F implements per spec.
