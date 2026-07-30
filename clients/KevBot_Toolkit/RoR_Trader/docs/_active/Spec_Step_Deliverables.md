# Spec — Step deliverables + the deliverables surface (board #184)

**Author:** M·auto (1805c263 lane), 2026-07-29 · **Implements:** F
**Depends on:** #182 (DONE — step object, accordion, step-action endpoints)
**Status:** Scoping → awaiting Kevin's stamp (chain step 2)

Two halves of one context, deliberately one task: **capture** a deliverable inside
the step that asks for it, then **surface** everything delivered in one place.

---

## 0. Decisions already made (do not re-litigate)

| # | Decision | Source |
|---|----------|--------|
| D1 | Deliverables are **OPTIONAL by default**; `*` marks the required ones. Bias toward optional until a process is well understood. | Kevin 07-28, comment 558 |
| D2 | The Complete button blocks **only** on unfilled *required* deliverables. Unfilled *optional* ones are **visible, not silent** — a soft prompt, never a gate. | Kevin 07-28, comment 558 |
| D3 | An agent that cannot supply a required deliverable **escalates on the existing path** (`⚠ Raise issue → M`, #182 T9). One road, not two. | Kevin 07-28, comment 558 |
| D4 | Half 2 ships in Kevin's order: **tab first** (functional benefit, low risk), **strip second**. | task description |
| D5 | The strip must not eat real estate, and must not displace a vision's subtask list. | task description |
| D6 | **Move the stamp into the step**: a stamp applied while step N is current attaches to step N (who + when). **No retroactive backfill.** | Kevin 07-29, comment 686 |

**Why optional-by-default is right** (recorded so a future reader does not "fix" it):
a required input encodes an assumption about the *shape* of the answer, usually made
before anyone has run the process. Demanding a file when a URL was the natural artifact
does not gather better evidence — it produces a blocked step and a workaround. A stamp
is a **decision** whose shape is known; a deliverable is **evidence** whose shape often
is not. Gating on a guess about form is friction, not honesty.

---

## 1. Data model — `deliverables` on a chain step

The #182 step object widens again. Purely additive, feature-detected, **no DDL**:
`dev_tasks.checklist` is already JSONB and already whitelisted for PATCH. A step with
no `deliverables` key behaves exactly as it does today.

```
step.deliverables : [ deliverable ]      -- ordered, optional, default absent

deliverable = {
  -- SPEC (authored: who asks, for what) — editable via the generic checklist PATCH
  id            : text      -- server-assigned stable id (uuid4 hex[:12]), like step.id
  kind          : text      -- 'text' | 'link' | 'file'
  label         : text      -- the ask, one line: "link to the merged PR"
  required      : bool      -- DEFAULT false (D1)
  hint          : text      -- optional placeholder / help line          default ""

  -- FILLED STATE (server-owned — see §2) — a generic PATCH may NEVER touch these
  value         : text|null -- text: the answer · link: the URL · file: the STORAGE PATH
  filename      : text|null -- file only
  size_bytes    : int|null  -- file only
  content_type  : text|null -- file only
  filled_at     : tstz|null
  filled_by     : text|null
}
```

**`value` for `kind='file'` is the storage object path, not a URL** (§4) — a URL would
either expire in the record or leak past the admin gate.

**Limits** (API-enforced, fail loud): ≤ 8 deliverables per step · `label` ≤ 120 chars ·
`text` value ≤ 4000 chars · `link` value must parse as `http(s)://` · file ≤ 10 MB and
content-type in the allow-list `png jpg jpeg gif webp pdf csv txt md json log zip`.

**Three kinds, no more.** Kevin named exactly text / link / file. Multi-field forms,
selects, dates and numbers are a non-goal (§8) — the moment this becomes a form builder
it stops being a step and starts being a product.

---

## 2. Server-owned vs author-owned — the rule that makes it trustworthy

This mirrors #182 exactly, and for the same reason: transition rules must be **API
refusals, not UI conventions**.

- The **generic** `PATCH /api/dev-tasks/{id}` (checklist) may add, edit, reorder or
  delete deliverable **specs** (`kind`/`label`/`required`/`hint`) on **not-yet-done**
  steps. `_prepare_checklist_patch` gains the deliverable analogue of
  `_STEP_COMPLETION_FIELDS`:
  `_DELIVERABLE_FILL_FIELDS = ("value","filename","size_bytes","content_type","filled_at","filled_by")`
  → a PATCH that changes any of them **409s**: *"a deliverable's value is managed by
  /steps/deliverables — a PATCH cannot fill or clear one."*
- Deleting a **filled** deliverable on a not-yet-done step: **allowed**, because the
  step is still open and the author may have asked for the wrong thing — but it posts a
  system comment naming what was dropped, so the audit trail survives the edit.
  Deleting any deliverable on a **completed** step: **409** (T4' immutability).
- New deliverables inserted onto an existing step must arrive **unfilled** (`value`
  null) — same shape as "a newly inserted step cannot start completed".
- `_ensure_step_ids` gains `_ensure_deliverable_ids` (same uuid4 hex[:12] pattern).

**Which steps may be filled?** Any step that is **not yet done** — not just the current
one. Evidence often arrives early, and refusing it would push people to park it in a
comment where the surface cannot see it. A **completed** step's deliverables are
immutable (409), consistent with T4'.

---

## 3. API

All admin-gated exactly like the existing dev-tasks routes.

### 3.1 Fill a text/link deliverable
```
POST /api/dev-tasks/{task_id}/steps/deliverables/{deliverable_id}
body: { value: str, actor: str }
```
Validates kind (`text`|`link` only — a file goes to §3.2), enforces the §1 limits, sets
`value` / `filled_at` / `filled_by`, writes the whole checklist array (never a partial
merge — memory `feedback_jsonb_partial_updates`), posts a system comment tagged to that
step's index: `deliverable "<label>" filled (by <actor>) on step N`.
Refuses: unknown id (404) · completed step (409) · wrong kind (409) · limit breach (400).

### 3.2 Upload a file deliverable
```
POST /api/dev-tasks/{task_id}/steps/deliverables/{deliverable_id}/upload
body: multipart file  (or {filename, content_type, b64} JSON — match whatever the
      screenshots helper ends up using, so the two share one client path)
```
Validates size + content-type allow-list **before** touching storage, uploads to §4,
then sets `value` = object path, `filename`, `size_bytes`, `content_type`, `filled_*`.
Same refusals as §3.1, plus 413 over-size and 415 disallowed type.

### 3.3 Clear a value
```
DELETE /api/dev-tasks/{task_id}/steps/deliverables/{deliverable_id}/value
```
Nulls the fill fields on a not-yet-done step (409 on a completed one). Does **not**
delete the stored object (§4, orphans).

### 3.4 Mint a read URL (files only)
```
GET /api/dev-tasks/{task_id}/steps/deliverables/{deliverable_id}/url
→ { url: <signed, 1h expiry> }
```
Called **on click**, never on board load. Keeps the 120-row board fetch cheap and keeps
the object behind the admin gate.

### 3.5 The completion gate — **T10**
`POST /steps/complete` (existing) gains one refusal, placed **after** the T8 stamp check:

> **T10.** A step with an **unfilled required** deliverable cannot be completed.
> `409: step "<title>" needs its required deliverable "<label>" before it can be
> completed (T10) — fill it, or ⚠ Raise issue → M if you cannot.`

The error text names the escalation path in-line (D3), so an agent that hits the wall
does not invent a workaround.

**Optional-unfilled is recorded, not blocked** (D2). On success, `complete_step` appends
to its existing system comment:
`… · 2 optional deliverables left unfilled: "<label>", "<label>"`.
This is the durable half of "visible, not silent" — a UI dialog nobody records is not
visibility. The UI confirm (§5.1) is the other half.

### 3.6 Step-aware stamps (D6)
Today's task-level `POST /{id}/stamp` (#136) ticks the first un-done kevin step directly
— it writes `done` with **no** `completed_at` / `completed_by`, and **no hand-off**. On a
#182 chain that is a divergence from the step machinery, and it is why the step's `stamp`
field reads `None` after Kevin stamps.

Change, **for chain tasks only** (`_is_process_chain` true):
1. Record `step.stamp = {required: <preserved>, state:'approved', by:'kevin', at:<now>}`
   on the first un-done kevin-owned step — so the chain carries **which** step was signed
   off, not merely that a sign-off happened.
2. Complete that step through the **same code path** as `/steps/complete` (extract the
   body of `complete_step` into a `_complete_current_step(c, task, actor)` helper and call
   it from both) — so the stamp gets `completed_at`/`completed_by`, the hand-off fires,
   and T10 applies to Kevin's step too.
3. Legacy (non-chain) tasks: byte-identical to today. **No retroactive backfill** of the
   42 legacy tasks (D6, and #182 Step 9's retrofit-on-touch rule).

---

## 4. Storage

Bucket **`task-deliverables`**, **private**. Path:
`task-{task_id}/step-{step_id}/{uuid4hex}-{sanitized_filename}`.

Precedent for the client pattern already exists in this repo —
`src/bar_cache_store.py:125` (`get_admin_client().storage.from_(BUCKET).upload(...)`),
bucket `dough-cache`. Reuse it; do not add a new storage client.

**Private, not public.** The board is admin-gated; a public-read bucket URL walks
straight past that gate, and deliverables are not guaranteed to be innocuous
screenshots. Cost of private = §3.4's click-time signed URL. Accepted.

**Orphans.** Clearing a value (§3.3), or deleting a not-yet-done step that carried an
upload, leaves the object in the bucket. That is deliberate — deleting on the delete path
would make an audit-trail edit destructive. A periodic orphan sweep is a **follow-up
chore**, explicitly named here rather than silently skipped (see §9).

**⚠ Coordinate with the screenshots spec.** `Spec_Task_Thread_Screenshots.md` (Backlog)
proposes its own `task-screenshots` bucket + upload endpoint. **Do not build two.** This
task lands the generic primitive first; the screenshots spec should be re-pointed at
`task-deliverables` when it comes off the backlog. That re-point is chain step 9.

---

## 5. UI — Half 1: the input inside the step

### 5.1 In the accordion (`ProcessChain`, `TaskDetailModal.tsx:985+`)
Inside an open step's body, **below the SOP markdown, above the Step-6 edit controls**:

- One row per deliverable: `kind` icon (`✎ text` / `🔗 link` / `📎 file`) · `label`
  (with a red `*` when `required`) · the control · fill provenance when filled
  (`✓ <filled_by> <relTime>`, styled like the existing `✓ {s.completed_by}` chip).
- **Controls** — `text`: a one-line input (multiline auto-grow if the value is long),
  saved on blur/Enter. `link`: a URL input, rendering as a clickable anchor once filled.
  `file`: a drop-zone/browse button; once filled, `📎 filename (size)` that calls §3.4 on
  click. Every filled control offers a small `✕ clear` (§3.3).
- **Only on not-yet-done steps are the controls live.** On a completed step the
  deliverables render read-only (values + provenance) — same immutability the rest of a
  completed step already shows.
- **Required-and-unfilled → the Complete button is disabled**, exactly as `stampBlocks`
  already does for T8, with `title` = the T10 message. Reuse the existing disabled/opacity
  treatment; do not invent a second visual language for "blocked".
- **Optional-and-unfilled → a confirm on Complete**: *"2 optional deliverables are
  unfilled — complete anyway?"* (D2). Never a block.

### 5.2 In the step form (`StepForm`, the "process module")
The authoring side: a **Deliverables** section with `＋ add deliverable` → rows of
[kind select] [label] [required checkbox] [hint]. Reorder/remove per row. This is the
only place deliverable specs are authored — no hand-edited JSON, the #182 Step 6 rule.

---

## 6. UI — Half 2 Phase 1: the deliverables **tab** (D4, ships first)

- `type Tab` gains `'deliverables'`; a fourth chip after `[summary][process][config]`,
  labelled `deliverables N` where N = filled count.
- **The chip renders only when the scoped task's chain declares ≥ 1 deliverable.** Tasks
  with none get no new chrome.
- Body: deliverables grouped by step, in chain order — `▸ Step 3 · F — <title>` then its
  rows. Each row: kind icon · label (`*` if required) · the value rendered by kind (text
  = the answer inline; link = anchor; file = `📎 filename (size)` → §3.4 on click) ·
  `filled_by · relTime`. Unfilled rows show a muted `— not provided` so the gaps are as
  visible as the fills.
- Empty state (chain has specs, none filled): *"nothing delivered yet."*
- **Scope follows the context panel.** On a vision showing "Summary", it shows the
  vision's own deliverables; selecting a subtask in the Pipeline strip re-scopes it, just
  as it already re-scopes the accordion. **Cross-subtask aggregation is a non-goal** (§8).

---

## 7. UI — Half 2 Phase 2: the **strip** under the context panel (D4, ships after Kevin's review)

Kevin's two constraints (D5) drive the whole layout, so state them as invariants:

**I1 — it must not eat real estate.** The strip is `flex: 'none'` with a **fixed ~56 px**
total height (one header line + one row of square cards), and it **renders nothing at all**
when the scoped task has zero deliverables. Cards are ~44 px squares in a single
horizontally-scrolling row (`overflow-x: auto`, `flex-wrap: nowrap`) — the row never grows
vertically no matter how many deliverables exist.

**I2 — a vision must still show its subtask list.** The strip is inserted **below the
context panel and above the existing Pipeline strip** (`TaskDetailModal.tsx:797`, the
`{vision && …}` block). The Pipeline strip's `maxHeight: '45%'` is **not** reduced; the
strip's 56 px comes out of the context panel's `flex: 1`. Verify on a vision with ≥ 6
subtasks that the Pipeline list is still scrollable and shows ≥ 4 rows.

- **Card** = kind icon, large and centered, with a tiny corner state mark: filled = the
  kind icon in full colour; required-unfilled = the icon muted with a red `*`;
  optional-unfilled = the icon at 40 % opacity.
- **Hover** = the detail: `label · step N (<owner>) · value or "not provided" · filled_by,
  relTime`. Use the existing `title` tooltip convention the rest of the modal uses — do
  not introduce a tooltip library for this.
- **Click** = the natural action: file → §3.4 signed URL in a new tab; link → the URL;
  text → expand the card into the tab (`setTab('deliverables')`). One click, no menu.

---

## 8. Non-goals

Multi-field forms, selects, dates, numbers · per-deliverable comment threads · version
history when a file is replaced (the new upload wins; the old object is an orphan) ·
cross-subtask aggregation on a vision · deleting storage objects from the UI · retrofit
of the 42 legacy tasks · any change to how legacy (non-chain) checklists behave.

---

## 9. Follow-ups this spec deliberately defers (named, not hidden)

1. **Orphan sweep** for `task-deliverables` objects whose deliverable was cleared or whose
   step was removed (§4).
2. **Re-point `Spec_Task_Thread_Screenshots.md`** at this bucket + endpoint instead of a
   second `task-screenshots` bucket (§4) — chain step 9.
3. **Per-step comment filter UI**, which consumes `dev_task_comments.step_order` already
   captured by #182 Step 3. It can ride along with this task's accordion work or stand
   alone; it is not blocked by anything here.

---

## 10. Lane, shipping, testing

**Lane.** `frontend/src/views/TaskDetailModal.tsx` + `src/api/routers/dev_tasks.py` are org
tooling — F implements per this M spec (PR #71 / #182 precedent). No engine or data-path
files. Cut the branch from **latest `origin/dev`** (board #153).

**Flag.** None. Pure addition, inert until a step declares a `deliverables` array — the
same argument #182 used and it held. The new upload endpoint is admin-gated like every
other dev-tasks route.

**Migration.** No DDL. One infra action: create the private `task-deliverables` bucket. A
doc-only migration note goes in `src/migrations/dev_tasks_step_deliverables.sql` recording
the widened step shape, matching the style of `dev_tasks_process_chain.sql`.

**Tests** — new module `src/test_step_deliverables_184.py`, mirroring
`src/test_process_chain_182.py`. Minimum coverage:
- a step with no `deliverables` key behaves byte-identically to today (back-compat);
- a generic PATCH that changes `value`/`filled_*` → 409;
- a newly inserted deliverable arriving filled → 400;
- fill/clear on a **completed** step → 409; on an open non-current step → **allowed**;
- **T10**: complete with a required-unfilled deliverable → 409 naming the label; with only
  optional-unfilled → succeeds, and the system comment names them;
- limits: over-size 413, disallowed content-type 415, non-URL `link` 400, >8 per step 400;
- §3.6: a task-level stamp on a chain task writes `step.stamp{state,by,at}` **and**
  `completed_at`/`completed_by` **and** hands off; on a legacy checklist it is unchanged.

**Acceptance.** On dev: author a chain step with one required link and one optional file →
Complete is blocked with the T10 message → fill the link → Complete succeeds and the
system comment names the unfilled optional one → the deliverables tab shows both rows →
(Phase 2) the strip shows two cards, and a vision with 6 subtasks still lists them.
