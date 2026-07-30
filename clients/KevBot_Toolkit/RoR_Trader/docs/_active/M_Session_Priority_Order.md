# M-Session Priority Order

**Source of truth for the order the M session takes work in.** Board #220 section 1
(M's queue) and section 3 (the rules) both RENDER this file — they do not restate it.
Edit here and the dashboard changes; there is no second copy to keep in step.

**Owner: M.** F authored the file to satisfy board #220's section-3 principle — *"That
ordering is itself a RULE, so it belongs rendered from a source file, not typed into the
page."* The content below is Kevin's ruling of 2026-07-30, transcribed; M maintains it
from here and Kevin corrects it.

## Priority order

Kevin, 2026-07-30, on the Ask-AI inbox: *"this Ask AI feature should probably get top
priority to respond to. Otherwise, I'll just have to come into the session and ask you
within the session, which would be duplicative and would not centralize the conversation
(because usually when I ask the AI within the task board, I'm looking for a response
pretty quickly)."*

1. **unanswered `@M` Ask AI** — he is waiting, and expects to be waiting briefly
2. **anything blocked on Kevin** that M can unblock by supplying scope or an answer
3. **alarms** — loop down, stranded branch, failed run parked in a lying status
4. **reviews of finished agent work** — the step that caught 5 real defects green suites passed
5. **dispatch/ship** — moving reviewed work out
6. routine classification and tidy

Kevin filed this as *"a first cut, for M to refine and Kevin to correct"* — so a change to
this list is expected, and belongs in this file rather than in a session's head.

## How M's intended order relates to this

The dashboard's queue reads `m_session_queue_order` (board #220 step 4) — M's **intended**
order, which is a plan and cannot be derived from the board. This list is the RULE that
plan should follow; the store is M's application of it to today's specific tasks. Where
they disagree, the disagreement is the interesting thing and is visible on the page: each
ranked row carries a `why`, which is what answers Kevin's *"why is that third?"*

A task with no row in the store falls back to board order (`priority_phase.priority_seq`).
Absence is a valid state — it means M has not formed an intention about it yet — and the
page says so rather than manufacturing a rank.
