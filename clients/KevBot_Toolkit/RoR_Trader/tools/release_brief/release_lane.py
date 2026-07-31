#!/usr/bin/env python3
"""WHO owns a ship step -- the release lane's owner set, defined ONCE. Board #269.

`R` is the release SESSION (the conductor) and `R-A` its headless executor
(charter §1, board #229). BOTH own ship steps, and the board carries both: chains
authored before the split name `R`, everything since names `R-A`.

WHY THIS IS ITS OWN MODULE, AND NOT A CONSTANT INSIDE ONE OF THE TOOLS
----------------------------------------------------------------------
Because it WAS a constant inside one of the tools, and that is precisely how
board #269 happened. The #229 split taught `car_ship_tick.py` and the frontend
about `R-A` and left `brief_gen.py` behind on `== "R"`. The consequence was not
a degraded generator, it was a BLIND one: on 07-31 it saw **zero cars on a board
with ten**, refusing chains whose step literally read `(R-A) Ship via a release
train`, while the sweep could happily TICK the very steps the generator could
not see. R hand-authored Waves 34, 35 and 36 for that one reason, and #249 sat
staged and stranded until a human noticed.

So the fix cannot be a third literal -- a third literal is the same trap moved
one file over. It has to be a single definition every reader imports.

It cannot live in either existing tool, either: `car_ship_tick` imports
`brief_gen` (for `_step_owner` and the branch resolver), so defining the set in
`car_ship_tick` and importing it back into `brief_gen` closes an import CYCLE
whose success depends on which module Python happens to load first. A leaf
module that imports nothing has no such ordering hazard, and both tools --
plus `backfill_car_status_249.py`, plus any future reader -- can depend on it.

THE ONE COPY THAT MUST REMAIN
-----------------------------
`frontend/src/views/AdminDispatchPage.tsx` exports `SHIP_STEP_OWNERS`, because
TypeScript cannot import a Python tuple. That copy is unavoidable and is PINNED:
`src/test_shipping_lane_unmerged_branch_245.py` parses both files and asserts the
two lists are equal. Inside the PYTHON tree there is exactly one definition, and
`test_brief_gen.py` (rail `t_269_one_definition_in_the_python_tree`) fails loudly
the moment a second one appears anywhere under the app root.
"""

SHIP_OWNERS = ("R", "R-A")


def is_ship_owner(owner):
    """True when a chain step's owner/role belongs to the release lane.

    Case-insensitive and None-safe: every call site was already hand-writing
    `(_step_owner(s) or "").upper()`, and the site that wrote `== "R"` after it
    instead of a membership test is board #269. Exposing the test as a function
    means the next reader gets the membership semantics for free rather than
    re-deriving them -- and re-deriving them is what went wrong.

    Deliberately EXACT beyond case: `R2`, `RA` and `R-B` are not release owners.
    This widens who may own a ship step; it does not change what a car IS
    (that is board #245's resolver).
    """
    return (owner or "").upper() in SHIP_OWNERS
