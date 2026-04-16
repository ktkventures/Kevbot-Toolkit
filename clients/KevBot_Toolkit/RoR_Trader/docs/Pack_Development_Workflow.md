# Pack Development Workflow

## Scope

This document covers **user packs** — new indicators contributed via the Pack Builder wizard, which live in `user_packs/<slug>/` and are registered through `pack_registry`.

For **built-in confluence templates** (EMA, MACD, VWAP, UT Bot, RVOL — computed directly by the unified engine), see the [Built-in Template Contract](#built-in-template-contract) section at the bottom. Built-in templates go through a stricter parameter-resolution contract than user packs and have different registration requirements.

## Purpose

This document describes the standardized process for creating, fixing, and validating user packs in the RoR Trader system. It distinguishes between the **user-facing AI fix flow** (what end users will eventually use) and the **dev workflow** (how Kevin and Claude collaborate on pack creation and iteration).

## Two Flows

### 1. User Flow (Production)
- User describes a pack idea in plain English in the Pack Builder wizard
- AI generates manifest, indicator.py, interpreter.py
- User clicks Install
- If issues arise, user clicks "Request Fix" with a description
- AI returns updated code, validation runs, install proceeds
- This flow MUST be self-contained and rely on prompts + validation alone

### 2. Dev Flow (This Document)
- Kevin drives initial pack creation through the wizard (same as user flow)
- When fixes are needed, Claude takes over directly:
  - Reads the actual pack code
  - Inspects DataFrame columns and engine flow on real data
  - Edits pack files directly in `user_packs/<slug>/`
  - Triggers a server reload
  - Verifies the fix works before declaring done
  - Documents the lesson in `pack_builder_context.md`
  - Adds a validation check to `pack_spec.py` or `pack_builder.py` to catch the mistake
- The user flow improves over time as validation rules accumulate from real failures

## Why Two Flows?

The AI fix flow has limitations:
- Only sees what's passed in the prompt
- Can't run scripts to verify behavior
- Can't read engine internals
- Doesn't know if the fix actually worked until validation runs (and validation only catches what we explicitly check)

Claude in dev mode has full access:
- Can read and edit any file
- Can run Python scripts to test column values, dtypes, and flow
- Can inspect the unified engine to understand semantics
- Can verify the fix works by running real data through the pack
- Each iteration's lessons go into the validation rules and context document, making the user flow stronger

## The Process

### Step 1: Create Pack via Wizard
Kevin uses the Pack Builder UI:
1. Pack Info (name, category, display type, description)
2. Generate Structure (AI proposes parameters/outputs/triggers)
3. Refine Structure (manual edits if needed)
4. Generate & Validate Code (AI generates Python, validation runs)
5. Review & Install

Initial generation should usually work for simple packs. If validation fails or behavior is wrong, move to Step 2.

### Step 2: Diagnose Issue (Claude)
Claude reads:
- `user_packs/<slug>/manifest.json`
- `user_packs/<slug>/indicator.py`
- `user_packs/<slug>/interpreter.py`
- Relevant engine code (`unified_engine.py`, `pack_registry.py`, etc.)

Then runs a diagnostic script to verify:
- Are the indicator columns being created correctly?
- Are the values in the expected range/format?
- Does the trigger boolean fire on expected bars?
- Does the level column (if any) have values on every bar?
- Does the engine receive the columns correctly?

### Step 3: Apply Fix Directly
Claude edits `user_packs/<slug>/indicator.py` (and/or manifest.json) directly.
After editing, trigger a server reload:
```bash
touch /home/kevin/projects/Kevbot-Toolkit/clients/KevBot_Toolkit/RoR_Trader/src/api/main.py
```

### Step 4: Verify the Fix
Claude runs a diagnostic script to confirm:
- Column values are correct
- Trigger fires as expected
- Engine receives the data

Then Kevin re-tests in the Strategy Builder UI.

### Step 5: Document the Lesson
For every issue we fix, update:

1. **`src/pack_builder_context.md`** — Add a clear rule explaining what to do (and what NOT to do)
2. **`src/pack_builder.py` `_STRUCTURE_SYSTEM_PROMPT` and rule list** — Reinforce the rule in the generation prompt
3. **`src/pack_builder.py` `_FIX_SYSTEM_PROMPT`** — Reinforce the rule in the fix prompt
4. **`src/pack_builder.py` `validate_parsed_response()`** — Add a validation check that programmatically catches the mistake (regex on the indicator code, or inspection of the manifest)

This ensures the AI flow gets stronger every iteration. Future packs created via the wizard or fixed via Request Fix will not repeat the mistake.

## Files That Strengthen Over Time

| File | Purpose |
|------|---------|
| `src/pack_builder_context.md` | Source of truth for pack architecture. Loaded into structure, code, and fix prompts. Add lessons here. |
| `src/pack_builder.py` `_STRUCTURE_SYSTEM_PROMPT` | Rules for AI Step 2 (structure generation) |
| `src/pack_builder.py` `generate_code_prompt` parts | Rules for AI Step 4 (code generation) |
| `src/pack_builder.py` `_FIX_SYSTEM_PROMPT` | Rules for AI Request Fix flow |
| `src/pack_builder.py` `validate_parsed_response` | Programmatic validation checks |
| `src/pack_spec.py` `validate_manifest` | Manifest schema validation |

## Common Pitfalls (Living List)

### Trigger Direction/Type
- All triggers must use `direction: "BOTH"` and `type: "BOTH"`
- Don't create separate LONG/SHORT or ENTRY/EXIT versions of the same trigger

### Level Columns for L-Type Triggers
- Use `_prev` suffix in the column name for L1 classification
- Do NOT use `.shift()` — the engine adds the 1-bar lag automatically
- Do NOT use `.where(trigger, np.nan)` — level must exist on every bar
- Pattern: `result["my_level_prev"] = result["close"]` (current value, _prev for naming)

### Internal Columns (Never Visualize)
- Level columns (referenced as `level_column` in triggers)
- Candle color columns (referenced in `plot_config.candle_color_column`)
- These should be in `indicator_columns` (so they get computed) but NOT in `column_color_map` or `plot_schema`
- The system auto-excludes them from chart overlays

### Pattern-Based Packs vs Indicator Packs
- Indicator packs (EMA, VWAP, RSI): naturally support L-type intra-bar entries
- Pattern packs (Swing 1-2-3, Strat): can support L-type ONLY if the pattern's "trigger condition" is a simple price level cross
- Pure pattern packs without a meaningful intra-bar level should only use C/CC variants

### Reserved Names
- Trigger prefixes, interpreter keys, and indicator columns must not collide with built-ins
- See `pack_spec.py` BUILTIN_TRIGGER_PREFIXES, BUILTIN_INTERPRETER_KEYS, BUILTIN_INDICATOR_COLUMNS
- Update these lists when new built-in or user packs are added

## Verification Script Template

When diagnosing a pack issue, use a script like this:

```python
import sys
sys.path.insert(0, 'src')
import os; os.chdir('src')
from dotenv import load_dotenv
load_dotenv(override=True)
import pack_registry
pack_registry.scan_and_load_all()

from data_loader import load_market_data, resample_to_timeframe
df = load_market_data('NVDA', days=2, timeframe='1Min', feed='sip', session='RTH')
df = resample_to_timeframe(df, '5Min')

pack = pack_registry.get_pack('<slug>')
params = {k: v.get('default') for k, v in pack.manifest.get('parameters_schema', {}).items()}
df = pack.indicator_func(df, **params)

# Inspect columns
for col in pack.manifest.get('indicator_columns', []):
    if col in df.columns:
        s = df[col]
        print(f'{col}: dtype={s.dtype}, non-NaN={s.notna().sum()}, sample={s.dropna().head(3).tolist()}')

# Run interpreter and triggers
states = pack.interpreter_func(df, **params)
print('State counts:', states.value_counts().to_dict())
triggers = pack.trigger_func(df, **params)
for k, v in (triggers or {}).items():
    print(f'  {k}: {v.sum()} fires')
```

This catches most issues quickly without needing to run a full backtest through the UI.

---

# Built-in Template Contract

Built-in confluence templates (EMA, MACD, VWAP, UT Bot, RVOL) are computed directly by the unified engine in `IncrementalIndicatorEngine` and `TriggerEvaluator`. They have a stricter parameter contract than user packs — adding a new built-in template or parameter requires updates in four specific places, and missing any one of them will raise at strategy load rather than silently drift.

## Why stricter

User pack indicators are pre-computed in the DataFrame by the batch pipeline and read as columns. If a user pack param is wrong, only that pack's output is wrong.

Built-in indicators run live inside the unified engine with O(1) incremental state. The engine reads parameters from a `params: dict[str, Any]` produced by `resolve_strategy_requirements(strategy)`. If any required key is missing, backtest and live silently diverge — one path reads the user's configured value, the other reads a hardcoded fallback. We explicitly don't allow this (see `unified_engine.py` `_INDICATOR_PARAM_SPEC` and the contract validator at the end of `resolve_strategy_requirements`).

## Adding a parameter to an existing template

Example: "I want to expose `ema_alpha_boost` on the EMA Price Position template."

1. Add the parameter to the template's parameter schema in `confluence_groups.py` (the `TEMPLATES` dict). Users can now set it in the Confluence Packs UI.
2. Add it to the `_INDICATOR_PARAM_SPEC` entry for the indicator family in `unified_engine.py`. Map the engine-side key (what the engine reads) to the group-side key (what the user sets):
   ```python
   _INDICATOR_PARAM_SPEC['ema']['params'].append(
       ('ema_alpha_boost', 'alpha_boost'),
   )
   ```
3. Read it strictly in the engine constructor — `params['ema_alpha_boost']`, never `.get(key, default)`.
4. Add or update the entry in the contract test (`test_params_contract` in `test_unified_parity.py`) so a missing key would fail CI.

The contract check at the end of `resolve_strategy_requirements` will refuse to return if any spec'd key is missing. Strategies using the affected indicator will fail at load time with a clear error pointing at the resolver — not quietly run with wrong values.

## Adding a new built-in indicator family

Example: "Bollinger Bands" as a first-class built-in (not a user pack).

Steps, in order:

1. **Compute logic** — `indicators.py`: batch function (`calculate_bbands(df, period, num_std)`). Used by backtest and chart rendering.
2. **Incremental state** — `unified_engine.py` `IncrementalIndicatorEngine`: add fields to `IndicatorState`, the `_update_indicators` branch (`if 'bbands' in self.required`), and write output columns `bb_upper`, `bb_mid`, `bb_lower`, etc.
3. **State classification** — `interpreters.py`: `interpret_bbands(df)` returning a Series of state labels (e.g. `"ABOVE_UPPER"`, `"INSIDE"`, `"BELOW_LOWER"`).
4. **Triggers** — `triggers.py`: `detect_bbands_triggers(df)` returning a dict of bool Series. For L-type (intra-bar level-cross) support, add entries to `INTRABAR_LEVEL_MAP` in `unified_engine.py` with the `param_key` field so column names parameterize correctly from the group.
5. **Template registration** — `confluence_groups.py`:
   - Add the template to the `TEMPLATES` dict with `interpreters`, `indicator_columns`, `triggers`, `display_type`, and user-editable `parameters_schema`.
   - Add to `OVERLAY_TEMPLATES` or `OSCILLATOR_TEMPLATES` for chart classification.
6. **Engine requirements mapping** — `unified_engine.py`:
   - Add to `TEMPLATE_REQUIREMENTS`: `'bbands': ({'bbands'}, 'BBANDS')` — indicator slug(s) and interpreter key.
   - Add to `TRIGGER_PREFIX_TO_TEMPLATE`: `'bbands': 'bbands'` — so `resolve_strategy_requirements` can route trigger IDs to the template.
7. **Parameter contract** — `unified_engine.py` `_INDICATOR_PARAM_SPEC`:
   ```python
   'bbands': {
       'params': [
           ('bbands_period', 'period'),
           ('bbands_num_std', 'num_std'),
       ],
       'templates': ('bbands',),
       'interpreters': ('BBANDS',),
   }
   ```
8. **Strict engine reads** — in `IncrementalIndicatorEngine.__init__`, read `params['bbands_period']` directly. No `.get(key, default)`.
9. **Chart overlay resolution** — `api/routers/strategies.py` `/chart-data`: the template's `indicator_columns` are already resolved via `group.parameters`, but if your column names are parameterized (like `ema_{period}`), add the resolution logic alongside the existing `ema_short`/`ema_mid`/`ema_long` block.
10. **Contract test** — extend `test_params_contract` in `test_unified_parity.py` with a fixture for the new indicator family so the round-trip is pinned.

The contract test guards step 7 specifically. If you add the template (steps 1-6) but forget step 7, any strategy using the new family will raise at load with a message pointing at `_INDICATOR_PARAM_SPEC`.

## Reviewing the contract at a glance

```
resolve_strategy_requirements(strategy)                      [the resolver]
  └── reads: strategy dict + user's confluence groups
  └── uses: _INDICATOR_PARAM_SPEC to know what to populate
  └── returns: params dict with EVERY key required by the engine
  └── raises: if any key is missing (contract violation)

IncrementalIndicatorEngine / TriggerEvaluator                [strict consumers]
  └── reads: params[key] (no .get defaults)
  └── raises: KeyError if a key is missing → always a resolver bug
```

Never add a `.get(key, default)` in the engine to "work around" a missing param. The right fix is always in `_INDICATOR_PARAM_SPEC` or the resolver.
