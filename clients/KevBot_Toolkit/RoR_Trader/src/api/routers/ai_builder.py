"""
AI Builder Router — Pack Builder wizard endpoints.

Provides 5 endpoints for the Pack Builder 5-step wizard:
  1. generate-structure  — AI proposes params/outputs/triggers (Step 2)
  2. generate-code       — AI generates manifest + indicator + interpreter (Step 4)
  3. fix                 — AI fixes validation errors (Step 4 auto-fix / Step 5 request fix)
  4. validate            — Run pack_spec validation (no AI call)
  5. install             — Write to user_packs/ and register
"""

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional

from api.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/packs/builder", tags=["pack-builder"])


# =============================================================================
# Request / Response Models
# =============================================================================

class GenerateStructureRequest(BaseModel):
    pack_name: str
    pack_type: str = "tf_confluence"
    category: str = ""
    display_type: str = "oscillator"
    description: str = ""
    pine_script: str = ""
    ai_model: str = "claude-sonnet"


class GenerateCodeRequest(BaseModel):
    pack_name: str
    slug: str
    pack_type: str = "tf_confluence"
    category: str = ""
    display_type: str = "oscillator"
    description: str = ""
    parameters: list = []
    outputs: list = []
    triggers: list = []
    pine_script: str = ""
    ai_model: str = "claude-sonnet"


class FixRequest(BaseModel):
    manifest: dict
    indicator_code: str
    interpreter_code: str
    validation_errors: list = []
    user_description: str = ""
    ai_model: str = "claude-sonnet"


class ValidateRequest(BaseModel):
    manifest: dict
    indicator_code: str
    interpreter_code: str


class InstallRequest(BaseModel):
    manifest: dict
    indicator_code: str
    interpreter_code: str
    pine_script_code: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================

def _categorize_error(error: str) -> str:
    """Assign a validation error to a UI category."""
    lower = error.lower()
    if any(kw in lower for kw in ("missing required", "invalid slug", "invalid pack_type",
                                   "must be", "not in", "collision", "reserved")):
        return "schema"
    if any(kw in lower for kw in ("disallowed import", "disallowed function", "disallowed module",
                                   "unsafe", "forbidden")):
        return "safety"
    if "not found" in lower or "function" in lower:
        return "functions"
    return "execution"


def _errors_to_validation_items(errors: list[str], all_pass: bool = False) -> list[dict]:
    """Convert error strings into the frontend ValidationItem format."""
    if all_pass and not errors:
        return [
            {"id": "schema-ok", "label": "Manifest Schema", "category": "schema", "status": "pass", "message": "All required fields present and valid"},
            {"id": "safety-ok", "label": "Python Safety", "category": "safety", "status": "pass", "message": "No disallowed imports or calls"},
            {"id": "functions-ok", "label": "Function Signatures", "category": "functions", "status": "pass", "message": "All declared functions found"},
        ]

    items = []
    seen_categories = set()
    for i, err in enumerate(errors):
        cat = _categorize_error(err)
        seen_categories.add(cat)
        items.append({
            "id": f"{cat}-{i}",
            "label": err.split(":")[0] if ":" in err else err[:60],
            "category": cat,
            "status": "fail",
            "message": err,
        })

    # Add passing items for categories without errors
    for cat, label in [("schema", "Manifest Schema"), ("safety", "Python Safety"), ("functions", "Function Signatures")]:
        if cat not in seen_categories:
            items.append({
                "id": f"{cat}-ok",
                "label": label,
                "category": cat,
                "status": "pass",
                "message": "Passed",
            })

    return items


def _parse_structure_json(raw_text: str) -> dict:
    """Extract and parse JSON from LLM response, tolerating markdown fencing."""
    text = raw_text.strip()

    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Remove // comments that some LLMs add
    clean_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    return json.loads(text)


# =============================================================================
# Endpoints
# =============================================================================

def _compute_user_pack_metrics(packs: dict, strategies: list,
                               alerts: list) -> dict:
    """Per-pack health metrics for the User Packs page cards.

    A strategy 'uses' a pack if either:
      - a confluence record's interpreter segment matches one of the
        pack's declared interpreters (`{tf}-{INTERP}-{state}` format) —
        this is the pack acting as a confluence GATE, OR
      - an entry/exit trigger ID resolves to the pack via longest-
        matching slug-or-trigger_prefix — the pack acting as a TRIGGER.

    Longest-match resolution is required so `swing_123_test_default_*`
    triggers count for `swing_123_test`, not for `swing_123` (whose
    slug is a strict prefix of the other).

    Returns {slug: {
        strategies_using: int,   # distinct strategies referencing it
        last_triggered:   iso|None,  # last live alert from a pack trigger
        triggered_7d:     int,       # alert count from pack triggers, 7d
        last_gated:       iso|None,  # last live alert from a strategy that
                                     # gates on this pack (gate was open)
    }}

    `alerts` should be a recent window (the endpoint passes ~7 days).
    A None timestamp therefore means "not in the window" — read as
    "not firing recently", which is exactly the signal the card wants.
    """
    from datetime import datetime, timezone, timedelta

    interp_to_slug: dict = {}      # interpreter name -> pack slug
    key_to_slug: dict = {}         # slug OR trigger_prefix -> pack slug
    for slug, pack in packs.items():
        m = pack.manifest or {}
        for interp in (m.get('interpreters') or []):
            interp_to_slug[interp] = slug
        key_to_slug[slug] = slug
        tp = m.get('trigger_prefix')
        if tp:
            key_to_slug[tp] = slug
    sorted_keys = sorted(key_to_slug.keys(), key=len, reverse=True)

    def _trigger_pack(trig) -> str | None:
        t = str(trig or '')
        for k in sorted_keys:
            if t == k or t.startswith(k + '_'):
                return key_to_slug[k]
        return None

    # --- structural usage scan ---
    pack_strats: dict = {slug: set() for slug in packs}       # used anywhere
    pack_gate_strats: dict = {slug: set() for slug in packs}  # used as a GATE
    for strat in strategies:
        sid = strat.get('id')
        cfg = strat.get('config') or {}
        if not isinstance(cfg, dict):
            continue
        # confluence interpreters → gate usage. split('-', 2) caps at 3
        # parts so dash-containing states (e.g. '<-2σ') stay intact.
        for f in ('confluence', 'general_confluences'):
            for rec in (cfg.get(f) or []):
                parts = str(rec).split('-', 2)
                if len(parts) >= 2:
                    s = interp_to_slug.get(parts[1])
                    if s:
                        pack_strats[s].add(sid)
                        pack_gate_strats[s].add(sid)
        # entry/exit triggers → trigger usage
        trigs: list = []
        for f in ('entry_trigger', 'entry_trigger_confluence_id', 'exit_trigger'):
            v = cfg.get(f)
            if v:
                trigs.append(v)
        for f in ('exit_triggers', 'exit_trigger_confluence_ids'):
            trigs.extend(cfg.get(f) or [])
        for t in trigs:
            s = _trigger_pack(t)
            if s:
                pack_strats[s].add(sid)

    # --- alert-driven liveness ---
    cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    last_trig: dict = {slug: None for slug in packs}
    trig_7d: dict = {slug: 0 for slug in packs}
    last_gated: dict = {slug: None for slug in packs}
    # strategy_id -> set of pack slugs it gates on (for last_gated)
    sid_to_gated: dict = {}
    for slug, sids in pack_gate_strats.items():
        for sid in sids:
            sid_to_gated.setdefault(sid, set()).add(slug)

    # Only count alerts from strategies that STILL EXIST. Alerts persist
    # after a strategy is deleted, so without this filter a pack used
    # only by now-deleted strategies would show "used in 0 · triggered
    # 1195×" — a contradictory card. Scoping to current strategies makes
    # the liveness metrics reflect the live fleet, not history.
    current_sids = {s.get('id') for s in strategies}

    for a in alerts:
        ts = a.get('timestamp')
        if not ts:
            continue
        if a.get('strategy_id') not in current_sids:
            continue
        # trigger attribution — which pack's trigger caused this alert
        tp = _trigger_pack(a.get('trigger_id'))
        if tp is not None:
            if last_trig[tp] is None or ts > last_trig[tp]:
                last_trig[tp] = ts
            if ts >= cutoff_7d:
                trig_7d[tp] += 1
        # gate attribution — a fired alert means every gate on that
        # strategy was open, including any pack used as a gate.
        for slug in sid_to_gated.get(a.get('strategy_id'), ()):
            if last_gated[slug] is None or ts > last_gated[slug]:
                last_gated[slug] = ts

    return {slug: {
        'strategies_using': len(pack_strats[slug]),
        'last_triggered': last_trig[slug],
        'triggered_7d': trig_7d[slug],
        'last_gated': last_gated[slug],
    } for slug in packs}


@router.get("/user-packs")
def list_user_packs(user=Depends(get_current_user)):
    """List all installed user packs with their metadata + health metrics.

    Per-pack metrics (2026-05-21) — was hardcoded strategies_using:0:
      - strategies_using: distinct strategies referencing the pack
      - last_triggered / triggered_7d: liveness as a TRIGGER
      - last_gated: liveness as a confluence GATE
    Lets the UserPacks page show at a glance whether a pack is wired
    in AND actually firing. Metrics derive from a 7-day alert window;
    a null timestamp means "not firing in the last 7 days".
    """
    import pack_registry
    packs = pack_registry.get_registered_packs()

    metrics: dict = {}
    try:
        user_id = (user.get('id') or user.get('sub')
                   if isinstance(user, dict) else None)
        if user_id:
            from db import get_admin_client
            from datetime import datetime, timezone, timedelta
            client = get_admin_client()
            strat_rows = client.table('strategies') \
                .select('id,config').eq('user_id', str(user_id)).execute()
            # Recent alerts (7d) — projected to the 3 fields the metric
            # needs; paginated (PostgREST caps responses at 1000).
            since = (datetime.now(timezone.utc)
                     - timedelta(days=7)).isoformat()
            alerts: list = []
            offset = 0
            while offset < 100_000:
                page = client.table('alerts') \
                    .select('strategy_id,trigger_id,timestamp') \
                    .eq('user_id', str(user_id)) \
                    .gte('timestamp', since) \
                    .order('timestamp') \
                    .range(offset, offset + 999).execute().data or []
                alerts.extend(page)
                if len(page) < 1000:
                    break
                offset += 1000
            metrics = _compute_user_pack_metrics(
                packs, strat_rows.data or [], alerts)
    except Exception as e:
        logger.warning("user-packs: metrics computation failed (%s) — "
                       "returning zeros", e)
        metrics = {}

    result = []
    for slug, pack in packs.items():
        m = pack.manifest
        pm = metrics.get(slug) or {}
        result.append({
            "slug": slug,
            "name": m.get("name", slug),
            "version": m.get("version", "1.0.0"),
            "pack_type": m.get("pack_type", "tf_confluence"),
            "category": m.get("category", ""),
            "display_type": m.get("display_type", "oscillator"),
            "description": m.get("description", ""),
            "is_valid": pack.is_valid,
            "validation_errors": pack.validation_errors,
            "parameters_schema": m.get("parameters_schema", {}),
            "plot_schema": m.get("plot_schema", {}),
            "outputs": m.get("outputs", []),
            "output_descriptions": m.get("output_descriptions", {}),
            "triggers": m.get("triggers", []),
            "indicator_columns": m.get("indicator_columns", []),
            "status": "private" if pack.is_valid else "verification",
            "strategies_using": pm.get("strategies_using", 0),
            "last_triggered": pm.get("last_triggered"),
            "triggered_7d": pm.get("triggered_7d", 0),
            "last_gated": pm.get("last_gated"),
        })
    return result


@router.get("/user-packs/{slug}/code")
def get_user_pack_code(slug: str, user=Depends(get_current_user)):
    """Return the actual file contents of a user pack."""
    import pack_registry
    pack = pack_registry.get_pack(slug)
    if not pack:
        raise HTTPException(status_code=404, detail=f"Pack '{slug}' not found")

    result = {"manifest": {}, "indicator_code": "", "interpreter_code": ""}

    manifest_path = pack.pack_dir / "manifest.json"
    if manifest_path.exists():
        result["manifest"] = json.loads(manifest_path.read_text())

    indicator_path = pack.pack_dir / "indicator.py"
    if indicator_path.exists():
        result["indicator_code"] = indicator_path.read_text()

    interpreter_path = pack.pack_dir / "interpreter.py"
    if interpreter_path.exists():
        result["interpreter_code"] = interpreter_path.read_text()

    return result


# =============================================================================
# Parity test — 4-quadrant pack validation
# =============================================================================

class ParityTestRequest(BaseModel):
    """Optional config for the 4-quadrant parity test. Sensible defaults
    cover most stock packs; crypto packs should pass an appropriate
    symbol + session."""
    symbol: str = "SPY"
    primary_tf: str = "1Min"
    secondary_tf: str = "15Min"
    days: int = 7
    session: str = "RTH"
    feed: str = "sip"
    warmup_bars: int = 200


@router.post("/user-packs/{slug}/parity-test")
def run_pack_parity_test_endpoint(
    slug: str,
    req: ParityTestRequest = Body(default_factory=ParityTestRequest),
    user=Depends(get_current_user),
):
    """Run the 4-quadrant parity test for a user pack and return the
    full report. Sync invocation — typical runtime 30-60s on 7d/SPY.

    Result schema (top-level):
      pack_id, symbol, primary_tf, secondary_tf, days,
      overall_verdict ('PASS'|'WARN'|'FAIL'),
      summary (one-liner),
      quadrants: {
        Q1: trigger / primary TF      (parity_score, verdict, ...)
        Q2: interpreter / primary TF  (parity_score, verdict, ...)
        Q3: interpreter / secondary TF (cross-TF shadow path)
        Q4: data fidelity (deferred — always SKIP)
      }

    Use this as the gating check before publishing a pack: a pack
    should reach overall_verdict == 'PASS' on its primary supported
    symbol/timeframe before being trusted in production.
    """
    import pack_registry
    pack = pack_registry.get_pack(slug)
    if pack is None:
        raise HTTPException(
            status_code=404, detail=f"Pack '{slug}' not registered")
    if not pack.is_valid:
        raise HTTPException(
            status_code=400,
            detail=(f"Pack '{slug}' has validation errors and cannot "
                    f"be tested: {pack.validation_errors}"))

    from parity_simulator import run_pack_parity_test_4q
    try:
        result = run_pack_parity_test_4q(
            pack_id=slug,
            symbol=req.symbol,
            primary_tf=req.primary_tf,
            secondary_tf=req.secondary_tf,
            days=req.days,
            session=req.session,
            feed=req.feed,
            warmup_bars=req.warmup_bars,
        )
    except Exception as e:
        logger.exception("[parity-test] %s failed: %s", slug, e)
        raise HTTPException(
            status_code=500,
            detail=f"Parity test crashed: {type(e).__name__}: {e}")

    # Persist the result so the user_packs detail tab can render the
    # last-known verdict on tab open without a fresh test run.
    try:
        from db import save_pack_parity_status
        user_id = (user.get('id') or user.get('sub')
                   if isinstance(user, dict) else None)
        if user_id:
            save_pack_parity_status(
                pack_slug=slug,
                user_id=str(user_id),
                overall_verdict=result.get('overall_verdict', 'FAIL'),
                summary=result.get('summary', ''),
                quadrants=result.get('quadrants', {}),
                test_config={
                    'symbol': req.symbol,
                    'primary_tf': req.primary_tf,
                    'secondary_tf': req.secondary_tf,
                    'days': req.days,
                    'session': req.session,
                    'feed': req.feed,
                    'warmup_bars': req.warmup_bars,
                },
            )
    except Exception as _e:
        # Persistence failure should never break the test response —
        # frontend still gets the fresh result, persistence will
        # retry on the next run.
        logger.warning("[parity-test] persistence failed for %s: %s",
                       slug, _e)
    return result


@router.get("/user-packs/{slug}/parity-status")
def get_pack_parity_status(
    slug: str,
    user=Depends(get_current_user),
):
    """Return the most recent saved parity-test result for this user/pack.

    Returns None (HTTP 200, body=null) if the user has never run the
    test on this pack — frontend should show a "no result yet" state
    and offer the "Run Validation" button.
    """
    import pack_registry
    if pack_registry.get_pack(slug) is None:
        raise HTTPException(
            status_code=404, detail=f"Pack '{slug}' not registered")

    from db import load_pack_parity_status
    user_id = (user.get('id') or user.get('sub')
               if isinstance(user, dict) else None)
    if not user_id:
        raise HTTPException(status_code=401, detail="No user context")
    return load_pack_parity_status(slug, str(user_id))


@router.post("/generate-structure")
def generate_structure(req: GenerateStructureRequest, user=Depends(get_current_user)):
    """Step 2: AI proposes pack structure from description."""
    from pack_builder import generate_structure_prompt
    from api.services.ai_provider import generate_completion, AIProviderError

    system_prompt, user_prompt = generate_structure_prompt(
        pack_name=req.pack_name,
        pack_type=req.pack_type,
        category=req.category,
        display_type=req.display_type,
        description=req.description,
        pine_script=req.pine_script,
    )

    try:
        raw_response = generate_completion(req.ai_model, system_prompt, user_prompt)
    except AIProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        parsed = _parse_structure_json(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Structure parse failed. Raw response:\n%s", raw_response[:500])
        raise HTTPException(
            status_code=422,
            detail=f"AI returned invalid JSON. Try regenerating. Parse error: {e}",
        )

    return {
        "parameters": parsed.get("parameters", []),
        "outputs": parsed.get("outputs", []),
        "triggers": parsed.get("triggers", []),
        "ai_message": parsed.get("summary", f"Proposed structure for {req.pack_name}: "
                                  f"{len(parsed.get('parameters', []))} parameters, "
                                  f"{len(parsed.get('outputs', []))} states, "
                                  f"{len(parsed.get('triggers', []))} triggers."),
    }


@router.post("/generate-code")
def generate_code(req: GenerateCodeRequest, user=Depends(get_current_user)):
    """Step 4: AI generates full pack code from refined structure."""
    from pack_builder import generate_code_prompt, parse_llm_response, validate_parsed_response
    from api.services.ai_provider import generate_completion, AIProviderError

    system_prompt, user_prompt = generate_code_prompt(
        pack_name=req.pack_name,
        slug=req.slug,
        pack_type=req.pack_type,
        category=req.category,
        display_type=req.display_type,
        description=req.description,
        parameters=req.parameters,
        outputs=req.outputs,
        triggers=req.triggers,
        pine_script=req.pine_script,
    )

    try:
        raw_response = generate_completion(req.ai_model, system_prompt, user_prompt, max_tokens=12000)
    except AIProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    success, parsed, parse_errors = parse_llm_response(raw_response)

    if not success:
        return {
            "manifest": parsed.get("manifest") or {},
            "indicator_code": parsed.get("indicator_code") or "",
            "interpreter_code": parsed.get("interpreter_code") or "",
            "pine_script_code": parsed.get("pine_script_code"),
            "validation": _errors_to_validation_items(parse_errors),
            "ai_message": f"Code generation partially failed: {'; '.join(parse_errors)}",
        }

    is_valid, validation_errors = validate_parsed_response(parsed)

    return {
        "manifest": parsed["manifest"],
        "indicator_code": parsed["indicator_code"],
        "interpreter_code": parsed["interpreter_code"],
        "pine_script_code": parsed.get("pine_script_code"),
        "validation": _errors_to_validation_items(validation_errors, all_pass=is_valid),
        "ai_message": (
            f"Code generated and validated. All checks passed!"
            if is_valid
            else f"Code generated with {len(validation_errors)} validation issue(s). Use Auto-Fix or Request Fix to resolve."
        ),
    }


@router.post("/fix")
def fix_code(req: FixRequest, user=Depends(get_current_user)):
    """Step 4 auto-fix / Step 5 request fix: AI corrects validation errors."""
    from pack_builder import generate_fix_prompt, parse_llm_response, validate_parsed_response
    from api.services.ai_provider import generate_completion, AIProviderError

    parsed_files = {
        "manifest": req.manifest,
        "indicator_code": req.indicator_code,
        "interpreter_code": req.interpreter_code,
    }

    system_prompt, user_prompt = generate_fix_prompt(
        parsed_files=parsed_files,
        validation_errors=req.validation_errors,
        user_description=req.user_description,
    )

    try:
        raw_response = generate_completion(req.ai_model, system_prompt, user_prompt, max_tokens=12000)
    except AIProviderError as e:
        raise HTTPException(status_code=502, detail=str(e))

    success, parsed, parse_errors = parse_llm_response(raw_response)

    if not success:
        return {
            "manifest": parsed.get("manifest") or req.manifest,
            "indicator_code": parsed.get("indicator_code") or req.indicator_code,
            "interpreter_code": parsed.get("interpreter_code") or req.interpreter_code,
            "pine_script_code": parsed.get("pine_script_code"),
            "validation": _errors_to_validation_items(parse_errors),
            "ai_message": f"Fix attempt failed to produce valid code: {'; '.join(parse_errors)}",
        }

    is_valid, validation_errors = validate_parsed_response(parsed)

    return {
        "manifest": parsed["manifest"],
        "indicator_code": parsed["indicator_code"],
        "interpreter_code": parsed["interpreter_code"],
        "pine_script_code": parsed.get("pine_script_code"),
        "validation": _errors_to_validation_items(validation_errors, all_pass=is_valid),
        "ai_message": (
            "Fix applied — all validation checks now pass!"
            if is_valid
            else f"Fix applied but {len(validation_errors)} issue(s) remain. You can try another fix attempt."
        ),
    }


@router.post("/validate")
def validate_pack(req: ValidateRequest, user=Depends(get_current_user)):
    """Re-validate pack code (no AI call). Used after manual edits."""
    from pack_builder import validate_parsed_response

    parsed = {
        "manifest": req.manifest,
        "indicator_code": req.indicator_code,
        "interpreter_code": req.interpreter_code,
    }

    is_valid, errors = validate_parsed_response(parsed)

    return {
        "is_valid": is_valid,
        "validation": _errors_to_validation_items(errors, all_pass=is_valid),
    }


@router.post("/install")
def install_pack(req: InstallRequest, user=Depends(get_current_user)):
    """Step 5: Write pack to user_packs/ and register in pipeline."""
    from pack_builder import validate_parsed_response, install_pack_from_parsed

    parsed = {
        "manifest": req.manifest,
        "indicator_code": req.indicator_code,
        "interpreter_code": req.interpreter_code,
        "pine_script_code": req.pine_script_code,
    }

    # Safety check: validate before install
    is_valid, errors = validate_parsed_response(parsed)
    if not is_valid:
        return {
            "success": False,
            "slug": req.manifest.get("slug", ""),
            "errors": errors,
        }

    success, slug, install_errors = install_pack_from_parsed(parsed)

    # Ensure confluence group exists in DB (register_pack creates in-memory
    # but _ensure_confluence_group may skip DB if no user context at startup)
    if success:
        try:
            from confluence_groups import load_confluence_groups, save_confluence_groups, ConfluenceGroup, PlotSettings
            groups = load_confluence_groups()
            default_id = f"{slug}_default"
            if not any(g.id == default_id for g in groups):
                manifest = req.manifest
                default_params = {k: v["default"] for k, v in manifest.get("parameters_schema", {}).items()}
                plot_schema = manifest.get("plot_schema", {})
                default_colors = {k: v["default"] for k, v in plot_schema.items() if v.get("type") == "color"}
                groups.append(ConfluenceGroup(
                    id=default_id,
                    base_template=slug,
                    version="Default",
                    description="",
                    enabled=True,
                    is_default=True,
                    parameters=default_params,
                    plot_settings=PlotSettings(colors=default_colors),
                ))
                save_confluence_groups(groups)
        except Exception as e:
            logger.warning("Failed to create DB confluence group for %s: %s", slug, e)

    # Auto-parity check: run a fast Q1+Q2 parity test on the just-installed
    # pack so the wizard can surface a verdict before the user trusts it.
    # Skipped on install failure or batch-only packs (no incremental_class).
    # Q3+Q4 are not run here — they need a cross-TF config the wizard doesn't
    # know; the user_packs detail page exposes the full 4Q test for those.
    parity_summary = None
    if success and slug:
        try:
            import pack_registry as _pr
            pack = _pr.get_pack(slug)
            if pack is not None and pack.incremental_class is not None:
                from parity_simulator import (
                    run_pack_parity_test,
                    run_quadrant_2_interpreter_primary,
                )
                # Q1: pick the first declared trigger as the entry trigger
                triggers_list = req.manifest.get('triggers', [])
                q1 = None
                if triggers_list:
                    base = triggers_list[0].get('base', '')
                    try:
                        q1_full = run_pack_parity_test(
                            pack_id=slug, entry_trigger=base,
                            symbol='SPY', timeframe='1Min', days=5,
                        )
                        q1 = {
                            'verdict': q1_full['verdict'],
                            'parity_score': q1_full['parity_score'],
                            'matched': len(q1_full['matched']),
                            'divergent': (len(q1_full['backtest_only'])
                                          + len(q1_full['live_only'])),
                        }
                    except Exception as _e:
                        logger.warning(
                            "[install] Q1 parity test crashed for %s: %s",
                            slug, _e)
                        q1 = {'verdict': 'ERROR', 'explanation': str(_e)}
                # Q2: interpreter parity
                q2 = None
                try:
                    q2_full = run_quadrant_2_interpreter_primary(
                        pack_id=slug, symbol='SPY', timeframe='1Min', days=5,
                    )
                    q2 = {
                        'verdict': q2_full['verdict'],
                        'parity_score': q2_full.get('parity_score'),
                        'compared': q2_full.get('compared'),
                        'matched': q2_full.get('matched'),
                    }
                except Exception as _e:
                    logger.warning(
                        "[install] Q2 parity test crashed for %s: %s",
                        slug, _e)
                    q2 = {'verdict': 'ERROR', 'explanation': str(_e)}
                # Q5: Synthetic-strategy probe — runs the FULL integration
                # path (TriggerEvaluator dispatch + parity-service replay
                # + position state machine + cross-TF shadow engine) on
                # a real symbol/window. Q1/Q2 catch math bugs in
                # isolation; Q5 catches integration bugs that have
                # historically slipped through. See Phase B drill in
                # docs/Parity_Trust_Roadmap_2026-04-29.md.
                q5 = None
                try:
                    user_id_for_probe = (
                        user.get('id') or user.get('sub')
                        if isinstance(user, dict) else None
                    )
                    if user_id_for_probe:
                        from api.services.synthetic_probe import run_pack_probe
                        probe_result = run_pack_probe(
                            slug, str(user_id_for_probe),
                            symbol='SPY', timeframe='1Min', days=7,
                            add_cross_tf_gate=False,  # don't pair on
                            # install-time — keep the test focused on
                            # the candidate pack's own fire path
                            cleanup=True,
                        )
                        # Map probe status → 4Q verdict ranks
                        score = probe_result.get('score')
                        st = probe_result.get('status')
                        if st == 'pass':
                            v = 'PASS'
                        elif st == 'partial' and score is not None and score >= 0.85:
                            v = 'WARN'
                        elif st in ('partial', 'fail'):
                            v = 'FAIL'
                        elif st == 'skipped':
                            v = 'SKIP'
                        else:
                            v = 'ERROR'
                        q5 = {
                            'verdict': v,
                            'parity_score': score,
                            'matched': probe_result.get('matched_count'),
                            'stored': probe_result.get('stored_count'),
                            'replay_only': probe_result.get('replay_only_count'),
                            'most_common_failing_gate': (
                                probe_result.get('most_common_failing_gate')),
                        }
                except Exception as _e:
                    logger.warning(
                        "[install] Q5 synthetic probe crashed for %s: %s",
                        slug, _e)
                    q5 = {'verdict': 'ERROR', 'explanation': str(_e)}

                # Roll up to a single verdict — overall is FAIL if any
                # explicit FAIL, WARN if any WARN, else PASS.
                parts = [p for p in (q1, q2, q5) if p]
                ranks = {'PASS': 1, 'SKIP': 0, 'WARN': 2, 'FAIL': 3,
                         'ERROR': 3}
                overall = 'PASS'
                if parts:
                    worst = max(ranks.get(p['verdict'], 0) for p in parts)
                    overall = {3: 'FAIL', 2: 'WARN', 1: 'PASS',
                               0: 'PASS'}[worst]
                parity_summary = {
                    'overall_verdict': overall,
                    'Q1': q1,
                    'Q2': q2,
                    'Q5': q5,
                    'note': ('Q5 = synthetic-strategy end-to-end probe. '
                             'Run the full 4-quadrant test (Q3 cross-TF) '
                             'from the user_packs detail page when ready.'),
                }
                # Persist as initial parity_status row so the detail page
                # opens with the install-time verdict already populated.
                try:
                    user_id = (user.get('id') or user.get('sub')
                               if isinstance(user, dict) else None)
                    if user_id:
                        from db import save_pack_parity_status
                        save_pack_parity_status(
                            pack_slug=slug,
                            user_id=str(user_id),
                            overall_verdict=overall,
                            summary=(
                                f'Install-time check: '
                                f'Q1={q1["verdict"] if q1 else "—"}, '
                                f'Q2={q2["verdict"] if q2 else "—"}, '
                                f'Q5={q5["verdict"] if q5 else "—"} '
                                f'(SPY/1Min)'),
                            quadrants={'Q1': q1, 'Q2': q2, 'Q5': q5},
                            test_config={
                                'symbol': 'SPY',
                                'primary_tf': '1Min',
                                'secondary_tf': '15Min',
                                'days': 5,
                                'session': 'RTH',
                                'install_time': True,
                            },
                        )
                except Exception as _e:
                    logger.warning(
                        "[install] persist parity status failed for %s: %s",
                        slug, _e)
        except Exception as _e:
            logger.warning(
                "[install] auto-parity gate failed for %s: %s",
                slug, _e)

    return {
        "success": success,
        "slug": slug,
        "errors": install_errors,
        "parity": parity_summary,
    }


# =============================================================================
# Chart Preview — run a pack's indicator/interpreter on sample data
# =============================================================================

class PackPreviewRequest(BaseModel):
    symbol: str = "NVDA"
    timeframe: str = "5Min"
    days: int = 5
    session: str = "RTH"


@router.post("/user-packs/{slug}/preview")
def pack_preview(slug: str, req: PackPreviewRequest, user=Depends(get_current_user)):
    """Run a user pack's indicator + interpreter on sample data for chart preview.

    Returns OHLCV bars with state classification and trigger events.
    """
    import pandas as pd
    import numpy as np
    import pack_registry
    from data_loader import load_market_data, resample_to_timeframe
    from confluence_groups import load_confluence_groups

    pack = pack_registry.get_pack(slug)
    if not pack or not pack.is_valid:
        raise HTTPException(404, f"Pack '{slug}' not found or invalid")

    if not pack.indicator_func or not pack.interpreter_func or not pack.trigger_func:
        raise HTTPException(400, f"Pack '{slug}' is missing required functions")

    # Load 1-min bars and resample (per CLAUDE.md: always resample from 1-min)
    try:
        df = load_market_data(req.symbol, days=req.days, timeframe='1Min',
                              feed='sip', session=req.session)
    except Exception as e:
        logger.exception("Failed to load market data for preview")
        raise HTTPException(500, f"Failed to load market data: {e}")

    if df is None or len(df) == 0:
        return {"bars": [], "triggers": [], "states": [],
                "indicator_columns": [], "display_type": "overlay"}

    # Resample to target timeframe
    if req.timeframe != '1Min':
        df = resample_to_timeframe(df, req.timeframe)

    if len(df) == 0:
        return {"bars": [], "triggers": [], "states": [],
                "indicator_columns": [], "display_type": "overlay"}

    # Get parameters — try confluence group first, fall back to manifest defaults
    params = {}
    try:
        groups = load_confluence_groups()
        group = next((g for g in groups if g.base_template == slug), None)
        if group:
            params = dict(group.parameters)
    except Exception as e:
        logger.debug("Could not load confluence groups for params: %s", e)
    if not params:
        params = {k: v.get("default") for k, v in pack.manifest.get("parameters_schema", {}).items()}
    # Filter out internal keys
    params = {k: v for k, v in params.items() if not k.startswith('_')}

    # Run indicator
    try:
        df = pack.indicator_func(df, **params)
    except Exception as e:
        logger.exception("Pack indicator failed for %s", slug)
        raise HTTPException(500, f"Indicator function failed: {e}")

    # Run interpreter (returns Series of state strings)
    states_series = None
    try:
        states_series = pack.interpreter_func(df, **params)
    except Exception as e:
        logger.warning("Pack interpreter failed for %s: %s", slug, e)

    # Run trigger detection (returns dict of boolean Series)
    trigger_dict = {}
    try:
        trigger_dict = pack.trigger_func(df, **params) or {}
    except Exception as e:
        logger.warning("Pack trigger detection failed for %s: %s", slug, e)

    # Build response
    manifest = pack.manifest
    indicator_columns = manifest.get("indicator_columns", [])
    display_type = manifest.get("display_type", "overlay")
    output_states = manifest.get("outputs", [])
    trigger_defs = {t["base"]: t for t in manifest.get("triggers", [])}
    trigger_prefix = manifest.get("trigger_prefix", slug)

    # Serialize bars
    bars = []
    reset_df = df.reset_index()
    for i, row in reset_df.iterrows():
        ts = row.get('timestamp', row.name)
        if hasattr(ts, 'isoformat'):
            ts = ts.isoformat()

        bar = {
            "timestamp": str(ts),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "volume": int(row.get("volume", 0)),
        }

        # Add state
        if states_series is not None and i < len(states_series):
            val = states_series.iloc[i] if hasattr(states_series, 'iloc') else None
            bar["state"] = str(val) if pd.notna(val) else None
        else:
            bar["state"] = None

        # Add indicator values
        for col in indicator_columns:
            if col in reset_df.columns:
                val = row.get(col)
                if pd.isna(val):
                    bar[col] = None
                elif isinstance(val, (bool, np.bool_)):
                    bar[col] = bool(val)
                elif isinstance(val, str):
                    bar[col] = val
                else:
                    try:
                        bar[col] = float(val)
                    except (TypeError, ValueError):
                        bar[col] = str(val)

        bars.append(bar)

    # Serialize trigger events — use original df index (DatetimeIndex)
    trigger_events = []
    for key, series in trigger_dict.items():
        # Strip trigger prefix to get base name
        base = key[len(trigger_prefix) + 1:] if key.startswith(trigger_prefix + "_") else key
        tdef = trigger_defs.get(base, {})

        try:
            fired = series[series == True]
            for idx in fired.index:
                ts = idx
                if hasattr(ts, 'isoformat'):
                    ts = ts.isoformat()
                trigger_events.append({
                    "timestamp": str(ts),
                    "trigger_id": key,
                    "trigger_name": tdef.get("name", key),
                    "direction": tdef.get("direction", "BOTH"),
                    "type": tdef.get("type", "ENTRY"),
                })
        except Exception as e:
            logger.warning("Failed to serialize triggers for %s: %s", key, e)

    # Sort trigger events by time
    trigger_events.sort(key=lambda e: e["timestamp"])

    # Include plot config for reference lines, line styles, etc.
    plot_config = manifest.get("plot_config", {})

    return {
        "bars": bars,
        "triggers": trigger_events,
        "states": output_states,
        "indicator_columns": indicator_columns,
        "display_type": display_type,
        "plot_config": plot_config,
    }
