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

@router.get("/user-packs")
def list_user_packs(user=Depends(get_current_user)):
    """List all installed user packs with their metadata."""
    import pack_registry
    packs = pack_registry.get_registered_packs()
    result = []
    for slug, pack in packs.items():
        m = pack.manifest
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

    return {
        "success": success,
        "slug": slug,
        "errors": install_errors,
    }
