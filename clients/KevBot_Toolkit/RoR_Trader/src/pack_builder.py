"""
Pack Builder — Prompt Assembly & Response Parser
==================================================

Generates structured prompts for LLMs to create confluence packs,
and parses LLM responses back into installable pack files.

Usage:
    from pack_builder import (
        generate_prompt,
        parse_llm_response,
        install_pack_from_parsed,
    )
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pack_spec import validate_manifest, validate_python_file, validate_function_exists


# =============================================================================
# CONTEXT DOCUMENT
# =============================================================================

def _load_context_document() -> str:
    """Load the architecture context document."""
    context_path = Path(__file__).parent / "pack_builder_context.md"
    if context_path.exists():
        return context_path.read_text()
    return "(Architecture context document not found)"


# =============================================================================
# PROMPT ASSEMBLY
# =============================================================================

def generate_prompt(
    pack_description: str,
    pack_type: str = "tf_confluence",
    pine_script: str = "",
    parameters: Optional[List[Dict]] = None,
    category: str = "",
    display_type: str = "overlay",
) -> str:
    """
    Assemble a structured prompt combining architecture context with user inputs.

    Args:
        pack_description: Plain-language description of desired indicator behavior
        pack_type: Pack type (currently only "tf_confluence")
        pine_script: Optional TradingView Pine Script code to translate
        parameters: Optional list of parameter definitions
            [{"name": "period", "type": "int", "default": 14, "label": "Period"}]
        category: Optional category hint (e.g., "Momentum", "Trend")
        display_type: How the indicator renders on charts
            ("overlay", "oscillator", or "hidden")

    Returns:
        Complete prompt string ready to copy to an LLM.
    """
    context = _load_context_document()

    # Build the user request section
    request_parts = []

    request_parts.append("## Your Task\n")
    request_parts.append(
        "Create a complete confluence pack for the RoR Trader platform "
        "based on the following description:\n"
    )
    request_parts.append(f"**Description:** {pack_description}\n")

    if category:
        request_parts.append(f"**Category:** {category}\n")

    if display_type == "overlay":
        request_parts.append(
            f"**Display Type:** `{display_type}` — "
            "This indicator draws lines/bands on the price chart. "
            "Set `display_type` to `\"overlay\"` in the manifest.\n"
        )
    elif display_type == "oscillator":
        request_parts.append(
            f"**Display Type:** `{display_type}` — "
            "This indicator renders in a separate pane below the price chart. "
            "Set `display_type` to `\"oscillator\"` in the manifest.\n"
        )
    else:
        request_parts.append(
            f"**Display Type:** `{display_type}` — "
            "This indicator has no chart rendering. "
            "Set `display_type` to `\"hidden\"` in the manifest.\n"
        )

    if parameters:
        request_parts.append("\n**User-specified parameters:**")
        for p in parameters:
            param_str = f"- `{p['name']}` ({p.get('type', 'int')}): {p.get('label', p['name'])}"
            if 'default' in p:
                param_str += f", default={p['default']}"
            if 'min' in p:
                param_str += f", min={p['min']}"
            if 'max' in p:
                param_str += f", max={p['max']}"
            request_parts.append(param_str)
        request_parts.append("")

    if pine_script.strip():
        request_parts.append("\n**TradingView Pine Script to translate:**")
        request_parts.append(
            "The user has provided Pine Script code below. Translate the "
            "indicator logic to Python following the Pine Script translation "
            "reference in the context document, then build the full pack "
            "spec around it.\n"
        )
        request_parts.append(f"```pine\n{pine_script.strip()}\n```\n")

    request_parts.append(
        "\n## Requirements\n"
        "1. Generate all three files: manifest.json, indicator.py, interpreter.py\n"
        "2. Follow the Pack Spec schema exactly as documented above\n"
        "3. Choose a descriptive slug (lowercase, underscores)\n"
        "4. Define 3-7 mutually exclusive output states\n"
        "5. Define 2-6 meaningful triggers (entry/exit signals)\n"
        "6. Use vectorized pandas/numpy operations where possible\n"
        "7. Only import pandas, numpy, and math — nothing else\n"
        "8. Return None for bars with insufficient data (NaN values)\n"
        "9. Trigger keys must be {trigger_prefix}_{base} matching manifest\n"
        "10. Do NOT use any reserved names listed in the reference\n"
        "11. Set display_type: 'overlay' if indicator draws on the price chart "
        "(EMAs, bands), 'oscillator' if it belongs in a separate pane "
        "(RSI, Stochastic), or 'hidden' if no chart rendering needed\n"
        "12. Include column_color_map mapping each plottable indicator_column "
        "to its plot_schema color key (omit non-plottable columns like bandwidth)\n"
        "13. Outputs MUST be mutually exclusive zones — every bar maps to "
        "exactly one state. Do NOT mix zone states with condition flags "
        "(e.g., don't have UPPER_ZONE + SQUEEZE as separate outputs if "
        "squeeze can overlap with any zone). Instead, encode conditions as "
        "triggers or combine them into the zone names (SQUEEZE_UPPER, etc.)\n"
        "14. Test your thresholds mentally against typical 1-minute intraday data "
        "— no single output state should dominate 90%+ of bars\n"
        "15. ALSO generate a TradingView Pine Script v5 equivalent of the indicator. "
        "Output it as a fourth code block fenced with ```pine. "
        "The Pine Script should produce the same visual output as indicator.py "
        "when loaded in TradingView. Include //@version=5 header and indicator() call.\n"
    )

    user_request = "\n".join(request_parts)

    # Combine context + request
    prompt = f"{context}\n\n---\n\n{user_request}"

    return prompt


# =============================================================================
# STRUCTURED PROMPT ASSEMBLY (for API endpoints)
# =============================================================================

_STRUCTURE_SYSTEM_PROMPT = """\
You are an expert at designing technical indicator packs for a trading platform.

A pack has three parts:
- **parameters**: Configurable inputs (type: int, float, str, or bool). Each needs: name, type, default, label. Optional: min, max.
- **outputs**: Mutually exclusive state classifications. Every bar maps to exactly one state. 3-7 states typical. Use UPPER_SNAKE_CASE.
- **triggers**: Trade entry/exit signals. Each needs: name, base (snake_case key), direction (LONG/SHORT/BOTH), type (ENTRY/EXIT), execution (bar_close).

Rules:
- Output states must be mutually exclusive (no overlapping conditions)
- Trigger base keys must be snake_case, unique, and descriptive
- CRITICAL: Trigger base keys must NOT start with the indicator name/prefix. The system automatically prepends a trigger_prefix. For example, if the pack is "RSI Zones" with trigger_prefix "rsi", use base "cross_above_midline" NOT "rsi_cross_above_midline" — the system creates "rsi_cross_above_midline" automatically.
- LONG triggers fire on bullish conditions, SHORT on bearish
- Include both ENTRY and EXIT triggers. EXIT triggers use type "EXIT", not "ENTRY"
- Each trigger should specify from_state and to_state — the output states it transitions between. For example, a trigger that fires when RSI crosses from BEARISH_NEUTRAL into BULLISH_NEUTRAL would have from_state="BEARISH_NEUTRAL", to_state="BULLISH_NEUTRAL"
- Parameters should cover the key tunables of the indicator
- Design for 1-minute intraday data — ensure no single state dominates 90%+ of bars

Respond with ONLY a JSON object. No markdown fencing, no explanation, no text outside the JSON.

Required format:
{
  "parameters": [{"name": "...", "type": "int|float|str|bool", "default": ..., "label": "...", "min": ..., "max": ...}],
  "outputs": [{"code": "STATE_NAME", "description": "When this state occurs"}],
  "triggers": [{"name": "Human Name", "base": "snake_case_key", "sentiment": "bullish|bearish|neutral", "direction": "LONG|SHORT|BOTH", "type": "ENTRY|EXIT", "execution": "bar_close", "from_state": "STATE_A", "to_state": "STATE_B"}],
  "summary": "Brief description of the proposed structure"
}
"""


def generate_structure_prompt(
    pack_name: str,
    pack_type: str = "tf_confluence",
    category: str = "",
    display_type: str = "oscillator",
    description: str = "",
    pine_script: str = "",
) -> tuple:
    """
    Build (system_prompt, user_prompt) for Step 2: Generate Structure.

    The LLM proposes parameters, outputs, and triggers based on the description.
    Returns JSON only — no code generation at this stage.
    """
    parts = []
    parts.append(f"Create a pack structure for: **{pack_name}**\n")
    parts.append(f"Description: {description}\n")
    if category:
        parts.append(f"Category: {category}")
    parts.append(f"Display type: {display_type}")
    parts.append(f"Pack type: {pack_type}")

    if pine_script.strip():
        parts.append(f"\nTradingView Pine Script reference:\n```pine\n{pine_script.strip()}\n```")
        parts.append("Use this Pine Script to inform the parameters, states, and triggers.")

    return _STRUCTURE_SYSTEM_PROMPT, "\n".join(parts)


def generate_code_prompt(
    pack_name: str,
    slug: str,
    pack_type: str = "tf_confluence",
    category: str = "",
    display_type: str = "oscillator",
    description: str = "",
    parameters: Optional[List[Dict]] = None,
    outputs: Optional[List[Dict]] = None,
    triggers: Optional[List[Dict]] = None,
    pine_script: str = "",
) -> tuple:
    """
    Build (system_prompt, user_prompt) for Step 4: Generate Code.

    Uses the full context document as system prompt and the refined
    structure from Step 3 as the user prompt.
    """
    context = _load_context_document()

    parts = []
    parts.append(f"## Your Task\n")
    parts.append(f"Generate a complete confluence pack for: **{pack_name}**\n")
    parts.append(f"Description: {description}")
    parts.append(f"Slug: `{slug}`")
    parts.append(f"Category: {category}")
    parts.append(f"Display type: {display_type}")
    parts.append(f"Pack type: {pack_type}\n")

    if parameters:
        parts.append("### Parameters (use these exactly)")
        for p in parameters:
            line = f"- `{p['name']}` ({p.get('type', 'int')}): {p.get('label', p['name'])}"
            if 'default' in p and p['default'] not in (None, ''):
                line += f", default={p['default']}"
            if 'min' in p and p['min'] not in (None, ''):
                line += f", min={p['min']}"
            if 'max' in p and p['max'] not in (None, ''):
                line += f", max={p['max']}"
            parts.append(line)
        parts.append("")

    if outputs:
        parts.append("### Output States (use these exactly)")
        for o in outputs:
            parts.append(f"- `{o['code']}` — {o.get('description', '')}")
        parts.append("")

    if triggers:
        parts.append("### Triggers (use these exactly)")
        for t in triggers:
            direction = t.get('direction', 'BOTH')
            ttype = t.get('type', 'ENTRY')
            execution = t.get('execution', 'bar_close')
            parts.append(f"- `{t['base']}` — {t.get('name', t['base'])} ({direction}, {ttype}, {execution})")
        parts.append("")

    if pine_script.strip():
        parts.append("### Pine Script Reference")
        parts.append(f"```pine\n{pine_script.strip()}\n```")
        parts.append("Translate this indicator logic to Python.\n")

    parts.append("### Requirements")
    parts.append("1. Generate all three files: manifest.json, indicator.py, interpreter.py")
    parts.append("2. Follow the Pack Spec schema exactly as documented above")
    parts.append(f"3. Use slug `{slug}` and trigger_prefix `{slug.split('_')[0]}` (or a short unique prefix)")
    parts.append("4. Use the parameters, outputs, and triggers defined above — do not invent new ones")
    parts.append("5. Use vectorized pandas/numpy operations where possible")
    parts.append("6. Only import pandas, numpy, and math — nothing else")
    parts.append("7. Return None for bars with insufficient data (NaN values)")
    parts.append("8. Outputs MUST be mutually exclusive — every bar maps to exactly one state")
    parts.append("9. Include column_color_map mapping each plottable indicator_column to its plot_schema color key")
    parts.append("10. CRITICAL — Trigger naming: In the manifest, trigger `base` must NOT start with the trigger_prefix. "
                 "The system automatically creates the full key as `{trigger_prefix}_{base}`. "
                 "Example: trigger_prefix='rsi', base='cross_above_midline' → key is 'rsi_cross_above_midline'. "
                 "WRONG: base='rsi_cross_above_midline' → would create 'rsi_rsi_cross_above_midline'.")
    parts.append("11. In detect_*_triggers(), the dict keys must be `{trigger_prefix}_{base}` matching the manifest triggers exactly")
    parts.append("12. Exit triggers must use type 'EXIT' in the manifest, not 'ENTRY'")

    user_prompt = "\n".join(parts)
    return context, user_prompt


_FIX_SYSTEM_PROMPT = """\
You are fixing a technical indicator pack for a trading platform.

The user will provide:
1. The current manifest.json, indicator.py, and interpreter.py code
2. A list of validation errors
3. Optionally, a description of what's wrong

Your job is to make minimal, targeted fixes to resolve the errors. Do NOT rewrite working code.

Return your corrected code in the same format:
```json
// manifest.json
{ ... }
```

```python
# indicator.py
...
```

```python
# interpreter.py
...
```

Rules:
- Only import pandas, numpy, and math
- Outputs must be mutually exclusive
- Trigger keys must be {trigger_prefix}_{base}
- Functions must match the names declared in the manifest
- Do not use open(), exec(), eval(), os, sys, subprocess, or any I/O
"""


def generate_fix_prompt(
    parsed_files: Dict,
    validation_errors: List[str],
    user_description: str = "",
) -> tuple:
    """
    Build (system_prompt, user_prompt) for the fix endpoint.

    Sends current code + validation errors to the LLM for targeted correction.
    """
    parts = []
    parts.append("## Current Pack Code\n")

    manifest = parsed_files.get("manifest")
    if manifest:
        parts.append("### manifest.json")
        parts.append(f"```json\n{json.dumps(manifest, indent=2)}\n```\n")

    indicator = parsed_files.get("indicator_code", "")
    if indicator:
        parts.append("### indicator.py")
        parts.append(f"```python\n{indicator}\n```\n")

    interpreter = parsed_files.get("interpreter_code", "")
    if interpreter:
        parts.append("### interpreter.py")
        parts.append(f"```python\n{interpreter}\n```\n")

    parts.append("## Validation Errors\n")
    for err in validation_errors:
        parts.append(f"- {err}")
    parts.append("")

    if user_description.strip():
        parts.append(f"## User Description\n{user_description}\n")

    parts.append("Fix these issues and return the corrected code. Make minimal changes.")

    return _FIX_SYSTEM_PROMPT, "\n".join(parts)


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def parse_llm_response(response_text: str) -> Tuple[bool, Dict, List[str]]:
    """
    Parse an LLM response to extract manifest.json, indicator.py, and interpreter.py.

    Tolerant of markdown formatting, extra commentary, and minor variations.

    Args:
        response_text: The full text response from the LLM

    Returns:
        (success, parsed_files, errors)
        parsed_files = {
            "manifest": dict (parsed JSON),
            "indicator_code": str (Python source),
            "interpreter_code": str (Python source),
            "pine_script_code": str or None (Pine Script source, optional),
        }
    """
    errors = []
    parsed = {
        "manifest": None,
        "indicator_code": None,
        "interpreter_code": None,
        "pine_script_code": None,
    }

    # Extract code blocks (fenced markdown or unfenced with comment headers)
    code_blocks = _extract_code_blocks(response_text)

    if not code_blocks:
        # Fallback: try to parse as unfenced code with comment headers
        code_blocks = _extract_unfenced_sections(response_text)

    if not code_blocks:
        return False, parsed, ["No code blocks found in response"]

    # Identify each block by content/language hints
    json_blocks = []
    python_blocks = []
    pine_blocks = []

    for lang, content in code_blocks:
        content_stripped = content.strip()

        # Detect JSON blocks
        if lang == "json" or content_stripped.startswith("{"):
            json_blocks.append(content_stripped)
        # Detect Pine Script blocks
        elif lang == "pine" or lang == "pinescript" or content_stripped.startswith("//@version"):
            pine_blocks.append(content_stripped)
        # Detect Python blocks
        elif lang == "python" or lang == "py" or "def " in content_stripped:
            python_blocks.append(content_stripped)

    # Extract Pine Script (optional — not required for success)
    if pine_blocks:
        parsed["pine_script_code"] = pine_blocks[0]

    # Parse manifest (first JSON block)
    if not json_blocks:
        errors.append("No JSON code block found (expected manifest.json)")
    else:
        try:
            # Try to parse, stripping any leading comment lines
            json_text = json_blocks[0]
            # Remove // comments that some LLMs add
            json_lines = []
            for line in json_text.split("\n"):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue
                json_lines.append(line)
            json_text = "\n".join(json_lines)

            parsed["manifest"] = json.loads(json_text)
        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse manifest JSON: {e}")

    # Identify indicator.py and interpreter.py from Python blocks
    if len(python_blocks) < 2:
        errors.append(
            f"Expected 2 Python code blocks (indicator.py + interpreter.py), "
            f"found {len(python_blocks)}"
        )
    else:
        # Heuristic: the block with a calculate_* function is indicator.py,
        # the block with interpret_* is interpreter.py
        for block in python_blocks:
            if _has_function_pattern(block, r"def calculate_\w+"):
                if parsed["indicator_code"] is None:
                    parsed["indicator_code"] = block
            if _has_function_pattern(block, r"def interpret_\w+"):
                if parsed["interpreter_code"] is None:
                    parsed["interpreter_code"] = block
            if _has_function_pattern(block, r"def detect_\w+"):
                if parsed["interpreter_code"] is None:
                    parsed["interpreter_code"] = block

        # Fallback: if heuristics didn't match, use positional order
        if parsed["indicator_code"] is None and len(python_blocks) >= 1:
            parsed["indicator_code"] = python_blocks[0]
            errors.append(
                "Warning: Could not identify indicator.py by function name, "
                "using first Python block"
            )
        if parsed["interpreter_code"] is None and len(python_blocks) >= 2:
            parsed["interpreter_code"] = python_blocks[1]
            errors.append(
                "Warning: Could not identify interpreter.py by function name, "
                "using second Python block"
            )

    # Check if both interpret and detect functions are in the same block
    # (some LLMs put them together, which is correct)
    if parsed["interpreter_code"]:
        has_interpret = bool(
            re.search(r"def interpret_\w+", parsed["interpreter_code"])
        )
        has_detect = bool(
            re.search(r"def detect_\w+", parsed["interpreter_code"])
        )
        if has_interpret and not has_detect:
            # Look for detect function in remaining blocks
            for block in python_blocks:
                if block != parsed["interpreter_code"] and block != parsed["indicator_code"]:
                    if _has_function_pattern(block, r"def detect_\w+"):
                        # Append to interpreter code
                        parsed["interpreter_code"] += "\n\n" + block
                        break

    success = (
        parsed["manifest"] is not None
        and parsed["indicator_code"] is not None
        and parsed["interpreter_code"] is not None
    )

    return success, parsed, errors


def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract fenced code blocks from markdown text.

    Returns list of (language, content) tuples.
    """
    # Match ```lang\n...\n``` blocks
    pattern = r"```(\w*)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for lang, content in matches:
        # Strip leading comment like "// manifest.json" or "# indicator.py"
        lines = content.split("\n")
        if lines and (
            lines[0].strip().startswith("//")
            or lines[0].strip().startswith("# ")
            and lines[0].strip().endswith(".py")
            or lines[0].strip().endswith(".json")
        ):
            content = "\n".join(lines[1:])

        results.append((lang.lower(), content))

    return results


def _extract_unfenced_sections(text: str) -> List[Tuple[str, str]]:
    """
    Extract code sections from unfenced LLM responses.

    Handles responses where the LLM outputs raw code with comment-style
    headers like:
        // manifest.json
        { ... }

        # indicator.py
        import pandas as pd
        ...

        # interpreter.py
        import pandas as pd
        ...

    Returns list of (language, content) tuples.
    """
    results = []

    # Look for section headers: "// manifest.json", "# indicator.py", etc.
    # These can appear at the start of a line, possibly with leading whitespace
    section_pattern = re.compile(
        r'^(?://\s*manifest\.json|#\s*indicator\.py|#\s*interpreter\.py)\s*$',
        re.MULTILINE,
    )

    matches = list(section_pattern.finditer(text))

    if not matches:
        # Also try to detect sections by looking for JSON start or import statements
        # after a blank line or at the start
        json_start = re.search(r'(?:^|\n)\s*\{', text)
        if json_start:
            # Try to extract JSON object
            brace_count = 0
            start_idx = text.index('{', json_start.start())
            for i in range(start_idx, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_text = text[start_idx:i + 1]
                        results.append(("json", json_text))
                        remaining = text[i + 1:]
                        break

            # Look for Python sections in remaining text
            if results:
                remaining = text[results[0][1].__len__() + start_idx + 1:]
                python_sections = re.split(
                    r'\n(?=#\s*(?:indicator|interpreter)\.py)',
                    remaining,
                )
                for section in python_sections:
                    section = section.strip()
                    if section and ('import ' in section or 'def ' in section):
                        # Strip leading header comment
                        lines = section.split('\n')
                        if lines and re.match(r'^#\s*\w+\.py', lines[0]):
                            section = '\n'.join(lines[1:]).strip()
                        results.append(("python", section))

        return results

    # We found comment headers — split text at each header
    for i, match in enumerate(matches):
        header = match.group().strip()
        start = match.end()

        # Content ends at the next section header or end of text
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        if "manifest" in header:
            # Strip any // comments from JSON content
            json_lines = []
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("//"):
                    continue
                json_lines.append(line)
            results.append(("json", "\n".join(json_lines).strip()))
        elif "indicator" in header or "interpreter" in header:
            results.append(("python", content))

    return results


def _has_function_pattern(code: str, pattern: str) -> bool:
    """Check if code contains a function matching the regex pattern."""
    return bool(re.search(pattern, code))


# =============================================================================
# VALIDATION
# =============================================================================

def validate_parsed_response(parsed: Dict) -> Tuple[bool, List[str]]:
    """
    Run full validation on parsed LLM response.

    Validates manifest schema, Python safety, function existence,
    and cross-checks between manifest and code.

    Args:
        parsed: Dict from parse_llm_response()

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []

    manifest = parsed.get("manifest")
    indicator_code = parsed.get("indicator_code")
    interpreter_code = parsed.get("interpreter_code")

    if not manifest:
        errors.append("No manifest found")
        return False, errors

    # 1. Validate manifest schema
    valid, manifest_errors = validate_manifest(manifest)
    errors.extend(manifest_errors)

    if not indicator_code:
        errors.append("No indicator code found")
    if not interpreter_code:
        errors.append("No interpreter code found")

    if not indicator_code or not interpreter_code:
        return False, errors

    # 2. Validate Python safety (write to temp files for AST checking)
    import tempfile
    import os

    for label, code in [("indicator.py", indicator_code), ("interpreter.py", interpreter_code)]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            py_valid, py_errors = validate_python_file(tmp_path)
            for err in py_errors:
                errors.append(f"{label}: {err}")

            # 3. Check function existence
            if label == "indicator.py" and manifest.get("indicator_function"):
                exists, err = validate_function_exists(
                    tmp_path, manifest["indicator_function"]
                )
                if not exists:
                    errors.append(f"{label}: {err}")

            if label == "interpreter.py":
                if manifest.get("interpreter_function"):
                    exists, err = validate_function_exists(
                        tmp_path, manifest["interpreter_function"]
                    )
                    if not exists:
                        errors.append(f"{label}: {err}")
                if manifest.get("trigger_function"):
                    exists, err = validate_function_exists(
                        tmp_path, manifest["trigger_function"]
                    )
                    if not exists:
                        errors.append(f"{label}: {err}")
        finally:
            os.unlink(tmp_path)

    # 4. Cross-validate manifest triggers against interpreter code
    if manifest and interpreter_code:
        prefix = manifest.get("trigger_prefix", "")
        trigger_list = manifest.get("triggers", [])
        for trig in trigger_list:
            base = trig.get("base", "")
            expected_key = f"{prefix}_{base}" if prefix else base
            if expected_key not in interpreter_code:
                errors.append(
                    f"Trigger key '{expected_key}' not found in interpreter code. "
                    f"The detect_*_triggers() function must return a dict with key '{expected_key}'"
                )

    # 5. Check indicator columns are referenced in indicator code
    if manifest and indicator_code:
        for col in manifest.get("indicator_columns", []):
            if col not in indicator_code:
                errors.append(
                    f"Indicator column '{col}' declared in manifest but not found in indicator code. "
                    f"The calculate_*() function must create a column named '{col}'"
                )

    return len(errors) == 0, errors


# =============================================================================
# INSTALLATION
# =============================================================================

def install_pack_from_parsed(parsed: Dict) -> Tuple[bool, str, List[str]]:
    """
    Write parsed pack files to user_packs/ directory and register.

    Args:
        parsed: Dict from parse_llm_response() that has passed validation

    Returns:
        (success, pack_slug, errors)
    """
    errors = []
    manifest = parsed["manifest"]
    slug = manifest["slug"]

    # Get user_packs directory
    import pack_registry
    packs_dir = pack_registry.get_user_packs_dir()
    pack_dir = packs_dir / slug

    # Check if pack already exists
    if pack_dir.exists():
        errors.append(
            f"Pack directory '{slug}' already exists. "
            f"Delete the existing pack first or choose a different slug."
        )
        return False, slug, errors

    try:
        # Create directory
        pack_dir.mkdir(parents=True)

        # Write manifest.json
        with open(pack_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Write indicator.py
        with open(pack_dir / "indicator.py", "w") as f:
            f.write(parsed["indicator_code"])

        # Write interpreter.py
        with open(pack_dir / "interpreter.py", "w") as f:
            f.write(parsed["interpreter_code"])

        # Write reference.pine (optional)
        if parsed.get("pine_script_code"):
            with open(pack_dir / "reference.pine", "w") as f:
                f.write(parsed["pine_script_code"])

        # Register the pack
        pack = pack_registry.load_single_pack(pack_dir)
        if pack.is_valid:
            pack_registry.register_pack(pack)
            pack_registry._user_packs[slug] = pack
            return True, slug, []
        else:
            errors.extend(pack.validation_errors)
            # Clean up on failure
            import shutil
            shutil.rmtree(pack_dir)
            return False, slug, errors

    except Exception as e:
        errors.append(f"Installation failed: {e}")
        # Clean up on failure
        import shutil
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        return False, slug, errors
