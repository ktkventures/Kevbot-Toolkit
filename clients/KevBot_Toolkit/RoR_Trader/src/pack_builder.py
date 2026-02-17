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
    )

    user_request = "\n".join(request_parts)

    # Combine context + request
    prompt = f"{context}\n\n---\n\n{user_request}"

    return prompt


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
        }
    """
    errors = []
    parsed = {
        "manifest": None,
        "indicator_code": None,
        "interpreter_code": None,
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

    for lang, content in code_blocks:
        content_stripped = content.strip()

        # Detect JSON blocks
        if lang == "json" or content_stripped.startswith("{"):
            json_blocks.append(content_stripped)
        # Detect Python blocks
        elif lang == "python" or lang == "py" or "def " in content_stripped:
            python_blocks.append(content_stripped)

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
