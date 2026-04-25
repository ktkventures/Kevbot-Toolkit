"""
Pack Spec — Schema Definitions and Validation for User Packs
=============================================================

Defines the manifest schema that all user-created confluence packs must
conform to, and provides validation functions for manifests and Python code.

A user pack consists of:
- manifest.json  — metadata, parameters, outputs, triggers (Pack Spec schema)
- indicator.py   — calculate_<slug>(df, **params) -> DataFrame
- interpreter.py — interpret_<slug>(df, **params) -> Series
                    detect_<slug>_triggers(df, **params) -> Dict[str, Series]
"""

import ast
import re
from pathlib import Path
from typing import List, Tuple


# =============================================================================
# MANIFEST SCHEMA
# =============================================================================

MANIFEST_REQUIRED_FIELDS = [
    "slug",
    "name",
    "category",
    "description",
    "pack_type",
    "interpreters",
    "trigger_prefix",
    "parameters_schema",
    "outputs",
    "output_descriptions",
    "triggers",
    "indicator_columns",
    "indicator_function",
    "interpreter_function",
    "trigger_function",
]

MANIFEST_OPTIONAL_FIELDS = [
    "author",
    "version",
    "created_at",
    "plot_schema",
    "requires_indicators",
    "display_type",
    "column_color_map",
    "plot_config",
    # Live-mode incremental computation. Optional: a pack without this is
    # explicitly batch-only (works in backtest, FAIL_SILENT in live). When
    # declared, the pack ships an indicator_incremental.py file containing
    # a class with the contract documented in pack_builder_context.md.
    "incremental_class",
    # Modular trigger-to-execution-type bridge. Each entry maps a trigger
    # base name to the indicator column the engine should treat as the
    # intra-bar level for L/LC executions. Manifest authors declare ONE
    # base trigger per logical signal; the registry materializes runtime
    # IDs (with _ib/_lc/_cc suffixes) automatically based on this map.
    # See pack_builder_context.md for the modular trigger pattern.
    "trigger_levels",
]


# Schema for the incremental_class manifest entry. Lightweight — full
# class shape is enforced at register-time in pack_registry, not here.
INCREMENTAL_CLASS_REQUIRED_FIELDS = ["class_name"]
INCREMENTAL_CLASS_OPTIONAL_FIELDS = ["module"]  # default: "indicator_incremental"

VALID_PACK_TYPES = ["tf_confluence"]

# Display types for charting:
#   overlay   — indicator lines drawn on price chart (EMA, BB, VWAP)
#   oscillator — separate pane below price chart (RSI, MACD, RVOL)
#   hidden    — no chart rendering (bar_count)
VALID_DISPLAY_TYPES = ["overlay", "oscillator", "hidden"]

VALID_TRIGGER_DIRECTIONS = ["LONG", "SHORT", "BOTH"]
VALID_TRIGGER_TYPES = ["ENTRY", "EXIT", "BOTH"]
VALID_TRIGGER_EXECUTIONS = ["bar_close", "intra_bar", "hybrid_market", "hybrid_limit", "close_close"]

VALID_PARAM_TYPES = ["int", "float", "str", "bool"]

# Built-in trigger prefixes that user packs cannot use.
# Source of truth: unified_engine.TRIGGER_PREFIX_TO_TEMPLATE — only the
# prefixes registered there belong to true built-ins. Earlier versions of
# this file mirrored *user-pack* prefixes into this set ("bb", "src", "st",
# "sw123", "strat", "rsi"), which made every user pack fail validation on
# the second load: the pack's own prefix would now appear "reserved", so
# the validator rejected it. That left supertrend, swing_123, bollinger
# bands, sr_channels, strat_assistant, and rsi_zones invalid in the API
# (preview returned 404; parity tests returned 0 fires) until the pack
# was re-saved with a brand-new prefix. Keep this list aligned with the
# engine, not with anything users have authored.
BUILTIN_TRIGGER_PREFIXES = {
    "ema", "ema_pp", "ema_pp_v2",
    "macd", "macd_hist",
    "vwap", "rvol",
    "utbot", "utbot_v2",
    "bar_count",
}

# Built-in indicator columns that user packs cannot collide with.
# Same rule as above: only columns produced by the engine's built-in
# indicators (in indicators.py) — never columns claimed by a user pack.
BUILTIN_INDICATOR_COLUMNS = {
    "ema_8", "ema_21", "ema_50",
    "macd_line", "macd_signal", "macd_hist",
    "vwap", "vwap_sd1_upper", "vwap_sd1_lower", "vwap_sd2_upper", "vwap_sd2_lower",
    "atr", "vol_sma", "rvol",
    "utbot_stop", "utbot_stop_prev", "utbot_direction",
}

# Built-in interpreter keys that user packs cannot collide with.
# Source of truth: interpreters.INTERPRETERS at module-load time (i.e. the
# keys defined statically in that file, before any user pack registers).
BUILTIN_INTERPRETER_KEYS = {
    "EMA_STACK", "EMA_PRICE_POSITION", "EMA_PRICE_POSITION_V2",
    "MACD_LINE", "MACD_HISTOGRAM",
    "VWAP", "RVOL",
    "UTBOT", "UTBOT_V2",
}

SLUG_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')
INTERPRETER_KEY_PATTERN = re.compile(r'^[A-Z][A-Z0-9_]*$')


# =============================================================================
# PYTHON CODE SAFETY
# =============================================================================

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "typing"}

DISALLOWED_CALLS = {
    "open", "exec", "eval", "compile", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals",
    "breakpoint", "input",
}

DISALLOWED_MODULES = {
    "os", "sys", "subprocess", "socket", "requests", "urllib",
    "http", "shutil", "pathlib", "importlib", "builtins",
    "ctypes", "pickle", "shelve", "signal", "threading",
    "multiprocessing", "asyncio", "io",
}

# Allowed builtins for restricted execution
SAFE_BUILTINS = {
    "range", "len", "min", "max", "abs", "round",
    "int", "float", "str", "bool", "list", "dict", "set", "tuple", "object",
    "enumerate", "zip", "sorted", "reversed", "any", "all",
    "isinstance", "print", "True", "False", "None",
    "map", "filter", "sum", "pow", "divmod",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
}


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def validate_manifest(manifest: dict) -> Tuple[bool, List[str]]:
    """
    Validate a manifest dict against the Pack Spec schema.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []

    # Check required fields
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors

    # Validate slug
    slug = manifest["slug"]
    if not isinstance(slug, str) or not SLUG_PATTERN.match(slug):
        errors.append(
            f"Invalid slug '{slug}': must match [a-z][a-z0-9_]* "
            f"(lowercase, start with letter, underscores OK)"
        )

    # Validate pack_type
    if manifest["pack_type"] not in VALID_PACK_TYPES:
        errors.append(
            f"Invalid pack_type '{manifest['pack_type']}': "
            f"must be one of {VALID_PACK_TYPES}"
        )

    # Validate trigger_prefix doesn't collide with built-ins
    prefix = manifest["trigger_prefix"]
    if prefix in BUILTIN_TRIGGER_PREFIXES:
        errors.append(
            f"Trigger prefix '{prefix}' collides with built-in pack. "
            f"Reserved prefixes: {BUILTIN_TRIGGER_PREFIXES}"
        )

    # Validate interpreters
    for interp_key in manifest["interpreters"]:
        if not INTERPRETER_KEY_PATTERN.match(interp_key):
            errors.append(
                f"Invalid interpreter key '{interp_key}': "
                f"must be UPPERCASE with underscores (e.g., 'RSI_ZONES')"
            )
        if interp_key in BUILTIN_INTERPRETER_KEYS:
            errors.append(
                f"Interpreter key '{interp_key}' collides with built-in. "
                f"Reserved: {BUILTIN_INTERPRETER_KEYS}"
            )

    # Validate outputs
    if not isinstance(manifest["outputs"], list) or len(manifest["outputs"]) == 0:
        errors.append("'outputs' must be a non-empty list of strings")

    # Validate output_descriptions matches outputs
    descs = manifest["output_descriptions"]
    if isinstance(descs, dict):
        for out in manifest["outputs"]:
            if out not in descs:
                errors.append(f"Missing output_description for output '{out}'")
    else:
        errors.append("'output_descriptions' must be a dict")

    # Validate triggers
    triggers = manifest["triggers"]
    if not isinstance(triggers, list):
        errors.append("'triggers' must be a list")
    else:
        for i, trig in enumerate(triggers):
            if not isinstance(trig, dict):
                errors.append(f"triggers[{i}] must be a dict")
                continue
            for req_field in ["base", "name", "direction", "type"]:
                if req_field not in trig:
                    errors.append(f"triggers[{i}] missing '{req_field}'")
            if trig.get("direction") not in VALID_TRIGGER_DIRECTIONS:
                errors.append(
                    f"triggers[{i}] invalid direction '{trig.get('direction')}': "
                    f"must be one of {VALID_TRIGGER_DIRECTIONS}"
                )
            if trig.get("type") not in VALID_TRIGGER_TYPES:
                errors.append(
                    f"triggers[{i}] invalid type '{trig.get('type')}': "
                    f"must be one of {VALID_TRIGGER_TYPES}"
                )
            execution = trig.get("execution", "bar_close")
            if execution not in VALID_TRIGGER_EXECUTIONS:
                errors.append(
                    f"triggers[{i}] invalid execution '{execution}': "
                    f"must be one of {VALID_TRIGGER_EXECUTIONS}"
                )

    # Validate trigger base doesn't repeat the trigger_prefix
    prefix = manifest.get("trigger_prefix", "")
    if isinstance(triggers, list) and prefix:
        for i, trig in enumerate(triggers):
            base = trig.get("base", "")
            if base.startswith(prefix + "_"):
                errors.append(
                    f"triggers[{i}] base '{base}' starts with trigger_prefix '{prefix}_'. "
                    f"The system auto-prepends the prefix — use '{base[len(prefix)+1:]}' instead"
                )

    # Validate parameters_schema
    params = manifest["parameters_schema"]
    if isinstance(params, dict):
        for key, spec in params.items():
            if not isinstance(spec, dict):
                errors.append(f"parameters_schema['{key}'] must be a dict")
                continue
            if "type" not in spec:
                errors.append(f"parameters_schema['{key}'] missing 'type'")
            elif spec["type"] not in VALID_PARAM_TYPES:
                errors.append(
                    f"parameters_schema['{key}'] invalid type '{spec['type']}': "
                    f"must be one of {VALID_PARAM_TYPES}"
                )
            if "default" not in spec:
                errors.append(f"parameters_schema['{key}'] missing 'default'")
            if "label" not in spec:
                errors.append(f"parameters_schema['{key}'] missing 'label'")
    else:
        errors.append("'parameters_schema' must be a dict")

    # Validate indicator_columns don't collide with built-ins
    for col in manifest["indicator_columns"]:
        if col in BUILTIN_INDICATOR_COLUMNS:
            errors.append(
                f"Indicator column '{col}' collides with built-in column"
            )

    # Validate function names are strings
    for func_field in ["indicator_function", "interpreter_function", "trigger_function"]:
        if not isinstance(manifest[func_field], str):
            errors.append(f"'{func_field}' must be a string (function name)")

    # Validate display_type if present
    display_type = manifest.get("display_type")
    if display_type is not None and display_type not in VALID_DISPLAY_TYPES:
        errors.append(
            f"Invalid display_type '{display_type}': "
            f"must be one of {VALID_DISPLAY_TYPES}"
        )

    # Validate column_color_map if present
    ccm = manifest.get("column_color_map")
    if ccm is not None:
        if not isinstance(ccm, dict):
            errors.append("'column_color_map' must be a dict mapping indicator_column -> plot_schema_key")
        else:
            for col, color_key in ccm.items():
                if col not in manifest["indicator_columns"]:
                    errors.append(
                        f"column_color_map key '{col}' not in indicator_columns"
                    )

    # Validate plot_config if present
    plot_config = manifest.get("plot_config")
    if plot_config is not None:
        if not isinstance(plot_config, dict):
            errors.append("'plot_config' must be a dict")
        else:
            # Validate band_fills
            for i, bf in enumerate(plot_config.get("band_fills", [])):
                if not isinstance(bf, dict):
                    errors.append(f"plot_config.band_fills[{i}] must be a dict")
                    continue
                for req in ("upper_column", "lower_column", "fill_color_key"):
                    if req not in bf:
                        errors.append(f"plot_config.band_fills[{i}] missing '{req}'")
                if bf.get("upper_column") and bf["upper_column"] not in manifest["indicator_columns"]:
                    errors.append(f"plot_config.band_fills[{i}].upper_column '{bf['upper_column']}' not in indicator_columns")
                if bf.get("lower_column") and bf["lower_column"] not in manifest["indicator_columns"]:
                    errors.append(f"plot_config.band_fills[{i}].lower_column '{bf['lower_column']}' not in indicator_columns")

            # Validate reference_lines
            for i, rl in enumerate(plot_config.get("reference_lines", [])):
                if not isinstance(rl, dict):
                    errors.append(f"plot_config.reference_lines[{i}] must be a dict")
                    continue
                if "value" not in rl:
                    errors.append(f"plot_config.reference_lines[{i}] missing 'value'")
                elif not isinstance(rl["value"], (int, float)):
                    errors.append(f"plot_config.reference_lines[{i}].value must be numeric")

            # Validate line_styles
            line_styles = plot_config.get("line_styles", {})
            if not isinstance(line_styles, dict):
                errors.append("plot_config.line_styles must be a dict")
            else:
                for col, style in line_styles.items():
                    if not isinstance(style, int) or style not in range(5):
                        errors.append(f"plot_config.line_styles['{col}'] must be 0-4 (Solid/Dotted/Dashed/LargeDashed/SparseDotted)")

            # Validate candle_color_column
            ccc = plot_config.get("candle_color_column")
            if ccc is not None and ccc not in manifest["indicator_columns"]:
                errors.append(f"plot_config.candle_color_column '{ccc}' not in indicator_columns")

    # Validate trigger_levels shape if present. Each entry must reference a
    # trigger base that exists in `triggers` and an indicator column that
    # exists in `indicator_columns`. Cross direction must be 'above' or
    # 'below'. The registry uses this to auto-generate L/LC suffixed
    # runtime trigger IDs at install time.
    trigger_levels = manifest.get("trigger_levels")
    if trigger_levels is not None:
        if not isinstance(trigger_levels, dict):
            errors.append(
                "'trigger_levels' must be a dict mapping trigger_base -> "
                "{level_column, cross}"
            )
        else:
            trigger_bases = {t.get("base") for t in manifest.get("triggers", [])
                             if isinstance(t, dict)}
            indicator_cols = set(manifest.get("indicator_columns", []))
            for base, lvl in trigger_levels.items():
                if base not in trigger_bases:
                    errors.append(
                        f"trigger_levels['{base}'] references a trigger base "
                        f"that is not in the triggers list"
                    )
                if not isinstance(lvl, dict):
                    errors.append(
                        f"trigger_levels['{base}'] must be a dict with "
                        f"'level_column' and 'cross'"
                    )
                    continue
                level_col = lvl.get("level_column")
                cross_dir = lvl.get("cross")
                if not level_col or not isinstance(level_col, str):
                    errors.append(
                        f"trigger_levels['{base}'].level_column must be "
                        f"a non-empty string"
                    )
                elif level_col not in indicator_cols:
                    errors.append(
                        f"trigger_levels['{base}'].level_column "
                        f"'{level_col}' is not in indicator_columns"
                    )
                if cross_dir not in ("above", "below"):
                    errors.append(
                        f"trigger_levels['{base}'].cross must be 'above' "
                        f"or 'below', got {cross_dir!r}"
                    )

    # Validate incremental_class shape if present. Existence of the actual
    # class file is checked at register time, not here — manifest-level
    # validation just confirms the declaration shape is sane.
    inc = manifest.get("incremental_class")
    if inc is not None:
        if not isinstance(inc, dict):
            errors.append(
                "'incremental_class' must be a dict with 'class_name' "
                "(and optional 'module', defaulting to 'indicator_incremental')"
            )
        else:
            for f in INCREMENTAL_CLASS_REQUIRED_FIELDS:
                if f not in inc:
                    errors.append(f"incremental_class missing required field '{f}'")
            cn = inc.get("class_name")
            if cn is not None and not isinstance(cn, str):
                errors.append("incremental_class.class_name must be a string")
            mod = inc.get("module")
            if mod is not None and not isinstance(mod, str):
                errors.append("incremental_class.module must be a string (Python module name without .py)")

    return len(errors) == 0, errors


# =============================================================================
# PYTHON CODE VALIDATION
# =============================================================================

def validate_python_file(file_path: str) -> Tuple[bool, List[str]]:
    """
    AST-validate a Python file for safety.

    Checks for disallowed imports, function calls, and attribute access.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []
    path = Path(file_path)

    if not path.exists():
        return False, [f"File not found: {file_path}"]

    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    for node in ast.walk(tree):
        # Check Import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root not in ALLOWED_IMPORTS:
                    errors.append(
                        f"Line {node.lineno}: Disallowed import '{alias.name}'. "
                        f"Only {ALLOWED_IMPORTS} are allowed."
                    )

        # Check ImportFrom statements
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root not in ALLOWED_IMPORTS:
                    errors.append(
                        f"Line {node.lineno}: Disallowed import from '{node.module}'. "
                        f"Only {ALLOWED_IMPORTS} are allowed."
                    )

        # Check function calls
        elif isinstance(node, ast.Call):
            func_name = _get_call_name(node)
            if func_name in DISALLOWED_CALLS:
                errors.append(
                    f"Line {node.lineno}: Disallowed function call '{func_name}'"
                )
            # Check for module.function calls (e.g., os.system)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name in DISALLOWED_MODULES:
                    errors.append(
                        f"Line {node.lineno}: Disallowed module access "
                        f"'{module_name}.{node.func.attr}'"
                    )

        # Check attribute access on dangerous names
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id == "__builtins__":
                    errors.append(
                        f"Line {node.lineno}: Disallowed access to '__builtins__'"
                    )

    return len(errors) == 0, errors


def validate_function_exists(file_path: str, func_name: str) -> Tuple[bool, str]:
    """
    Verify that a function is defined in the given Python file.

    Returns:
        (exists, error_message_if_not)
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    try:
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True, ""

    return False, f"Function '{func_name}' not found in {path.name}"


def _get_call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""
