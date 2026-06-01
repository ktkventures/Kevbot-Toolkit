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
import builtins as _py_builtins
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

# External imports allowed in indicator.py / interpreter.py.
# Intra-pack siblings (`indicator_incremental`) are added below to allow
# the batch indicator to delegate to the live-mode incremental class —
# this keeps both paths reading from one source of truth instead of
# maintaining two parallel implementations that could drift apart.
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "typing",
    "indicator_incremental",  # intra-pack sibling, see pack_builder_context.md
}

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

# Trigger-base name patterns that suggest a static-level cross
# (price-vs-line). Packs with triggers matching these patterns SHOULD
# declare `trigger_levels` (or `trigger_levels_phase2` for explicit
# non-static cases) so Hi-Fi Pass 2 can refine sub-second exit times.
# This is a HEURISTIC — pack authors can override by listing the
# trigger in `trigger_levels_phase2` to mark Phase-2 (indicator-vs-
# indicator or dynamic-during-bar) intentionally.
import re as _re
_CROSS_STYLE_PATTERN = _re.compile(
    r"(^|_)(cross|flip|above|below|breaks?|crosses?)(_|$)",
    _re.IGNORECASE,
)


def looks_cross_style(trigger_base: str) -> bool:
    """Return True if a trigger-base name looks like a level cross.

    Heuristic for surfacing manifest gaps. Used by `audit_trigger_levels`
    + the install-time warning hook in pack_registry.

    Examples that match: cross_short_up, mid_cross_bull, bull_flip,
    cross_above, breaks_above. Does NOT match: oversold,
    bullish_c2_detected, max_hold_bars.
    """
    if not trigger_base or not isinstance(trigger_base, str):
        return False
    return bool(_CROSS_STYLE_PATTERN.search(trigger_base))


def audit_trigger_levels(manifest: dict) -> List[str]:
    """Return a list of WARNINGS (not errors) about trigger_levels coverage.

    Phase 1 Hi-Fi infrastructure (2026-05-07) refines L-type signal
    exits ONLY when the pack manifest declares `trigger_levels` for the
    relevant trigger. This audit surfaces packs whose trigger-base
    names look like static-level crosses (per `looks_cross_style`)
    but lack a corresponding `trigger_levels` entry — likely an
    oversight in AI-assisted pack creation, NOT a deliberate Phase-2
    placement.

    A pack author who genuinely intends Phase-2 detection (e.g. for
    indicator-vs-indicator crosses in MACD, EMA stack, stochastic)
    can suppress this warning by listing the trigger base in a
    `trigger_levels_phase2` block (a separate top-level key).
    `trigger_levels_phase2` is a placeholder schema today (any dict
    is accepted); the audit only checks its KEYS to know which
    triggers are intentionally non-static.

    Returns: list of warning strings (one per omitted cross-style
    trigger). Empty list = pack is consistent.
    """
    warnings = []
    triggers = manifest.get("triggers", []) or []
    if not isinstance(triggers, list):
        return warnings
    declared_levels = manifest.get("trigger_levels") or {}
    declared_phase2 = manifest.get("trigger_levels_phase2") or {}
    if not isinstance(declared_levels, dict):
        declared_levels = {}
    if not isinstance(declared_phase2, dict):
        declared_phase2 = {}

    for t in triggers:
        if not isinstance(t, dict):
            continue
        base = t.get("base")
        if not base:
            continue
        if not looks_cross_style(base):
            continue
        if base in declared_levels:
            continue  # static-level cross declared, all good
        if base in declared_phase2:
            continue  # explicitly marked Phase 2, all good
        warnings.append(
            f"Trigger '{base}' looks like a level cross but is missing "
            f"from both 'trigger_levels' (Phase 1 / static-level) and "
            f"'trigger_levels_phase2' (intentional non-static marker). "
            f"Hi-Fi Pass 2 will skip this trigger; sub-second exit "
            f"refinement won't be available."
        )
    return warnings


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

    # Allowlist check: any builtin used must be in the pack sandbox's
    # SAFE_BUILTINS. Catches hasattr/getattr/vars/Exception/etc. that the
    # legacy DISALLOWED_CALLS denylist misses (see validate_builtin_usage).
    _bi_ok, _bi_errors = validate_builtin_usage(file_path)
    errors.extend(_bi_errors)

    return len(errors) == 0, errors


def validate_function_exists(file_path: str, func_name: str) -> Tuple[bool, str]:
    """
    Verify that a function or class is defined in the given Python file.

    Used to check that an indicator/interpreter/trigger function
    declared in the manifest actually exists in the pack code, and that
    an `incremental_class` declaration (a class, not a function) is
    likewise present.

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
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == func_name:
            return True, ""

    return False, f"'{func_name}' not found in {path.name}"


def _get_call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


# Dunders the pack loader explicitly re-adds to the restricted builtins
# (pack_registry._import_module_restricted), plus the module global Python
# always injects. Not in SAFE_BUILTINS but legitimately resolvable in a pack.
_LOADER_PROVIDED_NAMES = {
    "__import__", "__build_class__", "__name__", "__builtins__",
    "__file__", "__doc__", "__spec__", "__loader__", "__package__",
}
_ALL_PY_BUILTINS = set(dir(_py_builtins))


def _collect_bound_names(tree: ast.AST) -> set:
    """Names bound anywhere in the module — defs, imports, assignments,
    args, comprehension/for targets, except-as, global/nonlocal.

    Used to distinguish a *builtin reference* from a pack-defined name.
    Module-wide (not scope-precise) on purpose: an over-broad bound set
    only makes the builtin check leaner, never falsely strict.
    """
    bound: set = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                bound.add(a.asname or a.name)
        elif isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return bound


def validate_builtin_usage(file_path: str) -> Tuple[bool, List[str]]:
    """Flag free names that are Python builtins NOT in the pack sandbox.

    Packs execute with a restricted ``__builtins__`` — the loader installs
    only ``SAFE_BUILTINS`` (+ a few dunders). A name like ``hasattr`` /
    ``getattr`` / ``vars`` / ``Exception`` / ``AttributeError`` is a
    perfectly valid builtin in normal Python, so it passes import and the
    AST import/call checks — but raises ``NameError`` the instant that
    code path runs inside a pack.

    This is an *allowlist* check (vs the legacy DISALLOWED_CALLS denylist,
    which silently missed ``hasattr`` and produced a live exception storm
    on 2026-05-19). It catches the whole class deterministically at
    validate-time, before the pack is ever installed.
    """
    path = Path(file_path)
    if not path.exists():
        return False, [f"File not found: {file_path}"]
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    bound = _collect_bound_names(tree)
    allowed = SAFE_BUILTINS | _LOADER_PROVIDED_NAMES
    offenders: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name in bound or name in allowed:
                continue
            if name not in _ALL_PY_BUILTINS:
                continue  # pack-defined / imported elsewhere — not our job
            offenders.setdefault(name, node.lineno)

    errors = [
        f"Line {ln}: builtin '{name}' is not in the pack sandbox "
        f"(SAFE_BUILTINS) — it will raise NameError at runtime. "
        f"Allowed builtins: {', '.join(sorted(SAFE_BUILTINS))}."
        for name, ln in sorted(offenders.items(), key=lambda kv: kv[1])
    ]
    return len(errors) == 0, errors


# =============================================================================
# COLUMN CONTRACT VALIDATION (added 2026-04-27)
# =============================================================================
# Lesson from the ut_bot_v4 FLIP gap: a pack's interpreter can read columns
# that only the BATCH indicator writes — leaving the LIVE incremental class
# silent. The interpreter then mis-classifies live bars (e.g. ut_bot_v4
# emitted BULL_TREND on flip bars instead of BULL_FLIP, capping Q2 parity
# at ~0.87).
#
# This validator parses each pack file's column reads/writes and fails when
# the contract is violated:
#
#   { columns interpreter reads }
#       ⊆ ({ columns batch indicator writes }
#         ∪ { columns incremental_class.update_bar() returns })
#
# Catches the bug class deterministically at validate-time, before install.

# Column-name string-literal patterns we care about
_DF_GET_RE = re.compile(r'df\s*\.\s*get\s*\(\s*["\']([^"\']+)["\']')
_DF_INDEX_RE = re.compile(r'df\s*\[\s*["\']([^"\']+)["\']\s*\]')


def _extract_columns_read_from_interpreter(file_path: str) -> set:
    """Find every column the interpreter reads via df["X"] or df.get("X").

    String-pattern based (regex over source) — we don't try to track
    aliases or list comprehensions; for the contract check we just need
    the set of literal string keys referenced. False positives here
    only over-reports the contract, never misses violations.
    """
    path = Path(file_path)
    if not path.exists():
        return set()
    try:
        src = path.read_text()
    except Exception:
        return set()
    cols = set()
    for m in _DF_GET_RE.finditer(src):
        cols.add(m.group(1))
    for m in _DF_INDEX_RE.finditer(src):
        cols.add(m.group(1))
    return cols


def _extract_columns_written_by_indicator(file_path: str) -> set:
    """Find every column the batch indicator writes via:
       result[X] = ...    df[X] = ...    out[X] = ...
    """
    path = Path(file_path)
    if not path.exists():
        return set()
    try:
        src = path.read_text()
    except Exception:
        return set()
    cols = set()
    # result["X"] = ..., df["X"] = ..., out["X"] = ..., enriched["X"] = ...
    for m in re.finditer(
        r'(?:result|df|out|enriched|_df|new_df|res)\s*\[\s*["\']([^"\']+)["\']\s*\]\s*=',
        src,
    ):
        cols.add(m.group(1))
    return cols


def _extract_columns_returned_by_incremental(file_path: str) -> set:
    """AST-parse update_bar() to find the literal string keys in its
    return dict. Handles dict-literal return values; treats anything
    more dynamic conservatively (returns empty set, no contract enforcement)."""
    path = Path(file_path)
    if not path.exists():
        return set()
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except Exception:
        return set()
    cols: set = set()
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name != 'update_bar':
                continue
            # Walk the function body for `return {...}` nodes
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            cols.add(key.value)
    return cols


def _filter_to_pack_columns(cols: set, pack_slug: str,
                            trigger_prefix: str = '') -> set:
    """Drop OHLCV and standard engine columns from the contract check.

    The interpreter is allowed to read `close`, `high`, `low`, etc. without
    these appearing in update_bar() — the engine writes them itself. We
    only enforce the contract on pack-emitted columns, which heuristically
    means columns starting with the slug, the trigger_prefix, or `__` +
    either.
    """
    standard = {'open', 'high', 'low', 'close', 'volume', 'timestamp',
                'atr', 'ema', 'macd_line', 'macd_signal', 'macd_hist',
                'vwap', 'rvol', 'vol_sma'}
    out = set()
    for c in cols:
        if c in standard:
            continue
        if c.startswith(pack_slug) or c.startswith('_' + pack_slug):
            out.add(c)
            continue
        if trigger_prefix and (
            c.startswith(trigger_prefix) or c.startswith('_' + trigger_prefix)
            or c.startswith('__' + trigger_prefix)
        ):
            out.add(c)
            continue
    return out


def validate_column_contract(
    pack_dir: str | Path,
    manifest: dict | None = None,
) -> Tuple[bool, List[str]]:
    """Verify the indicator + interpreter + incremental_class agree on
    column shapes for live-mode parity.

    Args:
        pack_dir: directory containing manifest.json + indicator.py +
                  interpreter.py + (optional) indicator_incremental.py
        manifest: pre-loaded manifest dict; if None, reads from disk

    Returns (ok, errors). `errors` is empty when ok=True. Each error is a
    short string suitable for surfacing in the pack-builder wizard's
    validation panel.

    Skips silently when a file doesn't exist — that's the responsibility
    of the existing structural validation.
    """
    pack_dir = Path(pack_dir)
    indicator_path = pack_dir / 'indicator.py'
    interpreter_path = pack_dir / 'interpreter.py'
    incremental_path = pack_dir / 'indicator_incremental.py'

    if manifest is None:
        manifest_path = pack_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                import json
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
        else:
            manifest = {}

    pack_slug = manifest.get('slug', pack_dir.name)
    trigger_prefix = manifest.get('trigger_prefix', '')

    # Skip if no incremental class exists — pack is batch-only by design.
    # The existing `is_valid` flag and pack_registry FAIL_SILENT already
    # handle that case.
    if not incremental_path.exists():
        return True, []

    interp_reads_raw = _extract_columns_read_from_interpreter(
        str(interpreter_path))
    indicator_writes_raw = _extract_columns_written_by_indicator(
        str(indicator_path))
    incremental_returns_raw = _extract_columns_returned_by_incremental(
        str(incremental_path))

    # Filter to pack-emitted columns (skip OHLCV / standard engine cols)
    interp_reads = _filter_to_pack_columns(
        interp_reads_raw, pack_slug, trigger_prefix)
    indicator_writes = _filter_to_pack_columns(
        indicator_writes_raw, pack_slug, trigger_prefix)
    incremental_returns = _filter_to_pack_columns(
        incremental_returns_raw, pack_slug, trigger_prefix)

    errors: list = []

    # CONTRACT 1: every column the interpreter reads must appear somewhere
    # the engine could have written it — either the batch indicator (for
    # backtest mode) OR the incremental class (for live mode).
    missing_anywhere = interp_reads - indicator_writes - incremental_returns
    if missing_anywhere:
        errors.append(
            f"interpreter.py reads columns that neither indicator.py nor "
            f"indicator_incremental.py emit: {sorted(missing_anywhere)}. "
            f"This will cause runtime errors in both backtest and live mode."
        )

    # CONTRACT 2: every column the interpreter reads AND the batch
    # indicator emits MUST also be emitted by the incremental class.
    # Otherwise the pack works in backtest but mis-classifies in live —
    # the exact ut_bot_v4 FLIP gap.
    in_indicator_not_incremental = (
        interp_reads & indicator_writes) - incremental_returns
    if in_indicator_not_incremental:
        errors.append(
            f"Live-mode parity gap: interpreter.py reads columns that "
            f"indicator.py writes but indicator_incremental.update_bar() "
            f"does NOT return: {sorted(in_indicator_not_incremental)}. "
            f"The pack will pass backtest but mis-classify live bars. "
            f"Add these keys to update_bar()'s return dict (mirror the "
            f"batch path's __-prefixed aliases). See ut_bot_v4's "
            f"indicator_incremental.py for a worked example."
        )

    return (len(errors) == 0), errors


# Types the generic state codec round-trips safely. A pack class storing
# anything else in `__dict__` should either restrict itself to this set
# (preferred) or opt in to explicit serialize_state/restore_state methods.
_STATE_PROTOCOL_SAFE_TYPES = (int, float, bool, str, type(None), list, dict)


def validate_state_protocol(incremental_class) -> Tuple[List[str], List[str]]:
    """Verify a pack's incremental class is compatible with the persistent
    state codec (`user_pack_state_codec.py`).

    Returns `(errors, warnings)`. Errors block registration; warnings do not.

    Rules (Phase 2 lenient v1):
    - HARD ERROR: defining `serialize_state` without `restore_state` (or
      vice versa). The codec dispatches on both-present; an inconsistent
      override is silently ignored, which breaks the contract.
    - WARNING: instantiating the class with no args populates `__dict__`
      with a non-pickle-safe type (numpy array, instance of another
      class, etc.). Generic codec stores `dict(__dict__)`, which pickle
      handles for primitives + lists + dicts + pandas.Timestamp. Anything
      exotic should be migrated to the opt-in protocol.

    Pack class is expected to accept `**params` with all params optional
    (mirrors the existing pack convention). If the class can't be
    instantiated without args, we emit a single warning and skip the
    attr inspection — registration is unaffected.
    """
    errors: list = []
    warnings: list = []

    if incremental_class is None:
        return errors, warnings

    has_serialize = callable(getattr(incremental_class, 'serialize_state', None))
    has_restore = callable(getattr(incremental_class, 'restore_state', None))
    if has_serialize != has_restore:
        which = 'serialize_state' if has_serialize else 'restore_state'
        other = 'restore_state' if has_serialize else 'serialize_state'
        errors.append(
            f"{incremental_class.__name__} defines `{which}` without "
            f"`{other}`. The codec dispatches on both-present; defining "
            f"only one is a silent contract bug. Add the missing method "
            f"(see user_pack_state_codec docstring) or remove the existing "
            f"one to fall back to the generic codec."
        )
        return errors, warnings

    if has_serialize:
        # Explicit override — pack owns its own state shape. Skip attr walk.
        return errors, warnings

    try:
        eng = incremental_class()
    except Exception as e:
        warnings.append(
            f"{incremental_class.__name__} could not be instantiated with "
            f"no args ({e}). State-protocol attr-type check skipped. If "
            f"this pack works at runtime, fine — but ensure all `__init__` "
            f"params have defaults so the validator can introspect."
        )
        return errors, warnings

    # Detect attrs whose type isn't safe for the generic codec's pickle path.
    # pandas.Timestamp is allowed via lazy isinstance check (don't import
    # pandas unconditionally — keeps pack_spec light).
    try:
        import pandas as _pd
        _pd_Timestamp = _pd.Timestamp
    except Exception:
        _pd_Timestamp = None

    exotic: list = []
    for k, v in vars(eng).items():
        if isinstance(v, _STATE_PROTOCOL_SAFE_TYPES):
            continue
        if _pd_Timestamp is not None and isinstance(v, _pd_Timestamp):
            continue
        exotic.append(f"{k}: {type(v).__module__}.{type(v).__name__}")

    if exotic:
        warnings.append(
            f"{incremental_class.__name__} stores non-codec-safe attrs in "
            f"__dict__: {exotic}. The generic codec may fail at "
            f"snapshot/restore. Either restrict state to "
            f"(int|float|bool|str|None|list|dict|Timestamp) or implement "
            f"explicit serialize_state + restore_state methods."
        )

    return errors, warnings
