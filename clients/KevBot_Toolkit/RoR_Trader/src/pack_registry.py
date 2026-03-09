"""
Pack Registry — Hot-Load, Validate, and Register User Packs
=============================================================

Central registry that scans user_packs/, validates manifests and Python code,
dynamically imports modules, and registers user packs into the existing
TEMPLATES, INTERPRETERS, and indicator/trigger pipelines.

Usage:
    import pack_registry

    # At app startup:
    pack_registry.scan_and_load_all()

    # Query:
    packs = pack_registry.get_registered_packs()
    pack = pack_registry.get_pack("rsi_zones")

    # Manage:
    pack_registry.refresh_registry()
    pack_registry.delete_pack("rsi_zones")
"""

import importlib.util
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from pack_spec import (
    validate_manifest,
    validate_python_file,
    validate_function_exists,
    SAFE_BUILTINS,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegisteredPack:
    """A validated, loaded user pack ready for pipeline integration."""
    slug: str
    manifest: dict
    indicator_func: Optional[Callable] = None
    interpreter_func: Optional[Callable] = None
    trigger_func: Optional[Callable] = None
    pack_dir: Path = field(default_factory=Path)
    is_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


# =============================================================================
# MODULE-LEVEL REGISTRY
# =============================================================================

_user_packs: Dict[str, RegisteredPack] = {}
_registry_loaded: bool = False


# =============================================================================
# PATH HELPERS
# =============================================================================

def get_user_packs_dir() -> Path:
    """Get path to user_packs/ directory, creating if needed."""
    src_dir = Path(__file__).parent
    project_dir = src_dir.parent
    packs_dir = project_dir / "user_packs"
    packs_dir.mkdir(exist_ok=True)
    return packs_dir


# =============================================================================
# CORE REGISTRY OPERATIONS
# =============================================================================

def scan_and_load_all() -> Dict[str, RegisteredPack]:
    """
    Scan user_packs/ directory, validate and load all packs.

    Called at app startup and on manual refresh.

    Returns:
        Dict mapping slug -> RegisteredPack for all discovered packs.
    """
    global _user_packs, _registry_loaded

    packs_dir = get_user_packs_dir()
    discovered = {}

    for pack_dir in sorted(packs_dir.iterdir()):
        if not pack_dir.is_dir():
            continue
        # Skip __pycache__ and hidden directories
        if pack_dir.name.startswith((".", "__")):
            continue

        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        pack = load_single_pack(pack_dir)
        discovered[pack.slug] = pack

        if pack.is_valid:
            register_pack(pack)

    _user_packs = discovered
    _registry_loaded = True
    return discovered


def load_single_pack(pack_dir: Path) -> RegisteredPack:
    """
    Load and validate a single pack from its directory.

    Returns:
        RegisteredPack with is_valid=True if everything checks out,
        or is_valid=False with validation_errors populated.
    """
    errors = []
    slug = pack_dir.name

    # Load manifest
    manifest_path = pack_dir / "manifest.json"
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return RegisteredPack(
            slug=slug,
            manifest={},
            pack_dir=pack_dir,
            is_valid=False,
            validation_errors=[f"Failed to read manifest.json: {e}"],
        )

    # Validate manifest schema
    valid, manifest_errors = validate_manifest(manifest)
    errors.extend(manifest_errors)

    # Check slug matches directory name
    if manifest.get("slug") != slug:
        errors.append(
            f"Manifest slug '{manifest.get('slug')}' does not match "
            f"directory name '{slug}'"
        )

    # Validate and import indicator.py
    indicator_path = pack_dir / "indicator.py"
    indicator_func = None
    if not indicator_path.exists():
        errors.append("Missing indicator.py")
    else:
        py_valid, py_errors = validate_python_file(str(indicator_path))
        errors.extend(py_errors)

        func_name = manifest.get("indicator_function", "")
        if func_name:
            exists, err = validate_function_exists(str(indicator_path), func_name)
            if not exists:
                errors.append(err)

    # Validate and import interpreter.py
    interpreter_path = pack_dir / "interpreter.py"
    interpreter_func = None
    trigger_func = None
    if not interpreter_path.exists():
        errors.append("Missing interpreter.py")
    else:
        py_valid, py_errors = validate_python_file(str(interpreter_path))
        errors.extend(py_errors)

        interp_func_name = manifest.get("interpreter_function", "")
        if interp_func_name:
            exists, err = validate_function_exists(str(interpreter_path), interp_func_name)
            if not exists:
                errors.append(err)

        trig_func_name = manifest.get("trigger_function", "")
        if trig_func_name:
            exists, err = validate_function_exists(str(interpreter_path), trig_func_name)
            if not exists:
                errors.append(err)

    # If validation passed, import the modules
    if not errors:
        try:
            ind_module = _import_module_safely(
                indicator_path, f"user_pack_{slug}_indicator"
            )
            indicator_func = getattr(ind_module, manifest["indicator_function"], None)
            if indicator_func is None:
                errors.append(
                    f"Function '{manifest['indicator_function']}' not found "
                    f"after import of indicator.py"
                )
        except Exception as e:
            errors.append(f"Failed to import indicator.py: {e}")

        try:
            interp_module = _import_module_safely(
                interpreter_path, f"user_pack_{slug}_interpreter"
            )
            interpreter_func = getattr(
                interp_module, manifest["interpreter_function"], None
            )
            trigger_func = getattr(
                interp_module, manifest["trigger_function"], None
            )
            if interpreter_func is None:
                errors.append(
                    f"Function '{manifest['interpreter_function']}' not found "
                    f"after import of interpreter.py"
                )
            if trigger_func is None:
                errors.append(
                    f"Function '{manifest['trigger_function']}' not found "
                    f"after import of interpreter.py"
                )
        except Exception as e:
            errors.append(f"Failed to import interpreter.py: {e}")

    return RegisteredPack(
        slug=slug,
        manifest=manifest,
        indicator_func=indicator_func,
        interpreter_func=interpreter_func,
        trigger_func=trigger_func,
        pack_dir=pack_dir,
        is_valid=len(errors) == 0,
        validation_errors=errors,
    )


def register_pack(pack: RegisteredPack) -> None:
    """
    Register a validated pack into the pipeline registries.

    Injects into:
    - confluence_groups.TEMPLATES
    - interpreters.INTERPRETERS, INTERPRETER_FUNCS, TRIGGER_FUNCS
    - indicators.GROUP_INDICATOR_FUNCS
    - Auto-creates a ConfluenceGroup entry if needed
    """
    if not pack.is_valid:
        return

    manifest = pack.manifest
    slug = pack.slug

    # 1. Inject into TEMPLATES
    _inject_into_templates(manifest)

    # 2. Inject into INTERPRETERS and register functions
    _inject_into_interpreters(manifest, pack.interpreter_func, pack.trigger_func)

    # 3. Register group indicator function
    _inject_into_indicators(manifest, pack.indicator_func)

    # 4. Auto-create ConfluenceGroup if none exists
    _ensure_confluence_group(manifest)


def unregister_pack(slug: str) -> None:
    """Remove a pack from all registries."""
    pack = _user_packs.get(slug)
    if not pack:
        return

    manifest = pack.manifest

    # Remove from TEMPLATES
    from confluence_groups import TEMPLATES
    TEMPLATES.pop(slug, None)

    # Remove from INTERPRETERS and function registries
    from interpreters import (
        INTERPRETERS,
        unregister_interpreter,
        unregister_trigger_detector,
    )
    for interp_key in manifest.get("interpreters", []):
        INTERPRETERS.pop(interp_key, None)
        unregister_interpreter(interp_key)
        unregister_trigger_detector(interp_key)

    # Remove from GROUP_INDICATOR_FUNCS
    from indicators import unregister_group_indicator
    unregister_group_indicator(slug)

    _user_packs.pop(slug, None)


def delete_pack(slug: str) -> bool:
    """
    Delete a user pack from disk and unregister it.

    Returns True if successfully deleted.
    """
    pack = _user_packs.get(slug)
    if not pack:
        return False

    unregister_pack(slug)

    # Remove directory
    try:
        if pack.pack_dir.exists():
            shutil.rmtree(pack.pack_dir)
        return True
    except Exception as e:
        print(f"Error deleting user pack '{slug}': {e}")
        return False


def refresh_registry() -> Dict[str, RegisteredPack]:
    """
    Force re-scan and reload of all user packs.

    Unregisters all existing user packs first, then re-scans.
    """
    # Unregister all existing packs
    for slug in list(_user_packs.keys()):
        unregister_pack(slug)

    # Re-scan
    return scan_and_load_all()


def get_registered_packs() -> Dict[str, RegisteredPack]:
    """Get all currently registered user packs."""
    return _user_packs.copy()


def get_pack(slug: str) -> Optional[RegisteredPack]:
    """Get a specific registered pack."""
    return _user_packs.get(slug)


# =============================================================================
# SAFE MODULE IMPORT
# =============================================================================

def _import_module_safely(file_path: Path, module_name: str):
    """
    Import a Python module using importlib with restricted builtins.

    The file must have already passed AST validation via validate_python_file().
    """
    import builtins

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Restrict builtins to safe subset.
    # __import__ is needed for `import pandas` etc. to work at runtime;
    # the AST validation layer already restricts which modules can be imported.
    safe_builtins_dict = {
        name: getattr(builtins, name)
        for name in SAFE_BUILTINS
        if hasattr(builtins, name)
    }
    safe_builtins_dict["__import__"] = builtins.__import__
    module.__builtins__ = safe_builtins_dict

    spec.loader.exec_module(module)
    return module


# =============================================================================
# INJECTION HELPERS
# =============================================================================

def _inject_into_templates(manifest: dict) -> None:
    """Add manifest data as a TEMPLATES entry in confluence_groups."""
    from confluence_groups import TEMPLATES

    TEMPLATES[manifest["slug"]] = {
        "name": manifest["name"],
        "category": manifest["category"],
        "description": manifest["description"],
        "interpreters": manifest["interpreters"],
        "trigger_prefix": manifest["trigger_prefix"],
        "parameters_schema": manifest["parameters_schema"],
        "plot_schema": manifest.get("plot_schema", {}),
        "outputs": manifest["outputs"],
        "output_descriptions": manifest["output_descriptions"],
        "triggers": manifest["triggers"],
        "indicator_columns": manifest["indicator_columns"],
        "display_type": manifest.get("display_type", "overlay"),
        "column_color_map": manifest.get("column_color_map", {}),
        "_user_pack": True,
    }


def _inject_into_interpreters(
    manifest: dict,
    interpreter_func: Optional[Callable],
    trigger_func: Optional[Callable],
) -> None:
    """Add interpreter config and functions to interpreters registries."""
    from interpreters import (
        INTERPRETERS,
        InterpreterConfig,
        register_interpreter,
        register_trigger_detector,
    )

    for interp_key in manifest["interpreters"]:
        INTERPRETERS[interp_key] = InterpreterConfig(
            name=manifest["name"],
            description=manifest["description"],
            category=manifest["category"],
            requires_indicators=manifest.get("requires_indicators", []),
            outputs=manifest["outputs"],
            triggers=[
                f"{manifest['trigger_prefix']}_{t['base']}"
                for t in manifest["triggers"]
            ],
        )

        # Register wrapped functions that match the built-in (df) -> ... signature
        if interpreter_func:
            wrapper = _make_interpreter_wrapper(interpreter_func, manifest)
            register_interpreter(interp_key, wrapper)

        if trigger_func:
            wrapper = _make_trigger_wrapper(trigger_func, manifest)
            register_trigger_detector(interp_key, wrapper)


def _inject_into_indicators(
    manifest: dict,
    indicator_func: Optional[Callable],
) -> None:
    """Register the group indicator function."""
    from indicators import register_group_indicator

    if indicator_func:
        wrapper = _make_group_indicator_wrapper(indicator_func, manifest)
        register_group_indicator(manifest["slug"], wrapper)


def _ensure_confluence_group(manifest: dict) -> None:
    """Auto-create a ConfluenceGroup entry if one doesn't exist for this pack."""
    from confluence_groups import (
        load_confluence_groups,
        save_confluence_groups,
        ConfluenceGroup,
        PlotSettings,
    )

    groups = load_confluence_groups()
    default_id = f"{manifest['slug']}_default"

    if any(g.id == default_id for g in groups):
        return  # Already exists

    # Build default parameters from schema
    default_params = {
        key: spec["default"]
        for key, spec in manifest["parameters_schema"].items()
    }

    # Build default plot colors from schema
    plot_schema = manifest.get("plot_schema", {})
    default_colors = {
        key: spec["default"]
        for key, spec in plot_schema.items()
        if spec.get("type") == "color"
    }

    new_group = ConfluenceGroup(
        id=default_id,
        base_template=manifest["slug"],
        version="Default",
        description=f"Default {manifest['name']} configuration",
        enabled=True,
        is_default=False,
        parameters=default_params,
        plot_settings=PlotSettings(
            colors=default_colors,
            line_width=1,
            visible=True,
        ),
    )

    groups.append(new_group)
    save_confluence_groups(groups)


# =============================================================================
# WRAPPER FACTORIES
# =============================================================================
# User pack functions use (df, **params) signatures. Built-in pipeline expects
# (df) -> ... signatures. These wrappers bridge the gap.

def _make_interpreter_wrapper(
    interpreter_func: Callable,
    manifest: dict,
) -> Callable:
    """
    Create a wrapper matching the built-in interpret_xxx(df) -> Series signature.

    Uses default parameters from the manifest's parameters_schema.
    """
    default_params = {
        key: spec["default"]
        for key, spec in manifest["parameters_schema"].items()
    }

    def wrapper(df: pd.DataFrame) -> pd.Series:
        return interpreter_func(df, **default_params)

    wrapper.__name__ = f"interpret_{manifest['slug']}"
    wrapper.__doc__ = f"User pack interpreter: {manifest['name']}"
    return wrapper


def _make_trigger_wrapper(
    trigger_func: Callable,
    manifest: dict,
) -> Callable:
    """
    Create a wrapper matching the built-in detect_xxx_triggers(df) -> Dict signature.

    Uses default parameters from the manifest's parameters_schema.
    """
    default_params = {
        key: spec["default"]
        for key, spec in manifest["parameters_schema"].items()
    }

    def wrapper(df: pd.DataFrame) -> Dict[str, pd.Series]:
        return trigger_func(df, **default_params)

    wrapper.__name__ = f"detect_{manifest['slug']}_triggers"
    wrapper.__doc__ = f"User pack trigger detector: {manifest['name']}"
    return wrapper


def _make_group_indicator_wrapper(
    indicator_func: Callable,
    manifest: dict,
) -> Callable:
    """
    Create a wrapper matching the built-in _run_xxx_indicators(df, group) signature.

    Passes the group's parameters to the user pack indicator function.
    """
    def wrapper(df: pd.DataFrame, group) -> pd.DataFrame:
        # Check if indicator columns already exist
        existing = all(
            col in df.columns
            for col in manifest["indicator_columns"]
        )
        if existing:
            return df
        return indicator_func(df, **group.parameters)

    wrapper.__name__ = f"_run_{manifest['slug']}_indicators"
    wrapper.__doc__ = f"User pack indicators: {manifest['name']}"
    return wrapper
