"""
Execution Type Modules — pluggable entry/confirmation/bail protocols.

Each execution type defines:
  - How to detect an entry signal (bar-close boolean vs. level cross)
  - What fill price to use
  - Whether confirmation is needed and its parameters
  - What happens on non-confirmation (bail behavior)
  - Whether to skip stop/target on entry bar

Shared behavior (stop/target, trailing, signal exit, bar count exit)
stays in PositionStateMachine.

Three core modules:
  1. BarCloseExecution (C) — Entry at bar close when trigger fires
  2. LevelExecution (L0, L1) — Entry at indicator level when price crosses
  3. ConfirmedExecution (LC, CC, HM, HL) — Two-stage entry with confirmation

Future: execution types can apply to stops and targets too (context field).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EntryResult:
    """Result of checking entry conditions."""
    fired: bool
    fill_price: float = 0.0
    is_ltype: bool = False  # True if filled at indicator level (affects stop computation)


@dataclass
class ConfirmationConfig:
    """Confirmation parameters set on entry."""
    needs_confirm: bool = False
    confirm_bar_offset: int = 0     # 0 = same bar, 1 = next bar
    bail_action: str = 'exit_market'
    hold_seconds: int = 0


@dataclass
class BailResult:
    """Result of checking bail conditions."""
    should_bail: bool = False
    reason: str = ''
    fill_price: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ExecutionTypeModule:
    """Base class for execution type modules."""

    slug: str = ''
    name: str = ''
    description: str = ''
    exec_type_codes: tuple = ()
    contexts: tuple = ('entry',)  # Which contexts this type supports

    # Workflow steps for UI visualization
    steps: list = []

    # Parameters schema for UI editing
    parameters_schema: dict = {}

    def check_entry_signal(
        self,
        trigger_id: str,
        exec_type: str,
        c_triggers: Dict[str, bool],
        l_type_fills: Dict[str, float],
        current_values: Dict[str, float],
        strip_exec_suffix: Callable,
    ) -> EntryResult:
        """Check if entry conditions are met. Return EntryResult."""
        raise NotImplementedError

    def get_confirmation_config(
        self,
        trigger_id: str,
        exec_type: str,
        bar_count: int,
        get_variant_param: Callable,
    ) -> ConfirmationConfig:
        """Return confirmation configuration for this entry."""
        return ConfirmationConfig(needs_confirm=False)

    def should_check_confirmation(
        self,
        exec_type: str,
        entry_bar_count: int,
        current_bar_count: int,
        pending_confirm_bar: int,
    ) -> bool:
        """Whether to run the confirmation check on this bar."""
        return False

    def on_confirmation_failed(self, exec_type: str, state) -> None:
        """Set state flags when confirmation fails."""
        pass

    def on_confirmation_passed(self, exec_type: str, state) -> None:
        """Clear state flags when confirmation passes."""
        pass

    def check_bail_at_market(
        self,
        exec_type: str,
        state,
        bar_count: int,
        bar_open: float,
    ) -> Optional[BailResult]:
        """Check if a market-bail exit should happen at bar open."""
        return None

    def check_bail_exit(
        self,
        exec_type: str,
        state,
        bar_count: int,
        direction: str,
        high: float,
        low: float,
    ) -> Optional[BailResult]:
        """Check limit-based bail exits (priority 2/2b in check_exit)."""
        return None

    def skip_stop_target_on_entry_bar(self, exec_type: str) -> bool:
        """Whether to skip stop/target OHLC checks on the entry bar."""
        return False

    def to_dict(self) -> dict:
        """Serialize for API/frontend."""
        return {
            'slug': self.slug,
            'name': self.name,
            'description': self.description,
            'exec_type_codes': list(self.exec_type_codes),
            'contexts': list(self.contexts),
            'steps': self.steps,
            'parameters_schema': self.parameters_schema,
        }


# ═══════════════════════════════════════════════════════════════════════════
# BAR CLOSE EXECUTION [C]
# ═══════════════════════════════════════════════════════════════════════════

class BarCloseExecution(ExecutionTypeModule):
    """Entry at bar close price when trigger boolean fires."""

    slug = 'bar_close'
    name = 'Bar Close [C]'
    description = 'Enter at bar close price when the trigger condition is met at bar close.'
    exec_type_codes = ('C',)
    contexts = ('entry', 'exit_signal')

    steps = [
        {'action': 'check_trigger', 'label': 'Check bar-close trigger boolean', 'source': 'c_triggers'},
        {'action': 'fill', 'label': 'Fill at bar close price', 'price': 'close'},
        {'action': 'plot_marker', 'label': 'Plot entry marker', 'symbol': 'cross', 'color_key': 'entry_color'},
    ]

    parameters_schema = {
        'reference_bar': {'type': 'int', 'default': 0, 'options': [0, -1], 'label': 'Reference Bar'},
        'order_type': {'type': 'str', 'default': 'market', 'options': ['market', 'limit'], 'label': 'Order Type'},
    }

    def check_entry_signal(self, trigger_id, exec_type, c_triggers, l_type_fills,
                           current_values, strip_exec_suffix):
        base_trigger = strip_exec_suffix(trigger_id)
        if c_triggers.get(trigger_id, False) or c_triggers.get(base_trigger, False):
            return EntryResult(fired=True, fill_price=current_values.get('close', 0), is_ltype=False)
        return EntryResult(fired=False)

    def skip_stop_target_on_entry_bar(self, exec_type):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# LEVEL EXECUTION [L0, L1]
# ═══════════════════════════════════════════════════════════════════════════

class LevelExecution(ExecutionTypeModule):
    """Entry at indicator level when price crosses mid-bar."""

    slug = 'level'
    name = 'Level [L]'
    description = 'Enter at the indicator level price when price crosses the level within the bar. L0 uses current bar level, L1 uses previous bar level.'
    exec_type_codes = ('L0', 'L1')
    contexts = ('entry', 'exit_signal', 'stop', 'target')

    steps = [
        {'action': 'check_level_cross', 'label': 'Check if price crossed indicator level within bar', 'source': 'l_type_fills'},
        {'action': 'fill', 'label': 'Fill at indicator level price', 'price': 'level'},
        {'action': 'plot_marker', 'label': 'Plot entry marker at level', 'symbol': 'cross', 'color_key': 'entry_color'},
    ]

    parameters_schema = {
        'reference_bar': {'type': 'int', 'default': -1, 'options': [0, -1], 'label': 'Reference Bar'},
        'order_type': {'type': 'str', 'default': 'market', 'options': ['market', 'limit'], 'label': 'Order Type'},
        'hold_seconds': {'type': 'int', 'default': 0, 'min': 0, 'label': 'Hold Seconds'},
    }

    def check_entry_signal(self, trigger_id, exec_type, c_triggers, l_type_fills,
                           current_values, strip_exec_suffix):
        # L-type: check l_type_fills for level cross
        if l_type_fills:
            if trigger_id in l_type_fills:
                return EntryResult(fired=True, fill_price=l_type_fills[trigger_id], is_ltype=True)
        # Fallback to C-type boolean (bar-close confirmation of level cross)
        base_trigger = strip_exec_suffix(trigger_id)
        if c_triggers.get(trigger_id, False) or c_triggers.get(base_trigger, False):
            return EntryResult(fired=True, fill_price=current_values.get('close', 0), is_ltype=False)
        return EntryResult(fired=False)

    def skip_stop_target_on_entry_bar(self, exec_type):
        return True  # L-type entries skip stop/target on entry bar


# ═══════════════════════════════════════════════════════════════════════════
# CONFIRMED EXECUTION [LC, CC, HM, HL]
# ═══════════════════════════════════════════════════════════════════════════

class ConfirmedExecution(ExecutionTypeModule):
    """Two-stage entry: initial fill + bar-close confirmation.

    LC: Level cross entry, configurable confirmation offset and bail.
    CC: Bar-close entry, next-bar confirmation, configurable bail.
    HM: Legacy — LC with confirm_bar_offset=0, bail=exit_market.
    HL: Legacy — LC with confirm_bar_offset=0, bail=exit_limit.
    """

    slug = 'confirmed'
    name = 'Confirmed [LC/CC]'
    description = 'Two-stage entry: fill at level or close, then wait for bar-close confirmation. If unconfirmed, bail according to configured action.'
    exec_type_codes = ('LC', 'CC', 'HM', 'HL')
    contexts = ('entry',)

    steps = [
        {'action': 'check_signal', 'label': 'Check entry signal (level cross for LC, bar close for CC)'},
        {'action': 'fill', 'label': 'Fill at signal price'},
        {'action': 'plot_marker', 'label': 'Plot pending entry marker', 'symbol': 'circle', 'color_key': 'entry_color'},
        {'action': 'wait_for_confirmation', 'label': 'Wait for bar-close confirmation'},
        {'action': 'branch', 'label': 'Check confirmation',
         'if_confirmed': [
             {'action': 'plot_marker', 'label': 'Plot confirmed marker', 'symbol': 'cross', 'color_key': 'entry_color'},
         ],
         'if_not_confirmed': [
             {'action': 'bail', 'label': 'Execute bail action (market exit or limit at entry price)'},
             {'action': 'plot_marker', 'label': 'Plot bail marker', 'symbol': 'xcross', 'color_key': 'exit_stop_color'},
         ]},
    ]

    parameters_schema = {
        'confirm_bar_offset': {'type': 'int', 'default': 0, 'options': [0, 1], 'label': 'Confirmation Offset (bars)'},
        'bail_action': {'type': 'str', 'default': 'exit_market', 'options': ['exit_market', 'exit_limit'], 'label': 'Bail Action'},
        'order_type': {'type': 'str', 'default': 'market', 'options': ['market', 'limit'], 'label': 'Order Type'},
        'hold_seconds': {'type': 'int', 'default': 0, 'min': 0, 'label': 'Hold Seconds (LC only)'},
    }

    def check_entry_signal(self, trigger_id, exec_type, c_triggers, l_type_fills,
                           current_values, strip_exec_suffix):
        fill_price = None
        is_ltype = False

        if exec_type in ('LC', 'HM', 'HL'):
            # Level-based entry (same as L-type)
            if l_type_fills:
                # LC uses _ib suffix for level lookup
                if exec_type == 'LC':
                    ib_lookup = strip_exec_suffix(trigger_id) + '_ib'
                    if ib_lookup in l_type_fills:
                        fill_price = l_type_fills[ib_lookup]
                        is_ltype = True
                if fill_price is None and trigger_id in l_type_fills:
                    fill_price = l_type_fills[trigger_id]
                    is_ltype = True

        if fill_price is None:
            # CC uses bar-close entry, or fallback for LC/HM/HL
            base_trigger = strip_exec_suffix(trigger_id)
            if c_triggers.get(trigger_id, False) or c_triggers.get(base_trigger, False):
                fill_price = current_values.get('close', 0)

        if fill_price is None:
            return EntryResult(fired=False)
        return EntryResult(fired=True, fill_price=fill_price, is_ltype=is_ltype)

    def get_confirmation_config(self, trigger_id, exec_type, bar_count, get_variant_param):
        if exec_type == 'HM':
            # Legacy HM: same-bar confirmation, bail at market
            return ConfirmationConfig(
                needs_confirm=True, confirm_bar_offset=0,
                bail_action='exit_market')
        elif exec_type == 'HL':
            # Legacy HL: same-bar confirmation, bail at limit (entry price)
            return ConfirmationConfig(
                needs_confirm=True, confirm_bar_offset=0,
                bail_action='exit_limit')
        elif exec_type == 'LC':
            cbo = get_variant_param(trigger_id, 'LC', 'confirm_bar_offset', 0)
            bail = get_variant_param(trigger_id, 'LC', 'bail_action', 'exit_market')
            hold = get_variant_param(trigger_id, 'L', 'hold_seconds', 0)
            return ConfirmationConfig(
                needs_confirm=True, confirm_bar_offset=cbo,
                bail_action=bail, hold_seconds=hold)
        elif exec_type == 'CC':
            bail = get_variant_param(trigger_id, 'CC', 'bail_action', 'exit_market')
            return ConfirmationConfig(
                needs_confirm=True, confirm_bar_offset=1,
                bail_action=bail)
        return ConfirmationConfig(needs_confirm=False)

    def should_check_confirmation(self, exec_type, entry_bar_count,
                                  current_bar_count, pending_confirm_bar):
        if exec_type in ('HM', 'HL'):
            return entry_bar_count == current_bar_count
        if pending_confirm_bar >= 0 and current_bar_count == pending_confirm_bar:
            return True
        return False

    def on_confirmation_failed(self, exec_type, state):
        if exec_type == 'HM':
            state.pending_hm_exit = True
        elif exec_type == 'HL':
            state.pending_hl_limit = True
        # LC/CC: pending_confirm_bar stays set, check_exit handles bail

    def on_confirmation_passed(self, exec_type, state):
        if exec_type in ('LC', 'CC'):
            state.pending_confirm_bar = -1

    def check_bail_at_market(self, exec_type, state, bar_count, bar_open):
        # HM: always bail at market on next bar
        if state.pending_hm_exit:
            return BailResult(should_bail=True, reason='unconfirmed_hm', fill_price=bar_open)

        # LC/CC with exit_market bail: bail at market when past confirmation bar
        if (state.pending_confirm_bar >= 0 and
                bar_count > state.pending_confirm_bar and
                state.bail_action == 'exit_market'):
            reason = f'unconfirmed_{state.exec_type.lower()}'
            return BailResult(should_bail=True, reason=reason, fill_price=bar_open)

        return None

    def check_bail_exit(self, exec_type, state, bar_count, direction, high, low):
        ep = state.entry_price

        # HL limit exit at entry price
        if state.pending_hl_limit:
            if direction == 'LONG' and high >= ep:
                return BailResult(should_bail=True, reason='unconfirmed_hl', fill_price=ep)
            elif direction == 'SHORT' and low <= ep:
                return BailResult(should_bail=True, reason='unconfirmed_hl', fill_price=ep)

        # LC/CC limit-based bail
        if (state.pending_confirm_bar >= 0 and
                bar_count > state.pending_confirm_bar and
                state.bail_action in ('exit_limit', 'exit_limit_breakeven')):
            reason = f'unconfirmed_{state.exec_type.lower()}'
            if direction == 'LONG' and high >= ep:
                return BailResult(should_bail=True, reason=reason, fill_price=ep)
            elif direction == 'SHORT' and low <= ep:
                return BailResult(should_bail=True, reason=reason, fill_price=ep)

        return None

    def skip_stop_target_on_entry_bar(self, exec_type):
        # LC, HM, HL are L-type entries — skip stop/target on entry bar
        # CC is bar-close entry — don't skip
        return exec_type in ('LC', 'HM', 'HL')


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

_MODULES: Dict[str, ExecutionTypeModule] = {}
_ALL_MODULES: List[ExecutionTypeModule] = []


def register(module: ExecutionTypeModule):
    """Register a module for its exec_type_codes."""
    _ALL_MODULES.append(module)
    for code in module.exec_type_codes:
        _MODULES[code] = module


def get_module(exec_type: str) -> ExecutionTypeModule:
    """Get the module for a given exec_type code."""
    mod = _MODULES.get(exec_type)
    if mod is None:
        # Default to bar close
        mod = _MODULES.get('C')
    return mod


def list_modules() -> List[ExecutionTypeModule]:
    """List all unique registered modules."""
    return list(_ALL_MODULES)


# Auto-register built-in modules
register(BarCloseExecution())
register(LevelExecution())
register(ConfirmedExecution())
