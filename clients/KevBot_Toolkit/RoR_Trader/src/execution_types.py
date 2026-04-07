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
    display_code: str = ''  # Simplified code shown in UI (e.g., 'L' instead of 'L0'/'L1')
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
            'display_code': self.display_code or (self.exec_type_codes[0] if self.exec_type_codes else ''),
            'exec_type_codes': list(self.exec_type_codes),
            'contexts': list(self.contexts),
            'steps': self.steps,  # dict keyed by context (entry, exit_signal, stop, target)
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
    display_code = 'C'
    contexts = ('entry', 'exit_signal')

    steps = {
        'entry': [
            {'action': 'check_trigger', 'label': 'Check bar-close trigger boolean'},
            {'action': 'fill', 'label': 'Fill at bar close price', 'price': 'close'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: entry_{{direction}}_market', 'event': 'entry_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot entry marker', 'symbol': 'cross', 'color_key': 'entry_color'},
        ],
        'exit_signal': [
            {'action': 'check_trigger', 'label': 'Check bar-close exit trigger boolean'},
            {'action': 'fill', 'label': 'Fill at bar close price', 'price': 'close'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market', 'event': 'exit_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot exit marker', 'symbol': 'cross', 'color_key': 'exit_color'},
        ],
    }

    parameters_schema = {
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
    description = 'Enter at the indicator level price when price crosses the level within the bar. The confluence pack indicator determines which bar level is used.'
    exec_type_codes = ('L0', 'L1')
    display_code = 'L'
    contexts = ('entry', 'exit_signal', 'stop', 'target')

    steps = {
        'entry': [
            {'action': 'check_level_cross', 'label': 'Check if price crossed indicator level within bar'},
            {'action': 'fill', 'label': 'Fill at indicator level price', 'price': 'level'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: entry_{{direction}}_market', 'event': 'entry_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot entry marker at level', 'symbol': 'cross', 'color_key': 'entry_color'},
        ],
        'exit_signal': [
            {'action': 'check_level_cross', 'label': 'Check if price crossed exit level within bar'},
            {'action': 'fill', 'label': 'Fill at level price', 'price': 'level'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market', 'event': 'exit_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot exit marker', 'symbol': 'cross', 'color_key': 'exit_color'},
        ],
        'stop': [
            {'action': 'check_level_cross', 'label': 'Check if price crossed stop level within bar'},
            {'action': 'fill', 'label': 'Fill at stop level (gap-aware: fill at bar open if gapped past)'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market', 'event': 'exit_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot stop marker', 'symbol': 'xcross', 'color_key': 'exit_stop_color'},
        ],
        'target': [
            {'action': 'check_level_cross', 'label': 'Check if price crossed target level within bar'},
            {'action': 'fill', 'label': 'Fill at target level (gap-aware: fill at bar open if gapped past)'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market', 'event': 'exit_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot target marker', 'symbol': 'cross', 'color_key': 'exit_win_color'},
        ],
    }

    parameters_schema = {
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
# LEVEL-CLOSE EXECUTION [LC] (+ legacy HM, HL)
# ═══════════════════════════════════════════════════════════════════════════

class LevelCloseExecution(ExecutionTypeModule):
    """Level cross entry + bar-close confirmation.

    Enters at indicator level when price crosses mid-bar, then waits for
    bar-close confirmation. If unconfirmed, bails according to configured action.
    HM/HL are legacy aliases with hardcoded parameters.
    """

    slug = 'level_close'
    name = 'Level-Close [LC]'
    description = 'Enter at indicator level when price crosses, then confirm at bar close. Bail if unconfirmed.'
    exec_type_codes = ('LC', 'HM', 'HL')
    display_code = 'LC'
    contexts = ('entry',)

    steps = {
        'entry': [
            {'action': 'check_level_cross', 'label': 'Check if price crossed indicator level within bar'},
            {'action': 'fill', 'label': 'Fill at indicator level price'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: entry_{{direction}}_market', 'event': 'entry_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot pending entry marker', 'symbol': 'circle', 'color_key': 'entry_color'},
            {'action': 'wait_for_confirmation', 'label': 'Wait for bar-close confirmation (offset: confirm_bar_offset bars)'},
            {'action': 'branch', 'label': 'Check if bar close confirms direction',
             'if_confirmed': [
                 {'action': 'plot_marker', 'label': 'Plot confirmed marker', 'symbol': 'cross', 'color_key': 'entry_color'},
             ],
             'if_not_confirmed': [
                 {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market (bail)', 'event': 'exit_{{direction}}_market'},
                 {'action': 'bail', 'label': 'Execute bail action (market exit or limit at entry price)'},
                 {'action': 'plot_marker', 'label': 'Plot bail marker', 'symbol': 'xcross', 'color_key': 'exit_stop_color'},
             ]},
        ],
    }

    parameters_schema = {
        'order_type': {'type': 'str', 'default': 'market', 'options': ['market', 'limit'], 'label': 'Order Type'},
        'confirm_bar_offset': {'type': 'int', 'default': 0, 'options': [0, 1], 'label': 'Confirmation Offset (bars)'},
        'bail_action': {'type': 'str', 'default': 'exit_market', 'options': ['exit_market', 'exit_limit'], 'label': 'Bail Action'},
        'hold_seconds': {'type': 'int', 'default': 0, 'min': 0, 'label': 'Hold Seconds'},
    }

    def check_entry_signal(self, trigger_id, exec_type, c_triggers, l_type_fills,
                           current_values, strip_exec_suffix):
        fill_price = None
        is_ltype = False

        # Level-based entry
        if l_type_fills:
            if exec_type == 'LC':
                ib_lookup = strip_exec_suffix(trigger_id) + '_ib'
                if ib_lookup in l_type_fills:
                    fill_price = l_type_fills[ib_lookup]
                    is_ltype = True
            if fill_price is None and trigger_id in l_type_fills:
                fill_price = l_type_fills[trigger_id]
                is_ltype = True

        if fill_price is None:
            # Fallback to bar-close
            base_trigger = strip_exec_suffix(trigger_id)
            if c_triggers.get(trigger_id, False) or c_triggers.get(base_trigger, False):
                fill_price = current_values.get('close', 0)

        if fill_price is None:
            return EntryResult(fired=False)
        return EntryResult(fired=True, fill_price=fill_price, is_ltype=is_ltype)

    def get_confirmation_config(self, trigger_id, exec_type, bar_count, get_variant_param):
        if exec_type == 'HM':
            return ConfirmationConfig(needs_confirm=True, confirm_bar_offset=0, bail_action='exit_market')
        elif exec_type == 'HL':
            return ConfirmationConfig(needs_confirm=True, confirm_bar_offset=0, bail_action='exit_limit')
        else:  # LC
            cbo = get_variant_param(trigger_id, 'LC', 'confirm_bar_offset', 0)
            bail = get_variant_param(trigger_id, 'LC', 'bail_action', 'exit_market')
            hold = get_variant_param(trigger_id, 'L', 'hold_seconds', 0)
            return ConfirmationConfig(needs_confirm=True, confirm_bar_offset=cbo, bail_action=bail, hold_seconds=hold)

    def should_check_confirmation(self, exec_type, entry_bar_count, current_bar_count, pending_confirm_bar):
        if exec_type in ('HM', 'HL'):
            return entry_bar_count == current_bar_count
        return pending_confirm_bar >= 0 and current_bar_count == pending_confirm_bar

    def on_confirmation_failed(self, exec_type, state):
        if exec_type == 'HM':
            state.pending_hm_exit = True
        elif exec_type == 'HL':
            state.pending_hl_limit = True

    def on_confirmation_passed(self, exec_type, state):
        state.pending_confirm_bar = -1

    def check_bail_at_market(self, exec_type, state, bar_count, bar_open):
        if state.pending_hm_exit:
            return BailResult(should_bail=True, reason='unconfirmed_hm', fill_price=bar_open)
        if (state.pending_confirm_bar >= 0 and bar_count > state.pending_confirm_bar
                and state.bail_action == 'exit_market'):
            return BailResult(should_bail=True, reason=f'unconfirmed_{state.exec_type.lower()}', fill_price=bar_open)
        return None

    def check_bail_exit(self, exec_type, state, bar_count, direction, high, low):
        ep = state.entry_price
        if state.pending_hl_limit:
            if direction == 'LONG' and high >= ep:
                return BailResult(should_bail=True, reason='unconfirmed_hl', fill_price=ep)
            elif direction == 'SHORT' and low <= ep:
                return BailResult(should_bail=True, reason='unconfirmed_hl', fill_price=ep)
        if (state.pending_confirm_bar >= 0 and bar_count > state.pending_confirm_bar
                and state.bail_action in ('exit_limit', 'exit_limit_breakeven')):
            reason = f'unconfirmed_{state.exec_type.lower()}'
            if direction == 'LONG' and high >= ep:
                return BailResult(should_bail=True, reason=reason, fill_price=ep)
            elif direction == 'SHORT' and low <= ep:
                return BailResult(should_bail=True, reason=reason, fill_price=ep)
        return None

    def skip_stop_target_on_entry_bar(self, exec_type):
        return True  # LC, HM, HL are all L-type entries


# ═══════════════════════════════════════════════════════════════════════════
# CLOSE-CLOSE EXECUTION [CC]
# ═══════════════════════════════════════════════════════════════════════════

class CloseCloseExecution(ExecutionTypeModule):
    """Bar-close entry + next-bar-close confirmation.

    Enters at bar close price when trigger fires, then requires the next
    bar to close confirming the direction. If unconfirmed, bails.
    """

    slug = 'close_close'
    name = 'Close-Close [CC]'
    description = 'Enter at bar close, then confirm on next bar close. Bail at market if next bar does not confirm direction.'
    exec_type_codes = ('CC',)
    display_code = 'CC'
    contexts = ('entry',)

    steps = {
        'entry': [
            {'action': 'check_trigger', 'label': 'Check bar-close trigger boolean'},
            {'action': 'fill', 'label': 'Fill at bar close price'},
            {'action': 'fire_webhook', 'label': 'Fire webhook: entry_{{direction}}_market', 'event': 'entry_{{direction}}_market'},
            {'action': 'plot_marker', 'label': 'Plot pending entry marker', 'symbol': 'circle', 'color_key': 'entry_color'},
            {'action': 'wait_for_next_bar', 'label': 'Wait for next bar to close'},
            {'action': 'branch', 'label': 'Check if next bar confirms direction',
             'if_confirmed': [
                 {'action': 'plot_marker', 'label': 'Plot confirmed marker', 'symbol': 'cross', 'color_key': 'entry_color'},
             ],
             'if_not_confirmed': [
                 {'action': 'fire_webhook', 'label': 'Fire webhook: exit_{{direction}}_market (bail)', 'event': 'exit_{{direction}}_market'},
                 {'action': 'bail', 'label': 'Exit at market (bail)'},
                 {'action': 'plot_marker', 'label': 'Plot bail marker', 'symbol': 'xcross', 'color_key': 'exit_stop_color'},
             ]},
        ],
    }

    # CC has fixed behavior — no configurable parameters
    parameters_schema = {}

    def check_entry_signal(self, trigger_id, exec_type, c_triggers, l_type_fills,
                           current_values, strip_exec_suffix):
        # CC: bar-close entry only
        base_trigger = strip_exec_suffix(trigger_id)
        if c_triggers.get(trigger_id, False) or c_triggers.get(base_trigger, False):
            return EntryResult(fired=True, fill_price=current_values.get('close', 0), is_ltype=False)
        return EntryResult(fired=False)

    def get_confirmation_config(self, trigger_id, exec_type, bar_count, get_variant_param):
        bail = get_variant_param(trigger_id, 'CC', 'bail_action', 'exit_market')
        return ConfirmationConfig(needs_confirm=True, confirm_bar_offset=1, bail_action=bail)

    def should_check_confirmation(self, exec_type, entry_bar_count, current_bar_count, pending_confirm_bar):
        return pending_confirm_bar >= 0 and current_bar_count == pending_confirm_bar

    def on_confirmation_failed(self, exec_type, state):
        pass  # pending_confirm_bar stays set, check_exit handles bail

    def on_confirmation_passed(self, exec_type, state):
        state.pending_confirm_bar = -1

    def check_bail_at_market(self, exec_type, state, bar_count, bar_open):
        if (state.pending_confirm_bar >= 0 and bar_count > state.pending_confirm_bar
                and state.bail_action == 'exit_market'):
            return BailResult(should_bail=True, reason='unconfirmed_cc', fill_price=bar_open)
        return None

    def check_bail_exit(self, exec_type, state, bar_count, direction, high, low):
        # CC with limit bail (future feature — currently CC always bails at market)
        return None

    def skip_stop_target_on_entry_bar(self, exec_type):
        return False  # CC is bar-close entry, no skip


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
register(LevelCloseExecution())
register(CloseCloseExecution())
