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
# EXEC TYPE MANIFEST (scaffolding for Trade_Timestamps_Spec_2026-04-17)
# ═══════════════════════════════════════════════════════════════════════════
#
# Declarative metadata for each execution type. Currently reference-only —
# NOT consumed by runtime logic. Phase 1 of the Timestamps Spec will refactor
# the runtime (`check_entry_signal`, `get_confirmation_config`,
# `PositionStateMachine.check_entry`) to read these manifests as the single
# source of truth, and compute `fill_ts` via `compute_fill_ts(trigger_ts,
# bar_duration, hold_duration_s, behavior)` per the spec.
#
# The four manifests below describe the CURRENT behavior of C/L/LC/CC
# verbatim. When Phase 1 lands, no behavior changes — only the plumbing that
# propagates the manifest into the state machine.
#
# See: docs/Trade_Timestamps_Spec_2026-04-17.md — Part 5 "User-facing
# semantics & pack definitions" for field definitions and the AI Wizard UI
# that creates user-defined manifests.

@dataclass
class ExecTypeManifest:
    """Declarative description of an execution type's lifecycle.

    Every exec type — system-shipped (C/L/LC/CC) and user-created /
    AI-generated — is described by one of these. The runtime engine reads
    the manifest to determine fill timing, confirmation behavior, and which
    events to emit.

    Scaffolding note (2026-04-17): this dataclass is defined but not yet
    consumed by runtime code. Wiring happens in Phase 1 of the Timestamps
    Spec. Safe to use for documentation / tests / UI generation now.
    """
    code: str
    display_name: str
    description: str
    trigger_kind: str              # 'bar_close' | 'level_cross' | 'custom'
    behavior: str                  # 'A' (wait-then-fill) | 'B' (fill-then-validate)
    hold_duration_bars: int = 0
    hold_duration_seconds: int = 0
    fill_offset: str = 'immediate'     # 'immediate' | 'next_bar_open' | 'after_hold'
    events_emitted: List[str] = field(default_factory=lambda: ['fill_event'])
    early_exit_on_invalidation: bool = False
    source: str = 'system'             # 'system' | 'user' | 'ai_generated'
    created_by: Optional[str] = None
    created_at: Optional[str] = None


# Manifests describing CURRENT behavior of each built-in exec type.
# Populated as reference data; Phase 1 wires these into the runtime.

MANIFEST_C = ExecTypeManifest(
    code='C',
    display_name='Bar Close',
    description=(
        'Entry confirmed when the bar closes with the trigger condition '
        'true. Fills at the next bar\'s open (same instant as the firing '
        'bar\'s close for display purposes).'
    ),
    trigger_kind='bar_close',
    behavior='B',
    hold_duration_bars=0,
    hold_duration_seconds=0,
    fill_offset='next_bar_open',
    events_emitted=['fill_event'],
    early_exit_on_invalidation=False,
    source='system',
)

MANIFEST_L = ExecTypeManifest(
    code='L',
    display_name='Level Cross (Immediate)',
    description=(
        'Entry triggers intra-bar the moment price crosses an indicator '
        'level. Fills immediately at the cross price. No hold period, no '
        'confirmation required.'
    ),
    trigger_kind='level_cross',
    behavior='B',
    hold_duration_bars=0,
    hold_duration_seconds=0,
    fill_offset='immediate',
    events_emitted=['fill_event'],
    early_exit_on_invalidation=False,
    source='system',
)

MANIFEST_LC = ExecTypeManifest(
    code='LC',
    display_name='Level Cross (Confirmed)',
    description=(
        'Enters immediately on level cross (like L), but requires the '
        'price to still be on the entry side of the line at the next '
        'bar\'s close. If invalidated during the hold bar, exits with '
        'reason=validation_failed.'
    ),
    trigger_kind='level_cross',
    behavior='B',
    hold_duration_bars=1,
    hold_duration_seconds=0,
    fill_offset='immediate',
    events_emitted=['fill_event'],
    early_exit_on_invalidation=True,
    source='system',
)

MANIFEST_CC = ExecTypeManifest(
    code='CC',
    display_name='Close-Close Confirmed',
    description=(
        'Enters on the bar close that first produces the trigger, fills '
        'at the next bar\'s open (like C), but requires the following '
        'bar\'s close to also produce the trigger. If the follow-up bar '
        'invalidates, exits with reason=validation_failed.'
    ),
    trigger_kind='bar_close',
    behavior='B',
    hold_duration_bars=1,
    hold_duration_seconds=0,
    fill_offset='next_bar_open',
    events_emitted=['fill_event'],
    early_exit_on_invalidation=True,
    source='system',
)


EXEC_TYPE_MANIFESTS: Dict[str, ExecTypeManifest] = {
    'C': MANIFEST_C,
    'L': MANIFEST_L,
    'LC': MANIFEST_LC,
    'CC': MANIFEST_CC,
}


def get_manifest(exec_type: str) -> Optional[ExecTypeManifest]:
    """Return the manifest for a given exec_type code, or None if unknown.

    Scaffolding — not yet consumed by runtime. See module docstring.
    """
    return EXEC_TYPE_MANIFESTS.get(exec_type)


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

    # Workflow steps for UI visualization (dict keyed by context)
    steps: dict = {}

    # Parameters schema for UI editing (empty = no configurable params)
    parameters_schema: dict = {}

    # Detailed description for the Description tab
    detailed_description: dict = {}

    # Placeholder definitions — how each webhook placeholder is determined (for AI context)
    placeholder_definitions: dict = {}

    # Technical specs — fixed values/decisions visible to the user as a reference
    technical_specs: list = []

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
            'steps': self.steps,
            'parameters_schema': self.parameters_schema,
            'detailed_description': self.detailed_description,
            'placeholder_definitions': self.placeholder_definitions,
            'technical_specs': self.technical_specs,
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

    detailed_description = {
        'overview': 'The simplest execution type. When a confluence pack trigger fires at bar close, the trade is entered (or exited) at the bar close price. This is the most common execution type for strategies that don\'t need intra-bar precision.',
        'how_it_works': [
            'The confluence pack evaluates its trigger condition at the moment the bar closes.',
            'If the trigger fires, the execution type fills at the close price of that bar.',
            'A webhook event is sent immediately with the fill details.',
            'The position is then managed by stop/target/exit triggers until exit.',
        ],
        'fill_price': 'Bar close price — determined at the moment the bar closes. This is the last traded price of the bar.',
        'pros': ['Simple and predictable', 'No slippage within the bar', 'Works with any indicator'],
        'cons': ['May miss better intra-bar fill prices', 'Bar close can differ from the trigger level for crossover indicators'],
        'webhook_context': {
            'order_price': 'Bar close price at the moment the trigger fired',
            'quantity': 'Calculated from portfolio risk_per_trade ÷ |entry_price - stop_price|',
            'direction': 'Determined by the confluence pack trigger (LONG or SHORT)',
            'stop_price': 'Calculated by the stop loss pack using ATR, swing, or other method',
        },
        'determined_by_pack': ['When the trigger fires (timing)', 'Direction (LONG/SHORT)', 'Which indicator conditions must be met'],
        'determined_by_exec_type': ['Fill price (always bar close)', 'Webhook event type (entry_X_market)', 'No confirmation needed'],
    }

    placeholder_definitions = {
        'execution': {
            'label': 'Execution Type',
            'placeholders': {
                'exec_type': {'value': 'C', 'description': 'Always "C" for Bar Close execution'},
                'order_action': {'value': 'buy (LONG) / sell (SHORT) / close (exit)', 'description': 'Derived from direction and signal type. Entry LONG = buy, Entry SHORT = sell, Exit = close.'},
                'order_price': {'value': 'Bar close price', 'description': 'The last traded price of the bar when the trigger fired. This is the fill price for both entry and exit.'},
                'market_position': {'value': 'long / short / flat', 'description': 'After entry: "long" or "short". After exit: "flat".'},
                'event_type': {'value': 'entry_signal / exit_signal', 'description': 'Determined by whether this is an entry or exit event.'},
            },
        },
        'signal': {
            'label': 'Signal Context',
            'placeholders': {
                'symbol': {'value': 'From strategy', 'description': 'The ticker symbol being traded (e.g., NVDA, SPY). Set by the strategy.'},
                'direction': {'value': 'LONG or SHORT', 'description': 'Set by the confluence pack trigger. Determines order_action.'},
                'timeframe': {'value': 'From strategy', 'description': 'The bar timeframe (e.g., 1Min, 5Min). Set by the strategy.'},
                'trigger_name': {'value': 'Trigger ID', 'description': 'The specific trigger that fired (e.g., ema_pp_v2_cross_short_up). From the confluence pack.'},
                'timestamp': {'value': 'Bar close time', 'description': 'ISO timestamp of the bar close when the trigger fired.'},
            },
        },
        'order': {
            'label': 'Order Details',
            'placeholders': {
                'quantity': {'value': 'risk_per_trade ÷ |entry - stop|', 'description': 'Number of shares. Calculated from portfolio risk_per_trade divided by the distance between entry price and stop price.'},
                'stop_price': {'value': 'From stop loss pack', 'description': 'Calculated by the stop loss pack (ATR, swing, fixed dollar, or percentage method). Set at entry time.'},
                'atr': {'value': 'Current ATR value', 'description': 'Average True Range at the signal bar. Used for stop/target calculations.'},
                'risk_per_trade': {'value': 'From portfolio', 'description': 'Dollar amount risked per trade. Set at the portfolio level.'},
            },
        },
        'strategy': {
            'label': 'Strategy Context',
            'placeholders': {
                'strategy_name': {'value': 'Strategy display name', 'description': 'The name of the strategy that generated this signal.'},
                'strategy_id': {'value': 'Strategy numeric ID', 'description': 'Database ID of the strategy.'},
                'confluence_met': {'value': 'Comma-separated conditions', 'description': 'List of confluence conditions that were met at entry time (e.g., "5M-EMA_STACK-SML, 1H-MACD_LINE-M>S+").'},
            },
        },
        'portfolio': {
            'label': 'Portfolio Context',
            'placeholders': {
                'portfolio_name': {'value': 'Portfolio display name', 'description': 'The portfolio containing this strategy.'},
                'portfolio_id': {'value': 'Portfolio numeric ID', 'description': 'Database ID of the portfolio.'},
                'position_risk': {'value': 'Position risk amount', 'description': 'Dollar risk for this position within the portfolio.'},
            },
        },
    }

    technical_specs = [
        {'key': 'Order Type', 'value': 'Market', 'note': 'Fills at market price (bar close). No limit order placement.'},
        {'key': 'Fill Price', 'value': 'Bar close price', 'note': 'The last traded price when the bar closes.'},
        {'key': 'Reference Bar', 'value': 'Current bar', 'note': 'Evaluates the bar that just closed. Indicators use current bar values.'},
        {'key': 'Stop Calculation', 'value': 'Current bar indicators', 'note': 'ATR and other indicators from the bar that triggered entry.'},
        {'key': 'Entry Bar Protection', 'value': 'None', 'note': 'Stop/target are checked immediately — no skip on entry bar.'},
        {'key': 'Confirmation', 'value': 'None', 'note': 'No confirmation step. Entry is final at bar close.'},
        {'key': 'Webhook Event', 'value': 'entry_long_market / entry_short_market', 'note': 'Fires immediately on fill.'},
    ]

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

    parameters_schema = {}

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

    detailed_description = {
        'overview': 'Enters at the exact indicator level when price crosses it within the bar. More realistic fill prices for crossover-based strategies — instead of waiting for bar close, the trade fills at the moment the cross happens.',
        'how_it_works': [
            'The engine checks if the bar\'s high/low range crossed the indicator level.',
            'If price crossed the level, the trade fills at the indicator level price (not bar close).',
            'The confluence pack\'s indicator determines which level is used (current bar vs previous bar).',
            'Stop and target are NOT checked on the entry bar (the bar\'s OHLC includes pre-entry price action).',
        ],
        'fill_price': 'Indicator level price — the exact value of the indicator line at the moment of the cross. This is typically more favorable than bar close for crossover strategies.',
        'pros': ['More realistic fill prices', 'Better represents actual crossover entry timing', 'Standard for stops and targets (level cross detection)'],
        'cons': ['Requires the indicator to have a defined level (not all do)', 'Stop/target skip on entry bar means first-bar protection is delayed'],
        'webhook_context': {
            'order_price': 'Indicator level price at the cross moment',
            'quantity': 'Calculated from portfolio risk_per_trade ÷ |entry_price - stop_price|',
            'direction': 'Determined by the confluence pack trigger',
            'stop_price': 'Calculated using previous bar\'s indicators (not current bar, since entry happens mid-bar)',
        },
        'determined_by_pack': ['Which indicator level to cross', 'Whether to use current or previous bar level', 'Cross direction (above/below)'],
        'determined_by_exec_type': ['Fill at level price (not close)', 'Skip stop/target on entry bar', 'Webhook: entry_X_market'],
    }

    placeholder_definitions = {
        'execution': {
            'label': 'Execution Type',
            'placeholders': {
                'exec_type': {'value': 'L', 'description': 'Always "L" (L0 or L1 internally, based on indicator)'},
                'order_action': {'value': 'buy / sell / close', 'description': 'Same as C-type. Derived from direction and signal type.'},
                'order_price': {'value': 'Indicator level price', 'description': 'The indicator level at the moment of the cross. More favorable than bar close for crossover strategies.'},
                'market_position': {'value': 'long / short / flat', 'description': 'Same as C-type.'},
            },
        },
        'signal': {
            'label': 'Signal Context',
            'placeholders': {
                'symbol': {'value': 'From strategy', 'description': 'Ticker symbol.'},
                'direction': {'value': 'LONG or SHORT', 'description': 'From confluence pack trigger.'},
                'timeframe': {'value': 'From strategy', 'description': 'Bar timeframe.'},
                'trigger_name': {'value': 'Trigger ID', 'description': 'The trigger with _ib suffix (e.g., ema_pp_v2_cross_short_up_ib).'},
                'timestamp': {'value': 'Bar timestamp', 'description': 'Timestamp of the bar where the level cross was detected.'},
            },
        },
        'order': {
            'label': 'Order Details',
            'placeholders': {
                'quantity': {'value': 'risk_per_trade ÷ |entry - stop|', 'description': 'Shares. Stop is calculated from PREVIOUS bar indicators (since entry happens mid-bar).'},
                'stop_price': {'value': 'From stop loss pack (prev bar)', 'description': 'Stop uses previous bar indicators — current bar not yet closed at entry time.'},
                'atr': {'value': 'Previous bar ATR', 'description': 'ATR from previous bar (current bar incomplete at entry).'},
            },
        },
    }

    technical_specs = [
        {'key': 'Order Type', 'value': 'Market', 'note': 'Fills at market price (indicator level). No limit order.'},
        {'key': 'Fill Price', 'value': 'Indicator level', 'note': 'The exact indicator value at the cross. More favorable than bar close.'},
        {'key': 'Reference Bar', 'value': 'From confluence pack', 'note': 'The indicator determines current vs previous bar level. Not set by the execution type.'},
        {'key': 'Stop Calculation', 'value': 'Previous bar indicators', 'note': 'Uses previous bar ATR/indicators since entry happens mid-bar (current bar not closed).'},
        {'key': 'Entry Bar Protection', 'value': 'Yes — stop/target skipped', 'note': 'The entry bar\'s OHLC includes pre-entry price action, so stop/target are not checked until the next bar.'},
        {'key': 'Confirmation', 'value': 'None', 'note': 'No confirmation. Entry is final at level cross.'},
        {'key': 'Webhook Event', 'value': 'entry_long_market / entry_short_market', 'note': 'Fires immediately on fill at level.'},
        {'key': 'Also Used For', 'value': 'Stops and Targets', 'note': 'Level [L] is the default execution type for stop loss and take profit (level cross detection).'},
    ]

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

    parameters_schema = {}

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

    detailed_description = {
        'overview': 'Two-stage entry: first fills at the indicator level when price crosses (like L-type), then waits for bar close to confirm the direction. If the bar closes on the wrong side of the level, the position is exited (bail). This reduces false entries from temporary intra-bar spikes.',
        'how_it_works': [
            'Price crosses the indicator level within the bar → fill at the level price.',
            'Wait for the bar to close (or next bar, depending on confirm_bar_offset).',
            'If bar close is on the correct side of the level → CONFIRMED. Position continues.',
            'If bar close is on the wrong side → NOT CONFIRMED. Bail at market (or limit at entry price).',
        ],
        'fill_price': 'Indicator level price (same as L-type). The confirmation check happens after the fill.',
        'pros': ['Filters out false intra-bar crosses', 'Better entry quality than pure L-type', 'Configurable confirmation timing and bail behavior'],
        'cons': ['Bailed trades incur a small loss (entry → bail spread)', 'More complex to understand and verify'],
        'webhook_context': {
            'order_price': 'Indicator level price at the cross moment',
            'quantity': 'Calculated from portfolio risk_per_trade ÷ |entry_price - stop_price|',
            'on_confirm': 'Position continues — no additional webhook',
            'on_bail': 'exit_X_market webhook fires with bail reason',
        },
        'determined_by_pack': ['Which indicator level to cross', 'Cross direction'],
        'determined_by_exec_type': ['Fill at level price', 'Confirmation check at bar close', 'Bail behavior (market or limit at entry price)'],
    }

    placeholder_definitions = {
        'execution': {
            'label': 'Execution Type',
            'placeholders': {
                'exec_type': {'value': 'LC', 'description': 'Level-Close execution type'},
                'order_action': {'value': 'buy / sell / close', 'description': 'Entry: buy/sell. Bail: close. Confirmed exit: close.'},
                'order_price': {'value': 'Indicator level (entry) / market (bail)', 'description': 'Entry fills at the indicator level. If confirmation fails, bail fills at bar open (market) or entry price (limit).'},
            },
        },
        'confirmation': {
            'label': 'Confirmation',
            'placeholders': {
                'confirm_bar_offset': {'value': '0 (same bar) or 1 (next bar)', 'description': 'How many bars to wait before checking confirmation. 0 = check at same bar close. 1 = check at next bar close.'},
                'bail_action': {'value': 'exit_market or exit_limit', 'description': 'What to do if confirmation fails. Market = exit at next bar open. Limit = place limit order at entry price (breakeven exit).'},
            },
        },
        'order': {
            'label': 'Order Details',
            'placeholders': {
                'quantity': {'value': 'risk_per_trade ÷ |entry - stop|', 'description': 'Same as L-type. Stop uses previous bar indicators.'},
                'stop_price': {'value': 'From stop loss pack (prev bar)', 'description': 'Calculated from previous bar (entry happens mid-bar).'},
            },
        },
    }

    technical_specs = [
        {'key': 'Order Type', 'value': 'Market', 'note': 'Initial fill is at market (level price). Bail is at market or limit.'},
        {'key': 'Fill Price', 'value': 'Indicator level', 'note': 'Same as L-type for the initial entry fill.'},
        {'key': 'Confirmation', 'value': 'Same-bar close (default)', 'note': 'After filling at the level, waits for the bar to close. Checks if close is on the correct side of the indicator level.'},
        {'key': 'Confirmation Offset', 'value': '0 bars (same bar)', 'note': 'Can be 0 (same bar close) or 1 (next bar close). Fixed per execution type definition.'},
        {'key': 'Bail Action', 'value': 'Exit at market', 'note': 'If confirmation fails, exit at market (next bar open). Alternative: exit at limit (entry price for breakeven).'},
        {'key': 'Stop Calculation', 'value': 'Previous bar indicators', 'note': 'Same as L-type — entry happens mid-bar.'},
        {'key': 'Entry Bar Protection', 'value': 'Yes — stop/target skipped', 'note': 'Same as L-type.'},
        {'key': 'Webhook Events', 'value': 'entry → bail/exit', 'note': 'Entry webhook fires on fill. If bail: exit webhook fires. If confirmed: normal exit webhook on later exit.'},
    ]

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

    parameters_schema = {}

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

    detailed_description = {
        'overview': 'Two-stage entry: first fills at bar close (like C-type), then waits for the NEXT bar to close confirming the direction. If the next bar closes against the entry direction, bail at market. This adds an extra layer of confirmation to bar-close entries.',
        'how_it_works': [
            'The confluence pack trigger fires at bar close → fill at close price.',
            'Wait for the next bar to close.',
            'If next bar closes in the entry direction → CONFIRMED. Position continues.',
            'If next bar closes against the entry direction → NOT CONFIRMED. Bail at market (next bar open).',
        ],
        'fill_price': 'Bar close price (same as C-type). The confirmation uses the next bar\'s close.',
        'pros': ['Filters out single-bar reversals', 'Confirms momentum continues into next bar', 'Simple to understand'],
        'cons': ['Always delays confirmation by one full bar', 'Bailed trades lose the entry-to-bail spread', 'Cannot adjust confirmation offset (always next bar)'],
        'webhook_context': {
            'order_price': 'Bar close price when trigger fired',
            'on_confirm': 'Position continues — no additional webhook',
            'on_bail': 'exit_X_market webhook fires at next bar open',
        },
        'determined_by_pack': ['When the trigger fires', 'Direction'],
        'determined_by_exec_type': ['Fill at bar close', 'Next-bar confirmation required', 'Bail at market if unconfirmed'],
    }

    parameters_schema = {}

    placeholder_definitions = {
        'execution': {
            'label': 'Execution Type',
            'placeholders': {
                'exec_type': {'value': 'CC', 'description': 'Close-Close execution type'},
                'order_action': {'value': 'buy / sell / close', 'description': 'Entry: buy/sell at bar close. Bail: close at next bar open.'},
                'order_price': {'value': 'Bar close (entry) / bar open (bail)', 'description': 'Entry fills at bar close. If next bar doesn\'t confirm, bail at the following bar\'s open price.'},
            },
        },
        'confirmation': {
            'label': 'Confirmation',
            'placeholders': {
                'confirm_bar_offset': {'value': '1 (always next bar)', 'description': 'CC always waits for the next bar to close before confirming. Cannot be adjusted.'},
                'bail_action': {'value': 'exit_market (always)', 'description': 'CC always bails at market (next bar open) if unconfirmed. Cannot be adjusted.'},
            },
        },
        'order': {
            'label': 'Order Details',
            'placeholders': {
                'quantity': {'value': 'risk_per_trade ÷ |entry - stop|', 'description': 'Same as C-type. Stop uses current bar indicators (entry is at bar close).'},
                'stop_price': {'value': 'From stop loss pack', 'description': 'Calculated from current bar indicators (same as C-type).'},
            },
        },
    }

    technical_specs = [
        {'key': 'Order Type', 'value': 'Market', 'note': 'Entry fills at market (bar close). Bail fills at market (next bar open).'},
        {'key': 'Fill Price', 'value': 'Bar close price', 'note': 'Same as C-type for the initial entry.'},
        {'key': 'Confirmation', 'value': 'Next bar close (always)', 'note': 'After entry, waits for the next bar to close. Checks if it confirms the entry direction.'},
        {'key': 'Confirmation Offset', 'value': '1 bar (fixed)', 'note': 'Always waits exactly one bar. Cannot be adjusted.'},
        {'key': 'Bail Action', 'value': 'Exit at market (fixed)', 'note': 'If next bar doesn\'t confirm, exit at the following bar\'s open price. Always market, no limit option.'},
        {'key': 'Stop Calculation', 'value': 'Current bar indicators', 'note': 'Same as C-type — entry is at bar close, so current bar indicators are available.'},
        {'key': 'Entry Bar Protection', 'value': 'None', 'note': 'Same as C-type — no skip. Stop/target checked from entry bar.'},
        {'key': 'Webhook Events', 'value': 'entry → bail/exit', 'note': 'Entry webhook fires on fill. If bail: exit webhook on next bar open. If confirmed: normal exit webhook later.'},
    ]

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
