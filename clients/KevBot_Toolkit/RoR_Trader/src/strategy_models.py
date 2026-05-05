"""Strategy model definitions (M8.7).

Each strategy declares two model properties:
  - backtest_model: data source for `recompute_and_persist_stored_trades`
                    (the algo-history view). Safe to change without
                    affecting live execution; affects analytics only.
  - live_model:     how the live engine sources 1Min bars and handles
                    rebroadcasts.  Changes to live_model affect alert
                    firing and should be made with care.

Modular framing: strategies authored under model X stay valid even
when defaults shift — every strategy carries its declared models in
config, no behavior changes retroactively.

Wire-up state (2026-05-05):
  - Schema: live + recorded on every strategy via config JSONB
  - API: GET /api/strategies/models exposes the registry
  - Frontend ModelsCard: read-only selectors with disabled "(coming
    soon)" entries for available=False models
  - Engine + backtest dispatch: NOT yet branching on these fields
    (Phases C/D/E in docs/plans/synchronous-tickling-yeti.md).
    Strategies marked with available=False models don't break — they
    just continue running on current default behavior until the
    relevant phase ships and the model is flipped to available=True.

History note (2026-05-05): three M3 placeholder backtest IDs
(`rest_with_cache_overlay`, `cache_only`, `cache_first`) were removed
because they were never engine-implemented and were superseded by
`cache_locked` / `cache_corrected`.  No strategy ever selected them
in production.  `ws_first_lock` retained but permanently
`available=False` — superseded by `ws_agg_locked` which provides the
same first-write semantics on a more reliable data source.
"""

# Allowed values for `backtest_model` field on strategy.config
BACKTEST_MODELS = {
    'rest_only': {
        'label': 'REST default',
        'available': True,
        'default': True,
        'description': (
            'Polygon REST canonical 1Min bars. Current default. '
            'Includes all FINRA late-print corrections and '
            'end-of-day reconciliation. Recommended baseline for '
            'historical backtests of any duration.'
        ),
    },
    'rest_hifi': {
        'label': 'REST + Hi-Fi Pass 2',
        'available': True,
        'default': False,
        'description': (
            'REST bars with Hi-Fi 1-second refinement (Pass 2). '
            'Refines entry/exit timestamps using per-second data '
            'where available. Best for sub-minute strategies on '
            'REST data.'
        ),
    },
    'cache_locked': {
        'label': 'Cache (locked at close)',
        'available': False,  # flipped True when Phase E ships
        'default': False,
        'description': (
            'Read 1Min bars from live_bars cache (source IN '
            '("ws", "ws_agg")). Decision-time view of what the live '
            'engine actually saw — no late-print corrections rolled '
            'in. Best parity with what live alerts fired on. '
            'Limited to dates after the cache started recording '
            '(2026-04-30 onwards) and to symbols with active live '
            'tracking. Coming soon — needs Phase E (backtest '
            'dispatch wiring).'
        ),
    },
    'cache_corrected': {
        'label': 'Cache + REST backfill',
        'available': False,  # flipped True when Phase D + E ship
        'default': False,
        'description': (
            'Read 1Min bars from live_bars cache after the REST '
            'backfill job has filled in any gaps with '
            'source="rest_backfill" rows. Best of both worlds — '
            'engine-time alignment for bars that landed live, plus '
            'canonical REST values for any minute the WS path '
            'missed. Coming soon — needs Phase D (REST backfill) + '
            'Phase E (backtest dispatch).'
        ),
    },
}

# Allowed values for `live_model` field on strategy.config
LIVE_MODELS = {
    'ws_with_corrections': {
        'label': 'AM with corrections',
        'available': True,
        'default': True,
        'description': (
            'Engine consumes Polygon AM.<symbol> per-minute events. '
            'Polygon rebroadcast corrections within the 15-min FINRA '
            'window are applied via recompute-from-history (Option '
            'B, 2026-05-02 fix). Default. Indicator state updates '
            'silently as corrections arrive; original alerts remain '
            'fact. Note: AM coverage on high-volume symbols can be '
            'unreliable — see `ws_agg_locked` for an alternative.'
        ),
    },
    'ws_first_lock': {
        'label': 'WS first-write (legacy)',
        'available': False,  # permanent — superseded by ws_agg_locked
        'default': False,
        'description': (
            'Legacy concept: lock the bar at first AM write; ignore '
            'rebroadcasts. Superseded by `ws_agg_locked` which '
            'provides the same first-write semantics on a more '
            'reliable data source (per-second A.* aggregation '
            'instead of pre-aggregated AM). Will not be enabled.'
        ),
    },
    'ws_agg_locked': {
        'label': 'A-aggregated (locked)',
        'available': False,  # flipped True when Phase C ships
        'default': False,
        'description': (
            'Engine consumes 1Min bars built client-side from '
            'Polygon A.<symbol> per-second events. Locks at minute '
            'close — no rebroadcast corrections. Best for symbols '
            'where AM coverage is unreliable (e.g. AAPL/AMD/SPY '
            'have ~60% AM loss while A is 97%+). Latency: ~1s after '
            'minute close. Coming soon — needs Phase C (live engine '
            'dispatch wiring).'
        ),
    },
    'ws_agg_with_rest_backfill': {
        'label': 'A-aggregated + REST backfill',
        'available': False,  # flipped True when Phase C + D ship
        'default': False,
        'description': (
            'Same engine path as ws_agg_locked (engine fires on '
            'ws_agg-completed bars only). Adds a periodic '
            'background job that pulls REST canonical values for '
            'any minute the WS path missed and writes them to '
            'live_bars with source="rest_backfill". Backfill is '
            'cosmetic for charts/backtests — does NOT trigger '
            'alerts on already-fired-or-missed minutes. Coming soon '
            '— needs Phase C + D.'
        ),
    },
}


def get_default_backtest_model() -> str:
    for k, v in BACKTEST_MODELS.items():
        if v.get('default'):
            return k
    return 'rest_only'


def get_default_live_model() -> str:
    for k, v in LIVE_MODELS.items():
        if v.get('default'):
            return k
    return 'ws_with_corrections'


def is_valid_backtest_model(value: str) -> bool:
    return value in BACKTEST_MODELS


def is_valid_live_model(value: str) -> bool:
    return value in LIVE_MODELS


def get_model_status(model_id: str, kind: str) -> str:
    """Return one of 'available', 'coming_soon', 'unknown'.

    `kind` is 'live' or 'backtest'.  Used by admin pages and (eventual)
    Phase C/E dispatch gates so unknown IDs from a deleted experimental
    model fall back to default cleanly.
    """
    registry = LIVE_MODELS if kind == 'live' else BACKTEST_MODELS
    entry = registry.get(model_id)
    if entry is None:
        return 'unknown'
    return 'available' if entry.get('available') else 'coming_soon'
