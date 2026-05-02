"""Strategy model definitions (M8.7, 2026-05-02).

Each strategy declares two model properties:
  - backtest_model: data source for `recompute_and_persist_stored_trades`
                    (the algo-history view). Safe to change without
                    affecting live execution; affects analytics only.
  - live_model:     how the live engine handles WS rebroadcasts and
                    which data source it treats as authoritative.
                    Changes to live_model affect alert firing and should
                    be made with care.

This is the placeholder phase: the fields are stored on the strategy
but `recompute_and_persist_stored_trades` and the live engine ignore
them for now (always behave per current default). Wiring comes when
the cache read path (M8.7d) ships.

Modular framing: strategies authored under model X stay valid even
when defaults shift — every strategy carries its declared models in
config, no behavior changes retroactively.
"""

# Allowed values for `backtest_model` field on strategy.config
BACKTEST_MODELS = {
    'rest_only': {
        'label': 'REST only',
        'available': True,
        'default': True,
        'description': (
            'Polygon REST settled bars only. Current default behavior. '
            'Backtest reads REST aggregates which include all FINRA '
            'late-print corrections and end-of-day reconciliation.'
        ),
    },
    'rest_hifi': {
        'label': 'REST + Hi-Fi Pass 2',
        'available': True,
        'default': False,
        'description': (
            'REST bars with Hi-Fi 1-second refinement (Pass 2). Refines '
            'entry/exit timestamps using per-second data where available. '
            'Best for sub-minute strategies on REST data.'
        ),
    },
    'rest_with_cache_overlay': {
        'label': 'REST + WS cache overlay',
        'available': False,
        'default': False,
        'description': (
            'REST for periods before the live_bars cache existed; '
            'WS cache for periods after. Closest representation of '
            'what the live engine actually saw. Coming soon — needs '
            'M8.7d (cache read path in backtest).'
        ),
    },
    'cache_only': {
        'label': 'WS cache only',
        'available': False,
        'default': False,
        'description': (
            'Only periods since the cache started recording (2026-04-30 '
            'onwards). Useful for forward-only analysis where you only '
            'care about live-equivalent data. Coming soon.'
        ),
    },
    'cache_first': {
        'label': 'WS cache (decision-time)',
        'available': False,
        'default': False,
        'description': (
            'cache.first_close — the bar values at the moment the live '
            'engine first saw them, before any Polygon rebroadcast '
            'corrections. Most rigorous "what did live see" view. '
            'Coming soon.'
        ),
    },
}

# Allowed values for `live_model` field on strategy.config
LIVE_MODELS = {
    'ws_with_corrections': {
        'label': 'WS + Polygon corrections',
        'available': True,
        'default': True,
        'description': (
            'Apply Polygon WS rebroadcast corrections within the 15-min '
            'FINRA window via recompute-from-history (Option B). Default '
            'after the 2026-05-02 fix. Indicator state updates silently '
            'as corrections arrive; original alerts remain fact.'
        ),
    },
    'ws_first_lock': {
        'label': 'WS first-write (locked)',
        'available': False,
        'default': False,
        'description': (
            'Lock the bar at first WS write; ignore subsequent Polygon '
            'rebroadcasts. More predictable but loses corrections. '
            'Coming soon — awaiting Monday TV stability test outcome to '
            'decide whether this is the better default.'
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
