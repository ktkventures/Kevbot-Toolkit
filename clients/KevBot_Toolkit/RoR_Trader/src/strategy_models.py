"""Strategy model definitions (M8.7).

Each strategy declares THREE model properties (algo_model added 2026-05-07):
  - backtest_model: data source for `recompute_and_persist_stored_trades`
                    (the strategy's KPI baseline / "what should this
                    strategy theoretically produce on the broadest data").
                    Default: 'rest_hifi'. Safe to change without affecting
                    live execution.
  - algo_model:     data source for the cron's incremental algo-history
                    append (the strategy's "live-accountability" lane —
                    what the live engine SHOULD have produced on the
                    same data it saw). Default: 'cache_locked'. Decoupled
                    from backtest_model 2026-05-07 so the Divergence tab
                    can compare both lanes honestly.
  - live_model:     how the live engine sources 1Min bars and handles
                    rebroadcasts. Default: 'ws_agg_locked'. Changes
                    affect alert firing — make with care.

Same registry of models (BACKTEST_MODELS) drives both backtest_model
and algo_model fields — they're two consumers of the same model
catalogue.

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
        'default': False,  # demoted 2026-05-07 in favor of rest_hifi
        'description': (
            'Polygon REST canonical 1Min bars. '
            'Includes all FINRA late-print corrections and '
            'end-of-day reconciliation. Recommended baseline for '
            'historical backtests of any duration. Faster than '
            'rest_hifi but coarser timestamps on L-type events.'
        ),
    },
    'rest_hifi': {
        'label': 'REST + Hi-Fi Pass 2',
        'available': True,
        'default': True,  # promoted 2026-05-07 — broadest data + sub-second L-type timestamps
        'description': (
            'REST bars with Hi-Fi 1-second refinement (Pass 2). '
            'Refines entry/exit timestamps using per-second data '
            'where available. Default backtest_model — broadest '
            'historical coverage with sub-second L-type alignment '
            'on packs declaring trigger_levels (eppv3/eppv4/utv4 '
            'today). Confluence still bar-aligned (CB-fidelity is '
            'future work).'
        ),
    },
    'cache_locked': {
        'label': 'Cache (locked at close)',
        'available': True,  # Phase E preview shipped 2026-05-06
        'default': False,
        'description': (
            'Read primary + secondary TF bars from live_bars cache '
            '(source IN ("ws", "ws_agg")). Decision-time view of what '
            'the live engine actually saw — no late-print corrections '
            'rolled in. Best parity with what live alerts fired on, '
            'including L-type sub-second exit timestamps. Limited to '
            'dates after the cache started recording (2026-04-30 '
            'onwards) and to symbols with active live tracking. '
            'Cache volume runs ~5% under REST due to structural '
            'late-print effect; RVOL-based confluence may evaluate '
            'slightly differently than rest_only.'
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
        'default': False,  # was True; demoted 2026-05-05 in favor of ws_agg_locked
        'description': (
            'Engine consumes Polygon AM.<symbol> per-minute events. '
            'Polygon rebroadcast corrections within the 15-min FINRA '
            'window are applied via recompute-from-history (Option '
            'B, 2026-05-02 fix). NOT recommended on high-volume '
            'symbols today — Mon-1/Tue-1 validation showed AAPL/AMD/'
            'SPY/TSLA AM coverage falls to ~0% during RTH. Use '
            'ws_agg_locked instead for those symbols. Kept available '
            'for low-volume tickers (META, TSLL) where AM is healthy '
            'and saves the per-second processing cost.'
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
        'available': True,  # Phase C shipped 2026-05-05
        'default': True,    # promoted 2026-05-05 — AM proven unreliable on high-volume symbols
        'description': (
            'Engine consumes 1Min bars built client-side from '
            'Polygon A.<symbol> per-second events. Locks at minute '
            'close — no rebroadcast corrections. Default as of '
            '2026-05-05 after AM was observed delivering ~0% on '
            'high-volume RTH symbols (AAPL/AMD/SPY/TSLA). Latency: '
            '~1s after minute close. Selecting this on a strategy '
            'automatically subscribes the worker to A.<symbol> if '
            'not already. Close values are bit-identical to Polygon '
            'REST canonical 1Min on validated samples; OHL/volume '
            'diffs are sub-cent typical, structural late-print effect.'
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
    return 'rest_hifi'


def get_default_algo_model() -> str:
    """Default `algo_model` field — what the cron uses for the algo
    accountability lane. Distinct from backtest_model since 2026-05-07.

    Returns 'cache_locked' as the explicit default — closest to what
    the live engine actually sees, gives the strongest accountability
    check against live alerts. Strategies on tickers without cache
    coverage (cache_locked.available will eventually narrow per-symbol)
    fall back to backtest_model at dispatch time.
    """
    return 'cache_locked'


def get_default_live_model() -> str:
    for k, v in LIVE_MODELS.items():
        if v.get('default'):
            return k
    return 'ws_agg_locked'


def is_valid_backtest_model(value: str) -> bool:
    return value in BACKTEST_MODELS


def is_valid_algo_model(value: str) -> bool:
    """algo_model uses the same registry as backtest_model (same models,
    different consumer). No separate ALGO_MODELS dict."""
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
