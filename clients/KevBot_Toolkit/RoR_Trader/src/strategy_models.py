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
        'algo_default': True,  # default `algo_model` — decision-time mirror
                               # for the live-accountability lane (2026-05-21)
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
        'available': False,  # retired 2026-05-18 — see description
        'default': False,  # was True; demoted 2026-05-05 in favor of ws_agg_locked
        'description': (
            'Engine consumes Polygon AM.<symbol> per-minute events. '
            'Polygon rebroadcast corrections within the 15-min FINRA '
            'window are applied via recompute-from-history (Option '
            'B, 2026-05-02 fix). NOT recommended on high-volume '
            'symbols today — Mon-1/Tue-1 validation showed AAPL/AMD/'
            'SPY/TSLA AM coverage falls to ~0% during RTH. '
            'RETIRED 2026-05-18: the worker no longer subscribes to '
            'Polygon AM.* for stocks — ws_agg is the sole live_bars '
            'source. Subscribing AM created a second writer that '
            'contaminated the first_* decision-time columns. Do not '
            'select this model.'
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
        'default': False,   # demoted 2026-05-20 — ws_agg_reconciled is now default (Tier 3 §8.3 verified)
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
    'ws_agg_reconciled': {
        'label': 'A-aggregated (reconciled, TF-aware)',
        'available': True,   # Phase 4 flip 2026-05-20 — canary sid 151 verified
        'default': False,    # demoted 2026-05-28 in favor of ws_rest_spliced
                              # (M8 promotion — verifies WS bars against REST and
                              # splices REST values into indicator history when drift
                              # is caught; verify-only path is the floor of behavior)
        'description': (
            'Timeframe-aware single forward model. At 1Min+ this is '
            'identical to `ws_agg_locked` (lock at close, no '
            'corrections — cache is already 100% vs REST there per '
            'Backtest_Speed_Analysis_2026-05-19.md §3). At sub-minute, '
            'the engine fires on a decision-grade bar (per-strategy '
            'grace via `config.grace_seconds`, default 5s per the '
            '2026-05-19 RTH grace sweep) and reconciles indicator '
            'state via the O(1) `apply_last_bar_correction` path on '
            'a Polygon rebroadcast correction. Default as of '
            '2026-05-20 EOD — Tier 3 §8.3 carryover path validated '
            'on 7 real IN_POSITION strategies at restart; canaries '
            'verified firing without incident. See '
            '`docs/Spec_Live_Execution_Fidelity.md`.'
        ),
    },
    'rest_polled': {
        'label': 'REST polled (backtest-aligned) — SUPERSEDED',
        'available': False,  # superseded 2026-05-28 — see ws_rest_spliced
        'default': False,
        'description': (
            'SUPERSEDED 2026-05-28 by `ws_rest_spliced`. Pure-REST '
            'polling was registered 2026-05-27 but RTH probe data '
            'on 2026-05-28 showed REST latency too high (p90 8-23s) '
            'to drive sub-minute alerts. The chosen direction is the '
            'hybrid `ws_rest_spliced` — WS for speed, REST for '
            'indicator-state correction after grace. See '
            '`docs/breezy-dreaming-umbrella` plan and '
            '`docs/Findings_REST_vs_WS_Latency_2026-05-27.md`. '
            'Do not select.'
        ),
    },
    'ws_rest_spliced': {
        'label': 'WS-tip, REST-spliced (backtest-aligned)',
        'available': True,   # M1 registered 2026-05-28
        'default': True,     # promoted 2026-05-28 EOD (M8) — canary sid 151 (SPY
                              # 10Sec) + sid 174 (TSLA 1Min) verified end-to-end:
                              # first production CORRECTED event landed on TSLA
                              # with Δ=-$0.02, sid 151 fresh alerts verifying Δ=0.0
                              # consistently. drift_uncorrected stamps honestly
                              # surface sub-minute correction-rejection cases. See
                              # [[project_ws_rest_spliced_canary]].
        'description': (
            'WS-aggregated bars drive the live engine and alert dispatch '
            '(sub-second latency, same as `ws_agg_reconciled`). After '
            'each completed bar fires alerts, the REST verifier '
            'asynchronously fetches Polygon REST 1-sec aggregates for '
            'the same bar window. When REST data is available (typically '
            '2-10s after bar close, configurable via `grace_seconds`), '
            'the bar values in BarBuilder.history are REPLACED with the '
            'REST values, and the incremental indicator state is rewound '
            'and re-applied via the existing `apply_last_bar_correction` '
            'O(1) path. Net result: indicator state at any moment is ~99% '
            'REST-based, matching backtest by construction. Alerts that '
            'already fired on WS values are NOT retracted, but each gets '
            'stamped with `verification_status` ("verified" / "corrected" '
            '/ "rest_unavailable") and `verification_close_delta` for '
            'audit. Default grace per TF: 1Min+ 2s, 30Sec 3s, 10Sec 4s, '
            'sub-5Sec 5s; correction threshold 1 tick ($0.01); max wait '
            '60s. Per-strategy override via `config.grace_seconds`, '
            '`config.correction_threshold_dollars`, '
            '`config.rest_max_wait_seconds`. Implements what the '
            '`ws_agg_reconciled` spec described but never built — '
            'REST-side reconciliation in addition to the WS-rebroadcast '
            'side. Feature-flagged via `REST_VERIFY_ENABLED` env var '
            '(default OFF) so the new model is inert until explicitly '
            'enabled on the Railway worker. See `docs/breezy-dreaming-'
            'umbrella` plan.'
        ),
    },
}


# ─── Grace seconds config (LEF Phase 1 scaffolding, 2026-05-20) ──────────
# Per-strategy grace window for sub-minute force-close. Phase 1 ships only
# the field schema + resolver; the engine still reads ralph_engine.
# SUBMINUTE_FORCE_CLOSE_GRACE_SEC until Phase 2 wires `ws_agg_reconciled`
# to consume this. Resolution order in `resolve_grace_seconds`:
#   1. strategy.config.grace_seconds  (per-strategy override)
#   2. GRACE_SECONDS_PER_SYMBOL[symbol]  (per-ticker default)
#   3. GRACE_SECONDS_DEFAULT  (global default — 5)

GRACE_SECONDS_DEFAULT: int = 5  # validated 2026-05-19 (SPY/TSLA 10Sec RTH)

# Per-ticker overrides — empty by default; populate as shadow-sweep data
# warrants. SPY and TSLA both validated at the global default, so no
# overrides are needed today.
GRACE_SECONDS_PER_SYMBOL: dict = {}


def resolve_grace_seconds(strategy: dict) -> int:
    """Resolve the per-strategy sub-minute force-close grace (seconds).

    Phase 1 scaffolding consumed by Phase 2a (BarBuilder grace override)
    and Phase 2b (StrategyMonitor.grace_seconds for fire-at-strategy-
    grace). Returns the configured grace regardless of timeframe — the
    1Min+ "no grace" behavior is a wiring decision in the engine, not a
    property of the strategy config.

    Permissive lookup: top-level FIRST, then nested in `config`. The
    DB loader (load_strategies_monitoring_admin) flattens config JSONB
    onto the strategy dict, but other code paths may pass an unflattened
    dict — handle both. (Pre-2026-05-20-fix this only checked `config`,
    silently ignoring the canary's grace_seconds=3 and falling back to
    the global default 5.)
    """
    override = strategy.get('grace_seconds')
    if override is None:
        cfg = strategy.get('config') or {}
        override = cfg.get('grace_seconds')
    if override is not None:
        return int(override)
    symbol = strategy.get('symbol') or ''
    return GRACE_SECONDS_PER_SYMBOL.get(symbol, GRACE_SECONDS_DEFAULT)


def get_default_backtest_model() -> str:
    for k, v in BACKTEST_MODELS.items():
        if v.get('default'):
            return k
    return 'rest_hifi'


def get_default_algo_model() -> str:
    """Default `algo_model` field — what the cron uses for the algo
    accountability lane. Distinct from backtest_model since 2026-05-07.

    Resolved from the `algo_default` flag in BACKTEST_MODELS — currently
    'cache_locked', closest to what the live engine actually sees, gives
    the strongest accountability check against live alerts. Strategies on
    tickers without cache coverage (cache_locked.available will eventually
    narrow per-symbol) fall back to backtest_model at dispatch time.
    """
    for k, v in BACKTEST_MODELS.items():
        if v.get('algo_default'):
            return k
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


# ── Live-model capability: consumes WS-aggregated completed bars ──────────
# Single source of truth for the routing gates that decide whether a model's
# engine fires on WS-aggregated (client-side per-second → close) bars: the
# A.<symbol> subscription, the completed-1Min dispatch, and the >60s primary
# fan-out. These three gates were duplicated across ralph_engine and DRIFTED:
# `ws_rest_spliced` (the DEFAULT since 2026-05-28) and `ws_agg_reconciled` were
# added to the polygon-skip site but NOT to the three eligibility gates, so a
# standalone default-model 1Min+ primary got no A-sub / dispatch / fan-out and
# fell back to the flush's incomplete partial (Brandon audit P0+P1, 2026-07-15).
# The spec for both models states they "drive the live engine on ws_agg bars,
# same as ws_agg_locked" — so they belong here. Consumed via ralph_engine's
# flag-gated `_elig_ws_agg` (RORT_CANONICAL_PRIMARY_CLOSE) for a reversible roll.
WS_AGG_CONSUMER_MODELS: frozenset = frozenset({
    'ws_agg_locked',
    'ws_agg_with_rest_backfill',
    'ws_agg_reconciled',
    'ws_rest_spliced',
})


def model_consumes_ws_agg(live_model: str) -> bool:
    """True if this live model's engine fires on WS-aggregated completed bars
    (so it needs the A-sub + 1Min dispatch + >60s fan-out routing)."""
    return live_model in WS_AGG_CONSUMER_MODELS


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
