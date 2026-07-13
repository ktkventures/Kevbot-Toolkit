"""
FastAPI application for RoR Trader.

Phase 38: Wraps existing Python computation modules as REST endpoints.
The existing engines (unified_engine, ralph_engine, worker, db) are imported
and called directly — never modified.

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from src/ directory (same as Streamlit)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(
        title="RoR Trader API",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        redirect_slashes=False,
    )

    # CORS — allow frontend origins (including Railway dev URLs)
    import os as _os
    _extra_origins = [o.strip() for o in _os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",        # Local dev
            "https://rortrader.com",        # Production
            "https://www.rortrader.com",
        ] + _extra_origins,  # Railway dev URLs via CORS_ORIGINS env var
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Register routers ---
    from api.routers.auth import router as auth_router
    from api.routers.settings import router as settings_router
    from api.routers.packs import router as packs_router
    from api.routers.data import router as data_router
    from api.routers.backtest import router as backtest_router
    from api.routers.strategies import router as strategies_router
    from api.routers.dashboard import router as dashboard_router
    from api.routers.portfolios import router as portfolios_router
    from api.routers.requirements import router as requirements_router
    from api.routers.alerts import router as alerts_router
    from api.routers.monitor import router as monitor_router
    from api.routers.webhooks import router as webhooks_router
    from api.routers.webhook_groups import router as webhook_groups_router
    from api.routers.mass_builder import router as mass_builder_router
    from api.routers.ai_builder import router as ai_builder_router
    from api.routers.execution_types import router as execution_types_router
    from api.routers.data_health import router as data_health_router
    from api.routers.strategy_models_admin import router as strategy_models_admin_router
    from api.routers.recompute_jobs import router as recompute_jobs_router
    from api.routers.admin_parity import router as admin_parity_router
    from api.routers.strategy_health import router as strategy_health_router
    from api.routers.update_jobs import router as update_jobs_router
    from api.routers.system_settings import router as system_settings_router
    from api.routers.dev_tasks import router as dev_tasks_router
    from api.routers.bar_cache_admin import router as bar_cache_admin_router
    from api.routers.resampled_store_admin import router as resampled_store_admin_router
    from api.routers.trade_snapshots import router as trade_snapshots_router
    from api.routers.model_parity import router as model_parity_router

    app.include_router(auth_router)
    app.include_router(settings_router)
    app.include_router(packs_router)
    app.include_router(data_router)
    app.include_router(backtest_router)
    app.include_router(strategies_router)
    app.include_router(dashboard_router)
    app.include_router(portfolios_router)
    app.include_router(requirements_router)
    app.include_router(alerts_router)
    app.include_router(monitor_router)
    app.include_router(webhooks_router)
    app.include_router(webhook_groups_router)
    app.include_router(mass_builder_router)
    app.include_router(ai_builder_router)
    app.include_router(execution_types_router)
    app.include_router(data_health_router)
    app.include_router(strategy_models_admin_router)
    app.include_router(recompute_jobs_router)
    app.include_router(admin_parity_router)
    app.include_router(strategy_health_router)
    app.include_router(update_jobs_router)
    app.include_router(system_settings_router)
    app.include_router(dev_tasks_router)
    app.include_router(bar_cache_admin_router)
    app.include_router(resampled_store_admin_router)
    app.include_router(trade_snapshots_router)
    app.include_router(model_parity_router)

    # Load user packs at startup — registers indicators, interpreters,
    # triggers, and intra-bar level maps. DB group creation is skipped
    # (no user context at startup) and handled by the install endpoint.
    import pack_registry
    pack_registry.scan_and_load_all()

    # Mass Builder orphan cleanup — runs in a background thread so a slow
    # Supabase response at boot can't block uvicorn from binding (the
    # original sync call was hanging the container startup with no output
    # when Supabase was under load, producing persistent 502s even after
    # the deploy reported SUCCESS). The cleanup still runs on boot but
    # uvicorn starts serving requests immediately.
    import threading
    def _startup_cleanup():
        try:
            from mass_builder import cleanup_orphaned_mass_searches
            cleanup_orphaned_mass_searches()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Mass search startup cleanup failed (non-fatal): %s", e)
        # Same pattern for update_jobs — orphan any 'running' UAD jobs
        # from a previous worker process so the UI surfaces them and the
        # user can retry.
        try:
            from update_jobs import cleanup_orphaned_update_jobs
            cleanup_orphaned_update_jobs()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Update jobs orphan cleanup failed (non-fatal): %s", e)
    threading.Thread(target=_startup_cleanup, daemon=True, name="startup_orphan_cleanup").start()

    # Stale parity_status cleanup — bg parity threads are daemon=True and
    # die when the worker process exits. Any strategy left at PENDING from
    # a pre-deploy queue is silently orphaned; the badge keeps saying
    # "Computing" forever. Mark anything PENDING older than 10 minutes as
    # ERROR with a clear hint so the user can re-trigger via Run Parity.
    # Runs once at boot, in a background thread to avoid blocking uvicorn.
    def _stale_parity_cleanup():
        try:
            from datetime import datetime, timezone, timedelta
            import json as _json
            from db import get_admin_client, USE_DB
            if not USE_DB:
                return
            client = get_admin_client()
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
            r = (client.table('strategies')
                 .select('id,parity_status')
                 .not_.is_('parity_status', 'null')
                 .execute())
            rows = r.data or []
            stale = []
            for row in rows:
                ps = row.get('parity_status') or {}
                if isinstance(ps, str):
                    try:
                        ps = _json.loads(ps)
                    except Exception:
                        continue
                if ps.get('verdict') != 'PENDING':
                    continue
                computed_at = ps.get('computed_at') or ''
                if computed_at and computed_at < cutoff:
                    stale.append(row['id'])
            for sid in stale:
                client.table('strategies').update({
                    'parity_status': {
                        'verdict': 'ERROR',
                        'score': None,
                        'error': 'Background parity thread was killed (likely '
                                 'an API deploy). Click Run Parity again to '
                                 'retry.',
                        'computed_at': datetime.now(timezone.utc).isoformat(),
                        'params': {'last_n': 200, 'forward_test_only': False},
                    },
                }).eq('id', sid).execute()
            if stale:
                import logging
                logging.getLogger(__name__).warning(
                    "[STARTUP] cleared %d stale PENDING parity_status rows: %s",
                    len(stale), stale)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Stale parity cleanup failed (non-fatal): %s", e)
    threading.Thread(target=_stale_parity_cleanup, daemon=True,
                     name="startup_stale_parity_cleanup").start()

    # Health check
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
