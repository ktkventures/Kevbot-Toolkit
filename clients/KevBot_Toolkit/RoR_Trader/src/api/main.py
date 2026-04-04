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
    from api.routers.mass_builder import router as mass_builder_router
    from api.routers.ai_builder import router as ai_builder_router

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
    app.include_router(mass_builder_router)
    app.include_router(ai_builder_router)

    # Health check
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
