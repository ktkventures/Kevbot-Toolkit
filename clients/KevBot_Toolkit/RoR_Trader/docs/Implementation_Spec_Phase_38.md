# Phase 38: Frontend Migration — Implementation Spec

**Status:** Approved — Building on `dev` branch
**Date:** 2026-03-25
**Backup branch:** `dev-backup-pre-38-impl`
**Plan file:** `/home/kevin/.claude/plans/cheerful-cooking-creek.md`

---

## 1. Overview

Migrate the RoR Trader frontend from Streamlit to Next.js + FastAPI. The Python computation layer (engines, indicators, portfolio math, alert system) stays untouched — we build a REST API that wraps existing functions and wire the already-built V5 React pages to real data.

**Build order:** 38-0 → 38-1A-1 → 38-1A-2 → 38-1A-3 → 38-1A-4 → 38-1A-5 → 38-1B-1 → 38-1B-2 → 38-1B-3 → 38-1B-4 → 38-1B-5 → 38-1C-1 → 38-1C-2

**Tech stack:**
- Backend: FastAPI + uvicorn (wraps existing Python modules)
- Frontend: Next.js 14 + React 18 + TypeScript + Tailwind
- State: TanStack Query v5 (server) + Zustand v4 (client UI) + React Context (auth)
- Charts: lightweight-charts v4 (candlestick/trading) + recharts v2 (equity/analytics)
- Forms: React Hook Form v7 + Zod v3
- Auth: @supabase/supabase-js v2 (frontend) + python-jose (backend JWT validation)

---

## 2. Safety Guardrails

### Files That Must NOT Be Modified

| File | Lines | Why |
|------|-------|-----|
| `ralph_engine.py` | ~2,900 | Production alert engine, extensively QA'd |
| `unified_engine.py` | ~2,400 | Backtest/live engine, 27 parity tests |
| `worker.py` | ~600 | Railway worker, manages per-user engines |
| `db.py` | ~1,000 | Supabase DAL with RLS, thread-local auth |
| `indicators.py` | ~400 | Indicator calculations |
| `interpreters.py` | ~600 | State classification |
| `triggers.py` | ~500 | Trade generation state machine |
| `data_loader.py` | ~500 | Market data abstraction (Polygon/Alpaca/Mock) |
| `portfolios.py` | ~1,200 | Portfolio computation engine |
| `alerts.py` | ~800 | Alert CRUD + webhook delivery |
| `confluence_groups.py` | ~400 | Pack management + TEMPLATES |
| `general_packs.py` | ~300 | General pack management |
| `risk_management_packs.py` | ~300 | RM pack management |
| `analytics.py` | ~375 | Rolling metrics, Markov, market regimes |

**Rule:** FastAPI `import`s and `call`s these modules. It never modifies them.

### What Gets Modified

| File | Change |
|------|--------|
| `app.py` | Extract ~8 functions to `services.py`, replace with thin wrappers |
| `frontend/` | Wire V5 pages, add providers/hooks/charts |
| `requirements.txt` | Add FastAPI, uvicorn, python-jose, pydantic |

### What Gets Created

| Path | Purpose |
|------|---------|
| `src/api/` | FastAPI application (routers, schemas, deps) |
| `src/services.py` | Extracted business logic (no Streamlit deps) |
| `Dockerfile.api` | FastAPI container definition |
| `frontend/src/lib/` | API client, schemas, Supabase config |
| `frontend/src/hooks/` | React Query + mutation hooks |
| `frontend/src/providers/` | Auth, Query, Store providers |
| `frontend/src/charts/` | Chart components |
| `frontend/src/app/login/` | Login page |

---

## 3. Phase 38-0: Infrastructure Scaffold

### 3.1 What to Build

| Step | Function/File | Purpose |
|------|--------------|---------|
| 0A | `git branch dev-backup-pre-38-impl` | Safety backup |
| 0B | `src/services.py` | Extract core functions from app.py |
| 0C | `src/api/__init__.py`, `src/api/main.py` | FastAPI app factory, CORS, health |
| 0D | `src/api/deps.py` | JWT auth dependency, JWKS cache |
| 0E | `src/api/routers/auth.py` | Login, signup, refresh, /me |
| 0F | `src/api/schemas/auth.py` | Auth request/response models |
| 0G | `Dockerfile.api`, `requirements.txt` | Container + deps |
| 0H | `frontend/src/lib/api/client.ts` | Fetch wrapper with JWT interceptor |
| 0I | `frontend/src/lib/supabase.ts` | Supabase client singleton |
| 0J | `frontend/src/providers/AuthProvider.tsx` | Auth context (login, logout, session) |
| 0K | `frontend/src/providers/QueryProvider.tsx` | TanStack Query config |
| 0L | `frontend/src/app/layout.tsx` (modify) | Wrap with providers |
| 0M | `frontend/src/app/login/page.tsx` | Login page |
| 0N | `src/api/routers/settings.py` | GET/PUT /api/settings (proves pattern) |
| 0O | `src/api/schemas/settings.py` | Settings schema |

### 3.2 Function Extraction — `src/services.py`

Extract these from `app.py` into `src/services.py` (no Streamlit imports):

```python
# src/services.py — Core business logic, callable from both Streamlit and FastAPI

import pandas as pd
from datetime import datetime, timedelta, timezone

from data_loader import load_market_data, get_data_source, is_crypto
from indicators import run_all_indicators, run_indicators_for_group
from interpreters import INTERPRETERS, run_all_interpreters, detect_all_triggers
import general_packs as gp_module
from confluence_groups import load_confluence_groups, get_enabled_groups


def prepare_data_with_indicators(
    symbol: str, days: int = 30, seed: int = 42,
    start_date=None, end_date=None,
    timeframe: str = "1Min", data_feed: str = "sip",
    session: str = "RTH", secondary_tfs: tuple = ()
) -> pd.DataFrame:
    """Load market data and run full indicator/interpreter/trigger pipeline.
    Extracted from app.py:634 — identical logic, no @st.cache_data.
    """
    # ... (copy function body from app.py:660-729)


def get_secondary_tf_map(df: pd.DataFrame) -> dict:
    """Extract secondary TF map from column names containing '__'.
    Extracted from app.py:732.
    """
    # ... (copy function body from app.py:738-745)


def unified_trades(
    df: pd.DataFrame, strategy: dict,
    include_open_position: bool = True,
    last_bar_partial: bool = False,
    bar_cache=None, cache_metadata=None
) -> pd.DataFrame:
    """Trade generation via unified engine with MTF support.
    Extracted from app.py:748 (_unified_trades).
    """
    # ... (copy function body from app.py:771-797+)


def trades_df_from_stored(stored_trades: list) -> pd.DataFrame:
    """Reconstruct trades DataFrame from stored minimal records.
    Extracted from app.py:1475.
    """
    # ... (copy function body from app.py:1477-1482)


def get_strategy_trades(
    strat: dict, data_feed: str = "sip"
) -> pd.DataFrame:
    """Get trades for any modern strategy. Extracted from app.py:892.
    Note: data_feed is now an explicit param (no session state).
    """
    # ... (copy function body from app.py:897-926)
    # Replace _get_data_feed() with the data_feed parameter


def prepare_forward_test_data(
    strat: dict, data_feed: str = "sip",
    data_days_override: int = None
):
    """Load continuous data and split trades at forward test boundary.
    Extracted from app.py:825.
    """
    # ... (copy function body from app.py:837-889)
    # Replace _get_data_feed() with the data_feed parameter


def calculate_kpis(
    trades_df: pd.DataFrame,
    starting_balance: float = 10000,
    risk_per_trade: float = 100,
    total_trading_days: int = None
) -> dict:
    """Calculate primary strategy KPIs. Extracted from app.py:1730."""
    # ... (copy function body from app.py:1739-1805)


def calculate_secondary_kpis(trades_df: pd.DataFrame, kpis: dict) -> dict:
    """Calculate extended KPIs (Sharpe, Sortino, etc.). Extracted from app.py:1808."""
    # ... (copy function body)


def count_trading_days(df: pd.DataFrame) -> int:
    """Count unique trading days. Extracted from app.py:1723."""
    if len(df) == 0 or not hasattr(df.index, 'normalize'):
        return 1
    return max(df.index.normalize().nunique(), 1)
```

**After extraction, update `app.py`** to use thin wrappers:

```python
import services as _svc

@st.cache_data(ttl=3600, hash_funcs={tuple: hash})
def prepare_data_with_indicators(symbol, days=30, seed=42, ...):
    return _svc.prepare_data_with_indicators(symbol, days, seed, ...)

def _unified_trades(df, strategy, ...):
    return _svc.unified_trades(df, strategy, ...)

def calculate_kpis(trades_df, ...):
    return _svc.calculate_kpis(trades_df, ...)

# etc.
```

### 3.3 FastAPI App Factory — `src/api/main.py`

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    app = FastAPI(title="RoR Trader API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "https://rortrader.com",
            "https://www.rortrader.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from api.routers import auth, settings, packs, strategies, backtest
    from api.routers import data, portfolios, requirements, alerts
    from api.routers import monitor, webhooks, mass_builder
    app.include_router(auth.router)
    app.include_router(settings.router)
    # ... all routers

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app

app = create_app()
```

### 3.4 JWT Auth Dependency — `src/api/deps.py`

```python
import os
import time
import httpx
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, Header
from db import set_current_user

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# Simple: verify JWT with the project's JWT secret (symmetric HS256)
# Supabase projects use HS256 by default with the JWT secret from dashboard

def get_current_user(authorization: str = Header(...)):
    """Extract and validate Supabase JWT. Sets thread-local user context."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization[7:]
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No subject in token")

    # Set thread-local context for db.py RLS
    set_current_user(user_id, token)

    return {"id": user_id, "email": payload.get("email", "")}
```

**Critical:** All router functions are `def` (sync), not `async def`. FastAPI runs sync functions in a threadpool, giving each request its own thread — matching db.py's `threading.local()` pattern.

### 3.5 Auth Router — `src/api/routers/auth.py`

```python
from fastapi import APIRouter, HTTPException
from api.schemas.auth import LoginRequest, LoginResponse, RefreshRequest
import os

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Authenticate via Supabase Auth."""
    from supabase import create_client
    client = create_client(
        os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    try:
        result = client.auth.sign_in_with_password({
            "email": req.email, "password": req.password
        })
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {
        "access_token": result.session.access_token,
        "refresh_token": result.session.refresh_token,
        "expires_at": result.session.expires_at,
        "user": {"id": result.user.id, "email": result.user.email},
    }

@router.post("/refresh", response_model=LoginResponse)
def refresh(req: RefreshRequest):
    """Refresh an expired JWT."""
    from supabase import create_client
    client = create_client(
        os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    try:
        result = client.auth.refresh_session(req.refresh_token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {
        "access_token": result.session.access_token,
        "refresh_token": result.session.refresh_token,
        "expires_at": result.session.expires_at,
        "user": {"id": result.user.id, "email": result.user.email},
    }

@router.get("/me")
def me(user=Depends(get_current_user)):
    return user
```

### 3.6 Settings Router — `src/api/routers/settings.py`

```python
from fastapi import APIRouter, Depends
from api.deps import get_current_user

router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("/")
def get_settings(user=Depends(get_current_user)):
    from db import USE_DB, load_settings_db
    if USE_DB:
        return load_settings_db()
    # Fallback: load from JSON file
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'settings.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@router.put("/")
def update_settings(settings: dict, user=Depends(get_current_user)):
    from db import USE_DB, save_settings_db
    if USE_DB:
        save_settings_db(settings)
        return {"status": "saved"}
    import json, os
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'settings.json')
    with open(path, 'w') as f:
        json.dump(settings, f, indent=2)
    return {"status": "saved"}
```

### 3.7 Frontend Infrastructure

**`frontend/src/lib/api/client.ts`:**
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem('supabase_access_token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (res.status === 401) {
    // Attempt token refresh, then retry once
    const refreshed = await refreshToken();
    if (refreshed) {
      headers['Authorization'] = `Bearer ${localStorage.getItem('supabase_access_token')}`;
      const retry = await fetch(`${API_URL}${path}`, { ...options, headers });
      if (!retry.ok) throw new ApiError(retry.status, await retry.text());
      return retry.json();
    }
    // Refresh failed — redirect to login
    window.location.href = '/login';
    throw new ApiError(401, 'Session expired');
  }
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.json();
}
```

**`frontend/src/providers/AuthProvider.tsx`:**
```typescript
'use client';
import { createContext, useContext, useEffect, useState } from 'react';
import { createClient, Session, User } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType>(/* ... */);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // 1. Check localStorage for existing session on mount
  // 2. Listen to onAuthStateChange for token refresh
  // 3. Store access_token in localStorage for apiFetch
  // 4. Redirect to /login if no session
}
```

**`frontend/src/providers/QueryProvider.tsx`:**
```typescript
'use client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

export function QueryProvider({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5 * 60 * 1000,    // 5 min
        gcTime: 30 * 60 * 1000,       // 30 min
        retry: 1,
        refetchOnWindowFocus: false,
      },
    },
  }));
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

### 3.8 Dockerfile.api

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
ENV USE_DB=true
ENV PYTHONPATH=/app/src
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
```

### 3.9 Self-Test Checkpoints

- [ ] **0-1:** `src/services.py` exists with all 8 extracted functions
- [ ] **0-2:** `app.py` imports from `services.py` — Streamlit app still runs (`./run.sh`)
- [ ] **0-3:** `uvicorn api.main:app` starts without errors on port 8000
- [ ] **0-4:** `curl http://localhost:8000/health` returns `{"status":"ok"}`
- [ ] **0-5:** `curl -X POST http://localhost:8000/api/auth/login` with valid credentials returns JWT
- [ ] **0-6:** `curl -H "Authorization: Bearer <jwt>" http://localhost:8000/api/auth/me` returns user
- [ ] **0-7:** `curl -H "Authorization: Bearer <jwt>" http://localhost:8000/api/settings` returns settings
- [ ] **0-8:** Frontend `npm run dev` starts on port 3000
- [ ] **0-9:** Frontend `/login` page renders, can enter credentials
- [ ] **0-10:** After login, redirect to `/dashboard`, JWT stored in localStorage
- [ ] **0-11:** AuthProvider shows user email in console/UI

---

## 4. Phase 38-1A-1: Packs Backend + Frontend

### 4.1 What to Build — Backend

| Function | File | Purpose |
|----------|------|---------|
| `GET /api/packs/confluence-groups` | `routers/packs.py` | Load user's confluence groups |
| `PUT /api/packs/confluence-groups` | `routers/packs.py` | Save full group list |
| `GET /api/packs/confluence-groups/templates` | `routers/packs.py` | TEMPLATES registry |
| `GET /api/packs/confluence-groups/triggers/{direction}` | `routers/packs.py` | Entry/exit triggers |
| `GET /api/packs/general` | `routers/packs.py` | Load general packs |
| `PUT /api/packs/general` | `routers/packs.py` | Save general packs |
| `GET /api/packs/general/templates` | `routers/packs.py` | GP templates |
| `GET /api/packs/risk-management` | `routers/packs.py` | Load RM packs (stop + TP) |
| `PUT /api/packs/risk-management` | `routers/packs.py` | Save RM packs |
| `GET /api/packs/risk-management/templates` | `routers/packs.py` | RM templates |

**Pattern:** Confluence groups, general packs, and RM packs are stored as whole-list-per-user. Individual pack CRUD happens client-side; the API saves the full updated list. This matches the existing `save_confluence_groups_db()` pattern exactly.

**Templates response:** Serialize the Python `TEMPLATES` dict including: `id`, `name`, `category`, `indicator_key`, `interpreter_key`, `parameters` (with schema), `outputs`, `triggers`, `plot_settings_schema`. These are static — can be cached aggressively.

### 4.2 What to Build — Frontend

| File | Purpose |
|------|---------|
| `hooks/queries/usePacks.ts` | `useConfluenceGroups()`, `useGeneralPacks()`, `useRiskManagementPacks()`, `useTemplates()` |
| `hooks/mutations/usePackMutations.ts` | `useSaveConfluenceGroups()`, `useSaveGeneralPacks()`, `useSaveRiskManagementPacks()` |
| Wire: `confluence-packs/tf-confluence/page.tsx` | Replace mock data in V5 with query hooks |
| Wire: `confluence-packs/general/page.tsx` | Same pattern |
| Wire: `confluence-packs/stop-loss/page.tsx` | Same pattern |
| Wire: `confluence-packs/take-profit/page.tsx` | Same pattern |

**Page wiring pattern (all 4 pages):**
1. `page.tsx` renders V5 component directly (bypass VersionedPage)
2. V5 component calls `useConfluenceGroups()` / `useGeneralPacks()` / etc.
3. Replace hardcoded `TEMPLATE_DEFS` with `useTemplates()` query
4. CRUD: modify local state → call save mutation with full list → invalidate query
5. Loading: skeleton cards. Error: error banner above list.

### 4.3 Self-Test Checkpoints

- [ ] **1A1-1:** `GET /api/packs/confluence-groups` returns user's groups (or empty list for new user)
- [ ] **1A1-2:** `GET /api/packs/confluence-groups/templates` returns all 8 templates with schemas
- [ ] **1A1-3:** `PUT /api/packs/confluence-groups` with modified list persists changes
- [ ] **1A1-4:** Same pattern works for general packs and RM packs
- [ ] **1A1-5:** TF Confluence page loads real groups from API
- [ ] **1A1-6:** Create a new variation → save → reload → variation persists
- [ ] **1A1-7:** Delete a variation → save → reload → variation gone
- [ ] **1A1-8:** General Packs, Stop Loss, Take Profit pages all render with real data
- [ ] **1A1-9:** Template detail tabs show correct params, triggers, outputs from API
- [ ] **1A1-10:** Templates that should match Python TEMPLATES dict are identical

---

## 5. Phase 38-1A-2: Data + Backtest Backend

### 5.1 What to Build

| Function | File | Purpose |
|----------|------|---------|
| `GET /api/data/bars/{symbol}` | `routers/data.py` | Load OHLCV bars |
| `GET /api/data/source` | `routers/data.py` | Current data provider name |
| `POST /api/backtest/run` | `routers/backtest.py` | Full backtest pipeline |
| `BacktestRequest` | `schemas/backtest.py` | Request model |
| `BacktestResponse` | `schemas/backtest.py` | Response model |
| `TradeRecord` | `schemas/common.py` | Single trade record |
| `KPISet` | `schemas/common.py` | Primary + secondary KPIs |

### 5.2 Backtest Request Schema

```python
class BacktestRequest(BaseModel):
    symbol: str                        # "SPY", "BTC/USD"
    timeframe: str = "1Min"
    direction: Literal["LONG", "SHORT"]
    days: int = 90
    lookback_mode: str = "Days"        # "Days", "Date Range"
    lookback_start_date: str | None = None
    lookback_end_date: str | None = None
    session: str = "RTH"
    data_feed: str = "sip"

    # Triggers (confluence IDs)
    entry_trigger_confluence_id: str
    exit_trigger_confluence_ids: list[str] = []

    # Confluence conditions
    confluence: list[str] = []

    # Risk management (pack-based or inline)
    stop_loss_pack_id: str | None = None
    take_profit_pack_id: str | None = None
    stop_config: dict | None = None
    target_config: dict | None = None

    # Advanced
    bar_count_exit: int | None = None
    secondary_tfs: list[str] = []
```

### 5.3 Backtest Response Schema

```python
class TradeRecord(BaseModel):
    entry_time: str              # ISO 8601
    exit_time: str | None
    direction: str
    entry_price: float
    exit_price: float | None
    stop_price: float | None
    target_price: float | None
    r_multiple: float
    win: bool
    exit_reason: str             # "signal", "stop", "target", "bar_count", "open"
    exec_type: str               # "C", "L0", "L1", "HM", "HL"
    bars_held: int | None = None
    entry_trigger: str | None = None

class KPISet(BaseModel):
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_r: float
    total_r: float
    daily_r: float
    r_squared: float
    max_r_drawdown: float
    final_balance: float
    total_pnl: float

class EquityCurvePoint(BaseModel):
    trade_number: int
    timestamp: str
    cumulative_r: float

class BacktestResponse(BaseModel):
    trades: list[TradeRecord]
    kpis: KPISet
    secondary_kpis: dict          # Extended KPIs (Sharpe, Sortino, etc.)
    equity_curve: list[EquityCurvePoint]
    total_bars: int
    total_trading_days: int
    data_source: str
    chart_data: list[dict] | None = None  # OHLCV + selected indicators
```

### 5.4 Backtest Service — `src/api/services/backtest_service.py`

```python
import services as svc
from unified_engine import run_unified_backtest

def run_backtest(req: BacktestRequest, user) -> BacktestResponse:
    """Orchestrate full backtest pipeline."""
    # 1. Determine dates
    start_date, end_date = None, None
    if req.lookback_mode == "Date Range":
        start_date = datetime.fromisoformat(req.lookback_start_date)
        end_date = datetime.fromisoformat(req.lookback_end_date)

    # 2. Build strategy dict (matches saved strategy format)
    strategy = {
        'symbol': req.symbol,
        'direction': req.direction,
        'timeframe': req.timeframe,
        'entry_trigger_confluence_id': req.entry_trigger_confluence_id,
        'exit_trigger_confluence_ids': req.exit_trigger_confluence_ids,
        'confluence': req.confluence,
        'trading_session': req.session,
        'bar_count_exit': req.bar_count_exit,
        # ... stop/target config from packs or inline
    }

    # 3. Resolve stop/target from packs if needed
    if req.stop_loss_pack_id:
        # Load RM pack, extract stop config
        pass
    if req.take_profit_pack_id:
        # Load RM pack, extract target config
        pass

    # 4. Load + enrich data
    sec_tfs = tuple(sorted(req.secondary_tfs))
    df = svc.prepare_data_with_indicators(
        req.symbol, days=req.days, start_date=start_date,
        end_date=end_date, timeframe=req.timeframe,
        data_feed=req.data_feed, session=req.session,
        secondary_tfs=sec_tfs)

    if len(df) == 0:
        return BacktestResponse(trades=[], kpis=empty_kpis(), ...)

    # 5. Run unified engine
    trades_df = svc.unified_trades(df, strategy)

    # 6. Calculate KPIs
    trading_days = svc.count_trading_days(df)
    kpis = svc.calculate_kpis(trades_df, total_trading_days=trading_days)
    secondary = svc.calculate_secondary_kpis(trades_df, kpis)

    # 7. Build equity curve
    equity_curve = build_equity_curve(trades_df)

    # 8. Serialize trades
    trades = serialize_trades(trades_df)

    # 9. Optional: chart data (OHLCV + indicators for frontend chart)
    chart_data = serialize_chart_data(df) if req.include_chart_data else None

    return BacktestResponse(
        trades=trades, kpis=kpis, secondary_kpis=secondary,
        equity_curve=equity_curve, total_bars=len(df),
        total_trading_days=trading_days,
        data_source=get_data_source(), chart_data=chart_data)
```

### 5.5 DataFrame Serialization

```python
def serialize_trades(trades_df: pd.DataFrame) -> list[dict]:
    """Convert trades DataFrame to JSON-serializable list."""
    if len(trades_df) == 0:
        return []
    records = trades_df.copy()
    # Convert timestamps to ISO strings
    for col in ['entry_time', 'exit_time']:
        if col in records.columns:
            records[col] = records[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None)
    return records.to_dict(orient='records')

def serialize_chart_data(df: pd.DataFrame, indicators: list[str] = None) -> list[dict]:
    """Serialize OHLCV + selected indicator columns to JSON.
    Only sends columns the frontend needs for chart rendering.
    """
    cols = ['open', 'high', 'low', 'close', 'volume']
    if indicators:
        cols += [c for c in indicators if c in df.columns]
    subset = df[cols].copy()
    subset.index = subset.index.map(lambda x: x.isoformat())
    subset = subset.reset_index().rename(columns={'index': 'timestamp'})
    return subset.to_dict(orient='records')
```

### 5.6 Self-Test Checkpoints

- [ ] **1A2-1:** `GET /api/data/bars/SPY?timeframe=1Min&days=5` returns OHLCV JSON array
- [ ] **1A2-2:** `GET /api/data/source` returns "Polygon" or "Mock Data"
- [ ] **1A2-3:** `POST /api/backtest/run` with valid SPY config returns trades + KPIs
- [ ] **1A2-4:** KPIs from API match KPIs from Streamlit for identical inputs (±0.01 tolerance)
- [ ] **1A2-5:** Equity curve has correct cumulative R values
- [ ] **1A2-6:** Trade records have all required fields (exec_type, exit_reason, r_multiple)
- [ ] **1A2-7:** Empty result (no trades) returns gracefully with zero KPIs
- [ ] **1A2-8:** Backtest with secondary TFs returns cross-TF confluence data

---

## 6. Phase 38-1A-3: Strategy Builder Frontend

### 6.1 What to Build

| File | Purpose |
|------|---------|
| `charts/TradingChart.tsx` | Candlestick chart with indicator overlays + trade markers |
| `charts/ConfluenceHeatmap.tsx` | Stacked histogram heatmap pane |
| `charts/EquityCurve.tsx` | 3-segment equity curve (recharts) |
| `hooks/queries/useBacktest.ts` | `useRunBacktest()` mutation |
| Wire: `strategy-builder/page.tsx` | Full builder wiring |

### 6.2 TradingChart Component Design

```typescript
// charts/TradingChart.tsx
interface TradingChartProps {
  ohlcv: CandleData[];              // {time, open, high, low, close, volume}
  overlays?: OverlaySeries[];        // EMA, VWAP lines
  markers?: TradeMarker[];           // Entry/exit markers with colors
  confluenceData?: HeatmapData[];    // For heatmap pane
  height?: number;
  theme?: string;                    // Maps to CSS theme variables
}

// Uses lightweight-charts v4 imperative API via useRef + useEffect
// Multi-pane: price chart (main) + confluence heatmap (below)
// Markers: arrowUp (entry), arrowDown (exit), colored by exec_type
// Overlays: addLineSeries for each indicator
```

### 6.3 EquityCurve Component Design

```typescript
// charts/EquityCurve.tsx
interface EquityCurveProps {
  data: EquityCurvePoint[];
  segments?: {
    backtest?: { end: number };       // Trade index where BT ends
    forward?: { end: number };        // Trade index where FWD ends
  };
  showHWM?: boolean;
  showEdgeCheck?: boolean;
  showConfidenceBands?: boolean;
  height?: number;
}

// Uses recharts LineChart
// 3 segments: blue (BT), orange (FWD), green (Live/Alert)
// Gradient fill on BT and FWD segments
// Zero line, HWM line, edge check (21-MA + Bollinger)
```

### 6.4 Builder Wiring Flow

1. User configures: symbol, timeframe, direction, session
2. Analysis tabs fetch pack data via `useConfluenceGroups()`, `useGeneralPacks()`, etc.
3. User selects entry trigger, exit trigger(s), confluence conditions, stop/target packs
4. Click "Backtest" → `useRunBacktest.mutate(config)` → shows loading spinner
5. On success: render results panel — `EquityCurve`, `TradingChart` with markers, KPI cards, trade table
6. Click "Save" → `useCreateStrategy.mutate(strategyWithKPIs)` → redirect to My Strategies

### 6.5 Self-Test Checkpoints

- [ ] **1A3-1:** Strategy Builder page loads with pack selectors populated from API
- [ ] **1A3-2:** Entry/Exit analysis tabs show real triggers from confluence groups
- [ ] **1A3-3:** Backtest button sends correct request, shows loading state
- [ ] **1A3-4:** Results: equity curve renders with correct shape
- [ ] **1A3-5:** Results: candlestick chart shows OHLCV with trade markers
- [ ] **1A3-6:** Results: KPI cards show matching values
- [ ] **1A3-7:** Results: trade history table shows all fields
- [ ] **1A3-8:** Save button creates strategy in DB, appears in My Strategies
- [ ] **1A3-9:** Confluence heatmap renders below price chart (if conditions selected)
- [ ] **1A3-10:** Charts respect active theme (background, text colors, accent)

---

## 7. Phase 38-1A-4: Strategies Backend + Frontend

### 7.1 What to Build — Backend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/strategies` | GET | `load_strategies_db()` |
| `/api/strategies` | POST | `save_strategy_db()` |
| `/api/strategies/{id}` | GET | `get_strategy_by_id_db()` |
| `/api/strategies/{id}` | PUT | `update_strategy_db()` |
| `/api/strategies/{id}` | DELETE | `delete_strategy_db()` |
| `/api/strategies/{id}/duplicate` | POST | Load + save with new name |
| `/api/strategies/bulk-delete` | POST | Loop `delete_strategy_db()` |
| `/api/strategies/{id}/trades` | GET | `services.get_strategy_trades()` |
| `/api/strategies/{id}/trades/cached` | GET | Use `stored_trades` fast path |
| `/api/strategies/{id}/forward-test` | GET | `services.prepare_forward_test_data()` |
| `/api/strategies/{id}/chart-data` | GET | OHLCV + indicators for chart |
| `/api/strategies/{id}/analytics` | GET | Rolling metrics, Markov, streaks |

### 7.2 What to Build — Frontend

| File | Purpose |
|------|---------|
| `hooks/queries/useStrategies.ts` | `useStrategies()`, `useStrategy(id)`, `useStrategyTrades(id)` |
| `hooks/mutations/useStrategyMutations.ts` | Create, update, delete, duplicate, bulk-delete |
| `charts/SparkLine.tsx` | Tiny inline equity for strategy cards |
| `charts/DistributionChart.tsx` | R-distribution histogram (recharts) |
| Wire: `strategies/page.tsx` | My Strategies list |
| Wire: `strategies/[id]/page.tsx` | Strategy Detail (6 tabs) |

### 7.3 My Strategies Wiring

Replace `mockStrategies` with `useStrategies()`. Key mappings:
- Mock `winRate` → `strategy.kpis.win_rate`
- Mock `equityCurve` → `strategy.equity_curve_data` (stored on save)
- Mock `tags` → `strategy.tags` (array field in config JSONB)
- KPI mode dropdown: locally-computed view logic (no API call)
- Equity sparklines: `SparkLine.tsx` using `equity_curve_data`
- Bulk actions: `useDeleteStrategy.mutate()` for each selected

### 7.4 Strategy Detail Wiring — 6 Tabs

| Tab | Data Source | Charts |
|-----|-----------|--------|
| Equity & KPIs | `/strategies/{id}` + `/strategies/{id}/trades` | EquityCurve (3-segment), DistributionChart |
| Chart & Trades | `/strategies/{id}/chart-data` | TradingChart (full), ConfluenceHeatmap |
| Confluence Analysis | `/strategies/{id}/chart-data` (per-group) | TradingChart (per-indicator) |
| Configuration | `/strategies/{id}` | None (readonly display) |
| Alerts | `/alerts/strategy/{id}` | None (table) |
| Alert Analysis | `/strategies/{id}/forward-test` + `/alerts/strategy/{id}` | EquityCurve overlay |

Each tab uses React Query's `enabled` option to lazy-load: `enabled: activeTab === 'chart'`.

### 7.5 Self-Test Checkpoints

- [ ] **1A4-1:** My Strategies loads real strategies from API
- [ ] **1A4-2:** Strategy cards show correct KPIs, tags, equity sparklines
- [ ] **1A4-3:** KPI mode dropdown changes displayed metrics without API call
- [ ] **1A4-4:** Filter/sort by ticker, direction, tag works client-side
- [ ] **1A4-5:** Click card → navigates to Strategy Detail
- [ ] **1A4-6:** Strategy Detail: Equity & KPIs tab shows 3-segment curve + R-distribution
- [ ] **1A4-7:** Strategy Detail: Chart & Trades tab shows candlestick with trade markers
- [ ] **1A4-8:** Strategy Detail: all 6 tabs render without errors
- [ ] **1A4-9:** Edit/Delete/Duplicate actions work, list updates after mutation
- [ ] **1A4-10:** Bulk delete removes selected strategies
- [ ] **1A4-11:** Bulk "Create Portfolio" navigates to portfolio builder with selected strategies

---

## 8. Phase 38-1A-5: Dashboard + Settings Frontend

### 8.1 What to Build — Backend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/dashboard/summary` | GET | Aggregates strategies + portfolios |

Dashboard summary computes from existing data:
- Count of strategies, portfolios, active positions
- Combined equity curve (if portfolios exist)
- Performance health per strategy (from `classify_strategy_health()`)
- Recent alerts

### 8.2 Frontend Wiring

**Dashboard V6:**
- `useDashboardSummary()` query with `refetchInterval: 30000`
- Equity curve: `EquityCurve.tsx` (portfolio combined or strategy aggregate)
- Daily P&L: recharts `BarChart`
- P&L Calendar: custom component (grid of day cells, colored by P&L)
- Active Positions: table with `refetchInterval: 5000`
- Performance Health: SD deviation bars (green/orange/red)

**Settings:**
- Themes: already works, save preference via `useSaveSettings`
- Display: wire 5 tabs to Zustand store + API persistence
- Connections: `useMonitorStatus()` query with `refetchInterval: 5000`
- Account: `useAuth()` for user info

### 8.3 Self-Test Checkpoints

- [ ] **1A5-1:** Dashboard loads with real data (or empty state for new users)
- [ ] **1A5-2:** Equity curve renders from portfolio/strategy data
- [ ] **1A5-3:** Active positions update every 5 seconds
- [ ] **1A5-4:** Performance health shows SD deviation indicators
- [ ] **1A5-5:** Settings: theme change persists across reload
- [ ] **1A5-6:** Settings: display preferences save to API and load on next visit
- [ ] **1A5-7:** Settings: Connections shows engine status (running/stopped/error)
- [ ] **1A5-8:** All 9 themes render correctly with charts (no broken colors)

**Phase 1A Milestone — Complete Core Trading Loop**

---

## 9. Phase 38-1B: Portfolios & Alerts

### 9.1 Phase 38-1B-1: Portfolio Backend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/portfolios` | GET | `load_portfolios_db()` |
| `/api/portfolios` | POST | `save_portfolio_db()` |
| `/api/portfolios/{id}` | GET | `get_portfolio_by_id_db()` |
| `/api/portfolios/{id}` | PUT | `update_portfolio_db()` |
| `/api/portfolios/{id}` | DELETE | `delete_portfolio_db()` |
| `/api/portfolios/{id}/duplicate` | POST | Load + save with new name |
| `/api/portfolios/{id}/compute` | POST | Compute analytics (selective) |
| `/api/portfolios/{id}/trades` | GET | `get_portfolio_trades()` |
| `/api/portfolios/{id}/requirements/check` | POST | `evaluate_requirement_set()` |
| `/api/portfolios/{id}/health` | POST | `classify_strategy_health()` |
| `/api/portfolios/{id}/monte-carlo` | POST | `run_monte_carlo()` |
| `/api/portfolios/{id}/account` | GET | `get_account()` |
| `/api/portfolios/{id}/account/deposit` | POST | `add_ledger_entry()` |
| `/api/portfolios/{id}/account/ledger/{eid}` | DELETE | `remove_ledger_entry()` |
| `/api/portfolios/{id}/anomalies` | GET | `detect_portfolio_anomalies()` |

**Compute endpoint:** Accepts `include` array to specify which computations:
`["kpis", "equity_curve", "drawdown", "correlation", "daily_pnl", "strategy_health"]`

### 9.2 Phase 38-1B-2: Portfolio Frontend (3 pages)

**My Portfolios:** Same card pattern as My Strategies — equity sparklines, KPI modes, filters.
**Portfolio New/Edit:** React Hook Form with Zod. Strategy selector, requirement set picker, allocation config.
**Portfolio Detail:** 6 tabs — Live Dashboard, Performance, Strategies, Prop Firm Check, Account, Webhooks.

### 9.3 Phase 38-1B-3: Requirements Backend + Frontend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/requirements` | GET | `load_requirements_db()` |
| `/api/requirements` | POST | `save_requirement_set_db()` |
| `/api/requirements/{id}` | PUT | `update_requirement_set_db()` |
| `/api/requirements/{id}` | DELETE | `delete_requirement_set_db()` |

Frontend: CRUD page with pre-seeded prop firm rules (TTP, FTMO, custom).

### 9.4 Phase 38-1B-4: Alerts & Monitor Backend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/alerts` | GET | `load_alerts_db()` |
| `/api/alerts/strategy/{sid}` | GET | `get_alerts_for_strategy_db()` |
| `/api/alerts/{id}/acknowledge` | PUT | `update_alert_db()` |
| `/api/alerts/clear` | POST | `clear_alerts_db()` |
| `/api/alerts/config` | GET | `load_alert_config_db()` |
| `/api/alerts/config` | PUT | `save_alert_config_db()` |
| `/api/monitor/status` | GET | `load_monitor_status_db()` |
| `/api/monitor/start` | POST | `set_desired_state_db('running')` |
| `/api/monitor/stop` | POST | `set_desired_state_db('stopped')` |
| `/api/monitor/engine-state` | GET | `load_engine_state_db()` |

### 9.5 Phase 38-1B-5: Alerts & Webhooks Frontend

**Alerts & Signals:** 4 tabs — monitor control (start/stop with 5s polling), alert feed, strategy config, portfolio config.
**Webhook Templates:**

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/webhooks/templates` | GET | `load_webhook_templates_db()` |
| `/api/webhooks/templates` | POST | `save_webhook_template_db()` |
| `/api/webhooks/templates/{id}` | PUT | `update_webhook_template_db()` |
| `/api/webhooks/templates/{id}` | DELETE | `delete_webhook_template_db()` |
| `/api/webhooks/test` | POST | `send_webhook()` |
| `/api/webhooks/delivery-log` | GET | `get_webhook_delivery_log()` |

### 9.6 Self-Test Checkpoints — Phase 1B

- [ ] **1B-1:** Portfolio CRUD works end-to-end (create → view → edit → delete)
- [ ] **1B-2:** Portfolio compute returns KPIs + equity curve for real strategies
- [ ] **1B-3:** Portfolio detail: all 6 tabs render with real data
- [ ] **1B-4:** Prop Firm Check shows pass/fail per rule
- [ ] **1B-5:** Requirement set CRUD works
- [ ] **1B-6:** Alert feed shows real alerts from monitor
- [ ] **1B-7:** Monitor start/stop buttons change desired_state in DB
- [ ] **1B-8:** Engine status polls every 5s and shows running/stopped/error
- [ ] **1B-9:** Webhook template CRUD works
- [ ] **1B-10:** Test webhook sends to configured URL
- [ ] **1B-11:** Portfolio live dashboard updates via polling (10s)

**Phase 1B Milestone — Full Trading Workflow**

---

## 10. Phase 38-1C: Power User Tools

### 10.1 Phase 38-1C-1: Mass Builder Backend + Frontend

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/mass-builder/run` | POST | Start async search (background thread) |
| `/api/mass-builder/progress/{id}` | GET | Poll progress |
| `/api/mass-builder/cancel/{id}` | POST | Cancel running search |
| `/api/mass-builder/results` | GET | `load_mass_searches()` |
| `/api/mass-builder/results/{id}` | GET | `get_mass_search()` |
| `/api/mass-builder/results/{id}` | DELETE | `delete_mass_search()` |

**Async pattern:** `POST /run` starts a thread (same pattern as Streamlit mass builder), returns `{search_id, status: "running"}`. Frontend polls `/progress/{id}` every 2 seconds.

### 10.2 Phase 38-1C-2: Pack Builder + User Packs + Timeframes

| Endpoint | Method | Wraps |
|----------|--------|-------|
| `/api/packs/user` | GET | `pack_registry.scan_and_load_all()` |
| `/api/packs/user` | POST | Save user pack |
| `/api/timeframes` | GET | Timeframe config from settings |
| `/api/timeframes` | PUT | Save timeframe config |

**Pack Builder V7:** Wizard UI (Describe → Generate → Refine → Code → Install). The AI integration endpoints are placeholder for now — the wizard structure and UI render from V5, but the "Generate" steps will need a Claude API endpoint in a future phase.

**User Packs V5:** 8-tab detail view. Signal Validation and Parity Simulator tabs show placeholders until Phase 31F completes.

**Timeframes V5:** 17 TFs x 4 use cases grid. Simple config page — read/write to settings.

### 10.3 Self-Test Checkpoints — Phase 1C

- [ ] **1C-1:** Mass Builder runs async search, progress updates via polling
- [ ] **1C-2:** Mass Results shows completed searches with strategy cards
- [ ] **1C-3:** Cancel button stops running search
- [ ] **1C-4:** Pack Builder wizard renders all 5 steps (AI steps placeholder)
- [ ] **1C-5:** User Packs page lists installed custom packs
- [ ] **1C-6:** Timeframes grid shows all 17 TFs with use case toggles
- [ ] **1C-7:** Timeframe changes persist to settings

**Phase 1C Milestone — Feature Parity with Streamlit**

---

## 11. End-to-End Verification

After all phases complete:

1. [ ] Login persists across page reload (no session drops)
2. [ ] All 20 V5 pages render with real data from FastAPI
3. [ ] Strategy Builder: full flow (configure → backtest → view chart → save)
4. [ ] My Strategies: list, detail (6 tabs), edit, delete, duplicate, bulk actions
5. [ ] Portfolio: create with strategies, view detail (6 tabs), prop firm check
6. [ ] Alerts: monitor start/stop, alert feed with acknowledge
7. [ ] Webhooks: template CRUD, test delivery
8. [ ] Mass Builder: async run, progress, results
9. [ ] All 4 pack types: CRUD, template browsing, variation creation
10. [ ] Dashboard: equity curve, P&L, active positions, health indicators
11. [ ] Settings: all 4 sub-pages persist preferences
12. [ ] Charts: candlestick + overlays + markers render correctly in all 9 themes
13. [ ] Equity curves: 3-segment rendering (BT blue, FWD orange, Live green)
14. [ ] Engine status polling (5s) shows real-time state
15. [ ] Streamlit app still runs independently via `./run.sh`
16. [ ] Backtest KPIs match between Streamlit and Next.js for identical inputs
17. [ ] `ralph_engine.py`, `unified_engine.py`, `worker.py`, `db.py` have ZERO modifications
18. [ ] All CRUD operations persist to Supabase and survive page reload

---

## 12. Intervention Points

| After Phase | Kevin Reviews | Type |
|-------------|--------------|------|
| 38-0 | Login flow, theme persistence | Visual QA |
| 38-1A-3 | Strategy Builder with real backtest, charts | Visual QA |
| 38-1A-4 | My Strategies + Strategy Detail (6 tabs) | Visual QA |
| 38-1A-5 | Phase 1A milestone — full core loop | Milestone review |
| 38-1B-2 | Portfolio pages (3 pages, complex layouts) | Visual QA |
| 38-1B-5 | Phase 1B milestone — alerts working | Milestone review |
| 38-1C | Phase 1C milestone — feature parity | Milestone review + deprecation decision |

Between milestones: autonomous implementation using this spec's self-test checkpoints.
