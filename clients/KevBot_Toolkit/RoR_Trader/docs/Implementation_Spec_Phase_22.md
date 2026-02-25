# Phase 22: Web Deployment & Multi-User Infrastructure — Implementation Spec

**Version:** 0.1
**Date:** February 25, 2026
**Purpose:** Detailed, implementation-ready spec for deploying RoR Trader to production on Railway + Supabase. Covers database migration, authentication, worker service extraction, containerization, and DNS configuration.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 22A: Supabase Setup & Database Schema](#phase-22a-supabase-setup--database-schema)
3. [Phase 22B: Data Access Layer](#phase-22b-data-access-layer)
4. [Phase 22C: Authentication](#phase-22c-authentication)
5. [Phase 22D: Worker Service](#phase-22d-worker-service)
6. [Phase 22E: Containerization & Deployment](#phase-22e-containerization--deployment)
7. [Phase 22F: Data Migration & DNS](#phase-22f-data-migration--dns)
8. [Dependency Order](#dependency-order)
9. [File Map](#file-map)
10. [Testing & Verification](#testing--verification)
11. [Rollback Strategy](#rollback-strategy)

---

## Architecture Overview

### Current Architecture (Local)
```
┌─────────────────────────────────────────────────┐
│                  User's Machine                  │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │           Streamlit (app.py)              │   │
│  │  ┌─────────────┐  ┌──────────────────┐   │   │
│  │  │ UI Rendering │  │ Daemon Threads   │   │   │
│  │  │              │  │  - realtime_engine│   │   │
│  │  │              │  │  - webhook_server │   │   │
│  │  └─────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────┘   │
│                      │                           │
│  ┌──────────────┐    │    ┌──────────────────┐  │
│  │ alert_monitor │◄───┘   │   JSON Files     │  │
│  │ (subprocess)  │ PID    │  strategies.json  │  │
│  └──────────────┘ mgmt   │  portfolios.json  │  │
│                           │  alerts.json      │  │
│                           │  alert_config.json│  │
│                           │  settings.json    │  │
│                           │  ...              │  │
│                           └──────────────────┘  │
└─────────────────────────────────────────────────┘
```

### Target Architecture (Production)
```
┌──────────────────┐     ┌──────────────────────┐
│   Railway: Web   │     │   Railway: Worker    │
│   ────────────   │     │   ──────────────     │
│  Streamlit app   │     │  alert_monitor       │
│  (UI only)       │     │  realtime_engine     │
│                  │     │  webhook_server       │
│  Auth gate       │     │                      │
│  Read/write DB   │     │  Admin DB client     │
│  Monitor control │     │  Reads desired_state │
│                  │     │  Writes alerts/status│
└────────┬─────────┘     └────────┬─────────────┘
         │                         │
         │    ┌────────────────┐   │
         └────┤   Supabase     ├───┘
              │  ────────────  │
              │  PostgreSQL    │
              │  Auth (JWT)    │
              │  RLS policies  │
              └────────────────┘

┌──────────────┐     ┌──────────────┐
│  Namecheap   │────▶│  Railway     │
│  DNS CNAME   │     │  Custom      │
│              │     │  Domain+SSL  │
└──────────────┘     └──────────────┘
```

### Hosting Stack

| Component | Provider | Cost | Purpose |
|-----------|----------|------|---------|
| Web service | Railway | ~$5-7/mo | Streamlit app (UI, auth, DB queries) |
| Worker service | Railway | ~$2-5/mo | Alert monitor, streaming engine, webhook delivery |
| Database | Supabase (free tier) | $0/mo | PostgreSQL, Row Level Security |
| Authentication | Supabase Auth | $0/mo | Email/password, optional OAuth |
| Domain | Namecheap | (already owned) | Custom domain DNS |
| SSL | Railway (auto) | $0/mo | Let's Encrypt certificates |
| **Total** | | **~$8-13/mo** | |

### Git Workflow

```
main ──────────────────────────────────── (auto-deploys to Railway production)
  │
  └─ dev ─────────────────────────────── (auto-deploys to Railway staging, optional)
       │
       ├─ deploy/schema ─────── (22A work)
       ├─ deploy/db-layer ───── (22B work)
       ├─ deploy/auth ────────── (22C work)
       ├─ deploy/worker ──────── (22D work)
       └─ deploy/docker ──────── (22E work)
```

---

## Phase 22A: Supabase Setup & Database Schema

### Prerequisites
- Create a Supabase account at https://supabase.com
- Create a new project (region: US East for lowest latency to Railway)

### Dashboard Configuration
1. **Auth > Providers**: Enable Email provider, disable "Confirm email" for development
2. **Auth > URL Configuration**: Set site URL to custom domain (once deployed)
3. **Note credentials**: Project URL, anon key, service role key, direct DB connection string

### Database Schema

```sql
-- ============================================================
-- Phase 22A: RoR Trader Production Schema
-- ============================================================

-- Strategies
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'LONG',
    timeframe TEXT NOT NULL DEFAULT '1Min',
    strategy_origin TEXT NOT NULL DEFAULT 'standard',
    forward_testing BOOLEAN DEFAULT true,
    forward_test_start TIMESTAMPTZ,
    alert_tracking_enabled BOOLEAN DEFAULT false,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    kpis JSONB,
    stored_trades JSONB,
    equity_curve_data JSONB,
    live_executions JSONB,
    discrepancies JSONB,
    discrepancies_dismissed_at TIMESTAMPTZ,
    data_refreshed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_strategies_user ON strategies(user_id);
CREATE INDEX idx_strategies_symbol ON strategies(user_id, symbol);

-- Portfolios
CREATE TABLE portfolios (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    name TEXT NOT NULL,
    starting_balance NUMERIC DEFAULT 10000,
    compound_rate NUMERIC DEFAULT 1.0,
    requirement_set_id INTEGER,
    strategies JSONB DEFAULT '[]'::jsonb,
    cached_kpis JSONB,
    equity_curve_data JSONB,
    account JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_portfolios_user ON portfolios(user_id);

-- Requirement Sets (prop firm rules)
CREATE TABLE requirement_sets (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    built_in BOOLEAN DEFAULT false,
    firm_key TEXT,
    rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_requirements_user ON requirement_sets(user_id);

-- Alerts
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    strategy_id INTEGER NOT NULL,
    portfolio_id INTEGER,
    alert_type TEXT NOT NULL,
    source TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb,
    acknowledged BOOLEAN DEFAULT false,
    timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_alerts_user ON alerts(user_id);
CREATE INDEX idx_alerts_strategy ON alerts(user_id, strategy_id);
CREATE INDEX idx_alerts_timestamp ON alerts(user_id, timestamp DESC);

-- Alert Configuration (one row per user)
CREATE TABLE alert_config (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Webhook Templates
CREATE TABLE webhook_templates (
    id TEXT NOT NULL,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    category TEXT,
    is_default BOOLEAN DEFAULT false,
    payload_template TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (id, COALESCE(user_id, '00000000-0000-0000-0000-000000000000'::uuid))
);

-- Monitor Status (one row per user)
CREATE TABLE monitor_status (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    desired_state TEXT DEFAULT 'stopped',
    status JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- User Settings (one row per user)
CREATE TABLE user_settings (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Confluence Groups (one row per user)
CREATE TABLE confluence_groups (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    groups JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- General Packs (one row per user)
CREATE TABLE general_packs (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    packs JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Risk Management Packs (one row per user)
CREATE TABLE risk_management_packs (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL UNIQUE,
    packs JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Inbound Webhook Signals
CREATE TABLE inbound_signals (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    strategy_id INTEGER,
    payload JSONB NOT NULL,
    processed BOOLEAN DEFAULT false,
    received_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_inbound_signals_user ON inbound_signals(user_id, processed);

-- Pre-registration waitlist (public, no auth required)
CREATE TABLE waitlist (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- Row Level Security Policies
-- ============================================================

ALTER TABLE strategies ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own strategies" ON strategies FOR ALL USING (auth.uid() = user_id);

ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own portfolios" ON portfolios FOR ALL USING (auth.uid() = user_id);

ALTER TABLE requirement_sets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own or built-in requirements" ON requirement_sets
    FOR ALL USING (auth.uid() = user_id OR user_id IS NULL);

ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own alerts" ON alerts FOR ALL USING (auth.uid() = user_id);

ALTER TABLE alert_config ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own alert config" ON alert_config FOR ALL USING (auth.uid() = user_id);

ALTER TABLE webhook_templates ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own or default templates" ON webhook_templates
    FOR ALL USING (auth.uid() = user_id OR user_id IS NULL);

ALTER TABLE monitor_status ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own monitor status" ON monitor_status FOR ALL USING (auth.uid() = user_id);

ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own settings" ON user_settings FOR ALL USING (auth.uid() = user_id);

ALTER TABLE confluence_groups ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own confluence groups" ON confluence_groups FOR ALL USING (auth.uid() = user_id);

ALTER TABLE general_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own general packs" ON general_packs FOR ALL USING (auth.uid() = user_id);

ALTER TABLE risk_management_packs ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own rm packs" ON risk_management_packs FOR ALL USING (auth.uid() = user_id);

ALTER TABLE inbound_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users access own inbound signals" ON inbound_signals FOR ALL USING (auth.uid() = user_id);

-- Waitlist: public insert, no read
ALTER TABLE waitlist ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can join waitlist" ON waitlist FOR INSERT WITH CHECK (true);
```

### Seed Data (Built-In)

```sql
-- Built-in requirement sets (user_id IS NULL = shared)
INSERT INTO requirement_sets (user_id, name, built_in, firm_key, rules)
VALUES
    (NULL, 'Take Profit Trading (TTP) - 50K', true, 'ttp_50k', '...'::jsonb),
    (NULL, 'FTMO - 100K', true, 'ftmo_100k', '...'::jsonb);
-- (Exact rules JSON extracted from current requirements.json at migration time)
```

### Strategy Config JSONB Schema

The `config` JSONB column on the `strategies` table stores all strategy parameters that aren't needed for database-level filtering. This preserves the existing flat dict pattern while keeping the database schema clean:

```jsonc
{
    // Trigger configuration
    "entry_trigger": "cross_bull",
    "entry_trigger_confluence_id": "ema_stack_default_cross_bull",
    "exit_trigger": "cross_bear",
    "exit_trigger_confluence_id": "ema_stack_default_cross_bear",
    "exit_triggers": ["cross_bear"],
    "exit_trigger_confluence_ids": ["ema_stack_default_cross_bear"],
    "confluence": ["macd_default"],
    "general_confluences": ["GEN-rvol_default"],

    // Risk parameters
    "risk_per_trade": 100.0,
    "starting_balance": 10000.0,
    "stop_config": {"method": "atr", "atr_mult": 1.5},
    "target_config": {"method": "risk_reward", "rr_ratio": 2.0},
    "stop_atr_mult": 1.5,
    "bar_count_exit": null,

    // Data parameters
    "data_days": 30,
    "data_seed": 42,
    "lookback_mode": "days",
    "bar_count": 500,
    "trading_session": "RTH",

    // Webhook origin data (if strategy_origin == "webhook")
    "webhook_source": null,
    "webhook_matched_triggers": null
}
```

**Transformation helpers:**

```python
def _strategy_to_row(strat: dict) -> dict:
    """Split flat strategy dict into column + JSONB format for database."""
    # Fields that go into dedicated columns
    COLUMN_FIELDS = {
        'id', 'user_id', 'name', 'symbol', 'direction', 'timeframe',
        'strategy_origin', 'forward_testing', 'forward_test_start',
        'alert_tracking_enabled', 'kpis', 'stored_trades',
        'equity_curve_data', 'live_executions', 'discrepancies',
        'discrepancies_dismissed_at', 'data_refreshed_at',
        'created_at', 'updated_at',
    }
    row = {}
    config = {}
    for k, v in strat.items():
        if k in COLUMN_FIELDS:
            row[k] = v
        else:
            config[k] = v
    row['config'] = config
    return row

def _row_to_strategy(row: dict) -> dict:
    """Merge database row back into flat strategy dict."""
    strat = {k: v for k, v in row.items() if k != 'config'}
    if row.get('config'):
        strat.update(row['config'])
    return strat
```

---

## Phase 22B: Data Access Layer

### New File: `src/db.py`

```python
"""
Centralized database access layer for RoR Trader.
Wraps all Supabase/PostgreSQL operations.
"""
import os
import threading
from supabase import create_client, Client

# Toggle: "true" = use database, "false" = use JSON files (for incremental dev)
USE_DB = os.getenv("USE_DB", "false").lower() == "true"

# Supabase credentials from environment
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Per-thread user context (Streamlit runs each session in its own thread)
_local = threading.local()

def set_current_user(user_id: str, access_token: str):
    """Set the current user context for this thread."""
    _local.user_id = user_id
    _local.access_token = access_token

def get_current_user_id() -> str:
    """Get the current user's ID."""
    return getattr(_local, 'user_id', None)

def get_current_token() -> str:
    """Get the current user's access token."""
    return getattr(_local, 'access_token', None)

def get_client() -> Client:
    """Get a Supabase client authenticated as the current user.
    Uses the anon key + user's JWT for RLS enforcement.
    """
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    token = get_current_token()
    if token:
        client.auth.set_session(token, "")  # Set user JWT for RLS
    return client

def get_admin_client() -> Client:
    """Get a Supabase client with service role key (bypasses RLS).
    Used by the worker service only.
    """
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
```

### CRUD Rewiring Pattern

Each existing CRUD function gets a database path gated by `USE_DB`. Example for strategies:

**Current (`app.py`):**
```python
def load_strategies() -> list:
    if os.path.exists(STRATEGIES_FILE):
        with open(STRATEGIES_FILE, 'r') as f:
            return json.load(f)
    return []
```

**After:**
```python
def load_strategies() -> list:
    from db import USE_DB
    if USE_DB:
        return _load_strategies_db()
    # Original JSON path (unchanged)
    if os.path.exists(STRATEGIES_FILE):
        with open(STRATEGIES_FILE, 'r') as f:
            return json.load(f)
    return []

def _load_strategies_db() -> list:
    from db import get_client, get_current_user_id, _row_to_strategy
    client = get_client()
    result = client.table('strategies') \
        .select('*') \
        .eq('user_id', get_current_user_id()) \
        .order('id') \
        .execute()
    return [_row_to_strategy(r) for r in result.data]
```

### Functions to Rewire (Full List)

**`src/app.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `load_strategies()` | ~3025 | SELECT all for user |
| `_save_strategies(strategies)` | ~3035 | Bulk upsert (used by save/update/delete) |
| `get_strategy_by_id(sid)` | ~3045 | SELECT by id + user_id |
| `save_strategy(strat)` | ~3055 | INSERT (assigns next id) |
| `update_strategy(strat)` | ~3065 | UPDATE by id |
| `delete_strategy(sid)` | ~3075 | DELETE by id |
| `duplicate_strategy(sid)` | ~3085 | SELECT + INSERT with new id |
| `load_settings()` | ~2998 | SELECT from user_settings |
| `save_settings(settings)` | ~3010 | UPSERT to user_settings |

**`src/portfolios.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `load_portfolios()` | ~63 | SELECT all for user |
| `save_portfolio(port)` | ~80 | INSERT |
| `get_portfolio_by_id(pid)` | ~95 | SELECT by id |
| `update_portfolio(port)` | ~110 | UPDATE by id |
| `delete_portfolio(pid)` | ~125 | DELETE by id |
| `duplicate_portfolio(pid)` | ~140 | SELECT + INSERT |
| `load_requirements()` | ~160 | SELECT all (own + built-in) |
| `save_requirement_set(rs)` | ~180 | INSERT |
| `get_requirement_set_by_id(rid)` | ~195 | SELECT by id |
| `update_requirement_set(rs)` | ~210 | UPDATE by id |
| `delete_requirement_set(rid)` | ~225 | DELETE by id |

**`src/alerts.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `load_alert_config()` | ~237 | SELECT from alert_config |
| `save_alert_config(config)` | ~260 | UPSERT to alert_config |
| `load_alerts()` | ~430 | SELECT from alerts (with optional limit) |
| `save_alert(alert)` | ~460 | INSERT to alerts |
| `delete_alerts_for_strategy(sid)` | ~480 | DELETE from alerts WHERE strategy_id |
| `load_monitor_status()` | ~958 | SELECT from monitor_status |
| `save_monitor_status(status)` | ~975 | UPSERT to monitor_status |
| `load_webhook_templates()` | ~1000 | SELECT all (own + defaults) |
| Various strategy alert config getters/setters | ~280-420 | Operate on alert_config JSONB |

**`src/confluence_groups.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `load_confluence_groups()` | ~484 | SELECT from confluence_groups |
| `save_confluence_groups(groups)` | ~510 | UPSERT to confluence_groups |

**`src/alert_monitor.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `load_strategies()` (local copy) | ~170 | Uses admin client in worker |
| `get_strategy_by_id()` (local copy) | ~182 | Uses admin client in worker |

**`src/webhook_server.py`:**
| Function | Line | Operation |
|----------|------|-----------|
| `_load_inbound_signals()` | ~31 | SELECT from inbound_signals |
| `_save_inbound_signals()` | ~50 | INSERT to inbound_signals |

---

## Phase 22C: Authentication

### New File: `src/auth.py`

```python
"""
Authentication module for RoR Trader.
Wraps Supabase Auth with Streamlit session management.
"""
import streamlit as st
from db import get_client, set_current_user, SUPABASE_URL, SUPABASE_ANON_KEY

def check_auth() -> bool:
    """Check if the current session is authenticated.
    Returns True if user is logged in with valid JWT.
    """
    if 'auth_user' not in st.session_state:
        return False
    # Check JWT expiry and refresh if needed
    _refresh_session_if_needed()
    return st.session_state.get('auth_user') is not None

def render_auth_page():
    """Render the login/signup page."""
    st.set_page_config(page_title="RoR Trader", layout="centered")
    tab_login, tab_signup, tab_waitlist = st.tabs(["Login", "Sign Up", "Pre-Register"])

    with tab_login:
        _render_login_form()
    with tab_signup:
        _render_signup_form()
    with tab_waitlist:
        _render_waitlist_form()

def _render_login_form():
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In", use_container_width=True)
    if submitted and email and password:
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            result = client.auth.sign_in_with_password({
                "email": email, "password": password
            })
            _set_session(result.user, result.session)
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")

def _render_signup_form():
    with st.form("signup_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create Account", use_container_width=True)
    if submitted:
        if password != confirm:
            st.error("Passwords do not match")
        elif email and password:
            try:
                from supabase import create_client
                client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                result = client.auth.sign_up({
                    "email": email, "password": password
                })
                _set_session(result.user, result.session)
                _seed_defaults_for_new_user()
                st.rerun()
            except Exception as e:
                st.error(f"Signup failed: {e}")

def _render_waitlist_form():
    st.markdown("Interested in RoR Trader? Leave your email and we'll notify you.")
    with st.form("waitlist_form"):
        email = st.text_input("Email")
        name = st.text_input("Name (optional)")
        submitted = st.form_submit_button("Join Waitlist", use_container_width=True)
    if submitted and email:
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            client.table('waitlist').insert({"email": email, "name": name}).execute()
            st.success("You're on the list! We'll be in touch.")
        except Exception as e:
            if 'duplicate' in str(e).lower():
                st.info("You're already on the waitlist!")
            else:
                st.error(f"Error: {e}")

def _set_session(user, session):
    """Store auth state in Streamlit session."""
    st.session_state['auth_user'] = {
        'id': user.id,
        'email': user.email,
    }
    st.session_state['auth_access_token'] = session.access_token
    st.session_state['auth_refresh_token'] = session.refresh_token
    st.session_state['auth_expires_at'] = session.expires_at
    # Set thread-local user context for db.py
    set_current_user(user.id, session.access_token)

def _refresh_session_if_needed():
    """Refresh JWT if expired or close to expiry."""
    import time
    expires_at = st.session_state.get('auth_expires_at', 0)
    if time.time() > expires_at - 60:  # refresh 60s before expiry
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            result = client.auth.refresh_session(
                st.session_state.get('auth_refresh_token', '')
            )
            _set_session(result.user, result.session)
        except Exception:
            # Force re-login on refresh failure
            logout()

def logout():
    """Clear auth state."""
    for key in ['auth_user', 'auth_access_token', 'auth_refresh_token', 'auth_expires_at']:
        st.session_state.pop(key, None)

def get_user() -> dict:
    """Get current authenticated user info."""
    return st.session_state.get('auth_user')

def _seed_defaults_for_new_user():
    """Seed default data for a newly registered user."""
    from db import get_client, get_current_user_id
    # Check if user already has confluence groups (idempotent)
    client = get_client()
    uid = get_current_user_id()
    existing = client.table('confluence_groups').select('id').eq('user_id', uid).execute()
    if not existing.data:
        # Seed defaults (confluence groups, packs, settings)
        from confluence_groups import create_default_groups
        client.table('confluence_groups').insert({
            'user_id': uid, 'groups': create_default_groups()
        }).execute()
        # ... similar for general_packs, risk_management_packs, user_settings
```

### Auth Gate Integration in `app.py`

Insert at the very beginning of the main render flow (after imports, before `st.set_page_config`):

```python
from db import USE_DB
if USE_DB:
    from auth import check_auth, render_auth_page, get_user, set_current_user
    if not check_auth():
        render_auth_page()
        st.stop()
    # Ensure thread-local user context is set (survives session state across reruns)
    user = get_user()
    set_current_user(user['id'], st.session_state.get('auth_access_token', ''))
```

Add logout button in sidebar:
```python
if USE_DB:
    with st.sidebar:
        if st.button("Logout"):
            from auth import logout
            logout()
            st.rerun()
```

---

## Phase 22D: Worker Service

### New File: `src/worker.py`

The worker replaces `alert_monitor.py` as the standalone background process. It runs on Railway as a separate service.

```python
"""
RoR Trader Background Worker Service.

Runs on Railway as a persistent background service. Manages the alert
monitor and streaming engine for all users with monitoring enabled.
Communicates with the web service exclusively through the Supabase database.
"""
import logging
import os
import signal
import sys
import time

# Setup logging (Railway captures stdout/stderr)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('worker')

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Force database mode
os.environ['USE_DB'] = 'true'

from db import get_admin_client

POLL_INTERVAL = 30  # seconds — check for user monitor state changes
HEARTBEAT_INTERVAL = 30  # seconds — write heartbeat to DB

class WorkerManager:
    """Manages per-user monitoring instances."""

    def __init__(self):
        self._running = True
        self._user_monitors = {}  # user_id -> MonitorInstance
        self._last_heartbeat = 0
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        self._running = False

    def run(self):
        logger.info("Worker starting...")
        while self._running:
            try:
                self._poll_user_states()
                self._heartbeat()
            except Exception as e:
                logger.error("Worker loop error: %s", e)
            time.sleep(POLL_INTERVAL)
        self._shutdown()

    def _poll_user_states(self):
        """Check all users' desired monitor states and start/stop accordingly."""
        client = get_admin_client()
        result = client.table('monitor_status').select('*').execute()
        active_users = set()
        for row in result.data:
            uid = row['user_id']
            desired = row.get('desired_state', 'stopped')
            if desired == 'running':
                active_users.add(uid)
                if uid not in self._user_monitors:
                    self._start_user_monitor(uid, row)
            elif uid in self._user_monitors:
                self._stop_user_monitor(uid)
        # Stop monitors for users no longer in the table
        for uid in list(self._user_monitors):
            if uid not in active_users:
                self._stop_user_monitor(uid)

    def _start_user_monitor(self, user_id, status_row):
        """Start monitoring for a user."""
        logger.info("Starting monitor for user %s", user_id)
        # Import and initialize the monitoring pipeline for this user
        # (Reuses existing alert_monitor + realtime_engine logic)
        # ... implementation details ...

    def _stop_user_monitor(self, user_id):
        """Stop monitoring for a user."""
        logger.info("Stopping monitor for user %s", user_id)
        monitor = self._user_monitors.pop(user_id, None)
        if monitor:
            monitor.stop()

    def _heartbeat(self):
        """Write heartbeat to all active user monitor_status rows."""
        now = time.time()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL:
            return
        self._last_heartbeat = now
        client = get_admin_client()
        for uid in self._user_monitors:
            client.table('monitor_status').update({
                'status': {
                    'running': True,
                    'last_heartbeat': time.time(),
                    # ... other status fields
                },
                'updated_at': 'now()',
            }).eq('user_id', uid).execute()

    def _shutdown(self):
        """Graceful shutdown: stop all monitors."""
        logger.info("Shutting down all monitors...")
        for uid in list(self._user_monitors):
            self._stop_user_monitor(uid)
        logger.info("Worker shutdown complete")

if __name__ == '__main__':
    WorkerManager().run()
```

### UI Monitor Controls Update (`app.py`)

Replace subprocess/PID code in `_render_monitor_status_bar()` with database operations:

```python
# Start Monitor button
if st.button("Start Monitor"):
    from db import get_client, get_current_user_id
    client = get_client()
    client.table('monitor_status').upsert({
        'user_id': get_current_user_id(),
        'desired_state': 'running',
        'updated_at': 'now()',
    }).execute()
    st.rerun()

# Stop Monitor button
if st.button("Stop Monitor"):
    from db import get_client, get_current_user_id
    client = get_client()
    client.table('monitor_status').update({
        'desired_state': 'stopped',
    }).eq('user_id', get_current_user_id()).execute()
    st.rerun()

# Status display — read from database
status = load_monitor_status()  # Already rewired to read from DB in Phase 22B
is_running = status.get('status', {}).get('running', False)
last_heartbeat = status.get('status', {}).get('last_heartbeat')
```

---

## Phase 22E: Containerization & Deployment

### `Dockerfile` (Web Service)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vendored lightweight charts fork
COPY streamlit_lwc_fork/ ./streamlit_lwc_fork/
RUN pip install -e ./streamlit_lwc_fork

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY user_packs/ ./user_packs/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

### `Dockerfile.worker` (Worker Service)

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (worker needs all modules for pipeline)
COPY src/ ./src/
COPY config/ ./config/
COPY user_packs/ ./user_packs/

CMD ["python", "src/worker.py"]
```

### `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 10

[browser]
gatherUsageStats = false
```

### Railway Configuration

**Web Service:**
- Source: GitHub repo, root directory `clients/KevBot_Toolkit/RoR_Trader`
- Dockerfile: `Dockerfile`
- Port: 8501
- Custom domain: configured in dashboard
- Environment variables: shared variable group

**Worker Service:**
- Source: Same GitHub repo, same root directory
- Dockerfile: `Dockerfile.worker`
- No port exposed
- Environment variables: same shared variable group

**Shared Environment Variables:**
```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
ALPACA_API_KEY=PKxxxxx
ALPACA_SECRET_KEY=xxxxx
USE_DB=true
```

### `.gitignore` Updates

```gitignore
# Environment & secrets
.env
*.env

# Python
__pycache__/
*.pyc
.venv/

# Runtime data (now in database)
src/strategies.json
src/portfolios.json
src/requirements.json
src/alerts.json
src/alert_config.json
src/monitor_status.json
src/webhook_templates.json

# Logs
src/streaming_engine.log
*.log
```

---

## Phase 22F: Data Migration & DNS

### Migration Script: `src/migrate_json_to_db.py`

```python
"""
One-time migration: JSON files → Supabase PostgreSQL.
Run locally after creating your Supabase account and database schema.

Usage: USE_DB=true python src/migrate_json_to_db.py
"""
import json
import os
import sys

# Ensure environment is configured
os.environ['USE_DB'] = 'true'

from db import get_admin_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def migrate(user_id: str):
    """Migrate all JSON data for a given user_id."""
    client = get_admin_client()

    # 1. Strategies
    strategies = load_json(os.path.join(SCRIPT_DIR, 'strategies.json')) or []
    for strat in strategies:
        row = _strategy_to_row(strat)
        row['user_id'] = user_id
        client.table('strategies').insert(row).execute()
    print(f"  Migrated {len(strategies)} strategies")

    # 2. Requirements
    requirements = load_json(os.path.join(SCRIPT_DIR, 'requirements.json')) or []
    for req in requirements:
        row = {'user_id': user_id, 'name': req['name'], 'rules': req.get('rules', [])}
        if req.get('built_in'):
            row['built_in'] = True
            row['firm_key'] = req.get('firm_key')
            row['user_id'] = None  # Shared
        client.table('requirement_sets').insert(row).execute()
    print(f"  Migrated {len(requirements)} requirement sets")

    # 3. Portfolios
    portfolios = load_json(os.path.join(SCRIPT_DIR, 'portfolios.json')) or []
    for port in portfolios:
        row = _portfolio_to_row(port)
        row['user_id'] = user_id
        client.table('portfolios').insert(row).execute()
    print(f"  Migrated {len(portfolios)} portfolios")

    # 4. Alerts
    alerts = load_json(os.path.join(SCRIPT_DIR, 'alerts.json')) or []
    for alert in alerts:
        row = {
            'user_id': user_id,
            'strategy_id': alert.get('strategy_id'),
            'portfolio_id': alert.get('portfolio_id'),
            'alert_type': alert.get('type', 'unknown'),
            'source': alert.get('source'),
            'data': alert,
            'timestamp': alert.get('timestamp'),
        }
        client.table('alerts').insert(row).execute()
    print(f"  Migrated {len(alerts)} alerts")

    # 5. Alert Config
    alert_config = load_json(os.path.join(SCRIPT_DIR, 'alert_config.json'))
    if alert_config:
        client.table('alert_config').upsert({
            'user_id': user_id, 'config': alert_config
        }).execute()
        print("  Migrated alert config")

    # 6. Webhook Templates
    templates = load_json(os.path.join(SCRIPT_DIR, 'webhook_templates.json')) or []
    for tmpl in templates:
        row = {
            'id': tmpl.get('id'),
            'user_id': user_id if not tmpl.get('is_default') else None,
            'name': tmpl.get('name', ''),
            'category': tmpl.get('category'),
            'is_default': tmpl.get('is_default', False),
            'payload_template': tmpl.get('payload_template', ''),
        }
        client.table('webhook_templates').insert(row).execute()
    print(f"  Migrated {len(templates)} webhook templates")

    # 7. Settings
    settings = load_json(os.path.join(CONFIG_DIR, 'settings.json'))
    if settings:
        client.table('user_settings').upsert({
            'user_id': user_id, 'settings': settings
        }).execute()
        print("  Migrated settings")

    # 8. Confluence Groups
    groups = load_json(os.path.join(CONFIG_DIR, 'confluence_groups.json'))
    if groups:
        client.table('confluence_groups').upsert({
            'user_id': user_id, 'groups': groups
        }).execute()
        print("  Migrated confluence groups")

    # 9. General Packs
    gpacks = load_json(os.path.join(CONFIG_DIR, 'general_packs.json'))
    if gpacks:
        client.table('general_packs').upsert({
            'user_id': user_id, 'packs': gpacks
        }).execute()
        print("  Migrated general packs")

    # 10. Risk Management Packs
    rmpacks = load_json(os.path.join(CONFIG_DIR, 'risk_management_packs.json'))
    if rmpacks:
        client.table('risk_management_packs').upsert({
            'user_id': user_id, 'packs': rmpacks
        }).execute()
        print("  Migrated risk management packs")

    print("\nMigration complete!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python migrate_json_to_db.py <user_id>")
        print("  Get your user_id from Supabase Auth dashboard after signing up")
        sys.exit(1)
    user_id = sys.argv[1]
    print(f"Migrating data for user {user_id}...")
    migrate(user_id)
```

### DNS Configuration

1. **Railway Dashboard:** Add custom domain to web service → note the CNAME target (e.g., `web-xxxxx.railway.app`)
2. **Namecheap Dashboard:**
   - Go to Domain List → Manage → Advanced DNS
   - Add CNAME Record: Host=`@` (or `www`), Value=Railway CNAME target, TTL=Automatic
   - If using apex domain (no www), add ALIAS/ANAME record instead of CNAME
3. **Verify:** Railway auto-provisions SSL certificate once DNS propagates (~5-30 minutes)

---

## Dependency Order

```
22A: Schema ──────────────────────────────────────┐
                                                   │
22B: db.py + CRUD rewire ◄────────────────────────┘
  │
  ├── 22B-strategies (app.py)
  ├── 22B-portfolios (portfolios.py)         (these can be
  ├── 22B-alerts (alerts.py)                  done in parallel)
  ├── 22B-config (confluence_groups, settings)
  └── 22B-monitor (alert_monitor, webhook_server)
       │
       ▼
22C: Auth ────────────────────────────────────────┐
                                                   │
22D: Worker ◄─────────────────────────────────────┘
       │
       ▼
22E: Docker + Railway ────────────────────────────┐
                                                   │
22F: Migration + DNS ◄────────────────────────────┘
```

**Critical path:** 22A → 22B → 22C → 22D → 22E → 22F
**Parallelizable:** 22B sub-modules can be developed in parallel

---

## File Map

| File | Purpose | Approx Lines |
|------|---------|-------------|
| `src/app.py` | Main Streamlit app — auth gate, all CRUD, UI | ~11,600 |
| `src/alerts.py` | Alert CRUD, config CRUD, status CRUD, templates | ~1,200 |
| `src/portfolios.py` | Portfolio + requirements CRUD, computation engine | ~900 |
| `src/confluence_groups.py` | Confluence group CRUD, templates | ~650 |
| `src/alert_monitor.py` | Background alert polling (refactored for worker) | ~640 |
| `src/realtime_engine.py` | WebSocket streaming engine (moves to worker) | ~1,400 |
| `src/webhook_server.py` | Inbound webhook receiver | ~120 |
| **New files** | |
| `src/db.py` | Database access layer | ~150 |
| `src/auth.py` | Authentication module | ~200 |
| `src/worker.py` | Background worker entry point | ~200 |
| `src/migrate_json_to_db.py` | One-time migration script | ~150 |
| `src/db/schema.sql` | Database schema + RLS policies | ~200 |
| `Dockerfile` | Web service container | ~20 |
| `Dockerfile.worker` | Worker service container | ~15 |
| `.streamlit/config.toml` | Production Streamlit config | ~15 |

---

## Testing & Verification

### Local Development
1. Create Supabase project, run schema SQL
2. Set environment variables locally
3. Run `USE_DB=true streamlit run src/app.py`
4. Test each CRUD operation: create/read/update/delete strategies, portfolios, alerts
5. Test auth: signup, login, logout, multi-user isolation (two browsers)
6. Test monitor control: start/stop via database flag

### Docker Local
```bash
docker build -t ror-web .
docker build -f Dockerfile.worker -t ror-worker .
docker run -p 8501:8501 --env-file .env ror-web
docker run --env-file .env ror-worker
```
Verify: login → create strategy → start monitor → alerts fire → webhooks deliver

### Production Deploy
1. Push to Railway, verify both services build and start
2. Run migration script against production Supabase
3. Login with migrated account, verify all data present
4. Point DNS, verify SSL, verify custom domain
5. Start monitor, verify end-to-end alert flow during market hours
6. Test with second user account — verify data isolation

### Regression Checklist
- [ ] All strategy CRUD operations work (create, view, edit, delete, duplicate)
- [ ] Backtest pipeline runs and produces correct KPIs
- [ ] Forward testing data persists and updates correctly
- [ ] All portfolio operations work (create, view, edit, delete, builder)
- [ ] Requirement sets load (built-in + custom)
- [ ] Alert monitor starts and stops from UI
- [ ] Streaming engine connects to Alpaca WebSocket
- [ ] Alerts fire and appear in UI
- [ ] Webhooks deliver to configured endpoints
- [ ] Price charts render with correct data
- [ ] Settings persist across sessions
- [ ] Confluence groups save and load correctly
- [ ] User A cannot see User B's data
- [ ] App is accessible via custom domain with HTTPS

---

## Rollback Strategy

If production deployment fails:
1. **Railway:** Redeploy previous commit (Railway keeps deployment history)
2. **Data:** Supabase database is independent — can drop and recreate from migration
3. **Fallback:** Set `USE_DB=false` to revert to JSON file mode instantly
4. **DNS:** Revert Namecheap CNAME to previous target (or remove)

The `USE_DB` flag provides a zero-risk incremental migration path — any individual CRUD module can be toggled back to JSON without affecting others.
