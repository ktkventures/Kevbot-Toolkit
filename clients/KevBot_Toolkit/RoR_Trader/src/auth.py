"""
Authentication module for RoR Trader.

Wraps Supabase Auth with Streamlit session management.
Handles login, signup, session refresh, and new-user onboarding.
"""
import time
import streamlit as st
from db import (
    USE_DB, SUPABASE_URL, SUPABASE_ANON_KEY,
    set_current_user, get_current_user_id, get_client,
)


def check_auth() -> bool:
    """Check if the current session is authenticated.
    Returns True if user is logged in with a valid JWT.
    """
    if 'auth_user' not in st.session_state:
        return False
    _refresh_session_if_needed()
    return st.session_state.get('auth_user') is not None


def require_auth():
    """Auth gate: show login page and stop if not authenticated.
    Call this at the top of app.py's main flow when USE_DB is True.
    """
    if not check_auth():
        render_auth_page()
        st.stop()
    # Ensure thread-local user context is set for db.py
    # (must be re-set on each Streamlit rerun since threads may be reused)
    user = get_user()
    set_current_user(
        user['id'],
        st.session_state.get('auth_access_token', ''),
    )


def render_auth_page():
    """Render the login / signup / waitlist page.
    Note: set_page_config is called by main() before this is reached.
    """
    st.title("RoR Trader")

    tab_login, tab_signup, tab_waitlist = st.tabs([
        "Login", "Sign Up", "Pre-Register"
    ])

    with tab_login:
        _render_login_form()
    with tab_signup:
        _render_signup_form()
    with tab_waitlist:
        _render_waitlist_form()


def get_user() -> dict | None:
    """Get current authenticated user info: {id, email}."""
    return st.session_state.get('auth_user')


def logout():
    """Clear auth state and force re-login."""
    for key in [
        'auth_user', 'auth_access_token',
        'auth_refresh_token', 'auth_expires_at',
    ]:
        st.session_state.pop(key, None)


def render_logout_button():
    """Render a logout button. Call from within a sidebar context."""
    if st.button("Logout"):
        logout()
        st.rerun()


# ============================================================
# Internal Helpers
# ============================================================

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
                "email": email,
                "password": password,
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
        submitted = st.form_submit_button(
            "Create Account", use_container_width=True
        )

    if submitted:
        if not email or not password:
            st.warning("Please enter email and password.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            try:
                from supabase import create_client
                client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
                result = client.auth.sign_up({
                    "email": email,
                    "password": password,
                })
                if result.user and result.session:
                    _set_session(result.user, result.session)
                    _seed_defaults_for_new_user()
                    st.rerun()
                else:
                    st.info(
                        "Check your email for a confirmation link, "
                        "then log in."
                    )
            except Exception as e:
                st.error(f"Signup failed: {e}")


def _render_waitlist_form():
    st.markdown(
        "Interested in RoR Trader? Leave your email and "
        "we'll notify you when access is available."
    )
    with st.form("waitlist_form"):
        email = st.text_input("Email")
        name = st.text_input("Name (optional)")
        submitted = st.form_submit_button(
            "Join Waitlist", use_container_width=True
        )

    if submitted and email:
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            client.table('waitlist').insert(
                {"email": email, "name": name or None},
                returning='minimal',
            ).execute()
            st.success("You're on the list! We'll be in touch.")
        except Exception as e:
            if 'duplicate' in str(e).lower():
                st.info("You're already on the waitlist!")
            else:
                st.error(f"Error: {e}")


def _set_session(user, session):
    """Store auth state in Streamlit session."""
    st.session_state['auth_user'] = {
        'id': str(user.id),
        'email': user.email,
    }
    st.session_state['auth_access_token'] = session.access_token
    st.session_state['auth_refresh_token'] = session.refresh_token
    st.session_state['auth_expires_at'] = session.expires_at
    # Set thread-local user context so db.py picks up the JWT
    set_current_user(str(user.id), session.access_token)


def _refresh_session_if_needed():
    """Refresh JWT if expired or close to expiry (60s buffer)."""
    expires_at = st.session_state.get('auth_expires_at', 0)
    if time.time() > expires_at - 60:
        try:
            from supabase import create_client
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            refresh_token = st.session_state.get('auth_refresh_token', '')
            result = client.auth.refresh_session(refresh_token)
            _set_session(result.user, result.session)
        except Exception:
            # Force re-login on refresh failure
            logout()


def _seed_defaults_for_new_user():
    """Seed default configuration for a newly registered user.

    Uses the existing create_default_*() + save_*() functions which are
    now DB-aware. The load functions already handle "no data = create defaults
    and save" so this just triggers that path for each config type.

    Idempotent — each load function checks for existing data first.
    """
    try:
        # Trigger default seeding for each config type.
        # Each load_*() function creates and saves defaults if no data exists.
        from confluence_groups import load_confluence_groups
        load_confluence_groups()

        from general_packs import load_general_packs
        load_general_packs()

        from risk_management_packs import load_risk_management_packs
        load_risk_management_packs()

        # Settings need explicit seeding since load_settings() returns
        # in-memory defaults without persisting them
        from db import load_settings_db, save_settings_db
        existing = load_settings_db()
        if not existing:
            from app import SETTINGS_DEFAULTS
            save_settings_db(dict(SETTINGS_DEFAULTS))

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to seed defaults for new user: {e}"
        )
