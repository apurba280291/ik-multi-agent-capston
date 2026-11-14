"""Streamlit login / registration screens and session handling.

``require_login`` is called before anything expensive is loaded; unauthenticated visitors only ever see the login form
(it calls ``st.stop()``). The signed-in user lives in ``st.session_state`` for this browser tab only (no cookie, so a page
refresh signs you out) and expires after ``auth.session_minutes`` of inactivity. Each login starts a fresh chat session id,
so one user's conversation memory is never reachable from another account.
"""
from __future__ import annotations

import time
import uuid

import streamlit as st

from src.auth.service import AuthService, User
from src.auth.store import UserStore
from src.core.config import Settings

USER_KEY, SEEN_KEY, NOTICE_KEY = "auth_user", "auth_seen", "auth_notice"
CHAT_KEYS = ("messages", "portfolio", "pending")


def auth_enabled(settings: Settings) -> bool:
    value = settings.get("auth.enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "no"}
    return bool(value)


def session_expired(last_seen: float, now: float, minutes: float) -> bool:
    """Idle timeout; ``minutes <= 0`` disables it."""
    return minutes > 0 and now - last_seen > minutes * 60


@st.cache_resource
def get_auth_service(db_path: str, max_attempts: int, lockout_minutes: float, allow_registration: bool) -> AuthService:
    return AuthService(UserStore(db_path), max_attempts, lockout_minutes * 60, allow_registration)


def _service(settings: Settings) -> AuthService:
    return get_auth_service(
        str(settings.resolve_path(settings.get("auth.db_path", "data/users.sqlite3"))),
        int(settings.get("auth.max_failed_attempts", 5)),
        float(settings.get("auth.lockout_minutes", 15)),
        bool(settings.get("auth.allow_registration", True)),
    )


def _start_session(user: User) -> None:
    st.session_state[USER_KEY] = user
    st.session_state[SEEN_KEY] = time.time()
    for key in CHAT_KEYS:
        st.session_state.pop(key, None)
    st.session_state.messages, st.session_state.portfolio = [], []
    st.session_state.session_id = f"user-{user.id}-{uuid.uuid4().hex[:8]}"


def logout(notice: str = "") -> None:
    for key in (USER_KEY, SEEN_KEY, *CHAT_KEYS):
        st.session_state.pop(key, None)
    st.session_state.session_id = str(uuid.uuid4())
    if notice:
        st.session_state[NOTICE_KEY] = notice
    st.rerun()


def require_login(settings: Settings) -> User | None:
    """The signed-in user, or None when authentication is switched off. Stops the script if not signed in."""
    if not auth_enabled(settings):
        return None
    user: User | None = st.session_state.get(USER_KEY)
    if user is not None:
        if session_expired(st.session_state.get(SEEN_KEY, 0.0), time.time(), float(settings.get("auth.session_minutes", 60))):
            logout("Your session expired. Please sign in again.")
        st.session_state[SEEN_KEY] = time.time()
        return user
    _login_screen(_service(settings))
    st.stop()


def _login_screen(service: AuthService) -> None:
    st.title("Finnie")
    st.caption("Sign in to chat with your finance guide.")
    if notice := st.session_state.pop(NOTICE_KEY, ""):
        st.info(notice)
    tabs = st.tabs(["Sign in", "Create account"] if service.allow_registration else ["Sign in"])
    with tabs[0]:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username", autocomplete="username")
            password = st.text_input("Password", type="password", key="login_password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", key="login_submit")
        if submitted:
            result = service.login(username, password)
            if result.ok:
                _start_session(result.user)
                st.rerun()
            st.error(result.message)
    if service.allow_registration:
        with tabs[1]:
            with st.form("register_form"):
                new_name = st.text_input("Choose a username", key="register_username", autocomplete="username")
                new_pw = st.text_input("Choose a password", type="password", key="register_password", autocomplete="new-password",
                                       help="At least 8 characters; avoid common passwords and your username.")
                confirm = st.text_input("Repeat the password", type="password", key="register_confirm", autocomplete="new-password")
                created = st.form_submit_button("Create account", key="register_submit")
            if created:
                if new_pw != confirm:
                    st.error("The two passwords do not match.")
                else:
                    result = service.register(new_name, new_pw)
                    if result.ok:
                        _start_session(result.user)
                        st.rerun()
                    st.error(result.message)
    st.caption("Educational tool only - not financial advice. Accounts are stored locally on this server.")


def account_panel(settings: Settings, user: User | None) -> None:
    """Sidebar block: who is signed in, change password, sign out. Nothing is shown when auth is off."""
    if user is None:
        return
    service = _service(settings)
    st.markdown(f"Signed in as **{user.username}**")
    with st.expander("Account"):
        with st.form("password_form", clear_on_submit=True):
            old = st.text_input("Current password", type="password", key="pw_old")
            new = st.text_input("New password", type="password", key="pw_new")
            changed = st.form_submit_button("Change password", key="pw_submit")
        if changed:
            result = service.change_password(user.id, old, new)
            (st.success if result.ok else st.error)(result.message)
    if st.button("Sign out", key="logout"):
        logout()
