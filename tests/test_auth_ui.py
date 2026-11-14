from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.auth.service import AuthService  # noqa: E402
from src.auth.store import UserStore  # noqa: E402
from src.core.config import Settings  # noqa: E402
from src.web_app.auth_ui import auth_enabled, session_expired  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "web_app" / "app.py")
PASSWORD = "blue-train-42-lemon"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FINANCE_AUTH", "on")
    monkeypatch.setenv("FINANCE_AUTH_DB", str(tmp_path / "ui-users.sqlite3"))
    return AppTest.from_file(APP, default_timeout=90).run()


def register(app, name="alice", password=PASSWORD, confirm=None):
    app.text_input(key="register_username").set_value(name)
    app.text_input(key="register_password").set_value(password)
    app.text_input(key="register_confirm").set_value(password if confirm is None else confirm)
    return app.button(key="register_submit").click().run()


def sign_in(app, name, password):
    app.text_input(key="login_username").set_value(name)
    app.text_input(key="login_password").set_value(password)
    return app.button(key="login_submit").click().run()


def errors(app):
    return [e.value for e in app.error]


def test_visitor_sees_only_the_login_screen(app):
    assert not app.exception
    assert len(app.chat_input) == 0 and len(app.tabs) == 2  # "Sign in" / "Create account" only
    assert [t.label for t in app.tabs] == ["Sign in", "Create account"]
    assert "auth_user" not in app.session_state


def test_register_signs_in_and_opens_the_app(app):
    register(app)
    assert not app.exception and not errors(app)
    assert app.session_state["auth_user"].username == "alice"
    assert app.session_state["session_id"].startswith("user-")
    assert len(app.chat_input) == 1 and len(app.tabs) == 5
    assert any("alice" in m.value for m in app.sidebar.markdown)


def test_wrong_password_and_unknown_user_get_the_same_error(app):
    register(app)
    app.button(key="logout").click().run()
    assert len(app.chat_input) == 0
    sign_in(app, "alice", "wrong-password-1")
    first = errors(app)
    sign_in(app, "nobody", "wrong-password-1")
    assert first == errors(app) == ["Invalid username or password."]
    assert len(app.chat_input) == 0 and "auth_user" not in app.session_state


def test_mismatched_confirmation_and_weak_password_are_rejected(app):
    register(app, confirm="something-else-123")
    assert errors(app) == ["The two passwords do not match."] and len(app.chat_input) == 0
    register(app, password="short", confirm="short")
    assert "at least 8" in errors(app)[0] and len(app.chat_input) == 0


def test_sign_out_clears_the_conversation_and_users_are_isolated(app):
    register(app, "alice")
    app.chat_input[0].set_value("What is an ETF?").run()
    assert len(app.chat_message) == 2
    alice_session = app.session_state["session_id"]
    app.button(key="logout").click().run()
    assert "auth_user" not in app.session_state and not app.session_state["messages"] and len(app.chat_input) == 0
    register(app, "bob")
    assert app.session_state["auth_user"].username == "bob"
    assert app.session_state["session_id"] != alice_session and len(app.chat_message) == 0
    assert not app.session_state["portfolio"]


def test_signing_back_in_starts_a_fresh_conversation(app):
    register(app, "alice")
    first = app.session_state["session_id"]
    app.button(key="logout").click().run()
    sign_in(app, "alice", PASSWORD)
    assert app.session_state["auth_user"].username == "alice" and app.session_state["session_id"] != first


def test_lockout_is_shown_after_repeated_failures(app):
    register(app)
    app.button(key="logout").click().run()
    for _ in range(5):
        sign_in(app, "alice", "wrong-password-1")
    assert "Too many failed attempts" in errors(app)[0]
    sign_in(app, "alice", PASSWORD)  # right password, still locked
    assert "Too many failed attempts" in errors(app)[0] and len(app.chat_input) == 0


def test_idle_session_expires(app):
    register(app)
    assert len(app.chat_input) == 1
    app.session_state["auth_seen"] = 0.0  # last activity long ago
    app.run()
    assert len(app.chat_input) == 0 and "auth_user" not in app.session_state
    assert any("expired" in i.value for i in app.info)


def test_change_password_from_the_sidebar(app):
    register(app)
    app.text_input(key="pw_old").set_value(PASSWORD)
    app.text_input(key="pw_new").set_value("new-long-passphrase-7")
    app.button(key="pw_submit").click().run()
    assert any("updated" in s.value for s in app.success)
    app.button(key="logout").click().run()
    sign_in(app, "alice", PASSWORD)
    assert errors(app) == ["Invalid username or password."]
    sign_in(app, "alice", "new-long-passphrase-7")
    assert len(app.chat_input) == 1


# --- pure helpers -------------------------------------------------------------------------------------------------------
def test_session_expiry_rule():
    assert not session_expired(1000, 1000 + 59 * 60, 60) and session_expired(1000, 1000 + 61 * 60, 60)
    assert not session_expired(0, 10**9, 0)  # 0 disables the timeout


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), ("off", False), ("OFF", False), ("0", False), ("on", True), ("true", True)])
def test_auth_enabled_parsing(value, expected):
    assert auth_enabled(Settings({"auth": {"enabled": value}})) is expected
    assert auth_enabled(Settings({})) is True  # secure by default


def test_accounts_created_in_the_ui_are_ordinary_service_accounts(app, tmp_path):
    register(app)
    service = AuthService(UserStore(tmp_path / "ui-users.sqlite3"))
    assert service.login("alice", PASSWORD).ok
