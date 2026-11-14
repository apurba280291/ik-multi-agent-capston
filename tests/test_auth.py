import sqlite3

import pytest

from src.auth import passwords
from src.auth.service import INVALID, AuthService
from src.auth.store import UserStore


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def service(tmp_path, clock):
    return AuthService(UserStore(tmp_path / "users.sqlite3"), max_attempts=3, lockout_seconds=600, clock=clock)


# --- passwords ------------------------------------------------------------------------------------------
def test_hash_is_salted_not_plaintext_and_verifies():
    a, b = passwords.hash_password("correct horse battery"), passwords.hash_password("correct horse battery")
    assert a != b and "correct horse" not in a and a.startswith("scrypt$")  # random salt each time
    assert passwords.verify_password("correct horse battery", a)
    assert not passwords.verify_password("correct horse batterY", a)
    assert not passwords.verify_password("anything", "garbage") and not passwords.verify_password("x", "bcrypt$1$2$3$a$b")


def test_hash_parameters_travel_with_the_hash():
    stored = passwords.hash_password("another long passphrase")
    scheme, n, r, p, salt, digest = stored.split("$")
    assert (scheme, int(n), int(r), int(p)) == ("scrypt", passwords.N, passwords.R, passwords.P)


@pytest.mark.parametrize("password,fragment", [
    ("short", "at least 8"), ("x" * 200, "at most"), ("password123", "too common"), ("aaaaaaaaaa", "repetitive"),
    ("alice-is-great", "username"),
])
def test_password_policy_rejects(password, fragment):
    assert fragment in passwords.validate_password(password, "alice")


def test_password_policy_accepts_a_passphrase_and_username_rules():
    assert passwords.validate_password("blue-train-42-lemon", "alice") is None
    assert passwords.validate_username("alice") is None and passwords.validate_username("a.b_c-9") is None
    for bad in ("al", "Alice", "bad name", "-lead", "x" * 33, "semi;colon"):
        assert passwords.validate_username(bad)
    assert passwords.normalise_username("  Alice ") == "alice"


# --- registration and login ----------------------------------------------------------------------------------
def test_register_then_login_roundtrip(service):
    created = service.register("Alice", "blue-train-42-lemon")
    assert created.ok and created.user.username == "alice"
    ok = service.login("ALICE", "blue-train-42-lemon")  # username is case-insensitive
    assert ok.ok and ok.user.id == created.user.id and ok.user.last_login is not None


def test_duplicate_and_invalid_registrations(service):
    assert service.register("alice", "blue-train-42-lemon").ok
    assert "already taken" in service.register("Alice", "another-long-pass-9").message
    assert not service.register("bob", "short").ok and not service.register("b", "blue-train-42-lemon").ok


def test_registration_can_be_disabled(tmp_path):
    closed = AuthService(UserStore(tmp_path / "u.sqlite3"), allow_registration=False)
    assert not closed.register("alice", "blue-train-42-lemon").ok


def test_wrong_and_unknown_credentials_look_identical(service):
    service.register("alice", "blue-train-42-lemon")
    wrong, unknown = service.login("alice", "not-the-password"), service.login("nobody", "blue-train-42-lemon")
    assert wrong.message == unknown.message == INVALID and not wrong.ok and not unknown.ok
    assert not service.login("alice", "x" * 500).ok  # over-long input is rejected, not hashed


def test_password_hash_is_what_gets_stored(service, tmp_path):
    service.register("alice", "blue-train-42-lemon")
    raw = sqlite3.connect(tmp_path / "users.sqlite3").execute("SELECT password_hash FROM users").fetchone()[0]
    assert "blue-train" not in raw and raw.startswith("scrypt$")


# --- lockout ---------------------------------------------------------------------------------------------------------
def test_lockout_after_repeated_failures_and_recovery(service, clock):
    service.register("alice", "blue-train-42-lemon")
    assert service.login("alice", "bad-1").message == INVALID
    assert service.login("alice", "bad-2").message == INVALID
    locked = service.login("alice", "bad-3")
    assert "Too many failed attempts" in locked.message and "10 minutes" in locked.message
    assert not service.login("alice", "blue-train-42-lemon").ok  # even the right password is refused while locked
    clock.now += 601
    assert service.login("alice", "blue-train-42-lemon").ok      # lock expired
    assert service.login("alice", "bad-again").message == INVALID  # and the counter restarted


def test_lockout_does_not_reveal_whether_the_account_exists(service):
    messages = [service.login("ghost", f"guess-{i}").message for i in range(4)]
    assert messages[:2] == [INVALID, INVALID] and all("Too many failed attempts" in m for m in messages[2:])


def test_successful_login_resets_the_failure_count(service):
    service.register("alice", "blue-train-42-lemon")
    service.login("alice", "bad-1")
    service.login("alice", "bad-2")
    assert service.login("alice", "blue-train-42-lemon").ok
    assert service.login("alice", "bad-3").message == INVALID  # would have locked had the count survived


# --- account management ----------------------------------------------------------------------------------------------------
def test_change_password(service):
    user = service.register("alice", "blue-train-42-lemon").user
    assert not service.change_password(user.id, "wrong", "new-long-passphrase-7").ok
    assert "at least 8" in service.change_password(user.id, "blue-train-42-lemon", "short").message
    assert service.change_password(user.id, "blue-train-42-lemon", "new-long-passphrase-7").ok
    assert not service.login("alice", "blue-train-42-lemon").ok and service.login("alice", "new-long-passphrase-7").ok


def test_delete_account_needs_the_password(service):
    user = service.register("alice", "blue-train-42-lemon").user
    assert not service.delete_account(user.id, "nope").ok and service.store.count_users() == 1
    assert service.delete_account(user.id, "blue-train-42-lemon").ok and service.store.count_users() == 0
    assert not service.login("alice", "blue-train-42-lemon").ok


def test_admin_reset_and_unlock(service):
    service.register("alice", "blue-train-42-lemon")
    for i in range(3):
        service.login("alice", f"bad-{i}")
    service.admin_unlock("Alice")
    assert service.login("alice", "blue-train-42-lemon").ok
    assert service.admin_set_password("alice", "reset-by-admin-99").ok and service.login("alice", "reset-by-admin-99").ok
    assert not service.admin_set_password("ghost", "reset-by-admin-99").ok


def test_store_survives_reopening_and_usernames_are_unique(tmp_path):
    path = tmp_path / "persist.sqlite3"
    AuthService(UserStore(path)).register("alice", "blue-train-42-lemon")
    reopened = AuthService(UserStore(path))
    assert reopened.login("alice", "blue-train-42-lemon").ok
    assert reopened.store.create_user("alice", "x", 0.0) is None
