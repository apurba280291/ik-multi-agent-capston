"""Account operations with uniform error messages, lockout and timing equalisation.

* Login failures always say "Invalid username or password." - unknown user and wrong password are indistinguishable.
* Failed attempts are counted per submitted username (even if no such account exists), so lockout does not reveal
  which usernames are registered. After ``max_attempts`` failures the name is locked for ``lockout_seconds``.
* An unknown username still pays for one password hash, so response time does not reveal whether it exists.
Passwords are never logged.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from src.auth import passwords
from src.auth.store import UserStore
from src.utils.logging_utils import get_logger

log = get_logger("auth")

INVALID = "Invalid username or password."


@dataclass(frozen=True)
class User:
    id: int
    username: str
    created_at: float = 0.0
    last_login: float | None = None


@dataclass
class AuthResult:
    ok: bool
    message: str = ""
    user: User | None = None


def _user(row) -> User:
    return User(row["id"], row["username"], row["created_at"], row["last_login"])


class AuthService:
    def __init__(
        self,
        store: UserStore,
        max_attempts: int = 5,
        lockout_seconds: float = 15 * 60,
        allow_registration: bool = True,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.max_attempts = max(1, max_attempts)
        self.lockout_seconds = lockout_seconds
        self.allow_registration = allow_registration
        self._clock = clock
        self._dummy_hash: str | None = None

    # -- registration -----------------------------------------------------------------------------
    def register(self, username: str, password: str) -> AuthResult:
        if not self.allow_registration:
            return AuthResult(False, "Registration is disabled. Ask an administrator for an account.")
        name = passwords.normalise_username(username)
        problem = passwords.validate_username(name) or passwords.validate_password(password, name)
        if problem:
            return AuthResult(False, problem)
        row = self.store.create_user(name, passwords.hash_password(password), self._clock())
        if row is None:
            return AuthResult(False, "That username is already taken.")
        log.info("account created: %s", name)
        return AuthResult(True, "Account created.", _user(row))

    # -- login ---------------------------------------------------------------------------------------
    def login(self, username: str, password: str) -> AuthResult:
        name = passwords.normalise_username(username)
        now = self._clock()
        locked_until = self.store.locked_until(name)
        if locked_until > now:
            return AuthResult(False, self._locked_message(locked_until - now))
        row = self.store.get_by_username(name)
        stored = row["password_hash"] if row else self._dummy()
        # Over-long input is rejected before hashing, but still counts as a failure.
        verified = len(password) <= passwords.MAX_LEN and passwords.verify_password(password, stored)
        if row is not None and verified:
            self.store.clear_attempts(name)
            self.store.touch_login(row["id"], now)
            log.info("login ok: %s", name)
            return AuthResult(True, "", _user(self.store.get_by_id(row["id"])))
        locked = self.store.record_failure(name, now, self.max_attempts, self.lockout_seconds)
        log.warning("login failed: %s%s", name[:32], " (now locked)" if locked else "")
        return AuthResult(False, self._locked_message(self.lockout_seconds) if locked else INVALID)

    def _dummy(self) -> str:
        if self._dummy_hash is None:
            self._dummy_hash = passwords.hash_password("dummy-password-for-timing")
        return self._dummy_hash

    @staticmethod
    def _locked_message(seconds: float) -> str:
        minutes = max(1, math.ceil(seconds / 60))
        return f"Too many failed attempts. Try again in {minutes} minute{'s' if minutes != 1 else ''}."

    # -- account management ---------------------------------------------------------------------------
    def change_password(self, user_id: int, old_password: str, new_password: str) -> AuthResult:
        row = self.store.get_by_id(user_id)
        if row is None or not passwords.verify_password(old_password, row["password_hash"]):
            return AuthResult(False, "Current password is incorrect.")
        problem = passwords.validate_password(new_password, row["username"])
        if problem:
            return AuthResult(False, problem)
        self.store.set_password_hash(user_id, passwords.hash_password(new_password))
        log.info("password changed: %s", row["username"])
        return AuthResult(True, "Password updated.")

    def delete_account(self, user_id: int, password: str) -> AuthResult:
        row = self.store.get_by_id(user_id)
        if row is None or not passwords.verify_password(password, row["password_hash"]):
            return AuthResult(False, "Password is incorrect.")
        self.store.delete_user(user_id)
        log.info("account deleted: %s", row["username"])
        return AuthResult(True, "Account deleted.")

    # -- administration (used by scripts/manage_users.py) ----------------------------------------------
    def admin_set_password(self, username: str, new_password: str) -> AuthResult:
        name = passwords.normalise_username(username)
        row = self.store.get_by_username(name)
        if row is None:
            return AuthResult(False, "No such user.")
        problem = passwords.validate_password(new_password, name)
        if problem:
            return AuthResult(False, problem)
        self.store.set_password_hash(row["id"], passwords.hash_password(new_password))
        self.store.clear_attempts(name)
        return AuthResult(True, "Password updated.")

    def admin_unlock(self, username: str) -> None:
        self.store.clear_attempts(passwords.normalise_username(username))
