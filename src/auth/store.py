"""SQLite persistence for accounts and login-attempt counters (both keyed so unknown usernames are tracked too)."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_login    REAL
);
CREATE TABLE IF NOT EXISTS login_attempts (
    username     TEXT PRIMARY KEY,
    failures     INTEGER NOT NULL DEFAULT 0,
    locked_until REAL NOT NULL DEFAULT 0
);
"""


class UserStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        try:
            os.chmod(self.path, 0o600)  # owner-only where the OS supports it
        except OSError:
            pass

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- users ------------------------------------------------------------------------------------
    def create_user(self, username: str, password_hash: str, now: float) -> sqlite3.Row | None:
        """New row, or None if the username already exists."""
        try:
            with self._conn() as conn:
                cur = conn.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)", (username, password_hash, now))
                return conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
        except sqlite3.IntegrityError:
            return None

    def get_by_username(self, username: str) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def get_by_id(self, user_id: int) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()

    def list_users(self) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute("SELECT * FROM users ORDER BY id").fetchall()

    def count_users(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def set_password_hash(self, user_id: int, password_hash: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))

    def touch_login(self, user_id: int, now: float) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, user_id))

    def delete_user(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    # -- login attempts -----------------------------------------------------------------------------
    def locked_until(self, username: str) -> float:
        with self._conn() as conn:
            row = conn.execute("SELECT locked_until FROM login_attempts WHERE username = ?", (username,)).fetchone()
            return row["locked_until"] if row else 0.0

    def record_failure(self, username: str, now: float, max_attempts: int, lockout_seconds: float) -> bool:
        """Count a failed login; returns True if this failure locked the account."""
        with self._conn() as conn:
            row = conn.execute("SELECT failures, locked_until FROM login_attempts WHERE username = ?", (username,)).fetchone()
            previous = row["failures"] if row and row["locked_until"] <= now else 0  # an expired lock starts a fresh count
            failures = previous + 1
            locked = failures >= max_attempts
            conn.execute(
                "INSERT INTO login_attempts (username, failures, locked_until) VALUES (?, ?, ?) "
                "ON CONFLICT(username) DO UPDATE SET failures = excluded.failures, locked_until = excluded.locked_until",
                (username, 0 if locked else failures, now + lockout_seconds if locked else 0.0),
            )
            return locked

    def clear_attempts(self, username: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM login_attempts WHERE username = ?", (username,))
