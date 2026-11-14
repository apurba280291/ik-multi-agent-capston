"""Password hashing (scrypt from the standard library) and username/password policy.

Stored format: ``scrypt$N$r$p$<salt b64>$<hash b64>`` so the cost parameters travel with each hash and can be raised later.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re

N, R, P, DKLEN = 2**14, 8, 1, 32  # ~16 MB and ~50 ms per hash: slow enough to hurt guessing, fast enough for a login
SALT_BYTES = 16
MAX_MEMORY = 64 * 1024 * 1024
MIN_LEN, MAX_LEN = 8, 128  # the cap stops absurdly long inputs from being used to burn CPU

_USERNAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")
_COMMON = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890", "qwerty123", "qwertyuiop",
    "iloveyou", "letmein123", "welcome123", "admin123", "abc12345", "11111111", "00000000", "passw0rd", "p@ssw0rd",
}


def normalise_username(username: str) -> str:
    return (username or "").strip().lower()


def validate_username(username: str) -> str | None:
    """Problem with an already-normalised username, or None if it is acceptable."""
    if not _USERNAME.match(username):
        return "Username must be 3-32 characters: lowercase letters, digits, '.', '_' or '-', starting with a letter or digit."
    return None


def validate_password(password: str, username: str = "") -> str | None:
    """Problem with a new password, or None if it is acceptable. Length matters more than character mix."""
    if len(password) < MIN_LEN:
        return f"Password must be at least {MIN_LEN} characters."
    if len(password) > MAX_LEN:
        return f"Password must be at most {MAX_LEN} characters."
    lowered = password.lower()
    if lowered in _COMMON:
        return "That password is too common."
    if username and (lowered == username or username in lowered and len(username) >= 5):
        return "Password must not contain your username."
    if len(set(password)) < 4:
        return "Password is too repetitive."
    return None


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _derive(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=MAX_MEMORY)


def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    return "$".join(["scrypt", str(N), str(R), str(P), _b64(salt), _b64(_derive(password, salt, N, R, P, DKLEN))])


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check; a malformed stored hash simply fails."""
    try:
        scheme, n, r, p, salt, expected = stored.split("$")
        if scheme != "scrypt":
            return False
        expected_raw = base64.b64decode(expected)
        actual = _derive(password, base64.b64decode(salt), int(n), int(r), int(p), len(expected_raw))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected_raw)
