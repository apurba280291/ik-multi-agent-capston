"""Local user accounts: password hashing, SQLite user store, and an auth service with lockout."""
from src.auth.service import AuthResult, AuthService, User

__all__ = ["AuthResult", "AuthService", "User"]
