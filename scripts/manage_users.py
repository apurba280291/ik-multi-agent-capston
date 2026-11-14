"""Administer Finnie accounts from the command line (works with auth.allow_registration: false).

    python scripts/manage_users.py create alice          # prompts for the password (never on the command line)
    python scripts/manage_users.py list
    python scripts/manage_users.py reset-password alice
    python scripts/manage_users.py unlock alice          # clear a lockout after too many failed logins
    python scripts/manage_users.py delete alice
"""
from __future__ import annotations

import argparse
import getpass
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth import passwords  # noqa: E402
from src.auth.service import AuthService  # noqa: E402
from src.auth.store import UserStore  # noqa: E402
from src.core.config import load_settings  # noqa: E402


def ask_password(confirm: bool = True) -> str:
    password = getpass.getpass("Password: ")
    if confirm and getpass.getpass("Repeat password: ") != password:
        sys.exit("Passwords do not match.")
    return password


def stamp(value) -> str:
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M") if value else "never"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["create", "list", "reset-password", "unlock", "delete"])
    ap.add_argument("username", nargs="?")
    args = ap.parse_args()
    if args.action != "list" and not args.username:
        ap.error(f"'{args.action}' needs a username")

    settings = load_settings()
    store = UserStore(settings.resolve_path(settings.get("auth.db_path", "data/users.sqlite3")))
    service = AuthService(store, allow_registration=True)  # the CLI is the admin path, so it ignores allow_registration
    name = passwords.normalise_username(args.username or "")

    if args.action == "list":
        rows = store.list_users()
        print(f"{len(rows)} account(s) in {store.path}")
        for r in rows:
            print(f"  {r['id']:>3}  {r['username']:<32} created {stamp(r['created_at'])}  last login {stamp(r['last_login'])}")
    elif args.action == "create":
        result = service.register(name, ask_password())
        sys.exit(0 if print(result.message) is None and result.ok else 1)
    elif args.action == "reset-password":
        result = service.admin_set_password(name, ask_password())
        sys.exit(0 if print(result.message) is None and result.ok else 1)
    elif args.action == "unlock":
        service.admin_unlock(name)
        print(f"Cleared failed-login lockout for {name}.")
    elif args.action == "delete":
        row = store.get_by_username(name)
        if row is None:
            sys.exit("No such user.")
        if input(f"Delete account '{name}'? Type the username to confirm: ").strip().lower() != name:
            sys.exit("Cancelled.")
        store.delete_user(row["id"])
        print("Deleted.")


if __name__ == "__main__":
    main()
