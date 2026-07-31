"""API key storage and migration helpers."""

from __future__ import annotations

import os
from pathlib import Path

import keyring

_SERVICE = "GeminiLiveAgent"
_USERNAME = "api_key"


def get_api_key() -> str | None:
    """Return the API key from the environment or OS credential store."""
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        return env_key
    try:
        return keyring.get_password(_SERVICE, _USERNAME)
    except Exception:  # pragma: no cover - keyring failures should not crash the app
        return None


def set_api_key(key: str) -> None:
    """Store the API key in the OS credential store."""
    if not key:
        msg = "API key cannot be empty"
        raise ValueError(msg)
    keyring.set_password(_SERVICE, _USERNAME, key)


def migrate_dotenv() -> bool:
    """Move GOOGLE_API_KEY from a project-root .env file into the credential store."""
    dotenv_path = Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.is_file():
        return False

    content = dotenv_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == "GOOGLE_API_KEY":
            token = value.strip().strip("\"'")
            if token:
                set_api_key(token)
                return True
    return False
