"""Helpers for redacting sensitive values from logs and persisted metadata."""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|cookie|session|bearer)",
    re.IGNORECASE,
)

_TEXT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(https://api\.telegram\.org/bot)(\d{6,}:[^/\s]+)"),
        r"\1***",
    ),
    (
        re.compile(r"(bot\d{6,}:)([^\s/]+)"),
        r"\1***",
    ),
    (
        re.compile(r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"),
        r"\1***",
    ),
    (
        re.compile(r"(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"),
        r"\1***",
    ),
    (
        re.compile(r"(gh[pousr]_[A-Za-z0-9]{5})[A-Za-z0-9]*"),
        r"\1***",
    ),
    (
        re.compile(r"(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"),
        r"\1***",
    ),
    (
        re.compile(r"(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"),
        r"\1***",
    ),
    (
        re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
        r"\1***",
    ),
    (
        re.compile(
            r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
            r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?",
            re.IGNORECASE,
        ),
        r"\1***",
    ),
    (
        re.compile(
            r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
            r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
            r"=)['\"]?[^\s'\"]{8,}['\"]?",
            re.IGNORECASE,
        ),
        r"\1***",
    ),
    (
        re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}", re.IGNORECASE),
        r"\1***",
    ),
    (
        re.compile(r"(Basic )[A-Za-z0-9+/=]{8,}", re.IGNORECASE),
        r"\1***",
    ),
    (
        re.compile(r"://([^:\s]+:)[^@\s]{4,}(@)"),
        r"://\1***\2",
    ),
]


def is_sensitive_key(key: str) -> bool:
    """Return whether a key likely contains sensitive data."""
    return bool(_SENSITIVE_KEY_PATTERN.search(key))


def redact_sensitive_text(text: str, max_length: int | None = None) -> str:
    """Redact likely secrets from text and optionally truncate it."""
    redacted = text
    for pattern, replacement in _TEXT_PATTERNS:
        redacted = pattern.sub(replacement, redacted)

    if max_length is not None and len(redacted) > max_length:
        return redacted[:max_length] + "...<truncated>"
    return redacted


def redact_sensitive_value(value: Any, max_string_length: int | None = 1000) -> Any:
    """Recursively redact secrets from nested values used in logs or storage."""
    if isinstance(value, str):
        return redact_sensitive_text(value, max_length=max_string_length)

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(str(key)):
                result[str(key)] = "***"
            else:
                result[str(key)] = redact_sensitive_value(item, max_string_length)
        return result

    if isinstance(value, list):
        return [redact_sensitive_value(item, max_string_length) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_sensitive_value(item, max_string_length) for item in value)

    if isinstance(value, set):
        return {redact_sensitive_value(item, max_string_length) for item in value}

    return value
