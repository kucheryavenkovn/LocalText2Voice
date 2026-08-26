"""Secret + subtitle-text redaction for diagnostic output.

Logs and JSONL events must never contain:

* API keys, bearer tokens or secret HTTP headers;
* the full user subtitle text or full reference transcript.

This module provides deterministic helpers used by the subprocess diagnostics,
the event sink and the TTS snapshot builder.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

# Keys whose value is always removed from serialised dicts (voice configs,
# subprocess env, headers, ...).
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "token",
    "bearer",
    "authorization",
    "secret",
    "password",
    "passwd",
    "api-key",
    "x-api-key",
)

# A generous but safe upper bound: a single line of subtitle text can be long.
_MAX_TEXT_SNIPPET = 80

# Patterns recognised inside command-line arguments / free-form strings.
# NOTE: deliberately prefix-based — a generic ``[A-Za-z0-9]{32,}`` would also
# mask legitimate SHA-256 hashes, fingerprints and UUIDs we *want* to keep.
_TOKEN_PATTERN = re.compile(
    r"(?i)"
    r"(sk-[A-Za-z0-9_\-]{6,})"  # OpenAI-style keys
    r"|(Bearer\s+[A-Za-z0-9_\-\.=]{6,})"
    r"|(key-[A-Za-z0-9]{6,})"  # ElevenLabs-style
    r"|(AIza[0-9A-Za-z_\-]{20,})"  # Google API keys
)
_URL_CREDENTIAL_PATTERN = re.compile(r"://[^:@/\s]+:[^@/\s]+@")


def redact_text(text: str | None) -> str:
    """Return a *short, safe* representation of free-form text.

    Long subtitle/transcript bodies are truncated and never echoed in full. The
    idea is to give enough context to identify the cue ("Реплика 3…") without
    leaking the full user content.
    """
    if text is None:
        return ""
    cleaned = _TOKEN_PATTERN.sub(_mask_match, str(text))
    cleaned = _URL_CREDENTIAL_PATTERN.sub("://***:***@", cleaned)
    if len(cleaned) > _MAX_TEXT_SNIPPET:
        cleaned = cleaned[:_MAX_TEXT_SNIPPET].rstrip() + "…"
    return cleaned


def sha256_text(text: str | None) -> str:
    """SHA-256 of the full text (stable id without leaking content)."""
    if not text:
        return ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def redact_value(value: Any) -> Any:
    """Recursively redact secrets from a nested structure."""
    if isinstance(value, Mapping):
        return {str(k): _redact_mapped(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cleaned = [redact_value(item) for item in value]
        return type(value)(cleaned) if isinstance(value, tuple) else cleaned
    if isinstance(value, str):
        return redact_text(value)
    return value


def _redact_mapped(key: Any, value: Any) -> Any:
    key_text = str(key).casefold()
    if any(fragment in key_text for fragment in _SECRET_KEY_FRAGMENTS):
        return "***REDACTED***"
    return redact_value(value)


def redact_command(arguments: Iterable[Any]) -> list[str]:
    """Return a command line safe to log.

    Long opaque arguments are masked, embedded ``key=...``/``Authorization``
    fragments are scrubbed and URL credentials removed. Path-like values and
    flags are kept intact so the command remains useful for debugging.
    """
    safe: list[str] = []
    for raw in arguments:
        text = str(raw)
        text = _TOKEN_PATTERN.sub(_mask_match, text)
        text = _URL_CREDENTIAL_PATTERN.sub("://***:***@", text)
        safe.append(text)
    return safe


def _mask_match(match: re.Match[str]) -> str:
    for group in match.groups():
        if group:
            tail = group[-4:] if len(group) > 8 else ""
            return f"***REDACTED***…{tail}"
    return "***REDACTED***"


def safe_voice_snapshot(voice_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of a voice config suitable for logging."""
    cleaned = redact_value(dict(voice_config))
    # Drop full reference transcripts if any engine leaked them into the config.
    for key in ("reference_transcript", "transcript", "text"):
        if isinstance(cleaned, dict) and key in cleaned:
            value = cleaned[key]
            cleaned[key] = sha256_text(value) if isinstance(value, str) else "***"
    return cleaned if isinstance(cleaned, dict) else {}


__all__ = [
    "redact_command",
    "redact_text",
    "redact_value",
    "safe_voice_snapshot",
    "sha256_text",
]
