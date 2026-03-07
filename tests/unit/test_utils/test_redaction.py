"""Tests for sensitive value redaction helpers."""

from src.utils.redaction import redact_sensitive_text, redact_sensitive_value


def test_redact_sensitive_text_masks_telegram_bot_token_url():
    text = "GET https://api.telegram.org/bot123456:ABCDEFSECRET/sendMessage"

    redacted = redact_sensitive_text(text)

    assert "ABCDEFSECRET" not in redacted
    assert "https://api.telegram.org/bot***" in redacted


def test_redact_sensitive_value_masks_sensitive_keys_recursively():
    payload = {
        "headers": {
            "Authorization": "Bearer secret-token",
            "X-Trace": "ok",
        },
        "command": "echo ok",
    }

    redacted = redact_sensitive_value(payload)

    assert redacted["headers"]["Authorization"] == "***"
    assert redacted["headers"]["X-Trace"] == "ok"
    assert redacted["command"] == "echo ok"


def test_redact_sensitive_text_masks_claude_debug_command_dump():
    text = (
        'DEBUG_CMD_JSON:["/usr/bin/claude","--system-prompt","secret prompt",'
        '"--mcp-config","/root/ClaudeBot/config/mcp.json"]'
    )

    redacted = redact_sensitive_text(text)

    assert redacted == "DEBUG_CMD_JSON:[redacted]"
