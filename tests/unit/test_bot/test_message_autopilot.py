"""Tests for classic-mode autopilot workspace routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.handlers.message import handle_text_message
from src.config import create_test_config


@pytest.mark.asyncio
async def test_handle_text_message_switches_workspace_and_resumes_session(tmp_path):
    """Classic mode should persist autopilot workspace switches and reuse that session."""
    settings = create_test_config(approved_directory=str(tmp_path), agentic_mode=False)

    current_root = tmp_path / "ClaudeBot"
    current_root.mkdir()
    target_root = tmp_path / "FreelanceAggregator"
    target_root.mkdir()

    mock_response = MagicMock()
    mock_response.session_id = "session-new"
    mock_response.content = "Done."
    mock_response.tools_used = []

    claude_integration = AsyncMock()
    claude_integration.run_command = AsyncMock(return_value=mock_response)
    claude_integration._find_resumable_session = AsyncMock(
        return_value=SimpleNamespace(session_id="session-resumed")
    )

    autopilot_plan = SimpleNamespace(
        prompt="AUTOPILOT PROMPT",
        workspace_root=target_root,
        workspace_changed=True,
        profile=SimpleNamespace(),
        matched_playbook="test",
        should_checkpoint=False,
        should_verify=False,
        read_only=False,
    )
    project_automation = MagicMock()
    project_automation.build_automation_plan.return_value = autopilot_plan
    change_guard = MagicMock()
    features = SimpleNamespace(
        get_project_automation=lambda: project_automation,
        get_project_change_guard=lambda: change_guard,
    )

    update = MagicMock()
    update.effective_user.id = 123
    update.message.text = "почини тесты в FreelanceAggregator"
    update.message.message_id = 1
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()

    progress_msg = AsyncMock()
    progress_msg.delete = AsyncMock()
    final_msg = AsyncMock()
    update.message.reply_text.side_effect = [progress_msg, final_msg]

    context = MagicMock()
    context.user_data = {
        "current_directory": current_root,
        "claude_session_id": "old-session",
    }
    context.bot_data = {
        "settings": settings,
        "claude_integration": claude_integration,
        "storage": None,
        "rate_limiter": None,
        "audit_logger": None,
        "features": features,
    }

    await handle_text_message(update, context)

    claude_integration._find_resumable_session.assert_awaited_once_with(123, target_root)
    kwargs = claude_integration.run_command.call_args.kwargs
    assert kwargs["prompt"] == "AUTOPILOT PROMPT"
    assert kwargs["working_directory"] == target_root
    assert kwargs["session_id"] == "session-resumed"
    assert context.user_data["current_directory"] == target_root
