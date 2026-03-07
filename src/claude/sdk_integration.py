"""Claude Code Python SDK integration."""

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ProcessError,
    ResultMessage,
    ToolPermissionContext,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import ThinkingConfigAdaptive, ThinkingConfigEnabled
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk.types import StreamEvent

from ..config.settings import Settings
from ..security.validators import SecurityValidator
from ..utils.redaction import redact_sensitive_text
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)
from .monitor import _is_claude_internal_path, check_bash_directory_boundary

logger = structlog.get_logger()

_CLAUDE_STDERR_NOISE_PATTERNS = (
    re.compile(r"^DEBUG MODEL CHECK:", re.IGNORECASE),
    re.compile(r"^DEBUG_CMD_JSON:", re.IGNORECASE),
)


@dataclass
class ClaudeResponse:
    """Response from Claude Code SDK."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """Streaming update from Claude SDK."""

    type: str  # 'assistant', 'user', 'system', 'result', 'stream_delta'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


def _make_can_use_tool_callback(
    security_validator: SecurityValidator,
    working_directory: Path,
    approved_directory: Path,
) -> Any:
    """Create a can_use_tool callback for SDK-level tool permission validation.

    The callback validates file path boundaries and bash directory boundaries
    *before* the SDK executes the tool, providing preventive security enforcement.
    """
    _FILE_TOOLS = {"Write", "Edit", "Read", "create_file", "edit_file", "read_file"}
    _BASH_TOOLS = {"Bash", "bash", "shell"}

    async def can_use_tool(
        tool_name: str,
        tool_input: Dict[str, Any],
        context: ToolPermissionContext,
    ) -> Any:
        # File path validation
        if tool_name in _FILE_TOOLS:
            file_path = tool_input.get("file_path") or tool_input.get("path")
            if file_path:
                # Allow Claude Code internal paths (~/.claude/plans/, etc.)
                if _is_claude_internal_path(file_path):
                    return PermissionResultAllow()

                valid, _resolved, error = security_validator.validate_path(
                    file_path, working_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied file operation",
                        tool_name=tool_name,
                        file_path=file_path,
                        error=error,
                    )
                    return PermissionResultDeny(message=error or "Invalid file path")

        # Bash directory boundary validation
        if tool_name in _BASH_TOOLS:
            command = tool_input.get("command", "")
            if command:
                valid, error = check_bash_directory_boundary(
                    command, working_directory, approved_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied bash command",
                        tool_name=tool_name,
                        command=command,
                        error=error,
                    )
                    return PermissionResultDeny(
                        message=error or "Bash directory boundary violation"
                    )

        return PermissionResultAllow()

    return can_use_tool


class ClaudeSDKManager:
    """Manage Claude Code SDK integration."""

    # Class-level caches: path -> (mtime, content)
    _claude_md_cache: Dict[str, tuple] = {}
    _mcp_config_cache: Dict[str, tuple] = {}

    def __init__(
        self,
        config: Settings,
        security_validator: Optional[SecurityValidator] = None,
    ):
        """Initialize SDK manager with configuration."""
        self.config = config
        self.security_validator = security_validator

        # Set up environment for Claude Code SDK if API key is provided
        # If no API key is provided, the SDK will use existing CLI authentication
        if config.anthropic_api_key_str:
            os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key_str
            logger.info("Using provided API key for Claude SDK authentication")
        else:
            logger.info("No API key provided, using existing Claude CLI authentication")

    @staticmethod
    def _sanitize_stderr_line(line: str) -> Optional[str]:
        """Return a safe stderr line to keep for diagnostics."""
        sanitized = redact_sensitive_text(line, max_length=500).strip()
        if not sanitized:
            return None

        if any(pattern.search(sanitized) for pattern in _CLAUDE_STDERR_NOISE_PATTERNS):
            return None

        return sanitized

    @classmethod
    def _summarize_captured_stderr(cls, stderr_lines: List[str]) -> str:
        """Condense stderr lines into a short, deduplicated excerpt."""
        if not stderr_lines:
            return ""

        deduped: list[str] = []
        for line in stderr_lines:
            if deduped and deduped[-1] == line:
                continue
            deduped.append(line)

        return redact_sensitive_text("\n".join(deduped[-5:]), max_length=800)

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
        images: Optional[List[dict]] = None,
    ) -> ClaudeResponse:
        """Execute Claude Code command via SDK."""
        start_time = asyncio.get_event_loop().time()

        logger.info(
            "Starting Claude SDK command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            # Capture stderr from Claude CLI for better error diagnostics
            from collections import deque

            stderr_lines: deque[str] = deque(maxlen=12)

            def _stderr_callback(line: str) -> None:
                sanitized_line = self._sanitize_stderr_line(line)
                if not sanitized_line:
                    return

                stderr_lines.append(sanitized_line)
                if self.config.debug:
                    logger.debug("Claude CLI stderr", line=sanitized_line)

            # Build system prompt, loading CLAUDE.md from working directory if present
            base_prompt = self._build_system_prompt(working_directory)
            claude_md_path = Path(working_directory) / "CLAUDE.md"
            if claude_md_path.exists():
                cache_key = str(claude_md_path)
                current_mtime = claude_md_path.stat().st_mtime
                cached = ClaudeSDKManager._claude_md_cache.get(cache_key)
                if cached and cached[0] == current_mtime:
                    base_prompt += "\n\n" + cached[1]
                    logger.debug(
                        "Using cached CLAUDE.md content",
                        path=cache_key,
                    )
                else:
                    content = claude_md_path.read_text(encoding="utf-8")
                    ClaudeSDKManager._claude_md_cache[cache_key] = (
                        current_mtime,
                        content,
                    )
                    base_prompt += "\n\n" + content
                    logger.info(
                        "Loaded CLAUDE.md into system prompt",
                        path=cache_key,
                    )

            # When DISABLE_TOOL_VALIDATION=true, pass None for allowed/disallowed
            # tools so the SDK does not restrict tool usage (e.g. MCP tools).
            if self.config.disable_tool_validation:
                sdk_allowed_tools = None
                sdk_disallowed_tools = None
            else:
                sdk_allowed_tools = self.config.claude_allowed_tools
                sdk_disallowed_tools = self.config.claude_disallowed_tools

            # Build Claude Agent options
            options = ClaudeAgentOptions(
                max_turns=self.config.claude_max_turns or None,
                model=self.config.claude_model or None,
                max_budget_usd=self.config.claude_max_cost_per_request or None,
                cwd=str(working_directory),
                allowed_tools=sdk_allowed_tools,
                disallowed_tools=sdk_disallowed_tools,
                cli_path=self.config.claude_cli_path or None,
                include_partial_messages=stream_callback is not None,
                sandbox={
                    "enabled": self.config.sandbox_enabled,
                    "autoAllowBashIfSandboxed": True,
                    "excludedCommands": self.config.sandbox_excluded_commands or [],
                },
                system_prompt=base_prompt,
                setting_sources=["project"],
                stderr=_stderr_callback,
                thinking=(
                    ThinkingConfigEnabled(
                        type="enabled",
                        budget_tokens=self.config.claude_max_thinking_tokens,
                    )
                    if self.config.claude_thinking
                    and self.config.claude_max_thinking_tokens > 0
                    else ThinkingConfigAdaptive(type="adaptive")
                    if self.config.claude_thinking
                    else None
                ),
            )

            # Pass MCP server configuration if enabled
            if self.config.enable_mcp and self.config.mcp_config_path:
                options.mcp_servers = self._load_mcp_config(self.config.mcp_config_path)
                logger.info(
                    "MCP servers configured",
                    mcp_config_path=str(self.config.mcp_config_path),
                )

            # Wire can_use_tool callback for preventive tool validation
            if self.security_validator:
                options.can_use_tool = _make_can_use_tool_callback(
                    security_validator=self.security_validator,
                    working_directory=working_directory,
                    approved_directory=self.config.approved_directory,
                )

            # Resume previous session if we have a session_id
            if session_id and continue_session:
                options.resume = session_id
                logger.info(
                    "Resuming previous session",
                    session_id=session_id,
                )

            # Collect messages via ClaudeSDKClient
            messages: List[Message] = []

            async def _run_client() -> None:
                # Use connect(None) + query(prompt) pattern because
                # can_use_tool requires the prompt as AsyncIterable, not
                # a plain string. connect(None) uses an empty async
                # iterable internally, satisfying the requirement.
                client = ClaudeSDKClient(options)
                try:
                    await client.connect()
                    if self.config.claude_model:
                        await client.set_model(self.config.claude_model)
                    # Build multimodal content if images are provided
                    if images:
                        content: list = []
                        for img in images:
                            content.append(img)
                        content.append({"type": "text", "text": prompt})

                        async def _multimodal_iter():  # type: ignore[override]
                            yield {
                                "type": "user",
                                "message": {
                                    "role": "user",
                                    "content": content,
                                },
                                "parent_tool_use_id": None,
                            }

                        await client.query(_multimodal_iter())
                    else:
                        await client.query(prompt)

                    # Iterate over raw messages and parse them ourselves
                    # so that MessageParseError (e.g. from rate_limit_event)
                    # doesn't kill the underlying async generator. When
                    # parse_message raises inside the SDK's receive_messages()
                    # generator, Python terminates that generator permanently,
                    # causing us to lose all subsequent messages including
                    # the ResultMessage.
                    async for raw_data in client._query.receive_messages():
                        try:
                            message = parse_message(raw_data)
                        except MessageParseError as e:
                            logger.debug(
                                "Skipping unparseable message",
                                error=str(e),
                            )
                            continue

                        messages.append(message)

                        if isinstance(message, ResultMessage):
                            break

                        # Handle streaming callback
                        if stream_callback:
                            try:
                                await self._handle_stream_message(
                                    message, stream_callback
                                )
                            except Exception as callback_error:
                                logger.warning(
                                    "Stream callback failed",
                                    error=str(callback_error),
                                    error_type=type(callback_error).__name__,
                                )
                finally:
                    await client.disconnect()

            # Execute with timeout
            await asyncio.wait_for(
                _run_client(),
                timeout=self.config.claude_timeout_seconds,
            )

            # Extract cost, tools, and session_id from result message
            cost = 0.0
            tools_used: List[Dict[str, Any]] = []
            claude_session_id = None
            result_content = None
            for message in messages:
                if isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                    claude_session_id = getattr(message, "session_id", None)
                    result_content = getattr(message, "result", None)
                    current_time = asyncio.get_event_loop().time()
                    for msg in messages:
                        if isinstance(msg, AssistantMessage):
                            msg_content = getattr(msg, "content", [])
                            if msg_content and isinstance(msg_content, list):
                                for block in msg_content:
                                    if isinstance(block, ToolUseBlock):
                                        tools_used.append(
                                            {
                                                "name": getattr(
                                                    block, "name", "unknown"
                                                ),
                                                "timestamp": current_time,
                                                "input": getattr(block, "input", {}),
                                            }
                                        )
                    break

            # Fallback: extract session_id from StreamEvent messages if
            # ResultMessage didn't provide one (can happen with some CLI versions)
            if not claude_session_id:
                for message in messages:
                    msg_session_id = getattr(message, "session_id", None)
                    if msg_session_id and not isinstance(message, ResultMessage):
                        claude_session_id = msg_session_id
                        logger.info(
                            "Got session ID from stream event (fallback)",
                            session_id=claude_session_id,
                        )
                        break

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Use Claude's session_id if available, otherwise fall back
            final_session_id = claude_session_id or session_id or ""

            if claude_session_id and claude_session_id != session_id:
                logger.info(
                    "Got session ID from Claude",
                    claude_session_id=claude_session_id,
                    previous_session_id=session_id,
                )

            # Use ResultMessage.result if available, fall back to message extraction
            if result_content is not None:
                content = result_content
            else:
                content_parts = []
                for msg in messages:
                    if isinstance(msg, AssistantMessage):
                        msg_content = getattr(msg, "content", [])
                        if msg_content and isinstance(msg_content, list):
                            for block in msg_content:
                                if hasattr(block, "text"):
                                    content_parts.append(block.text)
                        elif msg_content:
                            content_parts.append(str(msg_content))
                content = "\n".join(content_parts)

            return ClaudeResponse(
                content=content,
                session_id=final_session_id,
                cost=cost,
                duration_ms=duration_ms,
                num_turns=len(
                    [
                        m
                        for m in messages
                        if isinstance(m, (UserMessage, AssistantMessage))
                    ]
                ),
                tools_used=tools_used,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Claude SDK command timed out",
                timeout_seconds=self.config.claude_timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"Claude SDK timed out after {self.config.claude_timeout_seconds}s"
            )

        except CLINotFoundError as e:
            logger.error("Claude CLI not found", error=str(e))
            error_msg = (
                "Claude Code not found. Please ensure Claude is installed:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "If already installed, try one of these:\n"
                "  1. Add Claude to your PATH\n"
                "  2. Create a symlink: ln -s $(which claude) /usr/local/bin/claude\n"
                "  3. Set CLAUDE_CLI_PATH environment variable"
            )
            raise ClaudeProcessError(error_msg)

        except ProcessError as e:
            error_str = redact_sensitive_text(str(e), max_length=2000)
            # Include captured stderr for better diagnostics
            captured_stderr = self._summarize_captured_stderr(list(stderr_lines))
            if captured_stderr:
                error_str = f"{error_str}\nStderr: {captured_stderr}"
            logger.error(
                "Claude process failed",
                error=error_str,
                exit_code=getattr(e, "exit_code", None),
                stderr_excerpt=captured_stderr or None,
                stderr_line_count=len(stderr_lines),
            )
            # Check if the process error is MCP-related
            if "mcp" in error_str.lower():
                raise ClaudeMCPError(f"MCP server error: {error_str}")
            raise ClaudeProcessError(f"Claude process error: {error_str}")

        except CLIConnectionError as e:
            error_str = redact_sensitive_text(str(e), max_length=2000)
            logger.error("Claude connection error", error=error_str)
            # Check if the connection error is MCP-related
            if "mcp" in error_str.lower() or "server" in error_str.lower():
                raise ClaudeMCPError(f"MCP server connection failed: {error_str}")
            raise ClaudeProcessError(f"Failed to connect to Claude: {error_str}")

        except CLIJSONDecodeError as e:
            error_str = redact_sensitive_text(str(e), max_length=2000)
            logger.error("Claude SDK JSON decode error", error=error_str)
            raise ClaudeParsingError(f"Failed to decode Claude response: {error_str}")

        except ClaudeSDKError as e:
            error_str = redact_sensitive_text(str(e), max_length=2000)
            logger.error("Claude SDK error", error=error_str)
            raise ClaudeProcessError(f"Claude SDK error: {error_str}")

        except Exception as e:
            exceptions = getattr(e, "exceptions", None)
            if exceptions is not None:
                # ExceptionGroup from TaskGroup operations (Python 3.11+)
                logger.error(
                    "Task group error in Claude SDK",
                    error=redact_sensitive_text(str(e), max_length=2000),
                    error_type=type(e).__name__,
                    exception_count=len(exceptions),
                    exceptions=[
                        redact_sensitive_text(str(ex), max_length=1000)
                        for ex in exceptions[:3]
                    ],
                )
                raise ClaudeProcessError(
                    "Claude SDK task error: "
                    f"{redact_sensitive_text(str(exceptions[0] if exceptions else e), max_length=2000)}"
                )

            error_str = redact_sensitive_text(str(e), max_length=2000)
            logger.error(
                "Unexpected error in Claude SDK",
                error=error_str,
                error_type=type(e).__name__,
            )
            raise ClaudeProcessError(f"Unexpected error: {error_str}")

    @staticmethod
    def _build_system_prompt(working_directory: Path) -> str:
        """Build a comprehensive system prompt matching Claude Code quality."""
        return f"""Primary working directory: {working_directory}

You are an expert software engineer with deep knowledge of many programming languages, frameworks, design patterns, and best practices.

# Core Principles

## Doing tasks
- You are highly capable. Users rely on you for complex tasks that would otherwise take too long.
- Do NOT propose changes to code you haven't read. Always read files before modifying them.
- Do not create files unless absolutely necessary. Prefer editing existing files.
- Avoid over-engineering. Only make changes that are directly requested or clearly necessary.
- Don't add features, refactor code, or make "improvements" beyond what was asked.
- Don't add error handling, fallbacks, or validation for scenarios that can't happen.
- Don't create helpers or abstractions for one-time operations.
- Three similar lines of code is better than a premature abstraction.

## Problem solving
- When encountering an obstacle, do NOT brute force. Consider alternative approaches.
- If an API call or test fails, do NOT retry the same action repeatedly. Diagnose the root cause.
- Break complex tasks into steps and execute them methodically.
- If something is ambiguous, make a reasonable assumption and proceed rather than getting stuck.

## Code quality
- Write safe, secure code. No command injection, XSS, SQL injection.
- Keep solutions simple and focused on the current task.
- Only add comments where logic isn't self-evident.
- Don't add docstrings, type annotations, or comments to code you didn't change.

## Tool usage
- Use relative paths.
- Prefer relative paths whenever possible inside the working directory.
- Use Read to read files, not cat/head/tail via Bash.
- Use Grep to search file contents, not grep via Bash.
- Use Glob to find files, not find via Bash.
- Use Write/Edit to modify files, not sed/awk via Bash.
- Reserve Bash for actual system commands and terminal operations.
- When multiple independent operations are needed, run them in parallel.

## Communication
- Be concise. Lead with the answer, not the reasoning.
- Skip filler words and preamble.
- If you can say it in one sentence, don't use three.
- Focus on decisions that need input, status updates, and errors.

## Git safety
- Never force push, reset --hard, or skip hooks unless explicitly asked.
- Prefer new commits over amending existing ones.
- Never commit .env files or credentials.
"""

    async def _handle_stream_message(
        self, message: Message, stream_callback: Callable[[StreamUpdate], None]
    ) -> None:
        """Handle streaming message from claude-agent-sdk."""
        try:
            if isinstance(message, AssistantMessage):
                # Extract content from assistant message
                content = getattr(message, "content", [])
                text_parts = []
                tool_calls = []

                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "input": getattr(block, "input", {}),
                                    "id": getattr(block, "id", None),
                                }
                            )
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)

                if text_parts or tool_calls:
                    update = StreamUpdate(
                        type="assistant",
                        content=("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                    await stream_callback(update)
                elif content:
                    # Fallback for non-list content
                    update = StreamUpdate(
                        type="assistant",
                        content=str(content),
                    )
                    await stream_callback(update)

            elif isinstance(message, StreamEvent):
                event = message.event or {}
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            update = StreamUpdate(
                                type="stream_delta",
                                content=text,
                            )
                            await stream_callback(update)

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if content:
                    update = StreamUpdate(
                        type="user",
                        content=content,
                    )
                    await stream_callback(update)

        except Exception as e:
            logger.warning("Stream callback failed", error=str(e))

    def _load_mcp_config(self, config_path: Path) -> Dict[str, Any]:
        """Load MCP server configuration from a JSON file.

        The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
        Uses mtime-based caching to avoid re-reading/parsing on every call.
        """
        import json

        try:
            cache_key = str(config_path)
            current_mtime = Path(config_path).stat().st_mtime
            cached = ClaudeSDKManager._mcp_config_cache.get(cache_key)
            if cached and cached[0] == current_mtime:
                logger.debug(
                    "Using cached MCP config",
                    path=cache_key,
                )
                return cached[1]

            with open(config_path) as f:
                config_data = json.load(f)
            result = config_data.get("mcpServers", {})
            ClaudeSDKManager._mcp_config_cache[cache_key] = (current_mtime, result)
            return result
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "Failed to load MCP config", path=str(config_path), error=str(e)
            )
            return {}
