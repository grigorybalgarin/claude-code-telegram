"""Deterministic shell command execution for agentic workspace operations."""

import asyncio
from pathlib import Path
from typing import List

from ...utils.redaction import redact_sensitive_text
from ..utils.html_format import escape_html
from .context import ShellActionResult


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    return redact_sensitive_text(text)


class ShellExecutor:
    """Execute shell commands in a workspace and capture structured results."""

    @staticmethod
    def tail_output(text: str, limit: int = 900) -> str:
        """Keep only the tail of command output for Telegram summaries."""
        normalized = text.strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[-limit:]

    async def execute(
        self,
        workspace_root: Path,
        command: str,
        timeout_seconds: int = 120,
    ) -> ShellActionResult:
        """Execute a deterministic shell action and capture a redacted result."""
        timed_out = False
        try:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-lc",
                command,
                cwd=workspace_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                timed_out = True
                process.kill()
                stdout, stderr = await process.communicate()
            returncode = process.returncode if process.returncode is not None else -1
            stdout_text = self.tail_output(
                _redact_secrets(stdout.decode("utf-8", errors="replace"))
            )
            stderr_text = self.tail_output(
                _redact_secrets(stderr.decode("utf-8", errors="replace"))
            )
            return ShellActionResult(
                command=command,
                returncode=returncode,
                success=not timed_out and returncode == 0,
                timed_out=timed_out,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
            )
        except Exception as e:
            return ShellActionResult(
                command=command,
                returncode=-1,
                success=False,
                timed_out=False,
                stdout_text="",
                stderr_text="",
                error=str(e),
            )

    @staticmethod
    def format_result_lines(
        title: str,
        workspace_root: Path,
        boundary_root: Path,
        result: ShellActionResult,
    ) -> List[str]:
        """Format a shell action result into Telegram-friendly HTML lines."""
        try:
            relative = workspace_root.resolve().relative_to(boundary_root)
            rel_path = "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            rel_path = str(workspace_root)

        if result.error:
            return [
                "❌ <b>Ошибка команды</b>",
                "",
                f"Действие: <code>{escape_html(title)}</code>",
                f"Ошибка: <code>{escape_html(result.error)}</code>",
            ]

        lines = [
            f"✅ <b>{escape_html(title)}</b>"
            if result.success
            else f"❌ <b>{escape_html(title)}</b>",
            "",
            f"Проект: <code>{escape_html(rel_path)}</code>",
            f"Команда: <code>{escape_html(result.command)}</code>",
            f"Код выхода: <code>{result.returncode}</code>",
        ]
        if result.timed_out:
            lines.append("Вышло время ожидания.")
        if result.stdout_text:
            lines.extend(
                ["", "<b>stdout</b>", f"<pre>{escape_html(result.stdout_text)}</pre>"]
            )
        if result.stderr_text:
            lines.extend(
                ["", "<b>stderr</b>", f"<pre>{escape_html(result.stderr_text)}</pre>"]
            )
        return lines

    @staticmethod
    def summarize(result: ShellActionResult, limit: int = 140) -> str:
        """Build a compact one-line summary of a shell action result."""
        if result.error:
            summary = result.error
        else:
            summary = result.stdout_text or result.stderr_text or ""
        compact = " ".join(summary.split())
        if not compact:
            compact = "нет вывода"
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."
