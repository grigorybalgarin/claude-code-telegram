"""Shared data structures for agentic execution modules.

Provides AgenticWorkspaceContext as the primary runtime context passed between
orchestrator and extracted modules, eliminating the need to thread individual
parameters or reference back to MessageOrchestrator.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...bot.features.operator_runtime import WorkspaceOperatorRuntime
from ...bot.features.project_automation import ProjectAutomationManager, ProjectProfile


@dataclass(frozen=True)
class ShellActionResult:
    """Structured result of a deterministic shell action."""

    command: str
    returncode: int
    success: bool
    timed_out: bool
    stdout_text: str
    stderr_text: str
    error: Optional[str] = None


@dataclass(frozen=True)
class VerifyStep:
    """One deterministic verification step for the current workspace."""

    label: str
    command: str
    logs_command: Optional[str] = None


@dataclass(frozen=True)
class VerifyReport:
    """Structured result of a full verification run."""

    results: List[tuple]  # List of (label, ShellActionResult)
    failed_step: Optional[VerifyStep]
    logs_result: Optional[ShellActionResult]

    @property
    def success(self) -> bool:
        return self.failed_step is None


@dataclass(frozen=True)
class ResolveResult:
    """Structured result of an autonomous resolve attempt."""

    initial_failure: Optional[VerifyStep]
    claude_response: Optional[Any]  # ClaudeResponse from facade (last attempt)
    final_report: Optional[VerifyReport]
    rollback_report: Optional[Any]  # ChangeGuardReport
    success: bool
    checkpoint_created: bool = False
    error: Optional[str] = None
    attempts: int = 1


@dataclass
class AgenticWorkspaceContext:
    """Runtime context for agentic workspace operations.

    Built by MessageOrchestrator from Telegram context and passed to
    extracted modules so they never need to reference the orchestrator.
    """

    current_directory: Path
    current_workspace: Path
    boundary_root: Path
    project_automation: Optional[ProjectAutomationManager]
    profile: Optional[ProjectProfile]
    operator_runtime: Optional[WorkspaceOperatorRuntime] = None

    # Optional deps injected from bot_data when needed
    claude_integration: Optional[Any] = None
    storage: Optional[Any] = None
    audit_logger: Optional[Any] = None
    change_guard: Optional[Any] = None

    def format_relative_path(self, path: Path) -> str:
        """Format a path relative to boundary_root for display."""
        try:
            relative = path.resolve().relative_to(self.boundary_root)
            return "/" if str(relative) == "." else relative.as_posix()
        except ValueError:
            return str(path)
