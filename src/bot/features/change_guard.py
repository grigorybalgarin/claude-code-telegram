"""Automatic checkpoint, verification, and rollback for project workspaces."""

from __future__ import annotations

import asyncio
import json
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.bot.utils.html_format import escape_html


@dataclass(frozen=True)
class GitCheckpoint:
    """Snapshot of a git workspace before Claude edits."""

    checkpoint_id: str
    repo_root: Path
    checkpoint_dir: Path
    head_ref: str
    patch_path: Path
    untracked_archive_path: Optional[Path]
    original_status: str


@dataclass(frozen=True)
class VerificationStepResult:
    """Result of a single verification command."""

    command: str
    success: bool
    returncode: int
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class ChangeGuardReport:
    """Summary of guard actions for a single request."""

    checkpoint_created: bool = False
    checkpoint_id: Optional[str] = None
    verification_results: List[VerificationStepResult] = field(default_factory=list)
    rollback_triggered: bool = False
    rollback_succeeded: bool = False
    failure_reason: Optional[str] = None
    rollback_error: Optional[str] = None


class ProjectChangeGuard:
    """Create git checkpoints, run verification, and rollback on failure."""

    def __init__(self, checkpoint_root: Optional[Path] = None) -> None:
        root = checkpoint_root or Path(tempfile.gettempdir()) / "claude-bot-checkpoints"
        self.checkpoint_root = root

    async def create_checkpoint(self, repo_root: Path) -> Optional[GitCheckpoint]:
        """Create a reversible checkpoint for a git repository."""
        if shutil.which("git") is None:
            return None

        repo_root = repo_root.resolve()
        head_result = await self._run_git(["rev-parse", "--verify", "HEAD"], repo_root)
        if head_result.returncode != 0:
            return None

        checkpoint_id = uuid.uuid4().hex[:12]
        checkpoint_dir = self.checkpoint_root / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        patch_path = checkpoint_dir / "tracked.patch"
        diff_result = await self._run_git(["diff", "--binary", "HEAD", "--"], repo_root)
        patch_path.write_text(diff_result.stdout, encoding="utf-8")

        untracked_result = await self._run_git(
            ["ls-files", "--others", "--exclude-standard", "-z"], repo_root
        )
        untracked_files = [
            Path(item) for item in untracked_result.stdout.split("\0") if item.strip()
        ]
        archive_path: Optional[Path] = None
        if untracked_files:
            archive_path = checkpoint_dir / "untracked.tar"
            with tarfile.open(archive_path, "w") as archive:
                for relative_path in untracked_files:
                    absolute_path = repo_root / relative_path
                    if absolute_path.exists():
                        archive.add(absolute_path, arcname=str(relative_path))

        status_result = await self._run_git(
            ["status", "--short", "--branch"], repo_root
        )
        metadata = {
            "checkpoint_id": checkpoint_id,
            "repo_root": str(repo_root),
            "head_ref": head_result.stdout.strip(),
            "original_status": status_result.stdout,
            "has_patch": patch_path.stat().st_size > 0,
            "has_untracked": bool(untracked_files),
        }
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        return GitCheckpoint(
            checkpoint_id=checkpoint_id,
            repo_root=repo_root,
            checkpoint_dir=checkpoint_dir,
            head_ref=head_result.stdout.strip(),
            patch_path=patch_path,
            untracked_archive_path=archive_path,
            original_status=status_result.stdout,
        )

    async def run_verification_commands(
        self, repo_root: Path, commands: List[str]
    ) -> List[VerificationStepResult]:
        """Run verification commands sequentially from the repository root."""
        results: List[VerificationStepResult] = []
        repo_root = repo_root.resolve()

        for command in commands:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-lc",
                command,
                cwd=repo_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            results.append(
                VerificationStepResult(
                    command=command,
                    success=process.returncode == 0,
                    returncode=process.returncode,
                    stdout_tail=self._tail(stdout.decode("utf-8", errors="replace")),
                    stderr_tail=self._tail(stderr.decode("utf-8", errors="replace")),
                )
            )
            if process.returncode != 0:
                break

        return results

    async def rollback(
        self, checkpoint: GitCheckpoint, reason: str
    ) -> ChangeGuardReport:
        """Rollback the repository to the checkpointed state."""
        report = ChangeGuardReport(
            checkpoint_created=True,
            checkpoint_id=checkpoint.checkpoint_id,
            rollback_triggered=True,
            failure_reason=reason,
        )

        reset_result = await self._run_git(
            ["reset", "--hard", checkpoint.head_ref], checkpoint.repo_root
        )
        clean_result = await self._run_git(["clean", "-fd"], checkpoint.repo_root)
        if reset_result.returncode != 0 or clean_result.returncode != 0:
            report.rollback_error = (
                reset_result.stderr.strip()
                or clean_result.stderr.strip()
                or "reset failed"
            )
            await self.cleanup_checkpoint(checkpoint)
            return report

        if checkpoint.patch_path.exists() and checkpoint.patch_path.stat().st_size > 0:
            apply_result = await self._run_git(
                ["apply", "--whitespace=nowarn", str(checkpoint.patch_path)],
                checkpoint.repo_root,
            )
            if apply_result.returncode != 0:
                report.rollback_error = (
                    apply_result.stderr.strip() or "git apply failed"
                )
                await self.cleanup_checkpoint(checkpoint)
                return report

        if (
            checkpoint.untracked_archive_path
            and checkpoint.untracked_archive_path.exists()
        ):
            with tarfile.open(checkpoint.untracked_archive_path, "r") as archive:
                archive.extractall(checkpoint.repo_root)

        report.rollback_succeeded = True
        await self.cleanup_checkpoint(checkpoint)
        return report

    async def cleanup_checkpoint(self, checkpoint: GitCheckpoint) -> None:
        """Delete checkpoint artifacts once they are no longer needed."""
        shutil.rmtree(checkpoint.checkpoint_dir, ignore_errors=True)

    def format_report_html(self, report: ChangeGuardReport) -> str:
        """Render a compact Telegram-friendly summary."""
        lines = ["🛡️ <b>Auto-Guard</b>"]
        if report.checkpoint_created:
            lines.append(
                f"Checkpoint: <code>{escape_html(report.checkpoint_id or 'created')}</code>"
            )

        if report.verification_results:
            lines.append("")
            lines.append("<b>Verification</b>")
            for result in report.verification_results:
                status = "✅" if result.success else "❌"
                lines.append(f"{status} <code>{escape_html(result.command)}</code>")

        if report.rollback_triggered:
            lines.append("")
            if report.rollback_succeeded:
                lines.append("↩️ Rollback executed automatically.")
            else:
                lines.append(
                    f"❌ Rollback failed: <code>{escape_html(report.rollback_error or 'unknown error')}</code>"
                )

        if report.failure_reason and not report.rollback_triggered:
            lines.append("")
            lines.append(f"Reason: <code>{escape_html(report.failure_reason)}</code>")

        return "\n".join(lines)

    async def _run_git(self, args: List[str], cwd: Path):
        """Run a git subprocess and capture text output."""
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return _ProcessResult(
            returncode=process.returncode,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _tail(text: str, limit: int = 600) -> str:
        """Keep only the tail of command output for summaries."""
        normalized = text.strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[-limit:]


@dataclass(frozen=True)
class _ProcessResult:
    """Internal subprocess result container."""

    returncode: int
    stdout: str
    stderr: str
