"""Tests for automatic checkpoint, verification, and rollback."""

import shutil
import subprocess
import sys

import pytest

from src.bot.features.change_guard import ProjectChangeGuard


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for change guard tests"
)


def _run_git(tmp_path, *args):
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


@pytest.mark.asyncio
async def test_checkpoint_and_rollback_restores_dirty_state(tmp_path):
    """Rollback should restore both tracked and untracked pre-run state."""
    _run_git(tmp_path, "init")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test")

    tracked = tmp_path / "app.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    _run_git(tmp_path, "add", "app.txt")
    _run_git(tmp_path, "commit", "-m", "init")

    tracked.write_text("before checkpoint\n", encoding="utf-8")
    untracked = tmp_path / "notes.txt"
    untracked.write_text("remember me\n", encoding="utf-8")

    guard = ProjectChangeGuard()
    checkpoint = await guard.create_checkpoint(tmp_path)

    assert checkpoint is not None

    tracked.write_text("after autopilot\n", encoding="utf-8")
    untracked.unlink()

    report = await guard.rollback(checkpoint, reason="verification failed")

    assert report.rollback_succeeded is True
    assert tracked.read_text(encoding="utf-8") == "before checkpoint\n"
    assert untracked.read_text(encoding="utf-8") == "remember me\n"


@pytest.mark.asyncio
async def test_run_verification_commands_stops_on_failure(tmp_path):
    """Verification should stop after the first failing command."""
    guard = ProjectChangeGuard()
    commands = [
        f"{sys.executable} -c \"print('ok')\"",
        f"{sys.executable} -c \"import sys; sys.exit(2)\"",
        f"{sys.executable} -c \"print('should not run')\"",
    ]

    results = await guard.run_verification_commands(tmp_path, commands)

    assert len(results) == 2
    assert results[0].success is True
    assert results[1].success is False
    assert results[1].returncode == 2
