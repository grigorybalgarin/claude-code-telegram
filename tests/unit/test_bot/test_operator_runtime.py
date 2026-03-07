"""Tests for persistent workspace operator background jobs."""

import asyncio

import pytest

from src.bot.features.operator_runtime import WorkspaceOperatorRuntime


async def _wait_for_terminal_status(
    runtime: WorkspaceOperatorRuntime,
    job_id: str,
    timeout: float = 5.0,
):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = runtime.get_job(job_id)
        if job and not job.is_active:
            return job
        await asyncio.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not finish in time")


@pytest.mark.asyncio
async def test_launch_job_persists_completion_and_log_tail(tmp_path):
    """Background jobs should persist completion status and log output."""
    runtime = WorkspaceOperatorRuntime(tmp_path / "operator_runtime")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    job = await runtime.launch_job(
        workspace_root=workspace_root,
        action_key="build",
        command="python3 -c \"print('build ok')\"",
        title="Build",
    )
    completed = await _wait_for_terminal_status(runtime, job.job_id)

    assert completed.status == "succeeded"
    assert completed.exit_code == 0
    assert "build ok" in runtime.read_log_tail(completed)


@pytest.mark.asyncio
async def test_stop_job_marks_background_process_stopped(tmp_path):
    """Stopping a running job should persist the stopped state."""
    runtime = WorkspaceOperatorRuntime(tmp_path / "operator_runtime")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    job = await runtime.launch_job(
        workspace_root=workspace_root,
        action_key="dev",
        command="python3 -c \"import time; print('running'); time.sleep(10)\"",
        title="Dev",
    )
    await asyncio.sleep(0.2)
    stopped = await runtime.stop_job(job.job_id)
    if stopped.is_active:
        stopped = await _wait_for_terminal_status(runtime, job.job_id)

    assert stopped.status == "stopped"
    assert stopped.stop_requested_at is not None


@pytest.mark.asyncio
async def test_launch_job_blocks_second_active_job_in_same_workspace(tmp_path):
    """Only one active background job should run per workspace."""
    runtime = WorkspaceOperatorRuntime(tmp_path / "operator_runtime")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    job = await runtime.launch_job(
        workspace_root=workspace_root,
        action_key="start",
        command="python3 -c \"import time; time.sleep(10)\"",
        title="Start",
    )

    with pytest.raises(RuntimeError):
        await runtime.launch_job(
            workspace_root=workspace_root,
            action_key="deploy",
            command="python3 -c \"print('deploy')\"",
            title="Deploy",
        )

    stopped = await runtime.stop_job(job.job_id)
    if stopped.is_active:
        await _wait_for_terminal_status(runtime, job.job_id)
