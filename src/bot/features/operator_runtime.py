"""Persistent background jobs for workspace operator actions."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Optional

import structlog

logger = structlog.get_logger(__name__)

_ACTIVE_STATUSES = {"starting", "running", "stopping", "verifying"}


@dataclass(frozen=True)
class OperatorJob:
    """Persisted background job metadata."""

    job_id: str
    workspace_root: Path
    action_key: str
    title: str
    command: str
    status: str
    created_at: str
    log_path: Path
    verification_command: Optional[str] = None
    verification_mode: Optional[str] = None
    pid: Optional[int] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    verification_status: Optional[str] = None
    verification_attempts: int = 0
    verification_started_at: Optional[str] = None
    verification_finished_at: Optional[str] = None
    verification_exit_code: Optional[int] = None
    verification_error: Optional[str] = None
    stop_requested_at: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Return True when the job is still running."""
        return self.status in _ACTIVE_STATUSES

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OperatorJob":
        """Deserialize a job from JSON payload."""
        return cls(
            job_id=str(payload["job_id"]),
            workspace_root=Path(str(payload["workspace_root"])).resolve(),
            action_key=str(payload["action_key"]),
            title=str(payload["title"]),
            command=str(payload["command"]),
            status=str(payload["status"]),
            created_at=str(payload["created_at"]),
            log_path=Path(str(payload["log_path"])).resolve(),
            verification_command=(
                str(payload["verification_command"])
                if payload.get("verification_command") is not None
                else None
            ),
            verification_mode=(
                str(payload["verification_mode"])
                if payload.get("verification_mode") is not None
                else None
            ),
            pid=payload.get("pid") if isinstance(payload.get("pid"), int) else None,
            started_at=(
                str(payload["started_at"]) if payload.get("started_at") is not None else None
            ),
            finished_at=(
                str(payload["finished_at"])
                if payload.get("finished_at") is not None
                else None
            ),
            exit_code=(
                payload.get("exit_code")
                if isinstance(payload.get("exit_code"), int)
                else None
            ),
            verification_status=(
                str(payload["verification_status"])
                if payload.get("verification_status") is not None
                else None
            ),
            verification_attempts=(
                int(payload["verification_attempts"])
                if payload.get("verification_attempts") is not None
                else 0
            ),
            verification_started_at=(
                str(payload["verification_started_at"])
                if payload.get("verification_started_at") is not None
                else None
            ),
            verification_finished_at=(
                str(payload["verification_finished_at"])
                if payload.get("verification_finished_at") is not None
                else None
            ),
            verification_exit_code=(
                payload.get("verification_exit_code")
                if isinstance(payload.get("verification_exit_code"), int)
                else None
            ),
            verification_error=(
                str(payload["verification_error"])
                if payload.get("verification_error") is not None
                else None
            ),
            stop_requested_at=(
                str(payload["stop_requested_at"])
                if payload.get("stop_requested_at") is not None
                else None
            ),
            error=str(payload["error"]) if payload.get("error") is not None else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize a job into JSON-friendly primitives."""
        payload = asdict(self)
        payload["workspace_root"] = str(self.workspace_root)
        payload["log_path"] = str(self.log_path)
        return payload


class WorkspaceOperatorRuntime:
    """Manage long-running workspace commands outside the bot process."""

    def __init__(self, state_root: Path):
        self.state_root = state_root.resolve()
        self.jobs_root = self.state_root / "jobs"
        self.logs_root = self.state_root / "logs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self._reconcile_callback: Optional[Callable[[OperatorJob], Awaitable[None]]] = None
        self.reconcile_jobs()

    def set_reconcile_callback(
        self,
        callback: Callable[[OperatorJob], Awaitable[None]],
    ) -> None:
        """Persist or report stale job recovery events."""
        self._reconcile_callback = callback
        for job in self.list_jobs():
            if job.status == "stale":
                self._schedule_reconcile_callback(job)

    async def launch_job(
        self,
        workspace_root: Path,
        action_key: str,
        command: str,
        title: Optional[str] = None,
        verification_command: Optional[str] = None,
        verification_mode: Optional[str] = None,
        verification_delay_seconds: float = 0.0,
        verification_retries: int = 1,
        verification_interval_seconds: float = 0.0,
    ) -> OperatorJob:
        """Launch a background command and persist its state."""
        workspace = workspace_root.resolve()
        active_job = self.get_latest_job(workspace, statuses=_ACTIVE_STATUSES)
        if active_job:
            raise RuntimeError(
                f"Workspace already has an active job: {active_job.action_key} "
                f"({active_job.job_id[:8]})"
            )

        job_id = uuid.uuid4().hex[:12]
        log_path = self.logs_root / f"{job_id}.log"
        job = OperatorJob(
            job_id=job_id,
            workspace_root=workspace,
            action_key=action_key,
            title=title or action_key.replace("_", " ").title(),
            command=command,
            status="starting",
            created_at=self._now(),
            log_path=log_path,
            verification_command=verification_command,
            verification_mode=verification_mode,
        )
        self._write_job(job)
        log_path.write_text(
            (
                f"[{job.created_at}] job={job.job_id} action={job.action_key}\n"
                f"workspace={job.workspace_root}\n"
                f"command={job.command}\n\n"
            ),
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["CLAUDE_OPERATOR_COMMAND"] = command
        env["CLAUDE_OPERATOR_STATE_PATH"] = str(self._job_path(job.job_id))
        env["CLAUDE_OPERATOR_WORKSPACE"] = str(workspace)
        env["CLAUDE_OPERATOR_LOG_PATH"] = str(log_path)
        if verification_command:
            env["CLAUDE_OPERATOR_VERIFY_COMMAND"] = verification_command
        if verification_mode:
            env["CLAUDE_OPERATOR_VERIFY_MODE"] = verification_mode
        env["CLAUDE_OPERATOR_VERIFY_DELAY_SECONDS"] = str(
            verification_delay_seconds
        )
        env["CLAUDE_OPERATOR_VERIFY_RETRIES"] = str(max(1, verification_retries))
        env["CLAUDE_OPERATOR_VERIFY_INTERVAL_SECONDS"] = str(
            max(0.0, verification_interval_seconds)
        )

        runner_path = Path(__file__).with_name("operator_job_runner.py").resolve()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(runner_path),
            cwd=workspace,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )

        running_job = replace(
            job,
            pid=process.pid,
            status="running",
            started_at=self._now(),
        )
        self._write_job(running_job)
        logger.info(
            "Started workspace operator job",
            job_id=running_job.job_id,
            action_key=running_job.action_key,
            workspace_root=str(running_job.workspace_root),
            pid=running_job.pid,
        )
        return running_job

    async def stop_job(self, job_id: str) -> OperatorJob:
        """Request a graceful stop for an active background job."""
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job not found: {job_id}")
        if not job.is_active:
            return job

        stopping_job = replace(
            job,
            status="stopping",
            stop_requested_at=self._now(),
        )
        self._write_job(stopping_job)

        if stopping_job.pid is not None:
            try:
                os.killpg(stopping_job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                os.kill(stopping_job.pid, signal.SIGTERM)

        await asyncio.sleep(0.25)
        self.reconcile_jobs()
        return self.get_job(job_id) or stopping_job

    def get_job(self, job_id: str) -> Optional[OperatorJob]:
        """Return a job by ID after refreshing its state."""
        self.reconcile_jobs()
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return self._read_job(path)

    def get_latest_job(
        self,
        workspace_root: Optional[Path] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> Optional[OperatorJob]:
        """Return the newest job matching the filters."""
        jobs = self.list_jobs(workspace_root=workspace_root, limit=100)
        if statuses is None:
            return jobs[0] if jobs else None
        status_set = set(statuses)
        for job in jobs:
            if job.status in status_set:
                return job
        return None

    def list_jobs(
        self,
        workspace_root: Optional[Path] = None,
        limit: int = 10,
    ) -> list[OperatorJob]:
        """List recent jobs, optionally scoped to one workspace."""
        self.reconcile_jobs()
        workspace = workspace_root.resolve() if workspace_root else None
        jobs = []
        for path in sorted(self.jobs_root.glob("*.json"), reverse=True):
            job = self._read_job(path)
            if workspace is not None and job.workspace_root != workspace:
                continue
            jobs.append(job)
            if len(jobs) >= limit:
                break
        jobs.sort(key=lambda item: item.created_at, reverse=True)
        return jobs

    def read_log_tail(self, job: OperatorJob, limit: int = 1200) -> str:
        """Read the tail of a job log for Telegram output."""
        try:
            content = job.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        compact = content.strip()
        if len(compact) <= limit:
            return compact
        return compact[-limit:]

    def reconcile_jobs(self) -> list[OperatorJob]:
        """Refresh persisted job states after restarts or abrupt exits."""
        reconciled: list[OperatorJob] = []
        for path in self.jobs_root.glob("*.json"):
            job = self._read_job(path)
            if not job.is_active:
                continue
            if job.pid is not None and self._is_process_alive(job.pid):
                continue
            refreshed = replace(
                job,
                status="stopped" if job.stop_requested_at else "stale",
                finished_at=job.finished_at or self._now(),
                verification_error=job.verification_error or job.error,
                error=job.error or "process not running after restart/reconcile",
            )
            self._write_job(refreshed)
            reconciled.append(refreshed)
            self._schedule_reconcile_callback(refreshed)
        return reconciled

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """Check whether a process still exists."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _now() -> str:
        """Return an ISO timestamp in UTC."""
        return datetime.now(UTC).isoformat()

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def _read_job(self, path: Path) -> OperatorJob:
        return OperatorJob.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def _write_job(self, job: OperatorJob) -> None:
        path = self._job_path(job.job_id)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(job.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _schedule_reconcile_callback(self, job: OperatorJob) -> None:
        if not self._reconcile_callback:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._reconcile_callback(job))
