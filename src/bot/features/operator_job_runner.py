"""Detached runner for persistent workspace operator jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, payload: dict[str, object]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _update_state(path: Path, **changes: object) -> dict[str, object]:
    payload = _read_state(path)
    payload.update(changes)
    _write_state(path, payload)
    return payload


def _log_line(handle, text: str) -> None:
    handle.write(f"[{_now()}] {text}\n")
    handle.flush()


def _run_command(command: str, workspace: Path, log_handle) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        ["/bin/sh", "-lc", command],
        cwd=workspace,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )


def _run_verification(
    state_path: Path,
    workspace: Path,
    log_handle,
    verify_command: str,
    retries: int,
    interval_seconds: float,
) -> tuple[bool, int, int]:
    _update_state(
        state_path,
        verification_status="running",
        verification_started_at=_now(),
        verification_finished_at=None,
        verification_exit_code=None,
        verification_error=None,
    )

    attempts = 0
    exit_code = 1
    for attempt in range(1, retries + 1):
        attempts = attempt
        _update_state(state_path, verification_attempts=attempts)
        _log_line(
            log_handle,
            f"verification attempt {attempt}/{retries}: {verify_command}",
        )
        result = subprocess.run(
            ["/bin/sh", "-lc", verify_command],
            cwd=workspace,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        exit_code = result.returncode
        if exit_code == 0:
            _update_state(
                state_path,
                verification_status="passed",
                verification_attempts=attempts,
                verification_exit_code=exit_code,
                verification_finished_at=_now(),
                verification_error=None,
            )
            return True, exit_code, attempts

        if attempt < retries:
            time.sleep(interval_seconds)

    _update_state(
        state_path,
        verification_status="failed",
        verification_attempts=attempts,
        verification_exit_code=exit_code,
        verification_finished_at=_now(),
        verification_error=f"verification failed after {attempts} attempt(s)",
    )
    return False, exit_code, attempts


def _run_with_while_running_verification(
    state_path: Path,
    workspace: Path,
    log_handle,
    command: str,
    verify_command: str | None,
    verify_delay_seconds: float,
    retries: int,
    interval_seconds: float,
) -> int:
    process = _run_command(command, workspace, log_handle)
    _log_line(log_handle, f"spawned child pid={process.pid}")

    verification_started = False
    if verify_command:
        time.sleep(max(0.0, verify_delay_seconds))
        if process.poll() is None:
            verification_started = True
            _run_verification(
                state_path,
                workspace,
                log_handle,
                verify_command,
                retries,
                interval_seconds,
            )

    exit_code = process.wait()
    if verify_command and not verification_started and exit_code == 0:
        _update_state(state_path, status="verifying")
        _run_verification(
            state_path,
            workspace,
            log_handle,
            verify_command,
            retries,
            interval_seconds,
        )

    return exit_code


def _run_with_after_exit_verification(
    state_path: Path,
    workspace: Path,
    log_handle,
    command: str,
    verify_command: str | None,
    retries: int,
    interval_seconds: float,
) -> tuple[int, bool]:
    process = _run_command(command, workspace, log_handle)
    exit_code = process.wait()
    verification_ok = True
    if exit_code == 0 and verify_command:
        _update_state(state_path, status="verifying")
        verification_ok, _verify_exit_code, _attempts = _run_verification(
            state_path,
            workspace,
            log_handle,
            verify_command,
            retries,
            interval_seconds,
        )
    return exit_code, verification_ok


def main() -> int:
    state_path = Path(os.environ["CLAUDE_OPERATOR_STATE_PATH"])
    workspace = Path(os.environ["CLAUDE_OPERATOR_WORKSPACE"]).resolve()
    log_path = Path(os.environ["CLAUDE_OPERATOR_LOG_PATH"]).resolve()
    command = os.environ["CLAUDE_OPERATOR_COMMAND"]
    verify_command = os.environ.get("CLAUDE_OPERATOR_VERIFY_COMMAND") or None
    verify_mode = os.environ.get("CLAUDE_OPERATOR_VERIFY_MODE") or None
    verify_delay_seconds = float(
        os.environ.get("CLAUDE_OPERATOR_VERIFY_DELAY_SECONDS", "0")
    )
    retries = max(1, int(os.environ.get("CLAUDE_OPERATOR_VERIFY_RETRIES", "1")))
    interval_seconds = max(
        0.0, float(os.environ.get("CLAUDE_OPERATOR_VERIFY_INTERVAL_SECONDS", "0"))
    )

    with log_path.open("a", encoding="utf-8", buffering=1) as raw_handle:
        log_handle = raw_handle
        _log_line(log_handle, f"runner started in {workspace}")

        verification_ok = True
        if verify_mode == "while_running":
            exit_code = _run_with_while_running_verification(
                state_path=state_path,
                workspace=workspace,
                log_handle=log_handle,
                command=command,
                verify_command=verify_command,
                verify_delay_seconds=verify_delay_seconds,
                retries=retries,
                interval_seconds=interval_seconds,
            )
        else:
            exit_code, verification_ok = _run_with_after_exit_verification(
                state_path=state_path,
                workspace=workspace,
                log_handle=log_handle,
                command=command,
                verify_command=verify_command,
                retries=retries,
                interval_seconds=interval_seconds,
            )

    payload = _read_state(state_path)
    stop_requested = bool(payload.get("stop_requested_at"))
    status = (
        "stopped" if stop_requested else ("succeeded" if exit_code == 0 else "failed")
    )
    error = None if status != "failed" else payload.get("error")

    if (
        not stop_requested
        and exit_code == 0
        and verify_command
        and verify_mode == "after_exit"
    ):
        if verification_ok:
            status = "succeeded"
        else:
            status = "failed"
            error = "post-action verification failed"

    if not stop_requested and exit_code != 0 and error is None:
        error = f"command exited with code {exit_code}"

    _update_state(
        state_path,
        status=status,
        exit_code=exit_code,
        finished_at=_now(),
        error=error,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
