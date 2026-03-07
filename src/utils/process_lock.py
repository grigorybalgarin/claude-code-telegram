"""Single-instance process lock for polling bot runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO

import structlog

logger = structlog.get_logger()

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX platforms
    fcntl = None


class SingleInstanceLock:
    """Prevent multiple bot processes from running on the same host."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._handle: TextIO | None = None

    def acquire(self) -> None:
        """Acquire a non-blocking exclusive lock."""
        if fcntl is None:  # pragma: no cover
            logger.warning(
                "fcntl unavailable; single-instance lock disabled",
                path=str(self.lock_path),
            )
            return

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.lock_path.open("a+", encoding="utf-8")

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"Another bot instance is already running "
                f"(lock: {self.lock_path})"
            ) from exc

        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(str(os.getpid()))
        self._handle.flush()

        logger.info("Acquired single-instance lock", path=str(self.lock_path))

    def release(self) -> None:
        """Release the lock if held."""
        if self._handle is None:
            return

        if fcntl is not None:  # pragma: no branch
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

        logger.info("Released single-instance lock", path=str(self.lock_path))
