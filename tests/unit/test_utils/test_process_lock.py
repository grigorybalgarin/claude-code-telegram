"""Tests for single-instance process lock."""

from pathlib import Path

import pytest

from src.utils.process_lock import SingleInstanceLock


def test_single_instance_lock_blocks_second_holder(tmp_path: Path):
    lock_path = tmp_path / "bot.lock"
    first = SingleInstanceLock(lock_path)
    second = SingleInstanceLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release()


def test_single_instance_lock_can_be_reacquired_after_release(tmp_path: Path):
    lock_path = tmp_path / "bot.lock"
    first = SingleInstanceLock(lock_path)
    second = SingleInstanceLock(lock_path)

    first.acquire()
    first.release()

    second.acquire()
    second.release()
