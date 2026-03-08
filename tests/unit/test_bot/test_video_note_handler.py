"""Tests for video note handler feature — frame extraction from videos."""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bot.features.video_note_handler import (
    MAX_VIDEO_SIZE,
    ExtractedFrames,
    _extract_frames,
    extract_frames_from_video,
)


def _mock_video(duration=10, file_size=1024, is_video_note=True):
    """Create a mock Telegram VideoNote or Video object."""
    if is_video_note:
        from telegram import VideoNote

        cls = VideoNote
    else:
        from telegram import Video

        cls = Video

    video = MagicMock(spec=cls)
    video.duration = duration
    video.file_size = file_size

    mock_file = AsyncMock()
    mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x00" * 512))
    video.get_file = AsyncMock(return_value=mock_file)
    return video


# --- extract_frames_from_video tests ---


class TestExtractFramesFromVideo:
    """Tests for the main extract_frames_from_video entry point."""

    async def test_extract_frames_success(self):
        """Successful frame extraction returns ExtractedFrames with base64 data."""
        video = _mock_video(duration=10, file_size=1024, is_video_note=True)

        fake_b64 = [
            base64.b64encode(b"frame1").decode(),
            base64.b64encode(b"frame2").decode(),
            base64.b64encode(b"frame3").decode(),
        ]

        with patch(
            "src.bot.features.video_note_handler._extract_frames",
            new_callable=AsyncMock,
            return_value=fake_b64,
        ):
            result = await extract_frames_from_video(video, caption=None)

        assert isinstance(result, ExtractedFrames)
        assert result.frame_count == 3
        assert result.frames_base64 == fake_b64
        assert result.duration == 10
        assert "video circle" in result.prompt
        assert "3 frames" in result.prompt

    async def test_extract_frames_video_note_vs_video(self):
        """VideoNote uses 'video circle' label, Video uses 'video'."""
        fake_b64 = [base64.b64encode(b"f").decode()]

        # VideoNote -> "video circle"
        video_note = _mock_video(duration=5, file_size=512, is_video_note=True)
        with patch(
            "src.bot.features.video_note_handler._extract_frames",
            new_callable=AsyncMock,
            return_value=fake_b64,
        ):
            result_note = await extract_frames_from_video(video_note)

        assert "video circle" in result_note.prompt
        assert "video," not in result_note.prompt.replace("video circle", "")

        # Video -> "video" (not "video circle")
        video = _mock_video(duration=5, file_size=512, is_video_note=False)
        with patch(
            "src.bot.features.video_note_handler._extract_frames",
            new_callable=AsyncMock,
            return_value=fake_b64,
        ):
            result_video = await extract_frames_from_video(video)

        assert "video circle" not in result_video.prompt
        assert "video" in result_video.prompt

    async def test_extract_frames_with_caption(self):
        """Caption is prepended to the prompt."""
        video = _mock_video(duration=5, file_size=512, is_video_note=True)
        fake_b64 = [base64.b64encode(b"f").decode()]

        with patch(
            "src.bot.features.video_note_handler._extract_frames",
            new_callable=AsyncMock,
            return_value=fake_b64,
        ):
            result = await extract_frames_from_video(
                video, caption="What's happening here?"
            )

        assert result.prompt.startswith("What's happening here?")

    async def test_extract_frames_file_too_large(self):
        """Videos exceeding MAX_VIDEO_SIZE are rejected before download."""
        video = _mock_video(
            duration=60,
            file_size=MAX_VIDEO_SIZE + 1,
            is_video_note=True,
        )

        with pytest.raises(ValueError, match="too large"):
            await extract_frames_from_video(video)

        video.get_file.assert_not_awaited()

    async def test_extract_frames_downloaded_bytes_too_large(self):
        """Videos whose downloaded bytes exceed the limit are rejected."""
        video = _mock_video(duration=10, file_size=1024, is_video_note=True)
        # Actual downloaded bytes are over the limit
        big_payload = bytearray(b"\x00" * (MAX_VIDEO_SIZE + 1))
        mock_file = AsyncMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=big_payload)
        video.get_file = AsyncMock(return_value=mock_file)

        with pytest.raises(ValueError, match="too large"):
            await extract_frames_from_video(video)

    async def test_extract_frames_no_frames_produced(self):
        """Raises ValueError when _extract_frames returns empty list."""
        video = _mock_video(duration=5, file_size=512, is_video_note=True)

        with patch(
            "src.bot.features.video_note_handler._extract_frames",
            new_callable=AsyncMock,
            return_value=[],
        ):
            with pytest.raises(ValueError, match="Failed to extract any frames"):
                await extract_frames_from_video(video)


# --- _extract_frames (ffmpeg subprocess) tests ---


class TestExtractFramesSubprocess:
    """Tests for the internal _extract_frames function using ffmpeg."""

    async def test_extract_frames_ffmpeg_success(self, tmp_path):
        """Successful ffmpeg run produces base64-encoded frames."""
        # We mock create_subprocess_exec and tempfile to control ffmpeg output
        fake_frame_data = b"\xff\xd8\xff\xe0JFIF"  # fake JPEG header bytes

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with (
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ) as mock_exec,
            patch(
                "src.bot.features.video_note_handler.tempfile.TemporaryDirectory"
            ) as mock_tmpdir,
        ):
            # Set up temp directory with fake frame files
            mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create fake frame files that ffmpeg would produce
            for i in range(3):
                frame_path = tmp_path / f"frame_{i + 1:03d}.jpg"
                frame_path.write_bytes(fake_frame_data)

            result = await _extract_frames(b"video-data", duration=10)

        assert len(result) == 3
        for b64_str in result:
            decoded = base64.b64decode(b64_str)
            assert decoded == fake_frame_data

        mock_exec.assert_called_once()
        # Verify ffmpeg was called with expected arguments
        call_args = mock_exec.call_args[0]
        assert call_args[0] == "ffmpeg"

    async def test_extract_frames_timeout(self):
        """TimeoutError from ffmpeg is converted to ValueError."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            with pytest.raises(ValueError, match="timed out"):
                await _extract_frames(b"video-data", duration=5)

        mock_proc.kill.assert_called_once()

    async def test_extract_frames_ffmpeg_error(self):
        """Non-zero ffmpeg return code raises ValueError."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: invalid data"))
        mock_proc.returncode = 1

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ):
            with pytest.raises(ValueError, match="ffmpeg error"):
                await _extract_frames(b"video-data", duration=5)

    async def test_extract_frames_no_output_files(self, tmp_path):
        """ffmpeg succeeds but produces no frame files."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with (
            patch(
                "asyncio.create_subprocess_exec",
                new_callable=AsyncMock,
                return_value=mock_proc,
            ),
            patch(
                "src.bot.features.video_note_handler.tempfile.TemporaryDirectory"
            ) as mock_tmpdir,
        ):
            mock_tmpdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)
            # No frame files created in tmp_path

            with pytest.raises(ValueError, match="no frames"):
                await _extract_frames(b"video-data", duration=5)

    async def test_extract_frames_zero_duration(self):
        """When duration is 0, a fallback fps filter is used."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: no input"))
        mock_proc.returncode = 1

        with patch(
            "asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=mock_proc,
        ) as mock_exec:
            with pytest.raises(ValueError):
                await _extract_frames(b"video-data", duration=0)

        # Verify fps=0.5 fallback was used (duration=0 branch)
        call_args = mock_exec.call_args[0]
        vf_index = list(call_args).index("-vf")
        vf_value = call_args[vf_index + 1]
        assert "fps=0.5" in vf_value
