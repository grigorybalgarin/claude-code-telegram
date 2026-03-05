"""Handle video notes (circles) and videos — extract frames for Claude analysis."""

import asyncio
import base64
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import structlog
from telegram import Video, VideoNote

logger = structlog.get_logger(__name__)

# Max frames to extract
MAX_FRAMES = 6
# Max dimension for frame resize
FRAME_MAX_SIZE = 1280
# Max video file size (20MB)
MAX_VIDEO_SIZE = 20 * 1024 * 1024


@dataclass
class ExtractedFrames:
    """Result of video frame extraction."""

    frames_base64: List[str]
    frame_count: int
    duration: int
    prompt: str
    temp_dir: Optional[str] = None


async def extract_frames_from_video(
    video: Union[VideoNote, Video],
    caption: Optional[str] = None,
) -> ExtractedFrames:
    """Download video and extract key frames as base64 JPEG images.

    1. Download video from Telegram
    2. Extract frames with ffmpeg at equal intervals
    3. Encode frames as base64
    4. Clean up temp files
    """
    # Check file size
    file_size = getattr(video, "file_size", None)
    if isinstance(file_size, int) and file_size > MAX_VIDEO_SIZE:
        raise ValueError(
            f"Video too large ({file_size / 1024 / 1024:.1f}MB). "
            f"Max: {MAX_VIDEO_SIZE // 1024 // 1024}MB."
        )

    duration = getattr(video, "duration", 0) or 0

    # Download video
    tg_file = await video.get_file()
    video_bytes = bytes(await tg_file.download_as_bytearray())

    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise ValueError(
            f"Video too large ({len(video_bytes) / 1024 / 1024:.1f}MB). "
            f"Max: {MAX_VIDEO_SIZE // 1024 // 1024}MB."
        )

    logger.info(
        "Extracting frames from video",
        file_size=len(video_bytes),
        duration=duration,
        is_video_note=isinstance(video, VideoNote),
    )

    frames_b64 = await _extract_frames(video_bytes, duration)

    if not frames_b64:
        raise ValueError("Failed to extract any frames from the video.")

    # Build prompt
    label = caption or ""
    is_circle = isinstance(video, VideoNote)
    video_type = "video circle" if is_circle else "video"

    prompt_parts = []
    if label:
        prompt_parts.append(label)
    prompt_parts.append(
        f"[{len(frames_b64)} frames extracted from a {video_type}, "
        f"duration: {duration}s]"
    )

    return ExtractedFrames(
        frames_base64=frames_b64,
        frame_count=len(frames_b64),
        duration=duration,
        prompt="\n".join(prompt_parts),
    )


async def _extract_frames(video_bytes: bytes, duration: int) -> List[str]:
    """Extract frames using ffmpeg and return as base64 strings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_file = tmp_path / "input.mp4"
        video_file.write_bytes(video_bytes)

        # Calculate fps filter for even distribution
        if duration > 0:
            # Extract MAX_FRAMES evenly spread across the video
            interval = max(duration / MAX_FRAMES, 0.5)
            fps_filter = f"fps=1/{interval}"
        else:
            fps_filter = "fps=0.5"

        output_pattern = str(tmp_path / "frame_%03d.jpg")

        cmd = [
            "ffmpeg",
            "-i", str(video_file),
            "-vf", f"{fps_filter},scale='min({FRAME_MAX_SIZE},iw)':'-1'",
            "-q:v", "2",
            "-frames:v", str(MAX_FRAMES),
            "-y",
            output_pattern,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode != 0:
            logger.warning(
                "ffmpeg failed",
                returncode=proc.returncode,
                stderr=stderr.decode()[-500:],
            )
            raise ValueError("Failed to extract frames from video (ffmpeg error).")

        # Collect frame files sorted by name
        frame_files = sorted(tmp_path.glob("frame_*.jpg"))

        if not frame_files:
            raise ValueError("ffmpeg produced no frames.")

        frames_b64 = []
        for frame_file in frame_files[:MAX_FRAMES]:
            frame_bytes = frame_file.read_bytes()
            frames_b64.append(base64.b64encode(frame_bytes).decode("utf-8"))

        logger.info("Frames extracted", count=len(frames_b64))
        return frames_b64
