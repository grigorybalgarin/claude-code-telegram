"""Handle video URL processing — extract transcripts from YouTube videos."""

import re
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# YouTube URL patterns
_YOUTUBE_PATTERNS = [
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})"
    ),
    re.compile(r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})"),
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})"
    ),
    re.compile(
        r"(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})"
    ),
]


@dataclass
class VideoTranscript:
    """Result of video transcript extraction."""

    video_id: str
    title: str
    transcript: str
    language: str
    duration_text: str


def extract_youtube_id(text: str) -> Optional[str]:
    """Extract YouTube video ID from a message containing a URL."""
    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


async def get_youtube_transcript(
    video_id: str, proxy_url: Optional[str] = None
) -> VideoTranscript:
    """Fetch transcript for a YouTube video.

    Tries Russian first, then English, then any available language.
    Uses youtube-transcript-api v1.x API.
    """
    import asyncio

    from youtube_transcript_api import YouTubeTranscriptApi

    def _fetch() -> VideoTranscript:
        proxy_config = None
        if proxy_url:
            from youtube_transcript_api.proxies import GenericProxyConfig

            proxy_config = GenericProxyConfig(
                http_url=proxy_url,
                https_url=proxy_url,
            )
        ytt = YouTubeTranscriptApi(proxy_config=proxy_config)

        # Try preferred languages first (ru, en), fall back to any
        fetched = None
        lang = "unknown"

        for langs in [("ru", "en"), None]:
            try:
                if langs:
                    fetched = ytt.fetch(video_id, languages=langs)
                else:
                    # Get any available transcript
                    transcript_list = ytt.list(video_id)
                    available = list(transcript_list)
                    if available:
                        fetched = ytt.fetch(
                            video_id, languages=(available[0].language_code,)
                        )
                if fetched:
                    lang = fetched.language_code
                    break
            except Exception:
                continue

        if fetched is None:
            raise ValueError(
                f"No transcript available for video {video_id}. "
                "The video may not have subtitles."
            )

        parts = []
        total_seconds = 0
        for snippet in fetched.snippets:
            parts.append(snippet.text)
            end = snippet.start + snippet.duration
            if end > total_seconds:
                total_seconds = end

        full_text = " ".join(parts)

        # Format duration
        mins = int(total_seconds) // 60
        secs = int(total_seconds) % 60
        duration_text = f"{mins}:{secs:02d}"

        title = f"YouTube video ({video_id})"

        return VideoTranscript(
            video_id=video_id,
            title=title,
            transcript=full_text,
            language=lang,
            duration_text=duration_text,
        )

    return await asyncio.to_thread(_fetch)
