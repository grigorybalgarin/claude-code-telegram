"""Tests for video handler feature — YouTube transcript extraction."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.bot.features.video_handler import (
    VideoTranscript,
    extract_youtube_id,
    get_youtube_transcript,
)


# --- extract_youtube_id tests ---


class TestExtractYoutubeId:
    """Tests for YouTube video ID extraction from various URL formats."""

    def test_extract_youtube_id_watch_url(self):
        """Standard watch URL: youtube.com/watch?v=VIDEO_ID."""
        text = "Check this out https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_shorts_url(self):
        """Shorts URL: youtube.com/shorts/VIDEO_ID."""
        text = "New short https://www.youtube.com/shorts/dQw4w9WgXcQ here"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_short_url(self):
        """Short URL: youtu.be/VIDEO_ID."""
        text = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_embed_url(self):
        """Embed URL: youtube.com/embed/VIDEO_ID."""
        text = "embed link: https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_mobile_url(self):
        """Mobile URL: m.youtube.com/watch?v=VIDEO_ID."""
        text = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_no_match(self):
        """Non-YouTube URLs return None."""
        assert extract_youtube_id("https://example.com/video") is None
        assert extract_youtube_id("https://vimeo.com/12345678") is None
        assert extract_youtube_id("just some text without links") is None
        assert extract_youtube_id("") is None

    def test_extract_youtube_id_invalid(self):
        """Malformed YouTube-like URLs that don't contain a valid 11-char ID."""
        # Too short ID
        assert extract_youtube_id("https://www.youtube.com/watch?v=short") is None
        # No ID at all
        assert extract_youtube_id("https://www.youtube.com/watch?v=") is None
        # Truncated domain
        assert extract_youtube_id("https://youtube.com/") is None

    def test_extract_youtube_id_without_protocol(self):
        """URLs without https:// prefix are still matched."""
        text = "www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_youtube_id(text) == "dQw4w9WgXcQ"

    def test_extract_youtube_id_embedded_in_text(self):
        """Video ID is extracted when URL is surrounded by other text."""
        text = "Hey look at this https://youtu.be/abc123DEF_- it's cool"
        assert extract_youtube_id(text) == "abc123DEF_-"


# --- get_youtube_transcript tests ---


def _make_snippet(text: str, start: float, duration: float):
    """Create a mock transcript snippet."""
    s = MagicMock()
    s.text = text
    s.start = start
    s.duration = duration
    return s


def _install_mock_ytt(mp, mock_ytt):
    """Inject a mock YouTubeTranscriptApi class into sys.modules.

    The source imports ``from youtube_transcript_api import YouTubeTranscriptApi``
    inside an ``asyncio.to_thread`` callback, so we fake the whole package.
    """
    mock_api_class = MagicMock(return_value=mock_ytt)
    mock_module = SimpleNamespace(YouTubeTranscriptApi=mock_api_class)
    mp.setitem(sys.modules, "youtube_transcript_api", mock_module)
    return mock_api_class


class TestGetYoutubeTranscript:
    """Tests for YouTube transcript fetching."""

    async def test_get_youtube_transcript_success(self):
        """Successful transcript fetch with preferred language (ru/en)."""
        snippet1 = _make_snippet("Hello", 0.0, 2.0)
        snippet2 = _make_snippet("world", 2.0, 3.0)

        mock_fetched = MagicMock()
        mock_fetched.language_code = "en"
        mock_fetched.snippets = [snippet1, snippet2]

        mock_ytt = MagicMock()
        mock_ytt.fetch = MagicMock(return_value=mock_fetched)

        with pytest.MonkeyPatch.context() as mp:
            _install_mock_ytt(mp, mock_ytt)
            result = await get_youtube_transcript("testVID12345")

        assert isinstance(result, VideoTranscript)
        assert result.video_id == "testVID12345"
        assert result.transcript == "Hello world"
        assert result.language == "en"
        assert result.duration_text == "0:05"
        assert "testVID12345" in result.title

        mock_ytt.fetch.assert_called_once_with(
            "testVID12345", languages=("ru", "en")
        )

    async def test_get_youtube_transcript_language_fallback(self):
        """Falls back to any available language when ru/en are unavailable."""
        snippet = _make_snippet("Bonjour", 0.0, 4.0)

        mock_fetched = MagicMock()
        mock_fetched.language_code = "fr"
        mock_fetched.snippets = [snippet]

        mock_available = MagicMock()
        mock_available.language_code = "fr"

        mock_transcript_list = MagicMock()
        mock_transcript_list.__iter__ = MagicMock(
            return_value=iter([mock_available])
        )

        mock_ytt = MagicMock()
        # First fetch (ru, en) raises, second fetch (fr) succeeds
        mock_ytt.fetch = MagicMock(
            side_effect=[Exception("No transcripts found"), mock_fetched]
        )
        mock_ytt.list = MagicMock(return_value=mock_transcript_list)

        with pytest.MonkeyPatch.context() as mp:
            _install_mock_ytt(mp, mock_ytt)
            result = await get_youtube_transcript("frenchVID123")

        assert result.language == "fr"
        assert result.transcript == "Bonjour"
        assert result.duration_text == "0:04"

    async def test_get_youtube_transcript_no_subtitles(self):
        """Raises ValueError when no transcripts are available at all."""
        mock_ytt = MagicMock()
        mock_ytt.fetch = MagicMock(side_effect=Exception("No transcripts"))
        mock_ytt.list = MagicMock(
            return_value=MagicMock(
                __iter__=MagicMock(return_value=iter([]))
            )
        )

        with pytest.MonkeyPatch.context() as mp:
            _install_mock_ytt(mp, mock_ytt)
            with pytest.raises(ValueError, match="No transcript available"):
                await get_youtube_transcript("noSubsVID123")

    async def test_get_youtube_transcript_duration_format(self):
        """Duration is formatted as M:SS correctly for longer videos."""
        snippet = _make_snippet("Long video content", 0.0, 125.0)

        mock_fetched = MagicMock()
        mock_fetched.language_code = "en"
        mock_fetched.snippets = [snippet]

        mock_ytt = MagicMock()
        mock_ytt.fetch = MagicMock(return_value=mock_fetched)

        with pytest.MonkeyPatch.context() as mp:
            _install_mock_ytt(mp, mock_ytt)
            result = await get_youtube_transcript("longVID12345")

        assert result.duration_text == "2:05"
