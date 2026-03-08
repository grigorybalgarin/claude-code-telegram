"""Async client for mem0 semantic memory API."""

from typing import Any, Dict, List, Optional

import httpx
import structlog

logger = structlog.get_logger()


class Mem0Client:
    """Thin async wrapper around the mem0 REST API."""

    def __init__(self, base_url: str, default_user_id: str = "grigorybalgarin"):
        self.base_url = base_url.rstrip("/")
        self.default_user_id = default_user_id
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def health(self) -> bool:
        try:
            client = self._get_client()
            resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search for relevant memories."""
        try:
            client = self._get_client()
            payload = {
                "query": query,
                "user_id": user_id or self.default_user_id,
                "limit": limit,
            }
            resp = await client.post(f"{self.base_url}/search", json=payload)
            if resp.status_code == 200:
                return resp.json().get("results", [])
            logger.warning("mem0 search failed", status=resp.status_code)
            return []
        except Exception as e:
            logger.warning("mem0 search error", error=str(e))
            return []

    async def add(
        self,
        text: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Store a new memory. Returns memory_id or None on failure."""
        try:
            client = self._get_client()
            payload = {
                "text": text,
                "user_id": user_id or self.default_user_id,
                "metadata": metadata or {},
            }
            resp = await client.post(f"{self.base_url}/add", json=payload)
            if resp.status_code == 200:
                return resp.json().get("memory_id")
            logger.warning("mem0 add failed", status=resp.status_code)
            return None
        except Exception as e:
            logger.warning("mem0 add error", error=str(e))
            return None

    async def get_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        try:
            client = self._get_client()
            params = {"user_id": user_id or self.default_user_id}
            resp = await client.get(f"{self.base_url}/all", params=params)
            if resp.status_code == 200:
                return resp.json().get("memories", [])
            return []
        except Exception as e:
            logger.warning("mem0 get_all error", error=str(e))
            return []

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def format_memories_for_prompt(
    memories: List[Dict[str, Any]], min_score: float = 0.3
) -> str:
    """Format search results into a context block for Claude prompt."""
    relevant = [m for m in memories if m.get("score", 0) >= min_score]
    if not relevant:
        return ""

    lines = [
        "[Relevant context from your persistent memory (Mem0 + Qdrant, already integrated and active)]:"
    ]
    for m in relevant:
        text = m.get("text", "")
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)
