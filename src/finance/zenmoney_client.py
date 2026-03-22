"""ZenMoney API client."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import httpx
import structlog

from .models import ZenAccount, ZenDiffResponse, ZenTag, ZenTransaction

logger = structlog.get_logger()


class ZenMoneyClient:
    """Async client for ZenMoney API v8."""

    BASE_URL = "https://api.zenmoney.ru/v8"

    def __init__(self, token: str, timeout: float = 30.0) -> None:
        self._token = token
        self._timeout = timeout
        self._tags_cache: Dict[str, ZenTag] = {}  # id -> ZenTag
        self._tags_by_title: Dict[str, str] = {}  # title.lower() -> id

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._token}",
        }

    async def fetch_diff(
        self, server_timestamp: int = 0
    ) -> ZenDiffResponse:
        """Fetch changes since last sync.

        Args:
            server_timestamp: Last known server timestamp (0 = full sync).

        Returns:
            Parsed diff response with transactions, accounts, tags.
        """
        payload = {
            "currentClientTimestamp": int(time.time()),
            "serverTimestamp": server_timestamp,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.BASE_URL}/diff/",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        result = self._parse_response(data)
        # Update tags cache
        for tag in result.tags:
            self._tags_cache[tag.id] = tag
            self._tags_by_title[tag.title.lower()] = tag.id
        return result

    async def health_check(self) -> bool:
        """Quick check that the token is valid."""
        try:
            result = await self.fetch_diff(server_timestamp=int(time.time()))
            return result.server_timestamp > 0
        except Exception:
            logger.exception("zenmoney_health_check_failed")
            return False

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def update_transactions(
        self, transactions: List[Dict[str, object]]
    ) -> ZenDiffResponse:
        """Push modified transactions to ZenMoney.

        Args:
            transactions: List of transaction dicts with at least 'id'
                and 'changed' fields plus any fields to update.

        Returns:
            Server diff response confirming the sync.
        """
        payload = {
            "currentClientTimestamp": int(time.time()),
            "serverTimestamp": 0,
            "transaction": transactions,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.BASE_URL}/diff/",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        logger.info(
            "zenmoney_transactions_updated",
            count=len(transactions),
        )
        return self._parse_response(data)

    async def set_transaction_category(
        self, transaction_id: str, tag_id: str
    ) -> ZenDiffResponse:
        """Change category (tag) of a single transaction.

        Args:
            transaction_id: ZenMoney transaction ID.
            tag_id: ZenMoney tag ID to assign.

        Returns:
            Server diff response.
        """
        tx_update = {
            "id": transaction_id,
            "tag": [tag_id] if tag_id else [],
            "changed": int(time.time()),
        }
        return await self.update_transactions([tx_update])

    async def bulk_set_categories(
        self, updates: List[Dict[str, str]]
    ) -> ZenDiffResponse:
        """Change categories for multiple transactions at once.

        Args:
            updates: List of {"transaction_id": ..., "tag_id": ...} dicts.

        Returns:
            Server diff response.
        """
        now = int(time.time())
        tx_list = [
            {
                "id": u["transaction_id"],
                "tag": [u["tag_id"]] if u.get("tag_id") else [],
                "changed": now,
            }
            for u in updates
        ]
        return await self.update_transactions(tx_list)

    async def mark_as_self_transfer(
        self,
        transaction_id: str,
        income_account: str,
        outcome_account: str,
    ) -> ZenDiffResponse:
        """Mark transaction as transfer between own accounts.

        This removes it from expense/income calculations in ZenMoney.

        Args:
            transaction_id: ZenMoney transaction ID.
            income_account: Destination account ID.
            outcome_account: Source account ID.
        """
        tx_update = {
            "id": transaction_id,
            "incomeAccount": income_account,
            "outcomeAccount": outcome_account,
            "changed": int(time.time()),
        }
        return await self.update_transactions([tx_update])

    def find_tag_id(self, category_name: str) -> Optional[str]:
        """Find tag ID by category name (case-insensitive).

        Requires tags to be loaded via fetch_diff() first.
        """
        return self._tags_by_title.get(category_name.lower())

    def get_all_tags(self) -> Dict[str, str]:
        """Return {tag_title: tag_id} mapping."""
        return {tag.title: tag.id for tag in self._tags_cache.values()}

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(data: dict) -> ZenDiffResponse:  # type: ignore[type-arg]
        """Parse raw JSON into typed models."""
        transactions = [
            ZenTransaction(
                id=t["id"],
                date=t["date"],
                income=t.get("income", 0),
                outcome=t.get("outcome", 0),
                income_account=t.get("incomeAccount"),
                outcome_account=t.get("outcomeAccount"),
                income_instrument=t.get("incomeInstrument"),
                outcome_instrument=t.get("outcomeInstrument"),
                payee=t.get("payee"),
                comment=t.get("comment"),
                tag_ids=[str(tid) for tid in (t.get("tag") or [])],
                merchant_id=t.get("merchant"),
                outcome_bank_id=t.get("outcomeBankID"),
                income_bank_id=t.get("incomeBankID"),
                source=t.get("source"),
                created=t.get("created"),
                changed=t.get("changed"),
            )
            for t in data.get("transaction", [])
        ]

        accounts = [
            ZenAccount(
                id=a["id"],
                title=a.get("title", ""),
                instrument=a.get("instrument"),
                balance=a.get("balance", 0),
                archive=a.get("archive", False),
                type=a.get("type"),
                company=a.get("company"),
            )
            for a in data.get("account", [])
        ]

        tags = [
            ZenTag(
                id=t["id"],
                title=t.get("title", ""),
                parent=t.get("parent"),
            )
            for t in data.get("tag", [])
        ]

        return ZenDiffResponse(
            server_timestamp=data.get("serverTimestamp", 0),
            transactions=transactions,
            accounts=accounts,
            tags=tags,
        )
