"""Finance service — high-level operations over ZenMoney API."""

from __future__ import annotations

import calendar
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

from .categorizer import TransactionCategorizer
from .models import ZenAccount, ZenDiffResponse, ZenTag, ZenTransaction
from .zenmoney_client import ZenMoneyClient

logger = structlog.get_logger()


class FinanceService:
    """High-level finance operations combining ZenMoney API + categorizer."""

    def __init__(self, token: str, rules_path: Optional[Path] = None) -> None:
        self._client = ZenMoneyClient(token)
        self._rules_path = rules_path
        self._categorizer: Optional[TransactionCategorizer] = None
        self._synced = False
        self._server_timestamp = 0
        self._accounts: List[ZenAccount] = []
        self._transactions: List[ZenTransaction] = []
        self._tags: List[ZenTag] = []

    async def ensure_synced(self) -> None:
        """Fetch full data from ZenMoney if not yet synced."""
        if self._synced:
            return
        diff = await self._client.fetch_diff(server_timestamp=0)
        self._server_timestamp = diff.server_timestamp
        self._accounts = diff.accounts
        self._transactions = diff.transactions
        self._tags = diff.tags
        if self._rules_path and self._rules_path.exists():
            tags_map = {t.id: t.title for t in diff.tags}
            self._categorizer = TransactionCategorizer.from_yaml(
                self._rules_path, tags=tags_map
            )
        self._synced = True
        logger.info(
            "finance_service_synced",
            tags=len(diff.tags),
            transactions=len(diff.transactions),
        )

    async def change_category(
        self, transaction_id: str, new_category: str
    ) -> Tuple[bool, str]:
        """Change category of a transaction in ZenMoney.

        Returns (success, message).
        """
        await self.ensure_synced()
        tag_id = self._client.find_tag_id(new_category)
        if not tag_id:
            available = ", ".join(sorted(self._client.get_all_tags().keys()))
            return False, (
                f"Категория '{new_category}' не найдена в ZenMoney. "
                f"Доступные: {available}"
            )
        try:
            await self._client.set_transaction_category(transaction_id, tag_id)
            return True, f"Категория изменена на '{new_category}'"
        except Exception as e:
            logger.exception("finance_change_category_failed", error=str(e))
            return False, f"Ошибка: {e}"

    async def bulk_change_categories(
        self, updates: List[Dict[str, str]]
    ) -> Tuple[int, int, List[str]]:
        """Bulk change categories. Each dict: {transaction_id, category}.

        Returns (success_count, fail_count, errors).
        """
        await self.ensure_synced()
        prepared = []
        errors = []
        for u in updates:
            tag_id = self._client.find_tag_id(u["category"])
            if not tag_id:
                errors.append(
                    f"Категория '{u['category']}' не найдена для транзакции {u['transaction_id']}"
                )
                continue
            prepared.append(
                {"transaction_id": u["transaction_id"], "tag_id": tag_id}
            )
        if prepared:
            try:
                await self._client.bulk_set_categories(prepared)
            except Exception as e:
                logger.exception("finance_bulk_update_failed", error=str(e))
                return 0, len(updates), [f"API error: {e}"]
        return len(prepared), len(errors), errors

    async def mark_self_transfer(
        self,
        transaction_id: str,
        income_account: str,
        outcome_account: str,
    ) -> Tuple[bool, str]:
        """Mark a transaction as transfer between own accounts."""
        try:
            await self._client.mark_as_self_transfer(
                transaction_id, income_account, outcome_account
            )
            return True, "Транзакция помечена как перевод между своими счетами"
        except Exception as e:
            logger.exception("finance_mark_transfer_failed", error=str(e))
            return False, f"Ошибка: {e}"

    async def get_account_balances(self) -> List[Dict[str, object]]:
        """Get balances of active (non-archived) accounts.

        Returns list of {title, balance, type} dicts.
        """
        await self.ensure_synced()
        return [
            {
                "title": acc.title,
                "balance": acc.balance,
                "type": acc.type or "unknown",
            }
            for acc in self._accounts
            if not acc.archive and acc.title
        ]

    async def get_tags(self) -> Dict[str, str]:
        """Get all ZenMoney tags {title: id}."""
        await self.ensure_synced()
        return self._client.get_all_tags()

    async def get_transactions_for_month(
        self, year: int, month: int
    ) -> List[ZenTransaction]:
        """Get all transactions for a specific month."""
        await self.ensure_synced()
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        return [
            t for t in self._transactions
            if start <= t.date <= end and not t.is_own_transfer
        ]

    async def get_all_transactions(self) -> List[ZenTransaction]:
        """Get all synced transactions."""
        await self.ensure_synced()
        return self._transactions

    def get_tag_name(self, tag_id: str) -> str:
        """Resolve tag ID to title."""
        for t in self._tags:
            if t.id == tag_id:
                return t.title
        return ""

    async def build_context(
        self,
        current_year: int,
        current_month: int,
        extra_months: Optional[List[tuple]] = None,
        today: Optional[date] = None,
    ) -> str:
        """Build full financial context text for AI prompt.

        Includes detailed transactions for current_month and any extra_months.
        """
        await self.ensure_synced()
        from .formatter import build_finance_context

        tags_map = {t.id: t.title for t in self._tags}
        return build_finance_context(
            transactions=self._transactions,
            accounts=self._accounts,
            tags_map=tags_map,
            current_month=current_month,
            current_year=current_year,
            extra_months=extra_months,
            today=today,
        )

    async def health_check(self) -> bool:
        """Check ZenMoney API connection."""
        return await self._client.health_check()
