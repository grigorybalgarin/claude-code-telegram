"""Transaction categorizer based on YAML rules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import structlog
import yaml

from .models import CategorizedTransaction, CategoryRule, ZenTag, ZenTransaction

logger = structlog.get_logger()


class TransactionCategorizer:
    """Categorize ZenMoney transactions using configurable rules."""

    def __init__(
        self,
        rules: Dict[str, list],  # type: ignore[type-arg]
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self._skip_payee: List[str] = rules.get("skip_payee_contains", [])
        self._skip_self_transfer: bool = rules.get("skip_self_transfers", True)
        self._manual_rules: List[dict] = rules.get("manual_rules", [])  # type: ignore[type-arg]
        self._payee_rules: List[dict] = rules.get("payee_rules", [])  # type: ignore[type-arg]
        self._tag_map: Dict[str, str] = rules.get("tag_map", {})
        self._income_rules: List[dict] = rules.get("income_rules", [])  # type: ignore[type-arg]
        self._income_skip_payee: List[str] = rules.get("income_skip_payee_contains", [])
        self._default_category: str = rules.get("default_expense_category", "Прочее")
        self._default_income_category: str = rules.get(
            "default_income_category", "Прочие поступления"
        )
        # tag_id -> tag_title lookup
        self._tags = tags or {}

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        tags: Optional[Dict[str, str]] = None,
    ) -> "TransactionCategorizer":
        """Load rules from YAML config file."""
        with open(path, encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
        return cls(rules, tags=tags)

    def set_tags(self, tags: List[ZenTag]) -> None:
        """Update tag lookup from ZenMoney data."""
        self._tags = {t.id: t.title for t in tags}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def categorize(self, tx: ZenTransaction) -> CategorizedTransaction:
        """Categorize a single transaction."""
        # 1. Own transfers between accounts — skip
        if self._skip_self_transfer and tx.is_own_transfer:
            return CategorizedTransaction(transaction=tx, skip=True)

        desc = tx.description.lower()

        # 2. Skip by payee pattern (self-transfers not caught by is_own_transfer)
        for pattern in self._skip_payee:
            if pattern.lower() in desc:
                return CategorizedTransaction(transaction=tx, skip=True)

        # 3. Manual rules (e.g. phone transfers needing user input)
        for rule in self._manual_rules:
            match_phone = rule.get("match_phone", "")
            if match_phone and match_phone in desc:
                return CategorizedTransaction(
                    transaction=tx,
                    needs_manual=True,
                    manual_options=rule.get("options", []),
                )

        # 4. Determine if expense or income
        if tx.outcome > 0:
            category = self._categorize_expense(tx, desc)
        elif tx.income > 0:
            category = self._categorize_income(tx, desc)
        else:
            return CategorizedTransaction(transaction=tx, skip=True)

        return CategorizedTransaction(transaction=tx, category=category)

    # ------------------------------------------------------------------
    # Expense categorization
    # ------------------------------------------------------------------

    def _categorize_expense(self, tx: ZenTransaction, desc: str) -> str:
        # Tag-based (ZenMoney already categorized)
        tag_cat = self._resolve_tag(tx)
        if tag_cat:
            return tag_cat

        # Payee pattern rules
        for rule in self._payee_rules:
            patterns = rule.get("patterns", [])
            for pat in patterns:
                if pat.lower() in desc:
                    return rule["category"]

        return self._default_category

    def _categorize_income(self, tx: ZenTransaction, desc: str) -> str:
        # Skip self-incoming
        for pattern in self._income_skip_payee:
            if pattern.lower() in desc:
                return "__skip__"

        # Tag-based
        tag_cat = self._resolve_tag(tx)
        if tag_cat:
            return tag_cat

        # Income rules
        for rule in self._income_rules:
            patterns = rule.get("patterns", [])
            for pat in patterns:
                if pat.lower() in desc:
                    return rule["category"]

        return self._default_income_category

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_tag(self, tx: ZenTransaction) -> Optional[str]:
        """Map ZenMoney tag IDs to our category names."""
        for tid in tx.tag_ids:
            tag_title = self._tags.get(tid, "")
            if tag_title and tag_title in self._tag_map:
                return self._tag_map[tag_title]
            if tag_title:
                return tag_title  # Use ZenMoney tag name as-is
        return None
