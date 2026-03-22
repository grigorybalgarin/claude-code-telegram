"""Finance data models."""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ZenTransaction(BaseModel):
    """Single transaction from ZenMoney API."""

    id: str
    date: date
    income: float = 0.0
    outcome: float = 0.0
    income_account: Optional[str] = None
    outcome_account: Optional[str] = None
    income_instrument: Optional[int] = None
    outcome_instrument: Optional[int] = None
    payee: Optional[str] = None
    comment: Optional[str] = None
    tag_ids: List[str] = Field(default_factory=list)
    merchant_id: Optional[str] = None
    outcome_bank_id: Optional[str] = None
    income_bank_id: Optional[str] = None
    source: Optional[str] = None
    created: Optional[int] = None
    changed: Optional[int] = None

    @property
    def description(self) -> str:
        """Combined payee + comment for pattern matching."""
        parts = [self.payee or "", self.comment or ""]
        return " ".join(p for p in parts if p).strip()

    @property
    def is_own_transfer(self) -> bool:
        """Check if both accounts are filled and different (inter-account)."""
        return bool(
            self.income_account
            and self.outcome_account
            and self.income_account != self.outcome_account
            and self.income > 0
            and self.outcome > 0
        )


class ZenAccount(BaseModel):
    """ZenMoney account."""

    id: str
    title: str
    instrument: Optional[int] = None
    balance: float = 0.0
    archive: bool = False
    type: Optional[str] = None
    company: Optional[int] = None


class ZenTag(BaseModel):
    """ZenMoney tag / category."""

    id: str
    title: str
    parent: Optional[str] = None


class ZenDiffResponse(BaseModel):
    """Parsed response from /v8/diff/."""

    server_timestamp: int = 0
    transactions: List[ZenTransaction] = Field(default_factory=list)
    accounts: List[ZenAccount] = Field(default_factory=list)
    tags: List[ZenTag] = Field(default_factory=list)


class CategorizedTransaction(BaseModel):
    """Transaction with resolved category."""

    transaction: ZenTransaction
    category: Optional[str] = None
    skip: bool = False
    needs_manual: bool = False
    manual_options: List[str] = Field(default_factory=list)


class CategoryRule(BaseModel):
    """Single categorization rule from YAML config."""

    name: str
    match_type: Literal[
        "payee_contains",
        "tag_name",
        "phone_transfer",
        "self_transfer",
        "amount_range",
    ]
    pattern: str = ""
    category: str = ""
    skip: bool = False
    manual: bool = False
    manual_options: List[str] = Field(default_factory=list)
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None


class FinanceReport(BaseModel):
    """Generated financial report."""

    period: str  # "day", "week", "month"
    start_date: date
    end_date: date
    total_income: float = 0.0
    total_expense: float = 0.0
    expenses_by_category: dict[str, float] = Field(default_factory=dict)
    income_by_category: dict[str, float] = Field(default_factory=dict)
    balance: float = 0.0
    account_balances: dict[str, float] = Field(default_factory=dict)
