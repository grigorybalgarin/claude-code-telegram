"""Finance context formatter — builds text prompt with transaction details."""

from __future__ import annotations

import calendar
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

from .models import ZenAccount, ZenTransaction

MONTH_NAMES_RU = {
    1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель",
    5: "Май", 6: "Июнь", 7: "Июль", 8: "Август",
    9: "Сентябрь", 10: "Октябрь", 11: "Ноябрь", 12: "Декабрь",
}


def _fmt_amount(amount: float) -> str:
    """Format amount with thousands separator."""
    if amount == int(amount):
        return f"{int(amount):,}".replace(",", " ")
    return f"{amount:,.2f}".replace(",", " ")


def _resolve_tag(tag_id: str, tags_map: Dict[str, str]) -> str:
    return tags_map.get(tag_id, "Без категории")


def format_month_block(
    transactions: List[ZenTransaction],
    tags_map: Dict[str, str],
    year: int,
    month: int,
    detailed: bool = True,
) -> str:
    """Format a single month's data: summary + optionally detailed transactions."""
    month_name = MONTH_NAMES_RU.get(month, str(month))
    lines: List[str] = []

    incomes: List[ZenTransaction] = []
    expenses: List[ZenTransaction] = []

    for t in sorted(transactions, key=lambda x: x.date):
        if t.is_own_transfer:
            continue
        if t.income > 0 and t.outcome == 0:
            incomes.append(t)
        elif t.outcome > 0:
            expenses.append(t)

    total_income = sum(t.income for t in incomes)
    total_expense = sum(t.outcome for t in expenses)
    balance = total_income - total_expense

    # Summary
    lines.append(f"### {month_name} {year} — Сводка")
    lines.append(f"Доходы: {_fmt_amount(total_income)} ₽")
    lines.append(f"Расходы: {_fmt_amount(total_expense)} ₽")
    lines.append(f"Баланс: {_fmt_amount(balance)} ₽")
    lines.append("")

    # Expenses by category
    exp_by_cat: Dict[str, float] = defaultdict(float)
    for t in expenses:
        cat = _resolve_tag(t.tag_ids[0], tags_map) if t.tag_ids else "Без категории"
        exp_by_cat[cat] += t.outcome

    if exp_by_cat:
        lines.append("Расходы по категориям:")
        for cat, amount in sorted(exp_by_cat.items(), key=lambda x: -x[1]):
            pct = int(amount / total_expense * 100) if total_expense else 0
            lines.append(f"  {cat}: {_fmt_amount(amount)} ₽ ({pct}%)")
        lines.append("")

    # Income by source
    inc_by_cat: Dict[str, float] = defaultdict(float)
    for t in incomes:
        cat = _resolve_tag(t.tag_ids[0], tags_map) if t.tag_ids else "Прочие поступления"
        inc_by_cat[cat] += t.income

    if inc_by_cat:
        lines.append("Доходы по источникам:")
        for cat, amount in sorted(inc_by_cat.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {_fmt_amount(amount)} ₽")
        lines.append("")

    # Detailed transactions
    if detailed:
        lines.append(f"### Все транзакции за {month_name} (детально)")
        for t in sorted(transactions, key=lambda x: x.date):
            if t.is_own_transfer:
                continue
            if t.income > 0 and t.outcome == 0:
                tx_type = "ДОХОД"
                amount = t.income
            else:
                tx_type = "РАСХОД"
                amount = t.outcome

            cat = _resolve_tag(t.tag_ids[0], tags_map) if t.tag_ids else "Без категории"
            desc = t.description or ""
            lines.append(
                f"  [{t.id}] {t.date} | {tx_type} | "
                f"{_fmt_amount(amount)} ₽ | {cat} | {desc}"
            )
        lines.append("")

    return "\n".join(lines)


def format_accounts_block(accounts: List[ZenAccount]) -> str:
    """Format account balances section."""
    lines = ["### Счета и балансы (из ZenMoney)"]

    active = [a for a in accounts if not a.archive and a.title]
    active.sort(key=lambda a: -a.balance)

    total = 0.0
    for acc in active:
        lines.append(f"  {acc.title}: {_fmt_amount(acc.balance)} ₽ [{acc.type or '?'}]")
        total += acc.balance

    lines.append("  ---")
    lines.append(f"  Итого (₽): {_fmt_amount(total)} ₽")
    lines.append("")
    return "\n".join(lines)


def format_last_n_days(
    transactions: List[ZenTransaction],
    today: date,
    days: int = 7,
) -> str:
    """Format summary for last N days."""
    start = today - timedelta(days=days)
    recent = [
        t for t in transactions
        if start <= t.date <= today and not t.is_own_transfer
    ]

    income = sum(t.income for t in recent if t.income > 0 and t.outcome == 0)
    expense = sum(t.outcome for t in recent if t.outcome > 0)

    lines = [f"### Последние {days} дней"]
    lines.append(f"Расходы: {_fmt_amount(expense)} ₽")
    lines.append(f"Доходы: {_fmt_amount(income)} ₽")
    lines.append("")
    return "\n".join(lines)


def build_finance_context(
    transactions: List[ZenTransaction],
    accounts: List[ZenAccount],
    tags_map: Dict[str, str],
    current_month: int,
    current_year: int,
    extra_months: Optional[List[tuple]] = None,
    today: Optional[date] = None,
) -> str:
    """Build full financial context for the AI prompt.

    Args:
        transactions: All transactions from ZenMoney.
        accounts: All accounts.
        tags_map: {tag_id: tag_title} mapping.
        current_month: Primary month to show in detail.
        current_year: Year for primary month.
        extra_months: Additional (year, month) tuples to include with details.
        today: Current date for "last 7 days" calc.
    """
    if today is None:
        today = date.today()
    if extra_months is None:
        extra_months = []

    # Determine which months to show in detail
    detail_months = {(current_year, current_month)}
    for ym in extra_months:
        detail_months.add(ym)

    # Group transactions by (year, month)
    by_month: Dict[tuple, List[ZenTransaction]] = defaultdict(list)
    for t in transactions:
        if not t.is_own_transfer:
            by_month[(t.date.year, t.date.month)].append(t)

    sections: List[str] = []
    sections.append("## Текущие финансовые данные")

    # Current month first (always detailed)
    key = (current_year, current_month)
    if key in by_month:
        sections.append(format_month_block(
            by_month[key], tags_map, current_year, current_month, detailed=True
        ))

    # Extra months (also detailed)
    for y, m in sorted(extra_months):
        if (y, m) == (current_year, current_month):
            continue
        key = (y, m)
        if key in by_month:
            sections.append(format_month_block(
                by_month[key], tags_map, y, m, detailed=True
            ))

    # Last 7 days
    sections.append(format_last_n_days(transactions, today))

    # Account balances
    sections.append(format_accounts_block(accounts))

    return "\n".join(sections)
