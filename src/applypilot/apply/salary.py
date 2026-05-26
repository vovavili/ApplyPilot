"""Salary parsing and compensation strategy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re


_MONEY_RE = re.compile(r"(?P<currency>[$€£])?\s*(?P<amount>\d[\d., ]*)(?P<suffix>[kK])?")
_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_NO_VALUE = {"", "negotiable", "n/a", "na", "none", "null", "open", "market"}


@dataclass(frozen=True)
class SalaryRange:
    minimum: int | None = None
    maximum: int | None = None
    currency: str = ""


def money_value(value: object) -> int | None:
    """Parse one configured salary number."""
    if value is None:
        return None

    text = str(value).strip()
    if text.casefold() in _NO_VALUE:
        return None

    match = _MONEY_RE.search(text)
    if not match:
        return None
    return _parse_number(match.group("amount"), bool(match.group("suffix")))


def parse_posted_salary(text: object) -> SalaryRange:
    """Extract a rough annual salary range from a posting salary string."""
    if text is None:
        return SalaryRange()

    raw = str(text)
    if not raw.strip():
        return SalaryRange()

    currency = _detect_currency(raw)
    parsed: list[tuple[int, bool]] = []
    for match in _MONEY_RE.finditer(raw):
        amount_text = match.group("amount").strip()
        if not amount_text:
            continue
        has_k = bool(match.group("suffix"))
        amount = _parse_number(amount_text, has_k)
        if amount is not None:
            parsed.append((amount, has_k))

    if not parsed:
        return SalaryRange(currency=currency)

    amounts = _annualize_amounts(raw, parsed)
    if not amounts:
        return SalaryRange(currency=currency)

    return SalaryRange(minimum=min(amounts), maximum=max(amounts), currency=currency)


def salary_rejection_reason(salary_text: object, compensation: dict) -> str | None:
    """Return a permanent rejection reason when posted salary is clearly too low."""
    private_minimum = money_value(compensation.get("private_minimum"))
    if not private_minimum:
        return None

    posted = parse_posted_salary(salary_text)
    if not posted.maximum:
        return None

    configured_currency = str(compensation.get("salary_currency", "")).upper()
    if configured_currency and posted.currency and configured_currency != posted.currency:
        return None

    if posted.maximum < private_minimum:
        return "posted_salary_below_private_minimum"
    return None


def format_money(amount: int | None, currency: str) -> str:
    if amount is None:
        return ""
    return f"{amount:,} {currency}".strip()


def _parse_number(amount_text: str, has_k: bool) -> int | None:
    normalized = _normalize_number(amount_text)
    if not normalized:
        return None

    try:
        amount = float(normalized)
    except ValueError:
        return None

    if has_k:
        amount *= 1000
    return int(round(amount))


def _normalize_number(text: str) -> str:
    value = re.sub(r"[^\d.,]", "", text.strip())
    if not value:
        return ""

    if "," in value and "." in value:
        decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        value = value.replace(thousands_separator, "")
        if decimal_separator == ",":
            value = value.replace(",", ".")
        return value

    if "," in value:
        parts = value.split(",")
        if len(parts[-1]) == 3:
            return "".join(parts)
        return value.replace(",", ".")

    if "." in value:
        parts = value.split(".")
        if len(parts) > 1 and len(parts[-1]) == 3:
            return "".join(parts)

    return value


def _annualize_amounts(raw: str, parsed: list[tuple[int, bool]]) -> list[int]:
    amounts = [amount for amount, _ in parsed]
    has_k_suffix = any(has_k for _, has_k in parsed)
    if has_k_suffix:
        amounts = [amount * 1000 if amount < 1000 else amount for amount in amounts]

    raw_lower = raw.casefold()
    if any(marker in raw_lower for marker in ("/hr", "hour", "hourly")):
        amounts = [amount * 2080 if amount < 1000 else amount for amount in amounts]
    elif any(marker in raw_lower for marker in ("/mo", "month", "monthly")):
        amounts = [amount * 12 if amount < 20000 else amount for amount in amounts]

    return [amount for amount in amounts if amount >= 1000]


def _detect_currency(text: str) -> str:
    for symbol, currency in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return currency

    text_upper = text.upper()
    for currency in ("EUR", "USD", "GBP"):
        if currency in text_upper:
            return currency
    return ""
