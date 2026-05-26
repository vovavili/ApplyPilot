from hypothesis import given
from hypothesis import strategies as st

from applypilot.apply.salary import money_value, parse_posted_salary, salary_rejection_reason


def test_money_value_handles_plain_and_k_values():
    assert money_value("90000") == 90000
    assert money_value("€90k") == 90000
    assert money_value("Negotiable") is None


def test_parse_posted_salary_range_with_currency():
    salary = parse_posted_salary("€80k - €100k / year")

    assert salary.minimum == 80000
    assert salary.maximum == 100000
    assert salary.currency == "EUR"


def test_parse_posted_salary_hourly_to_annual():
    salary = parse_posted_salary("$50/hr")

    assert salary.minimum == 104000
    assert salary.maximum == 104000
    assert salary.currency == "USD"


def test_salary_rejection_uses_private_minimum_without_public_range():
    reason = salary_rejection_reason(
        "€70,000 - €80,000",
        {"private_minimum": "90000", "salary_currency": "EUR"},
    )

    assert reason == "posted_salary_below_private_minimum"


def test_salary_rejection_ignores_unknown_or_mismatched_currency():
    assert salary_rejection_reason("", {"private_minimum": "90000", "salary_currency": "EUR"}) is None
    assert salary_rejection_reason("$70,000 - $80,000", {"private_minimum": "90000", "salary_currency": "EUR"}) is None


@given(
    low=st.integers(min_value=20_000, max_value=200_000),
    high=st.integers(min_value=20_000, max_value=250_000),
)
def test_parse_posted_salary_preserves_generated_eur_range_bounds(low, high):
    low, high = sorted((low, high))

    salary = parse_posted_salary(f"€{low:,} - €{high:,} per year")

    assert salary.minimum == low
    assert salary.maximum == high
    assert salary.currency == "EUR"


@given(
    private_minimum=st.integers(min_value=30_000, max_value=220_000),
    posted_max=st.integers(min_value=20_000, max_value=260_000),
)
def test_salary_rejection_is_based_on_posted_maximum(private_minimum, posted_max):
    posted_min = min(20_000, posted_max)
    reason = salary_rejection_reason(
        f"€{posted_min:,} - €{posted_max:,}",
        {"private_minimum": str(private_minimum), "salary_currency": "EUR"},
    )

    expected = "posted_salary_below_private_minimum" if posted_max < private_minimum else None
    assert reason == expected
