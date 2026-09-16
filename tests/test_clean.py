import pandas as pd

from src.clean import clean_text_series


def test_clean_text_series() -> None:
    """Validates vectorized text cleaning operations."""
    raw_data: pd.Series = pd.Series(
        [
            "INVESTIGATIVE CASE: Check https://example.com/audit for details!!!",
            "  Compliance audit report -- Section 404 Exception flagged.  ",
        ]
    )

    cleaned: pd.Series = clean_text_series(raw_data)

    assert cleaned.iloc[0] == "investigative case check for details"
    assert cleaned.iloc[1] == "compliance audit report section 404 exception flagged"
