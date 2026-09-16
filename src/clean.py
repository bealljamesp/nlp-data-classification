from typing import Final

import pandas as pd

# Compiled regex patterns for vectorized execution
URL_PATTERN: Final[str] = r"https?://\S+|www\.\S+"
PUNCT_PATTERN: Final[str] = r"[^\w\s]"


def clean_text_series(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series of text documents using vectorized string methods.
    Bans explicit loops and pandas.apply().
    """
    # Convert to lowercase via vectorized str accessor
    s = series.str.lower()

    # Strip URLs using vectorized regex replacement
    s = s.str.replace(URL_PATTERN, "", regex=True)

    # Strip punctuation
    s = s.str.replace(PUNCT_PATTERN, "", regex=True)

    # Strip excess whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s
