"""Unicode script-range filtering for multi-script name selection.

Selects the English/Latin-script name block from OCR results that may
contain both regional-script and English text.
"""

from __future__ import annotations

import re

# Characters considered as "allowed punctuation" in names
# (not counted toward Latin/non-Latin ratio)
NAME_PUNCTUATION = re.compile(r"[\s'\-.]")


def is_latin_text(text: str, threshold: float = 0.90) -> bool:
    """Determine if a text block is predominantly Latin script.

    A text block qualifies as English/Latin if >= threshold fraction of its
    significant characters (after stripping whitespace and allowed name
    punctuation) fall within U+0041-U+005A (A-Z) and U+0061-U+007A (a-z).

    Args:
        text: The text to check.
        threshold: Minimum fraction of Latin characters required (default 0.90).

    Returns:
        True if the text is predominantly Latin script.
    """
    # Strip whitespace and allowed punctuation
    stripped = NAME_PUNCTUATION.sub("", text)

    if not stripped:
        return False

    latin_count = sum(1 for c in stripped if "\u0041" <= c <= "\u005a" or "\u0061" <= c <= "\u007a")

    ratio = latin_count / len(stripped)
    return ratio >= threshold


def is_numeric_text(text: str, threshold: float = 0.80) -> bool:
    """Determine if a text block is predominantly numeric.

    Args:
        text: The text to check.
        threshold: Minimum fraction of digit characters required.

    Returns:
        True if the text is predominantly numeric.
    """
    stripped = re.sub(r"[\s/\-]", "", text)
    if not stripped:
        return False

    digit_count = sum(1 for c in stripped if c.isdigit())
    return (digit_count / len(stripped)) >= threshold


def select_latin_lines(text_lines: list[str], threshold: float = 0.90) -> list[str]:
    """From a list of OCR text lines, select only those in Latin script.

    This is position-independent: works regardless of whether the English
    name appears above or below the regional-script name.

    Args:
        text_lines: List of text lines from OCR (may include regional scripts).
        threshold: Latin character threshold.

    Returns:
        List of text lines that are predominantly Latin script.
    """
    return [line for line in text_lines if line.strip() and is_latin_text(line, threshold)]


def extract_english_name(text_lines: list[str], threshold: float = 0.90) -> str | None:
    """Extract the English/Latin name from multi-script OCR output.

    Filters text lines to find Latin-script lines, excludes numeric-heavy
    lines (which might be card numbers/dates), and joins the result.

    Args:
        text_lines: All text lines from a name zone.
        threshold: Latin character threshold.

    Returns:
        The English name string, or None if no Latin line found.
    """
    latin_lines = []
    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines that are predominantly numeric (dates, card numbers)
        if is_numeric_text(line, 0.5):
            continue
        # Skip very short lines (likely artifacts)
        if len(line) < 2:
            continue
        if is_latin_text(line, threshold):
            latin_lines.append(line)

    if not latin_lines:
        return None

    # Join multiple Latin lines (some names span two lines)
    return " ".join(latin_lines).strip()
