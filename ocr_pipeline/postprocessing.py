"""Post-processing: text cleanup, date normalization, guardian prefix extraction.

Transforms raw OCR text into clean, structured field values.
"""

from __future__ import annotations

import re
from datetime import datetime


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text: str) -> str:
    """Normalize a name field: uppercase, clean whitespace, remove artifacts."""
    text = normalize_whitespace(text)
    # Remove common OCR artifacts
    text = re.sub(r"[|\\/:;]", "", text)
    # Remove leading/trailing punctuation that isn't part of names
    text = text.strip(".,;:!?")
    # Normalize to uppercase (PAN/Aadhaar names are printed in caps)
    text = text.upper()
    return text


def normalize_pan_number(text: str) -> str:
    """Normalize PAN number: strip spaces, uppercase, fix common OCR errors."""
    text = re.sub(r"\s+", "", text)
    text = text.upper()

    # Common OCR substitution fixes for PAN
    # O → 0 in digit positions (positions 5-8 are digits)
    # 0 → O in letter positions (positions 0-4, 9 are letters)
    if len(text) == 10:
        chars = list(text)
        # Positions 0-4: should be letters
        for i in range(5):
            if chars[i] == "0":
                chars[i] = "O"
            elif chars[i] == "1":
                chars[i] = "I"
        # Positions 5-8: should be digits
        for i in range(5, 9):
            if chars[i] == "O" or chars[i] == "o":
                chars[i] = "0"
            elif chars[i] == "I" or chars[i] == "l":
                chars[i] = "1"
            elif chars[i] == "S" or chars[i] == "s":
                chars[i] = "5"
            elif chars[i] == "B":
                chars[i] = "8"
        # Position 9: should be a letter
        if chars[9] == "0":
            chars[9] = "O"
        elif chars[9] == "1":
            chars[9] = "I"
        text = "".join(chars)

    return text


def normalize_aadhaar_number(text: str) -> str:
    """Normalize Aadhaar number: strip spaces and non-digit characters."""
    # Keep only digits
    digits = re.sub(r"[^\d]", "", text)
    return digits


def normalize_date(text: str) -> str | None:
    """Normalize a date string to DD/MM/YYYY format.

    Handles common formats:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - DD.MM.YYYY
    - YYYY (year-only for some Aadhaar cards)

    Returns:
        Normalized date string or None if unparseable.
    """
    text = normalize_whitespace(text)
    # Remove common OCR artifacts
    text = re.sub(r"[|\\;:]", "", text)

    # Try DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    date_pattern = re.compile(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})")
    match = date_pattern.search(text)
    if match:
        day, month, year = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            pass

    # Try YYYY-MM-DD
    iso_pattern = re.compile(r"(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})")
    match = iso_pattern.search(text)
    if match:
        year, month, day = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            pass

    # Try year-only (some Aadhaar cards show only birth year)
    year_pattern = re.compile(r"\b(\d{4})\b")
    match = year_pattern.search(text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= datetime.now().year:
            return str(year)

    return None


def extract_guardian_info(text: str) -> tuple[str | None, str | None]:
    """Extract guardian relationship and name from Aadhaar text.

    Looks for patterns like:
    - S/o RAMESH KUMAR
    - D/o SURESH SHARMA
    - W/o RAHUL VERMA
    - C/o MANOJ SINGH

    Args:
        text: OCR text from guardian zone.

    Returns:
        Tuple of (relationship, guardian_name) or (None, None).
    """
    text = normalize_whitespace(text)

    # Pattern: S/o, D/o, W/o, C/o followed by name
    pattern = re.compile(
        r"(?:([SDWC])\s*[/\\]\s*[oO0])\s*[:\-]?\s*(.+)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        relationship = match.group(1).upper()
        name = normalize_name(match.group(2))
        relationship_map = {"S": "S/o", "D": "D/o", "W": "W/o", "C": "C/o"}
        return relationship_map.get(relationship, f"{relationship}/o"), name

    # Fallback: just return the text as name without relationship
    cleaned = normalize_name(text)
    if cleaned and len(cleaned) > 2:
        return None, cleaned

    return None, None


def normalize_gender(text: str) -> str | None:
    """Normalize gender field.

    Args:
        text: Raw OCR text from gender zone.

    Returns:
        Normalized gender string or None.
    """
    text = text.upper().strip()

    if "MALE" in text and "FEMALE" not in text:
        return "MALE"
    if "FEMALE" in text:
        return "FEMALE"
    if "TRANS" in text:
        return "TRANSGENDER"

    # Single character shortcuts sometimes seen
    cleaned = re.sub(r"[^A-Z]", "", text)
    if cleaned == "M":
        return "MALE"
    if cleaned == "F":
        return "FEMALE"

    return None
