"""Field validators: PAN regex, Aadhaar Verhoeff checksum, DOB, gender.

Each validator returns (is_valid, normalized_value).
"""

from __future__ import annotations

import re
from datetime import datetime

# PAN format: 5 letters + 4 digits + 1 letter
PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# Valid gender values
VALID_GENDERS = {"MALE", "FEMALE", "TRANSGENDER"}


# --- Verhoeff Checksum Tables ---
# Multiplication table
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

# Permutation table
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

# Inverse table
_VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def _verhoeff_checksum(number: str) -> bool:
    """Validate a number string using the Verhoeff checksum algorithm.

    The Verhoeff algorithm is used by UIDAI for Aadhaar number validation.
    A valid number has a checksum of 0.

    Args:
        number: String of digits to validate.

    Returns:
        True if the checksum is valid (equals 0).
    """
    c = 0
    # Process digits from right to left
    digits = [int(d) for d in reversed(number)]
    for i, digit in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][digit]]
    return c == 0


def validate_pan(value: str) -> tuple[bool, str]:
    """Validate PAN number format.

    Args:
        value: PAN number string.

    Returns:
        Tuple of (is_valid, normalized_value).
    """
    normalized = value.strip().upper()
    # Remove any spaces or special characters
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)

    is_valid = bool(PAN_REGEX.match(normalized))
    return is_valid, normalized


def validate_aadhaar(value: str) -> tuple[bool, str]:
    """Validate Aadhaar number: 12 digits + Verhoeff checksum.

    Args:
        value: Aadhaar number string (may contain spaces).

    Returns:
        Tuple of (is_valid, normalized_value).
    """
    # Strip all non-digit characters
    normalized = re.sub(r"[^\d]", "", value)

    if len(normalized) != 12:
        return False, normalized

    # Check all digits
    if not normalized.isdigit():
        return False, normalized

    # Aadhaar numbers cannot start with 0 or 1
    if normalized[0] in ("0", "1"):
        return False, normalized

    # Verhoeff checksum validation
    is_valid = _verhoeff_checksum(normalized)
    return is_valid, normalized


def validate_dob(value: str) -> tuple[bool, str]:
    """Validate date of birth.

    Accepts DD/MM/YYYY or YYYY (year-only).
    Validates plausible range: 1900 to current year, age >= 0.

    Args:
        value: Date string.

    Returns:
        Tuple of (is_valid, normalized_value).
    """
    value = value.strip()
    current_year = datetime.now().year

    # Year-only format
    if re.match(r"^\d{4}$", value):
        year = int(value)
        is_valid = 1900 <= year <= current_year
        return is_valid, value

    # DD/MM/YYYY format
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            dt = datetime(year, month, day)
            if dt.year < 1900 or dt > datetime.now():
                return False, value
            return True, f"{day:02d}/{month:02d}/{year}"
        except ValueError:
            return False, value

    return False, value


def validate_gender(value: str) -> tuple[bool, str]:
    """Validate gender field.

    Args:
        value: Gender string.

    Returns:
        Tuple of (is_valid, normalized_value).
    """
    normalized = value.strip().upper()
    is_valid = normalized in VALID_GENDERS
    return is_valid, normalized
