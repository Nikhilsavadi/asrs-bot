"""
holidays.py — Market holiday calendar for DAX, US30, Nikkei
═══════════════════════════════════════════════════════════════

Returns True if a given date is a market holiday.
Covers major holidays through 2027. Update annually.
"""

from datetime import date

# Fixed holidays (month, day) — apply every year
_GLOBAL_FIXED = {
    (1, 1),    # New Year's Day
    (12, 25),  # Christmas Day
    (12, 26),  # Boxing Day (EU)
}

# DAX-specific holidays (XETRA)
_DAX_FIXED = {
    (1, 1), (5, 1), (12, 24), (12, 25), (12, 26), (12, 31),
}

# US-specific fixed holidays
_US_FIXED = {
    (1, 1), (7, 4), (12, 25),
}

# Nikkei-specific (TSE) — many holidays
_NIKKEI_FIXED = {
    (1, 1), (1, 2), (1, 3),  # New Year
    (2, 11),   # National Foundation Day
    (2, 23),   # Emperor's Birthday
    (4, 29),   # Showa Day
    (5, 3),    # Constitution Memorial Day
    (5, 4),    # Greenery Day
    (5, 5),    # Children's Day
    (8, 11),   # Mountain Day
    (9, 23),   # Autumnal Equinox (approximate)
    (11, 3),   # Culture Day
    (11, 23),  # Labor Thanksgiving Day
}

# Variable holidays (exact dates) — update annually
_VARIABLE_HOLIDAYS = {
    # 2025 Good Friday / Easter Monday
    date(2025, 4, 18): {"DAX", "US30"},
    date(2025, 4, 21): {"DAX"},
    # 2025 US holidays
    date(2025, 1, 20): {"US30"},   # MLK Day
    date(2025, 2, 17): {"US30"},   # Presidents Day
    date(2025, 5, 26): {"US30"},   # Memorial Day
    date(2025, 6, 19): {"US30"},   # Juneteenth
    date(2025, 9, 1):  {"US30"},   # Labor Day
    date(2025, 11, 27): {"US30"},  # Thanksgiving
    # 2025 DAX holidays
    date(2025, 6, 9):  {"DAX"},    # Whit Monday
    date(2025, 10, 3): {"DAX"},    # German Unity Day
    # 2026 Good Friday / Easter Monday
    date(2026, 4, 3):  {"DAX", "US30"},
    date(2026, 4, 6):  {"DAX"},
    # 2026 US holidays
    date(2026, 1, 19): {"US30"},
    date(2026, 2, 16): {"US30"},
    date(2026, 5, 25): {"US30"},
    date(2026, 6, 19): {"US30"},
    date(2026, 7, 3):  {"US30"},   # July 4th observed
    date(2026, 9, 7):  {"US30"},
    date(2026, 11, 26): {"US30"},
    # 2026 DAX
    date(2026, 5, 25): {"DAX"},    # Whit Monday
    date(2026, 10, 3): {"DAX"},
    # 2027 Good Friday / Easter Monday
    date(2027, 3, 26): {"DAX", "US30"},
    date(2027, 3, 29): {"DAX"},
}


def is_holiday(d: date, instrument: str = "DAX") -> bool:
    """Check if a date is a market holiday for the given instrument."""
    inst = instrument.upper()
    month_day = (d.month, d.day)

    # Weekend
    if d.weekday() >= 5:
        return True

    # Variable holidays
    if d in _VARIABLE_HOLIDAYS:
        if inst in _VARIABLE_HOLIDAYS[d]:
            return True

    # Fixed holidays by instrument
    if inst == "DAX" and month_day in _DAX_FIXED:
        return True
    if inst == "US30" and month_day in _US_FIXED:
        return True
    if inst in ("NIKKEI", "JAPAN") and month_day in _NIKKEI_FIXED:
        return True

    # Global fixed
    if month_day in _GLOBAL_FIXED:
        return True

    return False


def is_trading_day(d: date, instrument: str = "DAX") -> bool:
    """Inverse of is_holiday — True if markets are open."""
    return not is_holiday(d, instrument)
