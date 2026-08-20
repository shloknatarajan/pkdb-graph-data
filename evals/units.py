"""Unit normalization for PK timecourse values and times.

Predictions are only numerically comparable to the gold data when they share a
physical dimension. `parse_value_unit` maps a unit string onto a
(dimension, factor-to-base) pair; two units are comparable iff their dimensions
match, and a prediction is converted into the gold unit by the ratio of factors.
"""

from __future__ import annotations

import re

# factor converts a value in the given unit into the dimension's base unit.
# base units: mass_conc -> microg/ml, molar -> nmol/l, mass_rate -> mg/hr.
_VALUE_UNITS: dict[str, tuple[str, float]] = {
    "ug/ml": ("mass_conc", 1.0),
    "microg/ml": ("mass_conc", 1.0),
    "mcg/ml": ("mass_conc", 1.0),
    "mg/l": ("mass_conc", 1.0),
    "ng/ml": ("mass_conc", 1e-3),
    "ug/l": ("mass_conc", 1e-3),
    "microg/l": ("mass_conc", 1e-3),
    "mcg/l": ("mass_conc", 1e-3),
    "ng/l": ("mass_conc", 1e-6),
    "pg/ml": ("mass_conc", 1e-6),
    "mg/ml": ("mass_conc", 1e3),
    "g/l": ("mass_conc", 1e3),
    "nmol/l": ("molar", 1.0),
    "nm": ("molar", 1.0),
    "umol/l": ("molar", 1e3),
    "micromol/l": ("molar", 1e3),
    "um": ("molar", 1e3),
    "mmol/l": ("molar", 1e6),
    "mm": ("molar", 1e6),
    "pmol/l": ("molar", 1e-3),
    "mg/hr": ("mass_rate", 1.0),
    "ug/hr": ("mass_rate", 1e-3),
    "microg/hr": ("mass_rate", 1e-3),
    "nmol/hr": ("molar_rate", 1.0),
    "umol/hr": ("molar_rate", 1e3),
    "percent": ("percent", 1.0),
    "%": ("percent", 1.0),
}

_TIME_UNITS: dict[str, float] = {  # -> hours
    "hr": 1.0,
    "h": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "min": 1.0 / 60.0,
    "mins": 1.0 / 60.0,
    "minute": 1.0 / 60.0,
    "minutes": 1.0 / 60.0,
    "s": 1.0 / 3600.0,
    "sec": 1.0 / 3600.0,
    "second": 1.0 / 3600.0,
    "seconds": 1.0 / 3600.0,
    "day": 24.0,
    "days": 24.0,
    "d": 24.0,
}


_SUPERSCRIPTS = str.maketrans({"⁻": "-", "¹": "1", "²": "2", "³": "3", "−": "-"})


def normalize_unit_string(unit: str | None) -> str | None:
    """Canonicalize a unit onto a `numerator/denominator` form.

    Handles the notations that actually appear on journal axes and in curator
    tables: 'microg/ml', 'µg/mL', 'µg ml-1', and 'ug ml⁻¹' all become 'ug/ml'.
    """
    if unit is None:
        return None
    s = unit.strip().lower()
    if not s:
        return None
    # Models often preserve the full axis title rather than only its unit token,
    # e.g. "% total dose recovered". It is still a dimensionless percentage.
    if s.startswith("%") or s.startswith("percent"):
        return "%"
    for micro in ("µ", "μ"):  # MICRO SIGN, GREEK SMALL LETTER MU
        s = s.replace(micro, "u")
    s = s.translate(_SUPERSCRIPTS)
    # Axis labels often retain publication-era punctuation (`mg./hr.`) or use
    # TeX-like exponent notation (`nmol l^-1`). Neither changes the unit.
    s = s.replace("^", "").replace(".", "")
    s = s.replace("·", " ").replace("*", " ")

    if "/" in s:
        s = s.replace(" ", "")
    else:
        # Exponent notation: a token like 'ml-1' is a denominator.
        numerator: list[str] = []
        denominator: list[str] = []
        for token in s.split():
            m = re.fullmatch(r"([a-z%]+)-1", token)
            if m:
                denominator.append(m.group(1))
            else:
                numerator.append(token)
        s = "".join(numerator) + ("/" + "".join(denominator) if denominator else "")

    return _canonicalize_denominator(s) or None


# Excretion-rate units are written 'mg/hr', 'mg/h' and 'mg/hour' interchangeably.
_DENOMINATOR_ALIASES = {
    "h": "hr",
    "hour": "hr",
    "hours": "hr",
    "litre": "l",
    "liter": "l",
}


def _canonicalize_denominator(s: str) -> str:
    if "/" not in s:
        return s
    num, _, den = s.partition("/")
    return f"{num}/{_DENOMINATOR_ALIASES.get(den, den)}"


def parse_value_unit(unit: str | None) -> tuple[str, float] | None:
    """Return (dimension, factor_to_base) or None if the unit is unrecognized."""
    s = normalize_unit_string(unit)
    if s is None:
        return None
    return _VALUE_UNITS.get(s)


def parse_time_unit(unit: str | None) -> float | None:
    """Return the factor converting the given time unit into hours."""
    s = normalize_unit_string(unit)
    if s is None:
        return None
    return _TIME_UNITS.get(s)


def value_conversion(pred_unit: str | None, gold_unit: str | None) -> float | None:
    """Factor multiplying a value in `pred_unit` to express it in `gold_unit`.

    None means the two units are not comparable (unknown, or different
    dimensions), which callers must treat as a unit error rather than silently
    scoring the numbers against each other.
    """
    if gold_unit is None:
        return None
    p, g = parse_value_unit(pred_unit), parse_value_unit(gold_unit)
    if p is None or g is None or p[0] != g[0]:
        return None
    return p[1] / g[1]


def time_conversion(pred_unit: str | None, gold_unit: str | None) -> float | None:
    """Factor multiplying a time in `pred_unit` to express it in `gold_unit`."""
    p = parse_time_unit(pred_unit)
    g = parse_time_unit(gold_unit)
    if p is None or g is None:
        return None
    return p / g
