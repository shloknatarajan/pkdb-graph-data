"""Regression tests for unit spellings observed in model extractions."""

from __future__ import annotations

import unittest

from .units import normalize_unit_string, value_conversion


class UnitNormalizationTest(unittest.TestCase):
    def test_publication_notation_variants(self) -> None:
        self.assertEqual(normalize_unit_string("nmol l^-1"), "nmol/l")
        self.assertEqual(normalize_unit_string("mg./hr."), "mg/hr")

    def test_mcg_per_liter_equals_ug_per_liter(self) -> None:
        self.assertEqual(value_conversion("mcg/L", "ug/l"), 1.0)

    def test_descriptive_percent_axis_label(self) -> None:
        self.assertEqual(
            value_conversion("% total paracetamol and metabolites", "percent"), 1.0
        )


if __name__ == "__main__":
    unittest.main()
