"""Tests for the dataset-independent public extraction interface."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .extractor import TimecourseExtractor


class TimecourseExtractorTest(unittest.TestCase):
    def test_provider_selects_its_default_model(self) -> None:
        client = SimpleNamespace()
        self.assertEqual(
            TimecourseExtractor(provider="openai", client=client).model,
            "gpt-5.6-sol",
        )
        self.assertEqual(
            TimecourseExtractor(provider="gemini", client=client).model,
            "gemini-3.6-flash",
        )
        self.assertEqual(
            TimecourseExtractor(provider="anthropic", client=client).model,
            "claude-opus-5",
        )

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported provider"):
            TimecourseExtractor(provider="unknown", client=SimpleNamespace())

    def test_callable_returns_structured_result_and_inventory_warning(self) -> None:
        response = SimpleNamespace(
            output_text=json.dumps({"curve_count": 2, "timecourses": []}),
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            id="response-1",
        )
        responses = SimpleNamespace(create=lambda **kwargs: response)
        extractor = TimecourseExtractor(
            client=SimpleNamespace(responses=responses), model="test-model"
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "graph.png"
            image.write_bytes(b"not-a-real-png")
            result = extractor(image, paper_text="caption")

        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.response_id, "response-1")
        self.assertEqual(result.to_dict(), {"curve_count": 2, "timecourses": []})
        self.assertEqual(result.warnings, ["model inventoried 2 curves but returned 0"])


if __name__ == "__main__":
    unittest.main()
