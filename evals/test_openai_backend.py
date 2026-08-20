"""No-network contract tests for the OpenAI timecourse backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .backends import OpenAIBackend
from .benchmark import Entry


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_text=json.dumps({"curve_count": 0, "timecourses": []}),
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
            id="response-1",
        )


class OpenAIBackendTest(unittest.TestCase):
    def test_sends_original_image_and_strict_schema(self) -> None:
        fake_responses = _FakeResponses()
        fake_client = SimpleNamespace(responses=fake_responses)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "figure.png"
            image.write_bytes(b"not-a-real-png")
            entry = Entry(
                id="figure-1",
                image=image,
                study_sid="study-1",
                paper_text="caption and methods",
                reference_title="Paper",
                substances=[],
                gold=[],
                raw={"figure": "Fig. 1"},
            )
            pred = OpenAIBackend(client=fake_client).predict(entry)

        self.assertIsNone(pred.error)
        self.assertEqual(pred.timecourses, [])
        request = fake_responses.kwargs
        assert request is not None
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(request["reasoning"], {"effort": "high"})
        self.assertFalse(request["store"])
        image_part = request["input"][0]["content"][0]
        self.assertEqual(image_part["detail"], "original")
        self.assertTrue(image_part["image_url"].startswith("data:image/png;base64,"))
        output_format = request["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertTrue(output_format["strict"])
        self.assertNotIn(
            "minimum", output_format["schema"]["properties"]["curve_count"]
        )

    def test_optional_vocabulary_is_general_and_explicit(self) -> None:
        fake_responses = _FakeResponses()
        extractor = OpenAIBackend(
            client=SimpleNamespace(responses=fake_responses), substance_hint=True
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "figure.png"
            image.write_bytes(b"not-a-real-png")
            entry = Entry(
                id="figure-1",
                image=image,
                study_sid="study-1",
                paper_text="",
                reference_title="",
                substances=[{"name": "paracetamol"}],
                gold=[],
                raw={},
            )
            extractor.predict(entry)

        request = fake_responses.kwargs
        assert request is not None
        task = request["input"][0]["content"][1]["text"]
        self.assertIn("Canonical substance vocabulary", task)
        self.assertIn("paracetamol", task)


if __name__ == "__main__":
    unittest.main()
