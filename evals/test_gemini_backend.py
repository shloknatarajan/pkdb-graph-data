"""No-network contract test for the Gemini timecourse backend."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from .backends import GeminiBackend
from .benchmark import Entry


class GeminiBackendTest(unittest.TestCase):
    def test_uses_inline_image_shared_schema_and_high_thinking(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=json.dumps({"curve_count": 0, "timecourses": []}),
                usage=None,
            )

        backend = GeminiBackend(
            client=SimpleNamespace(interactions=SimpleNamespace(create=create))
        )
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "graph.png"
            image.write_bytes(b"not-a-real-png")
            entry = Entry(
                id="figure-1",
                image=image,
                study_sid="study-1",
                paper_text="caption",
                reference_title="Paper",
                substances=[],
                gold=[],
                raw={},
            )
            pred = backend.predict(entry)

        self.assertIsNone(pred.error)
        self.assertEqual(captured["model"], "gemini-3.6-flash")
        self.assertEqual(captured["generation_config"], {"thinking_level": "high"})
        self.assertEqual(captured["input"][0]["type"], "image")
        self.assertEqual(captured["input"][0]["mime_type"], "image/png")
        self.assertEqual(captured["response_format"]["mime_type"], "application/json")
        self.assertIn(
            "curve_count", captured["response_format"]["schema"]["properties"]
        )


if __name__ == "__main__":
    unittest.main()
