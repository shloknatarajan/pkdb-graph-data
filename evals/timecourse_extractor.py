"""Single-file, provider-selectable graph-to-timecourse extraction.

This module contains the public callable, extraction prompt, output schema, result
format, image encoding, response parsing, and provider-specific API calls. It has no
PK-DB benchmark or scoring dependencies.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "curve_count": {"type": "integer"},
        "timecourses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "panel": {"type": ["string", "null"]},
                    "visual_label": {"type": ["string", "null"]},
                    "substance": {"type": "string"},
                    "intervention": {"type": "string"},
                    "tissue": {"type": "string"},
                    "group": {"type": "string"},
                    "measurement_type": {"type": "string"},
                    "time_unit": {"type": "string"},
                    "value_unit": {"type": "string"},
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "number"},
                                "mean": {"type": ["number", "null"]},
                                "value": {"type": ["number", "null"]},
                                "sd": {"type": ["number", "null"]},
                                "se": {"type": ["number", "null"]},
                            },
                            "required": ["time", "mean", "value", "sd", "se"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "panel",
                    "visual_label",
                    "substance",
                    "intervention",
                    "tissue",
                    "group",
                    "measurement_type",
                    "time_unit",
                    "value_unit",
                    "points",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["curve_count", "timecourses"],
    "additionalProperties": False,
}

SYSTEM = """You are a pharmacokinetics data curator. You read concentration-time \
figures from published PK studies and recover the underlying numeric timecourses, \
exactly as a human curator digitizing the plot would."""

INSTRUCTIONS = """\
Recover every visibly plotted timecourse in this figure. The figure is the authority
for how many series and points exist; use paper text only to identify their metadata.

Before writing JSON, silently inventory every panel, legend entry, and distinct
marker/line style. Use that inventory to set `curve_count`, then verify that it equals
the number of objects in `timecourses`.

What counts as a timecourse:
- Return one object per visually distinct plotted series, including the same compound
  shown for different treatments or groups.
- Do not create a series merely because a compound or group is discussed in the paper.
- Error bars, shaded uncertainty/significance regions, fitted lines over an existing
  marker series, and derived ratios not visibly plotted are not additional series.
- For multi-panel figures, inspect every panel and identify it in `panel` using the
  printed panel letter/title, or null for a single unlabelled panel.

How to read the plot:
- Check whether the y-axis is linear or logarithmic before reading values. Many PK
  figures use a log y-axis; on one, a point halfway between 1 and 10 is ~3.2, not 5.5.
- Read the value at each marker, not at the interpolating line. Report one point per
  plotted marker.
- Never infer a regular sampling schedule. Different series may have different marker
  counts and sampling times. Do not invent points from a line or from the paper.
- If markers carry error bars, report the marker center as `mean`, and put the error
  bar half-width in `sd` or `se` — the paper's caption says which it is.
- If the curve is a single subject rather than a group mean, report `value` instead
  of `mean` and leave `sd`/`se` null.

Labels and metadata:
- `visual_label` — copy the legend/near-line label verbatim, or null if none is shown.
- `value_unit` and `time_unit` — copy the axis units exactly as printed.
- `substance` — write the full generic compound name when it can be established; omit
  parenthetical abbreviations. If a canonical vocabulary is supplied, copy the
  matching entry exactly. Otherwise preserve ambiguity rather than guessing.
- `intervention` — write a human-readable treatment description (substance, dose,
  route, formulation, and comparator when available). Never invent a database code.
- `tissue` — the matrix sampled: plasma, serum, urine, saliva.
- `group` — the subject group, if the figure splits subjects into groups; otherwise "all".
- `measurement_type` — usually "concentration"; use "secretion rate" for excretion-rate plots.

Do not merge distinct visible series or split one visible series. Silently recheck the
curve inventory, axis scale, and marker count before responding. Respond with JSON
matching the schema exactly, and nothing else."""

DEFAULT_MODELS = {
    "openai": "gpt-5.6-sol",
    "gemini": "gemini-3.6-flash",
    "anthropic": "claude-opus-5",
}
SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def build_extraction_text(
    *,
    paper: str | None,
    figure_name: str | None = None,
    paper_title: str | None = None,
    substance_vocabulary: Sequence[str] | None = None,
) -> str:
    """Build the dataset-independent user turn."""
    parts = [INSTRUCTIONS]
    if figure_name:
        parts += ["", f"Figure: {figure_name}"]
    if paper_title:
        parts.append(f"Paper: {paper_title}")
    if substance_vocabulary:
        parts += [
            "",
            "Canonical substance vocabulary (use an exact entry when applicable): "
            + ", ".join(substance_vocabulary),
        ]
    if paper:
        parts += ["", "--- BEGIN PAPER TEXT ---", paper, "--- END PAPER TEXT ---"]
    return "\n".join(parts)


def truncate_paper(text: str, max_chars: int | None) -> str:
    """Clip a long paper from the middle while retaining its beginning and end."""
    if not text or max_chars is None or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n\n[... paper text truncated ...]\n\n" + text[-tail:]


def extract_json(text: str) -> dict:
    """Parse JSON, tolerating a provider that wraps it in prose or a code fence."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"no JSON object found in response ({len(text)} chars)")


@dataclass
class ExtractionResult:
    """Structured extraction plus provider metadata and validation warnings."""

    timecourses: list[dict]
    curve_count: int
    model: str
    raw_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"curve_count": self.curve_count, "timecourses": self.timecourses}


class TimecourseExtractor:
    """Extract visible timecourses with OpenAI, Gemini, or Anthropic.

    The class is independent of the benchmark. Select a provider once, then call the
    object with any supported graph image and optional publication context.
    """

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str | None = None,
        reasoning_effort: str = "high",
        image_detail: str = "original",
        max_output_tokens: int = 32000,
        client: Any | None = None,
    ) -> None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"unsupported provider {provider!r}; choose from {sorted(DEFAULT_MODELS)}"
            )
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.reasoning_effort = reasoning_effort
        self.image_detail = image_detail
        self.max_output_tokens = max_output_tokens
        self.client = client or self._default_client(provider)

    @staticmethod
    def _default_client(provider: str) -> Any:
        if provider == "openai":
            from openai import OpenAI

            return OpenAI()
        if provider == "gemini":
            from google import genai

            return genai.Client()
        import anthropic

        return anthropic.Anthropic()

    def __call__(
        self,
        image: Path | str,
        *,
        paper_text: str | None = None,
        figure_name: str | None = None,
        paper_title: str | None = None,
        substance_vocabulary: Sequence[str] | None = None,
    ) -> ExtractionResult:
        return self.extract(
            image,
            paper_text=paper_text,
            figure_name=figure_name,
            paper_title=paper_title,
            substance_vocabulary=substance_vocabulary,
        )

    def extract(
        self,
        image: Path | str,
        *,
        paper_text: str | None = None,
        figure_name: str | None = None,
        paper_title: str | None = None,
        substance_vocabulary: Sequence[str] | None = None,
    ) -> ExtractionResult:
        image_path = Path(image)
        mime_type = mimetypes.guess_type(image_path.name)[0]
        if mime_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"unsupported image type for {image_path}")
        image_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
        task = build_extraction_text(
            paper=paper_text,
            figure_name=figure_name,
            paper_title=paper_title,
            substance_vocabulary=substance_vocabulary,
        )
        if self.provider == "openai":
            raw_text, usage, response_id = self._call_openai(image_b64, mime_type, task)
        elif self.provider == "gemini":
            raw_text, usage, response_id = self._call_gemini(image_b64, mime_type, task)
        else:
            raw_text, usage, response_id = self._call_anthropic(
                image_b64, mime_type, task
            )
        if not raw_text:
            raise ValueError("model returned no structured output")
        data = extract_json(raw_text)
        timecourses = data["timecourses"]
        curve_count = data["curve_count"]
        warnings = []
        if curve_count != len(timecourses):
            warnings.append(
                f"model inventoried {curve_count} curves but returned {len(timecourses)}"
            )
        return ExtractionResult(
            timecourses=timecourses,
            curve_count=curve_count,
            model=self.model,
            raw_text=raw_text,
            usage=usage,
            response_id=response_id,
            warnings=warnings,
        )

    def _call_openai(
        self, image_b64: str, mime_type: str, task: str
    ) -> tuple[str, dict[str, Any], str | None]:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            store=False,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{image_b64}",
                            "detail": self.image_detail,
                        },
                        {"type": "input_text", "text": task},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "graph_timecourses",
                    "schema": OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        )
        return (
            response.output_text or "",
            self._usage_dict(getattr(response, "usage", None)),
            getattr(response, "id", None),
        )

    def _call_gemini(
        self, image_b64: str, mime_type: str, task: str
    ) -> tuple[str, dict[str, Any], str | None]:
        response = self.client.interactions.create(
            model=self.model,
            system_instruction=SYSTEM,
            input=[
                {"type": "image", "mime_type": mime_type, "data": image_b64},
                {"type": "text", "text": task},
            ],
            generation_config={"thinking_level": self.reasoning_effort},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": OUTPUT_SCHEMA,
            },
        )
        return (
            response.output_text or "",
            self._usage_dict(getattr(response, "usage", None)),
            getattr(response, "id", None),
        )

    def _call_anthropic(
        self, image_b64: str, mime_type: str, task: str
    ) -> tuple[str, dict[str, Any], str | None]:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": image_b64,
                },
            },
            {"type": "text", "text": task},
        ]
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.reasoning_effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
            messages=[{"role": "user", "content": content}],
        ) as stream:
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise ValueError("model refused the extraction request")
        raw_text = next(
            (block.text for block in message.content if block.type == "text"), ""
        )
        return (
            raw_text,
            self._usage_dict(getattr(message, "usage", None)),
            getattr(message, "id", None),
        )

    @staticmethod
    def _usage_dict(usage: Any | None) -> dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            return usage.model_dump(exclude_none=True)
        return {
            key: value
            for key in ("input_tokens", "output_tokens")
            if (value := getattr(usage, key, None)) is not None
        }
