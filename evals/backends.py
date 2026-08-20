"""Model backends that turn a benchmark Entry into predicted timecourses.

Three real backends:

  openai      Calls the Responses API directly. Needs OPENAI_API_KEY. Sends the
              graph at original detail and constrains the answer with a JSON schema.

  anthropic   Calls the Messages API directly. Needs credentials (ANTHROPIC_API_KEY
              or an `ant auth login` profile). Sends the figure as an image block
              and constrains the answer with a JSON schema.
  claude-cli  Shells out to the authenticated `claude` CLI in headless mode. Slower
              and coarser, but runs wherever Claude Code already works.

Plus two synthetic backends (`oracle`, `noisy-oracle`) that read the gold labels.
They exist to check that the scorer says 1.0 when the answer is right and degrades
smoothly as the answer gets worse.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import Entry
from .prompt import OUTPUT_SCHEMA, SYSTEM, build_task_text, truncate_paper
from .timecourse_extractor import TimecourseExtractor, extract_json

DEFAULT_MAX_TOKENS = 32000


@dataclass
class Prediction:
    """One model's answer for one figure, plus what it cost to get it."""

    id: str
    model: str
    timecourses: list[dict] = field(default_factory=list)
    raw_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "timecourses": self.timecourses,
            "usage": self.usage,
            "elapsed_s": round(self.elapsed_s, 2),
            "error": self.error,
            "raw_text": self.raw_text if self.error else "",
        }


class Backend:
    name = "base"

    def __init__(
        self,
        model: str,
        *,
        paper_chars: int | None = 40000,
        substance_hint: bool = False,
    ):
        self.model = model
        self.paper_chars = paper_chars
        self.substance_hint = substance_hint

    def paper_for(self, entry: Entry) -> str:
        return truncate_paper(entry.paper_text, self.paper_chars)

    def predict(self, entry: Entry) -> Prediction:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicBackend(Backend):
    """Thin benchmark adapter for the reusable Anthropic extractor."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        *,
        effort: str = "high",
        client: Any | None = None,
        **kw,
    ):
        super().__init__(model, **kw)
        self.effort = effort
        self.extractor = TimecourseExtractor(
            provider="anthropic",
            model=model,
            reasoning_effort=effort,
            max_output_tokens=DEFAULT_MAX_TOKENS,
            client=client,
        )

    def predict(self, entry: Entry) -> Prediction:
        started = time.monotonic()
        pred = Prediction(id=entry.id, model=self.model)
        try:
            result = self.extractor(
                entry.image,
                paper_text=self.paper_for(entry),
                figure_name=entry.raw.get("figure", entry.id),
                paper_title=entry.reference_title,
                substance_vocabulary=(
                    entry.substance_names if self.substance_hint else None
                ),
            )
            pred.raw_text = result.raw_text
            pred.usage = result.usage
            pred.timecourses = result.timecourses
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the run
            pred.error = f"{type(exc).__name__}: {exc}"
        finally:
            pred.elapsed_s = time.monotonic() - started
        return pred


class OpenAIBackend(Backend):
    """Responses API call: original-detail image + paper + strict JSON output."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.6-sol",
        *,
        effort: str = "high",
        image_detail: str = "original",
        client: Any | None = None,
        **kw,
    ):
        super().__init__(model, **kw)
        self.effort = effort
        self.image_detail = image_detail
        self.extractor = TimecourseExtractor(
            provider="openai",
            model=model,
            reasoning_effort=effort,
            image_detail=image_detail,
            max_output_tokens=DEFAULT_MAX_TOKENS,
            client=client,
        )

    def predict(self, entry: Entry) -> Prediction:
        started = time.monotonic()
        pred = Prediction(id=entry.id, model=self.model)
        try:
            result = self.extractor(
                entry.image,
                paper_text=self.paper_for(entry),
                figure_name=entry.raw.get("figure", entry.id),
                paper_title=entry.reference_title,
                substance_vocabulary=(
                    entry.substance_names if self.substance_hint else None
                ),
            )
            pred.raw_text = result.raw_text
            pred.usage = result.usage
            pred.timecourses = result.timecourses
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the run
            pred.error = f"{type(exc).__name__}: {exc}"
        finally:
            pred.elapsed_s = time.monotonic() - started
        return pred


class GeminiBackend(Backend):
    """Thin benchmark adapter for the reusable Gemini extractor."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3.6-flash",
        *,
        thinking_level: str = "high",
        client: Any | None = None,
        **kw,
    ):
        super().__init__(model, **kw)
        self.thinking_level = thinking_level
        self.extractor = TimecourseExtractor(
            provider="gemini",
            model=model,
            reasoning_effort=thinking_level,
            max_output_tokens=DEFAULT_MAX_TOKENS,
            client=client,
        )

    def predict(self, entry: Entry) -> Prediction:
        started = time.monotonic()
        pred = Prediction(id=entry.id, model=self.model)
        try:
            result = self.extractor(
                entry.image,
                paper_text=self.paper_for(entry),
                figure_name=entry.raw.get("figure", entry.id),
                paper_title=entry.reference_title,
                substance_vocabulary=(
                    entry.substance_names if self.substance_hint else None
                ),
            )
            pred.raw_text = result.raw_text
            pred.usage = result.usage
            pred.timecourses = result.timecourses
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the run
            pred.error = f"{type(exc).__name__}: {exc}"
        finally:
            pred.elapsed_s = time.monotonic() - started
        return pred


class ClaudeCLIBackend(Backend):
    """Headless `claude -p`, reading the figure and paper off disk with the Read tool."""

    name = "claude-cli"

    def __init__(self, model: str = "opus", *, timeout_s: int = 900, **kw):
        super().__init__(model, **kw)
        self.timeout_s = timeout_s

    def _prompt(self, entry: Entry, paper_path: Path | None) -> str:
        lines = [f"Read the figure image at {entry.image.resolve()}."]
        if paper_path is not None:
            lines.append(f"Read the paper text at {paper_path}.")
        lines += [
            "",
            SYSTEM,
            "",
            build_task_text(entry, paper=None, substance_hint=self.substance_hint),
            "",
            "Output ONLY a single JSON object matching this schema, with no prose "
            "and no code fence:",
            json.dumps(OUTPUT_SCHEMA),
        ]
        return "\n".join(lines)

    def predict(self, entry: Entry) -> Prediction:
        started = time.monotonic()
        pred = Prediction(id=entry.id, model=self.model)
        tmp: Path | None = None
        try:
            paper = self.paper_for(entry)
            if paper:
                fd, name = tempfile.mkstemp(suffix=".txt", prefix=f"{entry.study_sid}_")
                tmp = Path(name)
                with os.fdopen(fd, "w") as fh:
                    fh.write(paper)

            proc = subprocess.run(
                [
                    "claude",
                    "-p",
                    self._prompt(entry, tmp),
                    "--allowedTools",
                    "Read",
                    "--output-format",
                    "json",
                    "--model",
                    self.model,
                    "--add-dir",
                    str(entry.image.parent.resolve()),
                ]
                + (["--add-dir", str(tmp.parent)] if tmp else []),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode != 0:
                pred.error = f"claude exited {proc.returncode}: {proc.stderr[-500:]}"
                return pred

            envelope = json.loads(proc.stdout)
            if envelope.get("is_error"):
                pred.error = f"claude error: {envelope.get('result', '')[:500]}"
                return pred
            pred.raw_text = envelope.get("result", "")
            usage = envelope.get("usage", {})
            pred.usage = {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cost_usd": envelope.get("total_cost_usd"),
            }
            pred.timecourses = extract_json(pred.raw_text).get("timecourses", [])
        except subprocess.TimeoutExpired:
            pred.error = f"timeout after {self.timeout_s}s"
        except Exception as exc:  # noqa: BLE001
            pred.error = f"{type(exc).__name__}: {exc}"
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            pred.elapsed_s = time.monotonic() - started
        return pred


class OracleBackend(Backend):
    """Returns the gold answer. A correct scorer must give this a perfect score."""

    name = "oracle"

    def __init__(self, model: str = "oracle", **kw):
        super().__init__(model, **kw)

    def predict(self, entry: Entry) -> Prediction:
        return Prediction(
            id=entry.id, model=self.model, timecourses=entry.raw["timecourses"]
        )


class NoisyOracleBackend(Backend):
    """Gold answer with multiplicative noise on the values, for metric sensitivity checks."""

    name = "noisy-oracle"

    def __init__(
        self, model: str = "noisy-oracle", *, noise: float = 0.15, seed: int = 0, **kw
    ):
        super().__init__(model, **kw)
        self.noise = noise
        self.seed = seed

    def predict(self, entry: Entry) -> Prediction:
        import numpy as np

        rng = np.random.default_rng(abs(hash((entry.id, self.seed))) % (2**32))
        tcs = json.loads(json.dumps(entry.raw["timecourses"]))
        for tc in tcs:
            for p in tc["points"]:
                for key in ("mean", "value"):
                    if p.get(key) is not None:
                        p[key] = float(p[key] * (1.0 + rng.normal(0, self.noise)))
        return Prediction(id=entry.id, model=self.model, timecourses=tcs)


BACKENDS = {
    "openai": OpenAIBackend,
    "gemini": GeminiBackend,
    "anthropic": AnthropicBackend,
    "claude-cli": ClaudeCLIBackend,
    "oracle": OracleBackend,
    "noisy-oracle": NoisyOracleBackend,
}


def build_backend(kind: str, model: str | None, **kw) -> Backend:
    if kind not in BACKENDS:
        raise SystemExit(f"unknown backend {kind!r}; choose from {sorted(BACKENDS)}")
    cls = BACKENDS[kind]
    return cls(model, **kw) if model else cls(**kw)
