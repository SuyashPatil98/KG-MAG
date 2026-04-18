"""
KG-MAG — OpenAI LLM Client
==========================
Thin wrapper around the OpenAI Python SDK.

Design principles
-----------------
- Retry with exponential backoff on rate-limit / transient errors
- Structured extraction: system prompt + user prompt → parsed output
- Token tracking for cost attribution
- Streaming support (optional, for live frontends)
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import structlog
from openai import OpenAI

from backend.core.config import get_settings

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.5


class LLMClient:
    """
    Singleton-ish wrapper for OpenAI API calls.
    Import and instantiate once per process.
    Supports mock mode for testing when API key is invalid/missing.
    """

    def __init__(self, use_mock: bool = False) -> None:
        cfg = get_settings()
        self._use_mock = (
            use_mock or not cfg.openai_api_key or cfg.openai_api_key.strip() == ""
        )
        self._model = cfg.llm_model
        self._max_tokens = cfg.max_article_tokens
        self._token_log: list[dict] = []

        if self._use_mock:
            logger.warning("LLMClient initialized in MOCK MODE - no real API calls")
            self._client = None
        else:
            try:
                self._client = OpenAI(api_key=cfg.openai_api_key)
                logger.info("LLMClient initialized with OpenAI API", model=self._model)
            except Exception as e:
                logger.warning(
                    "Failed to initialize OpenAI client, using mock mode", error=str(e)
                )
                self._client = None
                self._use_mock = True

    def _mock_response(self, tag: str) -> str:
        """Deterministic fallback content used when API access is unavailable."""
        if tag == "planner":
            return (
                '{"title": "Engineering-Grade Deep Dive", '
                '"subtitle": "Grounded technical analysis from your corpus", '
                '"target_audience": "software engineers and technical leaders", '
                '"estimated_reading_time": 10, '
                '"sections": ['
                '"System Context and Problem Framing", '
                '"Core Mechanisms and Data Flow", '
                '"Implementation Patterns and Constraints", '
                '"Trade-offs, Failure Modes, and Mitigations", '
                '"Production Hardening and Next Steps"], '
                '"seo_keywords": ["technical architecture", "implementation patterns", "system trade-offs"]}'
            )

        if tag == "critic_ground":
            return (
                '{"is_grounded": true, "supporting_chunk_ids": [], "confidence": 0.6}'
            )

        if tag == "writer_conclusion":
            return (
                "The strongest technical systems emerge when teams connect theory, implementation, "
                "and operational constraints into one coherent design. Grounding claims in source "
                "evidence improves both accuracy and decision quality, which is what turns a good "
                "prototype into dependable production software."
            )

        return (
            "The section synthesizes source-backed mechanisms, implementation detail, and practical "
            "trade-offs in a clear technical narrative [CITE:mock_chunk]."
        )

    # ── Core completion ───────────────────────────────────────────────────────

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tag: str = "generic",
    ) -> str:
        """
        Send a system+user prompt, return assistant text.
        Retries on rate-limit errors with exponential backoff.
        """
        mt = max_tokens or self._max_tokens

        if self._use_mock:
            return self._mock_response(tag)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=mt,
                )
                elapsed = time.perf_counter() - t0

                usage = getattr(resp, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

                self._token_log.append(
                    {
                        "tag": tag,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                logger.debug(
                    "LLM call",
                    tag=tag,
                    in_tok=input_tokens,
                    out_tok=output_tokens,
                    elapsed_s=round(elapsed, 2),
                )

                text = ""
                if resp.choices and resp.choices[0].message:
                    text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("OpenAI returned empty text")
                return text

            except Exception as e:
                wait = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI call failed, retrying",
                    attempt=attempt,
                    wait_s=wait,
                    error=str(e),
                )
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(wait)

        raise RuntimeError("LLM call failed after max retries")

    # ── Structured JSON extraction ────────────────────────────────────────────

    def complete_json(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        tag: str = "json",
    ) -> Any:
        """
        Ask the model to respond with a JSON object.
        Strips markdown fences if the model wraps it.
        """
        import json
        import re

        # Reinforce JSON-only output in the system prompt
        json_system = (
            system.rstrip()
            + "\n\nRespond ONLY with valid JSON. No explanation, no markdown code fences."
        )
        raw = self.complete(
            json_system, user, max_tokens=max_tokens, temperature=0.2, tag=tag
        )

        # Strip ```json ... ``` if model ignores instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("JSON parse failed", raw_snippet=raw[:200], error=str(e))
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Stream text tokens as they are generated.
        Yields text deltas (str), use in async contexts with asyncio.
        """
        mt = max_tokens or self._max_tokens
        if self._use_mock:
            yield self._mock_response("stream")
            return

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=mt,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            text = getattr(delta, "content", None)
            if text:
                yield text

    # ── Token accounting ──────────────────────────────────────────────────────

    def reset_token_log(self) -> None:
        """Reset per-process token accounting before a new generation run."""
        self._token_log = []

    def token_summary(
        self,
        include_tag_prefixes: tuple[str, ...] | None = None,
    ) -> dict:
        records = self._token_log
        if include_tag_prefixes:
            prefixes = tuple(p.lower() for p in include_tag_prefixes)
            records = [
                r
                for r in self._token_log
                if str(r.get("tag", "")).lower().startswith(prefixes)
            ]

        total_in = sum(r["input_tokens"] for r in records)
        total_out = sum(r["output_tokens"] for r in records)
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "calls": len(records),
            "breakdown": records,
        }
