"""
KG-MAG — Gemini LLM Client
==========================
Thin wrapper around the Google Generative AI SDK.

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
from google import genai
from google.genai import types

from backend.core.config import get_settings

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.5


class LLMClient:
    """
    Singleton-ish wrapper for Gemini API calls.
    Import and instantiate once per process.
    Supports mock mode for testing when API key is invalid/missing.
    """

    def __init__(self, use_mock: bool = False) -> None:
        cfg = get_settings()
        self._use_mock = use_mock or not cfg.gemini_api_key or cfg.gemini_api_key.strip() == ""
        self._model = cfg.llm_model
        self._max_tokens = cfg.max_article_tokens
        self._token_log: list[dict] = []
        
        if self._use_mock:
            logger.warning("LLMClient initialized in MOCK MODE - no real API calls")
            self._client = None
        else:
            try:
                self._client = genai.Client(api_key=cfg.gemini_api_key)
                logger.info("LLMClient initialized with Gemini API", model=self._model)
            except Exception as e:
                logger.warning("Failed to initialize Gemini client, using mock mode", error=str(e))
                self._client = None
                self._use_mock = True

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

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                t0 = time.perf_counter()
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=mt,
                    ),
                )
                elapsed = time.perf_counter() - t0

                usage = getattr(resp, "usage_metadata", None)
                input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
                output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)

                self._token_log.append({
                    "tag": tag,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "elapsed_s": round(elapsed, 2),
                })
                logger.debug(
                    "LLM call",
                    tag=tag,
                    in_tok=input_tokens,
                    out_tok=output_tokens,
                    elapsed_s=round(elapsed, 2),
                )

                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("Gemini returned empty text")
                return text

            except Exception as e:
                wait = _BASE_DELAY * (2 ** (attempt - 1))
                logger.warning("Gemini call failed, retrying", attempt=attempt, wait_s=wait, error=str(e))
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
        Ask Gemini to respond with a JSON object.
        Strips markdown fences if Gemini wraps it.
        """
        import json
        import re

        # Reinforce JSON-only output in the system prompt
        json_system = (
            system.rstrip()
            + "\n\nRespond ONLY with valid JSON. No explanation, no markdown code fences."
        )
        raw = self.complete(json_system, user, max_tokens=max_tokens, temperature=0.2, tag=tag)

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
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=mt,
            ),
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    # ── Token accounting ──────────────────────────────────────────────────────

    def token_summary(self) -> dict:
        total_in = sum(r["input_tokens"] for r in self._token_log)
        total_out = sum(r["output_tokens"] for r in self._token_log)
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "calls": len(self._token_log),
            "breakdown": self._token_log,
        }