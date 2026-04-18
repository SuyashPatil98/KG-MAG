"""
KG-MAG — Image Generation Tool (Nanobananpro)
===============================================
Generates header and section images from text prompts.

Design
------
- Async-first: uses httpx.AsyncClient for non-blocking calls
- Retry logic with exponential backoff
- Saves images to artifacts directory and returns public URLs
- Falls back gracefully if image generation fails (article still completes)
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Optional

import httpx
import structlog

from backend.core.config import get_settings

logger = structlog.get_logger(__name__)

_RETRY_DELAYS = [1.0, 2.0, 4.0]


class ImageGenerationTool:
    """
    Wraps the Nanobananpro image generation API.

    Expected API response (adapt to actual Nanobananpro spec):
    {
      "status": "success",
      "image_url": "https://...",   // OR
      "image_b64": "<base64>",      // base64-encoded PNG/JPG
      "width": 1024,
      "height": 512
    }
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._api_key = cfg.nanobananpro_api_key
        self._api_url = str(cfg.nanobananpro_api_url)
        self._artifacts_path = cfg.artifacts_path
        self._artifacts_path.mkdir(parents=True, exist_ok=True)

    def _is_google_generate_content_endpoint(self) -> bool:
        url = self._api_url.lower()
        return "generativelanguage.googleapis.com" in url and ":generatecontent" in url

    def _request_spec(
        self,
        prompt: str,
        width: int,
        height: int,
        style: str,
    ) -> tuple[str, dict[str, str], dict]:
        """
        Build provider-specific request URL, headers, and payload.
        """
        if self._is_google_generate_content_endpoint():
            url = self._api_url
            if "key=" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}key={self._api_key}"

            headers = {
                "Content-Type": "application/json",
            }
            # Gemini image endpoints are prompt-driven; width/height/style are encoded in prompt guidance.
            enriched_prompt = (
                f"{prompt} "
                f"Render style: {style}. Target dimensions: {width}x{height}. "
                "Return one high-quality image."
            )
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": enriched_prompt,
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            return url, headers, payload

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "style": style,
            "num_images": 1,
        }
        return self._api_url, headers, payload

    # ── Prompt engineering ────────────────────────────────────────────────────

    def build_header_prompt(
        self,
        topic: str,
        tone: str = "informative",
        section_headings: list[str] | None = None,
        content_brief: list[str] | None = None,
    ) -> str:
        key_points = ", ".join((section_headings or [])[:6]).strip()
        if not key_points:
            key_points = topic

        grounded_brief = " | ".join((content_brief or [])[:8]).strip()
        if not grounded_brief:
            grounded_brief = key_points

        return (
            f"Create one journal research-grade vector ecosystem diagram for an article about '{topic}'. "
            f"Tone: {tone}. "
            f"Key concepts to represent: {key_points}. "
            f"The diagram need not include everything mentioned below, but should capture the most relevant things and must be cohesive with the content and the topic."
            f"Grounding context from article evidence: {grounded_brief}. "
            "The diagram must explain the full ecosystem across: "
            "AI research, data science experimentation, model development, and data engineering operations. "
            "Compose as layered system architecture with explicit flow and feedback loops: "
            "data sources and ingestion -> data quality and feature/embedding pipelines -> "
            "experimentation/training/evaluation -> deployment/inference -> monitoring/observability -> "
            "feedback to research and iteration. "
            "Prioritize explaining the provided article evidence; do not introduce unrelated concepts. "
            "Include governance/reproducibility cues (lineage, versioning, evaluation gates) as visual modules. "
            "Visual grammar: clean vector lines, modular blocks, directional arrows, "
            "clear hierarchy, publication-ready composition, high information density without clutter. "
            "Style: technical figure for a peer-reviewed systems paper. "
            "Prefer minimal, legible labels only where necessary for clarity. "
            "Avoid photorealism, avoid decorative backgrounds, avoid logos/watermarks. "
            "Use neutral scientific palette and high contrast for readability. "
            "Aspect ratio 16:9."
        )

    def build_section_prompt(self, heading: str, topic: str) -> str:
        return (
            f"A technical editorial illustration for the section '{heading}' in an article about '{topic}'. "
            "Style: polished concept visualization with clear structure, credible technical motifs, "
            "and focused composition. Keep it understandable for engineers and product teams. "
            "No text overlays, no logos, no watermarks. Square or 4:3 aspect ratio."
        )

    # ── Core generation ───────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 512,
        style: str = "photorealistic",
    ) -> Optional[str]:
        """
        Generate an image and return its local file path (or URL).
        Returns None on failure so the pipeline can continue without images.
        """
        url, headers, payload = self._request_spec(prompt, width, height, style)

        async with httpx.AsyncClient(timeout=60.0) as client:
            for delay in _RETRY_DELAYS:
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    return await self._handle_response(data, prompt)

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        logger.warning("Image API rate limited — retrying", delay=delay)
                        await asyncio.sleep(delay)
                        continue
                    logger.error(
                        "Image API HTTP error",
                        status=e.response.status_code,
                        body=e.response.text[:200],
                    )
                    return None
                except httpx.RequestError as e:
                    err_text = str(e).lower()
                    # DNS/host resolution failures are not transient in most local/dev setups.
                    if any(
                        marker in err_text
                        for marker in [
                            "name or service not known",
                            "nodename nor servname",
                            "temporary failure in name resolution",
                            "no address associated with hostname",
                        ]
                    ):
                        logger.error(
                            "Image API DNS resolution failed — skipping retries",
                            error=str(e),
                        )
                        return None

                    logger.error("Image API request error", error=str(e))
                    await asyncio.sleep(delay)

        logger.error("Image generation failed after retries", prompt=prompt[:80])
        return None

    async def _handle_response(self, data: dict, prompt: str) -> Optional[str]:
        """
        Parse API response — handles both URL and base64 formats.
        Saves base64 images to disk and returns the local path.
        """
        if data.get("status") not in ("success", None):
            logger.warning("Image API non-success status", data=data)
            return None

        # Google Generative Language response format
        if b64 := self._extract_gemini_image_b64(data):
            return self._save_b64_image(b64, prompt)

        # Direct URL (CDN-hosted)
        if url := data.get("image_url"):
            return url

        # Base64 encoded image
        if b64 := data.get("image_b64") or data.get("image"):
            return self._save_b64_image(b64, prompt)

        logger.warning("Unexpected image API response format", keys=list(data.keys()))
        return None

    @staticmethod
    def _extract_gemini_image_b64(data: dict) -> Optional[str]:
        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            return None

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue

                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict):
                    b64 = inline.get("data")
                    if isinstance(b64, str) and b64:
                        return b64

        return None

    def _save_b64_image(self, b64_data: str, prompt: str) -> str:
        """Decode and save a base64 image; return the file path."""
        # Strip data URI prefix if present
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]

        img_bytes = base64.b64decode(b64_data)
        fname = f"img_{uuid.uuid4().hex[:12]}.png"
        dest = self._artifacts_path / fname
        dest.write_bytes(img_bytes)
        logger.debug("Saved generated image", path=str(dest))
        return f"/artifacts/{fname}"

    # ── Batch helpers ─────────────────────────────────────────────────────────

    async def generate_article_images(
        self,
        topic: str,
        section_headings: list[str],
        tone: str = "informative",
        max_section_images: int = 0,
        content_brief: list[str] | None = None,
    ) -> dict[str, Optional[str]]:
        """
        Generate header + section images concurrently.
        Returns dict mapping 'header' and each heading to its image URL/path.
        """
        tasks: dict[str, asyncio.Task] = {}

        # Primary research-style vector diagram (single image by default)
        header_prompt = self.build_header_prompt(
            topic,
            tone,
            section_headings,
            content_brief=content_brief,
        )
        tasks["header"] = asyncio.create_task(
            self.generate(header_prompt, width=1024, height=576, style="vector")
        )

        # Section images (limit to avoid excessive API calls / cost)
        if max_section_images > 0:
            for heading in section_headings[:max_section_images]:
                section_prompt = self.build_section_prompt(heading, topic)
                tasks[heading] = asyncio.create_task(
                    self.generate(section_prompt, width=800, height=600)
                )

        results: dict[str, Optional[str]] = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                logger.error("Image task failed", key=key, error=str(e))
                results[key] = None

        return results
