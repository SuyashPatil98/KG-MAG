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

    # ── Prompt engineering ────────────────────────────────────────────────────

    def build_header_prompt(self, topic: str, tone: str = "informative") -> str:
        return (
            f"A stunning, professional vector diagram for a Medium article about '{topic}'. "
            f"Tone: {tone}. "
            "Style: high-quality editorial photography or digital illustration. "
            "Clean, modern, no text overlays. Wide aspect ratio (16:9). "
            "Suitable for a technology/science publication."
        )

    def build_section_prompt(self, heading: str, topic: str) -> str:
        return (
            f"An illustrative image for the section '{heading}' in an article about '{topic}'. "
            "Style: clean infographic or conceptual illustration. "
            "Minimal, professional, no text. Square or 4:3 aspect ratio."
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

        async with httpx.AsyncClient(timeout=60.0) as client:
            for delay in _RETRY_DELAYS:
                try:
                    resp = await client.post(
                        self._api_url, json=payload, headers=headers
                    )
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

        # Direct URL (CDN-hosted)
        if url := data.get("image_url"):
            return url

        # Base64 encoded image
        if b64 := data.get("image_b64") or data.get("image"):
            return self._save_b64_image(b64, prompt)

        logger.warning("Unexpected image API response format", keys=list(data.keys()))
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
        max_section_images: int = 3,
    ) -> dict[str, Optional[str]]:
        """
        Generate header + section images concurrently.
        Returns dict mapping 'header' and each heading to its image URL/path.
        """
        tasks: dict[str, asyncio.Task] = {}

        # Header image
        header_prompt = self.build_header_prompt(topic, tone)
        tasks["header"] = asyncio.create_task(
            self.generate(header_prompt, width=1200, height=630)
        )

        # Section images (limit to avoid excessive API calls / cost)
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
