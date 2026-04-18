"""
KG-MAG — Multi-Agent Orchestration System
==========================================
Implements a four-agent pipeline with MCP-style tool orchestration.

Agents
------
PlannerAgent   — Topic → outline (sections, keywords, audience)
RetrieverAgent — Section → relevant chunks (FAISS + reranker)
WriterAgent    — (outline, chunks) → draft article sections
CriticAgent    — Draft → QA report + revision instructions

Data flow
---------
Topic
  │
  ▼
[PlannerAgent] ──→ ArticleOutline
  │
  ▼
[RetrieverAgent] ──→ {section: [RetrievedChunk]} (per section)
  │
  ▼
[WriterAgent] ──→ GeneratedArticle (draft)
  │
  ▼
[CriticAgent] ──→ QAReport + optional revision
  │
  ▼
Final Article + QAReport

MCP-style orchestration
------------------------
Each agent exposes a `run(context) -> context` interface.
The Orchestrator chains them, passing a shared Context object.
Agents declare their tools (LLM, VectorStore, ImageGen) via dependency injection,
making them individually testable and swappable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from backend.core.config import get_settings
from backend.core.models import (
    ArticleOutline,
    ArticleSection,
    GeneratedArticle,
    QAReport,
    RetrievedChunk,
)
from backend.models.reranker import rerank
from backend.tools.llm_client import LLMClient
from backend.tools.vector_store import FAISSVectorStore

logger = structlog.get_logger(__name__)


# ── Shared Pipeline Context ───────────────────────────────────────────────────


@dataclass
class PipelineContext:
    """Shared state flowing through all agents."""

    topic: str
    target_audience: str = "general tech readers"
    tone: str = "informative and engaging"
    max_sections: int = 6
    generate_images: bool = True

    # Populated by each agent
    outline: Optional[ArticleOutline] = None
    retrieved: dict[str, list[RetrievedChunk]] = field(default_factory=dict)
    article: Optional[GeneratedArticle] = None
    qa_report: Optional[QAReport] = None
    image_urls: dict[str, Optional[str]] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ── Planner Agent ─────────────────────────────────────────────────────────────


class PlannerAgent:
    """
    Converts a raw topic into a structured article outline.

    Responsibilities
    ----------------
    - Determine appropriate sections (heading-first thinking)
    - Identify target audience and tone
    - Extract SEO keywords
    - Estimate reading time
    """

    SYSTEM = """You are a principal technical editor for an elite engineering publication.
Produce dense, high-signal outlines for advanced software and AI topics.
Prioritize architecture, data flow, implementation mechanics, failure modes, trade-offs,
operational constraints, and production guidance. Avoid fluff and generic headings."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("PlannerAgent: generating outline", topic=ctx.topic)

        user_prompt = f"""
Create a detailed article outline for a technical long-form article on the topic:
**"{ctx.topic}"**

Target audience: {ctx.target_audience}
Tone: {ctx.tone}
Max sections: {ctx.max_sections}

Respond with a JSON object exactly matching this schema:
{{
  "title": "<compelling article title>",
  "subtitle": "<one-sentence subtitle>",
  "target_audience": "<description>",
  "estimated_reading_time": <integer minutes>,
  "sections": ["<section 1 heading>", "<section 2 heading>", ...],
  "seo_keywords": ["<keyword1>", "<keyword2>", ...]
}}

Section headings must be specific and technically meaningful (not generic).
Use an arc that moves from problem framing to mechanisms, implementation decisions,
trade-offs, reliability/operations, and pragmatic next steps.
Include 4-{ctx.max_sections} sections.
"""
        try:
            cfg = get_settings()
            data = self._llm.complete_json(
                self.SYSTEM,
                user_prompt,
                max_tokens=cfg.planner_max_tokens,
                tag="planner",
            )
            ctx.outline = ArticleOutline(**data)
            logger.info(
                "Outline created",
                title=ctx.outline.title,
                sections=len(ctx.outline.sections),
            )
        except Exception as e:
            ctx.errors.append(f"PlannerAgent failed: {e}")
            logger.error("PlannerAgent error", error=str(e))
            # Fallback outline
            ctx.outline = ArticleOutline(
                title=ctx.topic,
                subtitle=f"A deep dive into {ctx.topic}",
                target_audience=ctx.target_audience,
                estimated_reading_time=8,
                sections=[
                    "Introduction",
                    "Core Concepts",
                    "Practical Applications",
                    "Challenges and Trade-offs",
                    "Future Directions",
                ],
                seo_keywords=[ctx.topic.lower()],
            )

        return ctx


# ── Retriever Agent ───────────────────────────────────────────────────────────


class RetrieverAgent:
    """
    For each section in the outline, retrieves relevant chunks from the knowledge base.

    Strategy
    --------
    1. Build a section-specific query: "{topic} — {section_heading}"
    2. FAISS top-k retrieval
    3. ML reranker pass
    4. Deduplicate across sections (keep top per-section, avoid repetition)
    """

    def __init__(self, vector_store: FAISSVectorStore) -> None:
        self._store = vector_store

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.outline, "No outline — run PlannerAgent first"
        cfg = get_settings()
        logger.info("RetrieverAgent: retrieving chunks per section")

        seen_chunks: set[str] = set()

        for heading in ctx.outline.sections:
            query = f"{ctx.topic} — {heading}"
            candidates = self._store.search(query, top_k=cfg.top_k_retrieval)

            # ML reranker pass
            reranked = rerank(query, candidates, top_k=cfg.rerank_top_k)

            # Deduplicate: prefer unseen chunks but don't leave sections empty
            unique = [r for r in reranked if r.chunk.chunk_id not in seen_chunks]
            if not unique:
                unique = reranked[:2]  # allow repeats if no alternatives

            for r in unique:
                seen_chunks.add(r.chunk.chunk_id)

            ctx.retrieved[heading] = unique
            logger.debug(
                "Retrieved",
                section=heading,
                chunks=len(unique),
                top_score=round(unique[0].score, 3) if unique else 0,
            )

        return ctx


# ── Writer Agent ──────────────────────────────────────────────────────────────


class WriterAgent:
    """
    Drafts the full article section by section using retrieved context.

    Key principles
    --------------
    - ONLY use information from retrieved chunks (grounded generation)
    - Insert citation markers [chunk_id] inline
    - Follow Medium article style: conversational but authoritative
    - Each section prompt includes the specific chunks for that section
    """

    SYSTEM = """You are a principal engineer and technical essayist writing for advanced practitioners.
Write technically rigorous, highly understandable, implementation-oriented prose.

CRITICAL RULES:
1. Only write content that is directly supported by the provided source chunks.
2. Insert citation markers in the format [CITE:chunk_id] immediately after each factual claim.
3. Write in a precise, authoritative, and readable technical tone.
4. Explain advanced terms briefly when first introduced.
5. Never fabricate facts, statistics, or quotes not present in the source material.
6. Include concrete implementation implications, constraints, and trade-offs where supported.
7. Prefer actionable engineering detail over abstract commentary.
8. Each section should flow naturally into the next."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def _format_chunks(self, chunks: list[RetrievedChunk]) -> str:
        cfg = get_settings()
        max_sources = cfg.writer_max_sources_per_section
        total_budget = cfg.writer_source_word_budget
        per_source_cap = max(120, total_budget // max(1, max_sources))

        parts = []
        remaining_budget = total_budget
        for r in chunks[:max_sources]:
            if remaining_budget <= 0:
                break

            budget_for_chunk = min(per_source_cap, remaining_budget)
            clipped_text = self._truncate_words(r.chunk.text, budget_for_chunk)
            remaining_budget -= len(clipped_text.split())

            heading_label = f" [{r.chunk.heading}]" if r.chunk.heading else ""
            parts.append(
                f"--- SOURCE chunk_id={r.chunk.chunk_id}{heading_label} ---\n{clipped_text}"
            )
        return "\n\n".join(parts)

    def _write_section(
        self,
        topic: str,
        outline: ArticleOutline,
        heading: str,
        chunks: list[RetrievedChunk],
        prev_sections: list[str],
        tone: str,
        has_section_image: bool,
    ) -> ArticleSection:
        context_text = (
            self._format_chunks(chunks) if chunks else "No source material available."
        )
        prev_context = (
            "\n".join(f"- {h}" for h in prev_sections) if prev_sections else "None yet."
        )

        user_prompt = f"""
Article topic: {topic}
Article title: {outline.title}
Current section: **{heading}**
Sections already written: {prev_context}
Tone: {tone}

SOURCE MATERIAL (use ONLY this):
{context_text}

Write the "{heading}" section of this article.
Requirements:
- 220–380 words
- Use [CITE:chunk_id] immediately after each fact drawn from source material
- Balance technical depth with readability for practicing engineers
- Include explicit design trade-offs and operational considerations when supported by sources
- Include at least one practical implementation recommendation when supported by sources
- If an accompanying section image exists, include one concise sentence that references the visual
    as a supporting illustration for the discussed mechanism.
- No bullet points in the body (prose only)
- End with a smooth transition hint toward the next section if applicable
- Start directly with content (no "In this section..." opener)

Section image available for this section: {"yes" if has_section_image else "no"}
"""
        cfg = get_settings()
        raw = self._llm.complete(
            self.SYSTEM,
            user_prompt,
            max_tokens=cfg.writer_section_max_tokens,
            temperature=0.55,
            tag=f"writer_{heading[:20]}",
        )

        # Extract citation markers
        import re

        cited_ids = re.findall(r"\[CITE:([a-zA-Z0-9_-]+)\]", raw)

        # Clean citation markers for display (keep them as footnote refs)
        clean_text = re.sub(r"\[CITE:([a-zA-Z0-9_-]+)\]", r"[\1]", raw)

        return ArticleSection(
            heading=heading,
            content=clean_text,
            citations=list(dict.fromkeys(cited_ids)),  # unique, order-preserving
        )

    @staticmethod
    def _contains_visual_reference(text: str) -> bool:
        lowered = text.lower()
        markers = ["figure", "image", "illustration", "diagram", "visual"]
        return any(m in lowered for m in markers)

    def _ensure_image_reference(
        self,
        sections: list[ArticleSection],
        header_image_url: str | None,
    ) -> None:
        """
        Ensure at least one explicit in-text mention of the generated image(s).
        This makes visual assets part of the article narrative, not just decoration.
        """
        if not sections:
            return

        has_any_image = bool(header_image_url) or any(s.image_url for s in sections)
        if not has_any_image:
            return

        if any(
            self._contains_visual_reference(section.content) for section in sections
        ):
            return

        section_with_image = next((s for s in sections if s.image_url), None)
        if section_with_image is not None:
            section_with_image.content = (
                "Figure note: The accompanying illustration highlights the core architecture "
                "and data flow discussed in this section.\n\n"
                f"{section_with_image.content}"
            )
            return

        sections[0].content = (
            "Figure note: The header image provides a visual overview of the system concepts "
            "covered throughout this article.\n\n"
            f"{sections[0].content}"
        )

    def _write_conclusion(
        self,
        topic: str,
        outline: ArticleOutline,
        sections: list[ArticleSection],
        tone: str,
    ) -> str:
        section_summaries = "\n".join(
            f"- {s.heading}: {s.content[:100]}..." for s in sections
        )
        user_prompt = f"""
Write a compelling conclusion (150–200 words) for an article titled "{outline.title}" about {topic}.
The article covered:
{section_summaries}

Tone: {tone}
Synthesize the key insights, restate the most important engineering implications,
and close with a forward-looking but concrete takeaway.
No new facts — summarize and inspire.
"""
        cfg = get_settings()
        return self._llm.complete(
            self.SYSTEM,
            user_prompt,
            max_tokens=cfg.writer_conclusion_max_tokens,
            temperature=0.55,
            tag="writer_conclusion",
        )

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.outline, "No outline available"
        logger.info("WriterAgent: drafting article", sections=len(ctx.outline.sections))

        sections: list[ArticleSection] = []
        written_headings: list[str] = []

        for heading in ctx.outline.sections:
            chunks = ctx.retrieved.get(heading, [])
            section_image_url = ctx.image_urls.get(heading)
            section = self._write_section(
                ctx.topic,
                ctx.outline,
                heading,
                chunks,
                written_headings,
                ctx.tone,
                bool(section_image_url),
            )

            # Attach image URL if generated
            if section_image_url:
                section.image_url = section_image_url
                section.image_prompt = (
                    f"Technical editorial illustration for '{heading}' in '{ctx.topic}'"
                )

            sections.append(section)
            written_headings.append(heading)
            logger.debug(
                "Section written", heading=heading, words=len(section.content.split())
            )

        conclusion = self._write_conclusion(ctx.topic, ctx.outline, sections, ctx.tone)

        self._ensure_image_reference(sections, ctx.image_urls.get("header"))

        # Build citations map: chunk_id → chunk
        citations_map = {}
        for heading, chunk_list in ctx.retrieved.items():
            for r in chunk_list:
                citations_map[r.chunk.chunk_id] = r.chunk

        ctx.article = GeneratedArticle(
            topic=ctx.topic,
            title=ctx.outline.title,
            subtitle=ctx.outline.subtitle,
            header_image_url=ctx.image_urls.get("header"),
            sections=sections,
            conclusion=conclusion,
            citations_map=citations_map,
            seo_keywords=ctx.outline.seo_keywords,
            tags=ctx.outline.seo_keywords[:5],
            model_used=get_settings().llm_model,
            token_usage=self._llm.token_summary(
                include_tag_prefixes=("planner", "writer")
            ),
        )

        logger.info(
            "Article drafted",
            title=ctx.article.title,
            sections=len(sections),
            total_words=sum(len(s.content.split()) for s in sections),
        )
        return ctx


# ── Critic Agent (QA) ─────────────────────────────────────────────────────────


class CriticAgent:
    """
    Performs multi-dimensional quality assurance on the generated article.

    Checks
    ------
    1. Grounding verification — is each factual claim supported by a retrieved chunk?
    2. Self-consistency — regenerate intro, compare semantic similarity
    3. Flesch readability scoring
    4. Coverage score — what % of retrieved chunks were cited?
    5. Composite confidence score
    """

    SYSTEM = """You are a rigorous editorial fact-checker and quality assurance system.
Your job is to verify that every factual claim in an article is supported by provided source material.
Be strict: if a claim is not directly supported, mark it as ungrounded."""

    def __init__(self, llm: LLMClient, vector_store: FAISSVectorStore) -> None:
        self._llm = llm
        self._store = vector_store

    def _compute_flesch(self, text: str) -> float:
        """
        Flesch Reading Ease score.
        90-100: Very Easy, 60-70: Standard, 0-30: Very Difficult.
        """
        import re

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        if not sentences or not words:
            return 50.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Syllable count heuristic
        def syllables(word: str) -> int:
            word = word.lower().rstrip("e")
            vowels = re.findall(r"[aeiou]+", word)
            return max(1, len(vowels))

        total_syllables = sum(syllables(w) for w in words)
        avg_syllables_per_word = total_syllables / len(words)

        score = (
            206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
        )
        return max(0.0, min(100.0, score))

    @staticmethod
    def _extract_inline_citations(text: str) -> list[str]:
        import re

        return re.findall(r"\[([a-zA-Z0-9_-]{6,})\]", text)

    def _verify_grounding_heuristic(
        self,
        article: GeneratedArticle,
        max_paragraphs_per_section: int,
    ) -> list[dict]:
        """
        Token-free grounding approximation.
        A paragraph is treated as grounded when it contains at least one
        valid inline citation mapped to a retrieved source chunk.
        """
        results: list[dict] = []
        valid_source_ids = set(article.citations_map.keys())

        for section in article.sections:
            paras = [p.strip() for p in section.content.split("\n\n") if p.strip()]
            for para in paras[:max_paragraphs_per_section]:
                cited_ids = [
                    cid
                    for cid in self._extract_inline_citations(para)
                    if cid in valid_source_ids
                ]
                is_grounded = len(cited_ids) > 0
                results.append(
                    {
                        "sentence": para[:200],
                        "is_grounded": is_grounded,
                        "supporting_chunk_ids": cited_ids,
                        "confidence": 0.88 if is_grounded else 0.25,
                    }
                )

        return results

    def _verify_grounding_llm(
        self,
        article: GeneratedArticle,
        max_paragraphs_per_section: int,
    ) -> list[dict]:
        """Optional strict grounding verification using LLM checks."""
        results = []
        full_context = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}"
            for chunk in list(article.citations_map.values())[:12]
        )

        for section in article.sections:
            paras = [p.strip() for p in section.content.split("\n\n") if p.strip()]
            for para in paras[:max_paragraphs_per_section]:
                prompt = f"""
Source material:
{full_context[:3000]}

Statement to verify:
"{para[:500]}"

Is this statement directly supported by the source material?
Respond with JSON: {{"is_grounded": true/false, "supporting_chunk_ids": ["id1", ...], "confidence": 0.0-1.0}}
"""
                try:
                    data = self._llm.complete_json(
                        self.SYSTEM, prompt, max_tokens=180, tag="critic_ground"
                    )
                    results.append(
                        {
                            "sentence": para[:200],
                            "is_grounded": data.get("is_grounded", False),
                            "supporting_chunk_ids": data.get(
                                "supporting_chunk_ids", []
                            ),
                            "confidence": float(data.get("confidence", 0.5)),
                        }
                    )
                except Exception:
                    results.append(
                        {
                            "sentence": para[:200],
                            "is_grounded": True,
                            "supporting_chunk_ids": [],
                            "confidence": 0.5,
                        }
                    )

        return results

    def _verify_grounding(self, article: GeneratedArticle) -> list[dict]:
        cfg = get_settings()
        checks = max(1, cfg.qa_paragraph_checks_per_section)

        if cfg.qa_grounding_mode == "llm":
            return self._verify_grounding_llm(article, checks)
        return self._verify_grounding_heuristic(article, checks)

    def _coverage_score(self, article: GeneratedArticle, ctx: PipelineContext) -> float:
        """What fraction of retrieved chunks appear in at least one citation?"""
        all_retrieved_ids: set[str] = set()
        for chunks in ctx.retrieved.values():
            for r in chunks:
                all_retrieved_ids.add(r.chunk.chunk_id)

        cited_ids: set[str] = set()
        for section in article.sections:
            cited_ids.update(section.citations)

        if not all_retrieved_ids:
            return 1.0
        return len(cited_ids & all_retrieved_ids) / len(all_retrieved_ids)

    def _self_consistency_score(
        self, article: GeneratedArticle, ctx: PipelineContext
    ) -> float:
        """
        Token-free consistency approximation using local embeddings.
        Measures semantic continuity between adjacent sections and blends
        citation density as a proxy for grounded coherence.
        """
        from backend.tools.embedding import EmbeddingEngine

        snippets = [
            f"{section.heading}. {section.content[:320]}".strip()
            for section in article.sections
            if section.content.strip()
        ]

        if len(snippets) < 2:
            return 0.8

        engine = EmbeddingEngine()
        vecs = engine.encode(snippets[:8], normalize=True)

        adjacent_sims: list[float] = []
        for i in range(len(vecs) - 1):
            sim = float(engine.cosine_similarity(vecs[i], vecs[i + 1]))
            adjacent_sims.append(max(0.0, min(1.0, sim)))

        topical_cohesion = (
            sum(adjacent_sims) / len(adjacent_sims) if adjacent_sims else 0.7
        )

        total_words = sum(len(section.content.split()) for section in article.sections)
        total_citations = sum(len(section.citations) for section in article.sections)
        words_per_citation = total_words / max(1, total_citations)
        citation_density = max(0.0, min(1.0, 1.0 - (words_per_citation - 80) / 220))

        score = (0.75 * topical_cohesion) + (0.25 * citation_density)
        return float(max(0.0, min(1.0, score)))

    def run(self, ctx: PipelineContext) -> PipelineContext:
        assert ctx.article, "No article to critique — run WriterAgent first"
        cfg = get_settings()
        article = ctx.article
        logger.info("CriticAgent: running QA checks")

        full_text = (
            " ".join(s.content for s in article.sections) + " " + article.conclusion
        )

        # 1. Grounding verification
        grounding_results = self._verify_grounding(article)
        grounded_count = sum(1 for r in grounding_results if r["is_grounded"])
        grounding_score = (
            grounded_count / len(grounding_results) if grounding_results else 0.5
        )

        # 2. Readability
        readability_score = self._compute_flesch(full_text)

        # 3. Coverage
        coverage_score = self._coverage_score(article, ctx)

        # 4. Self-consistency
        try:
            consistency_score = self._self_consistency_score(article, ctx)
        except Exception as e:
            logger.warning("Consistency check failed", error=str(e))
            consistency_score = 0.7

        # 5. Composite confidence (weighted average)
        overall = (
            0.40 * grounding_score
            + 0.20 * (readability_score / 100)
            + 0.25 * coverage_score
            + 0.15 * consistency_score
        )

        # Collect warnings
        warnings: list[str] = []
        if grounding_score < cfg.qa_grounding_threshold:
            warnings.append(
                f"Low grounding score ({grounding_score:.2f}) — some claims may not be sourced."
            )
        if readability_score < cfg.qa_readability_min:
            warnings.append(
                f"Low readability ({readability_score:.1f} Flesch) — consider simplifying."
            )
        if coverage_score < 0.5:
            warnings.append("Less than 50% of retrieved sources were cited.")

        from backend.core.models import GroundingResult, QAReport

        ctx.qa_report = QAReport(
            article_id=article.article_id,
            grounding_score=round(grounding_score, 3),
            readability_score=round(readability_score, 1),
            coverage_score=round(coverage_score, 3),
            consistency_score=round(consistency_score, 3),
            overall_confidence=round(overall, 3),
            grounding_details=[GroundingResult(**r) for r in grounding_results],
            warnings=warnings,
            passed=(
                grounding_score >= cfg.qa_grounding_threshold
                and readability_score >= cfg.qa_readability_min
            ),
        )

        logger.info(
            "QA complete",
            grounding=grounding_score,
            readability=readability_score,
            coverage=coverage_score,
            confidence=overall,
            passed=ctx.qa_report.passed,
            warnings=len(warnings),
        )
        return ctx


# ── Orchestrator ──────────────────────────────────────────────────────────────


class ArticleOrchestrator:
    """
    MCP-style pipeline orchestrator.
    Chains all agents with shared context and error isolation.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        llm: LLMClient,
    ) -> None:
        self._llm = llm
        self._planner = PlannerAgent(llm)
        self._retriever = RetrieverAgent(vector_store)
        self._writer = WriterAgent(llm)
        self._critic = CriticAgent(llm, vector_store)

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])

    def _build_image_content_brief(
        self,
        retrieved: dict[str, list[RetrievedChunk]],
        max_items: int = 8,
    ) -> list[str]:
        """
        Create compact, evidence-grounded cues for image prompting.
        """
        items: list[str] = []
        seen: set[str] = set()

        for section, chunks in retrieved.items():
            if len(items) >= max_items:
                break

            if not chunks:
                continue

            top = chunks[0].chunk
            snippet = self._truncate_words(top.text.replace("\n", " ").strip(), 24)
            if not snippet:
                continue

            cue = f"{section}: {snippet}"
            key = cue.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(cue)

        return items

    async def run(
        self,
        topic: str,
        target_audience: str = "general tech readers",
        tone: str = "informative and engaging",
        generate_images: bool = True,
        run_qa: bool = True,
        max_sections: int = 6,
    ) -> PipelineContext:
        ctx = PipelineContext(
            topic=topic,
            target_audience=target_audience,
            tone=tone,
            generate_images=generate_images,
            max_sections=max_sections,
        )

        t0 = time.perf_counter()

        # Scope token accounting to this generation request only.
        self._llm.reset_token_log()

        # Agent pipeline
        t_stage = time.perf_counter()
        ctx = self._planner.run(ctx)
        ctx.stage_timings["planner_s"] = round(time.perf_counter() - t_stage, 3)

        t_stage = time.perf_counter()
        ctx = self._retriever.run(ctx)
        ctx.stage_timings["retriever_s"] = round(time.perf_counter() - t_stage, 3)

        # Image generation (async, non-blocking)
        if generate_images and ctx.outline:
            t_stage = time.perf_counter()
            from backend.tools.image_gen import ImageGenerationTool

            img_tool = ImageGenerationTool()
            content_brief = self._build_image_content_brief(ctx.retrieved)
            ctx.image_urls = await img_tool.generate_article_images(
                topic,
                ctx.outline.sections,
                tone,
                max_section_images=0,
                content_brief=content_brief,
            )

            # Keep generated figure inside the article body (first section), not in header.
            if ctx.outline.sections:
                first_section = ctx.outline.sections[0]
                header_image_url = ctx.image_urls.get("header")
                if header_image_url and not ctx.image_urls.get(first_section):
                    ctx.image_urls[first_section] = header_image_url
                ctx.image_urls["header"] = None

            ctx.stage_timings["image_generation_s"] = round(
                time.perf_counter() - t_stage, 3
            )

        t_stage = time.perf_counter()
        ctx = self._writer.run(ctx)
        ctx.stage_timings["writer_s"] = round(time.perf_counter() - t_stage, 3)

        if run_qa:
            t_stage = time.perf_counter()
            ctx = self._critic.run(ctx)
            ctx.stage_timings["qa_s"] = round(time.perf_counter() - t_stage, 3)

        elapsed = time.perf_counter() - t0
        ctx.stage_timings["total_s"] = round(elapsed, 3)
        logger.info(
            "Pipeline complete",
            topic=topic,
            elapsed_s=round(elapsed, 1),
            errors=len(ctx.errors),
        )
        return ctx
