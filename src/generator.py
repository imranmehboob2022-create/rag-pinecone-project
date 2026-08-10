"""
generator.py
------------
Takes retrieved chunks + the user's question and asks an LLM (Groq or
OpenAI) to answer STRICTLY from that context. This is the main
hallucination-prevention layer, enforced through:
  1. A system prompt that forbids using outside knowledge.
  2. An explicit fallback instruction to say the answer isn't in the
     document if the context doesn't support an answer.
  3. A post-hoc confidence score derived from retrieval similarity,
     shown to the user alongside the answer.
"""

from dataclasses import dataclass
from typing import List

from src.vector_store import RetrievedChunk

NOT_FOUND_MESSAGE = "The answer is not available in the provided document."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
You must answer the user's question using ONLY the provided context
excerpts below. Do not use any outside knowledge, do not guess, and
do not fill gaps with assumptions.

Rules:
- If the context fully or partially supports an answer, answer concisely and cite which excerpt(s) you used, like [Excerpt 2].
- If the context does NOT contain enough information to answer, respond with exactly:
  "{NOT_FOUND_MESSAGE}"
- Never fabricate page numbers, facts, or details not present in the context.
"""


@dataclass
class GeneratedAnswer:
    answer: str
    grounded: bool          # False if the model returned the "not available" fallback
    confidence: float       # 0..1, derived from retrieval similarity scores
    sources: List[RetrievedChunk]


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i}] (document: {c.document_name}, page: {c.page_number}, similarity: {c.score:.2f})\n{c.text}"
        )
    return "\n\n".join(parts)


def _estimate_confidence(chunks: List[RetrievedChunk]) -> float:
    """Simple, transparent confidence heuristic: average of the top
    retrieval similarity scores. Not a substitute for real calibration,
    but gives the user a meaningful, explainable signal."""
    if not chunks:
        return 0.0
    top = chunks[:3]
    return round(sum(c.score for c in top) / len(top), 3)


class AnswerGenerator:
    def __init__(self, provider: str, groq_api_key: str = "", groq_model: str = "",
                 openai_api_key: str = "", openai_model: str = ""):
        self.provider = provider.lower()
        self.groq_model = groq_model
        self.openai_model = openai_model

        if self.provider == "groq":
            from groq import Groq
            if not groq_api_key:
                raise EnvironmentError("GROQ_API_KEY is not set.")
            self.client = Groq(api_key=groq_api_key)
        elif self.provider == "openai":
            from openai import OpenAI
            if not openai_api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    def _call_llm(self, user_prompt: str) -> str:
        if self.provider == "groq":
            resp = self.client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()

        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, question: str, chunks: List[RetrievedChunk]) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=NOT_FOUND_MESSAGE,
                grounded=False,
                confidence=0.0,
                sources=[],
            )

        context_block = _build_context_block(chunks)
        user_prompt = (
            f"Context excerpts from the document:\n\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer strictly from the context above."
        )

        raw_answer = self._call_llm(user_prompt)
        grounded = NOT_FOUND_MESSAGE.lower() not in raw_answer.lower()
        confidence = _estimate_confidence(chunks) if grounded else 0.0

        return GeneratedAnswer(
            answer=raw_answer,
            grounded=grounded,
            confidence=confidence,
            sources=chunks if grounded else [],
        )
