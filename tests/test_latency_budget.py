"""The latency budget bounds time spent correcting, not the first verification.

On 20 October 2026 one verification took 20-40 s on free tiers (several judgements under a
15-requests-per-minute limit). With the clock started before it, the 12-second budget had
always run out by the time a correction was needed, and 4 of 93 answers that one prune would
have fixed were withheld.
"""
from __future__ import annotations

import time

from pramana.confidence.fusion import ConfidenceModel
from pramana.correction.policy import CorrectionExecutor, CorrectionPolicy
from pramana.detection.decomposer import RuleBasedDecomposer
from pramana.detection.verifier import GroundingVerifier, KeywordNLIBackend
from pramana.generation.base import Completion, GenerationResponse
from pramana.generation.drafting import DraftGenerator, DraftingPolicy
from pramana.ingestion.chunking import chunk_text
from pramana.pipeline import PramanaPipeline
from pramana.retrieval.hybrid import HybridRetriever, MultilingualRetriever

POLICY = "The adult membership fee is 600 rupees per year. A member may borrow at most 5 books at a time."
ANSWER = "The adult membership fee is 600 rupees per year. Members may bring pets into the reading room."


class _Provider:
    name = "fixed"

    def __init__(self, *replies: str):
        self.queue = list(replies)

    def generate(self, request):
        text = self.queue.pop(0) if self.queue else ""
        return GenerationResponse(completions=[Completion(text)], model="m", provider=self.name)

    def capabilities(self, model):
        return set()


class _SlowJudge:
    """Keyword NLI that takes a while, like a hosted judge under a rate limit."""

    def __init__(self, delay: float):
        self.inner = KeywordNLIBackend()
        self.delay = delay

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def score(self, premise, hypothesis, language):
        time.sleep(self.delay)
        return self.inner.score(premise, hypothesis, language)


def _pipeline(delay: float, budget_ms: float) -> PramanaPipeline:
    retriever = MultilingualRetriever()
    index = HybridRetriever(language="en", top_k=3)
    index.add(chunk_text(POLICY, doc_id="rules", language="en"))
    retriever.add_language("en", index)
    provider = _Provider(ANSWER)
    return PramanaPipeline(
        retriever=retriever, generator=DraftGenerator(provider, DraftingPolicy(n_samples=0)),
        decomposer=RuleBasedDecomposer(), verifier=GroundingVerifier(backend=_SlowJudge(delay)),
        confidence=ConfidenceModel(), policy=CorrectionPolicy(max_iterations=2, latency_budget_ms=budget_ms),
        executor=CorrectionExecutor(provider=provider), fail_closed=False,
    )


def test_slow_first_verification_does_not_use_up_the_correction_budget():
    result = _pipeline(delay=0.05, budget_ms=10).run("What is the adult membership fee?", language="en")
    assert result.stop_reason != "latency_budget"
    assert result.correction_attempts >= 1
    assert "pets" not in result.final_answer.lower()


def test_budget_still_stops_a_second_correction():
    pipeline = _pipeline(delay=0.05, budget_ms=10)
    pipeline.policy.max_iterations = 3
    result = pipeline.run("What is the adult membership fee?", language="en")
    assert result.correction_attempts <= 2
