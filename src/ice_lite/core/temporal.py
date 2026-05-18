import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np

class ValidityState(Enum):
    VALID    = "VALID"
    TEMPORAL = "TEMPORAL"
    EXPIRED  = "EXPIRED"

class DocumentKind(Enum):
    STATIC    = "STATIC"
    VERSIONED = "VERSIONED"
    EVENT     = "EVENT"

@dataclass
class Document:
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    doc_type: str = "general"
    kind: DocumentKind = DocumentKind.STATIC
    supersedes_id: Optional[str] = None

    def validity_state(self, query_time: datetime) -> ValidityState:
        has_window = self.valid_from is not None or self.valid_until is not None

        if not has_window:
            return ValidityState.VALID

        before_window = self.valid_from and query_time < self.valid_from
        after_window  = self.valid_until and query_time > self.valid_until

        if after_window or before_window:
            return ValidityState.EXPIRED

        if self.kind == DocumentKind.EVENT:
            return ValidityState.TEMPORAL

        return ValidityState.VALID

    def age_in_days(self, reference: datetime) -> float:
        delta = reference - self.created_at
        return max(0.0, delta.total_seconds() / 86400)

@dataclass
class ScoredDocument:
    document: Document
    vector_score: float
    recency_score: float
    decay_score: float
    validity_state: ValidityState
    validity_multiplier: float
    final_score: float
    reason: str

class TemporalConfig:
    def __init__(
        self,
        decay_half_life_days: float = 30.0,
        temporal_weight: float = 0.35,
        max_age_days: Optional[float] = None,
        enforce_validity: bool = True,
        validity_boost: float = 1.2,
        min_vector_score: float = 0.15,
        event_min_raw_vector_score: float = 0.20,
    ):
        self.decay_half_life_days = decay_half_life_days
        self.temporal_weight = temporal_weight
        self.max_age_days = max_age_days
        self.enforce_validity = enforce_validity
        self.validity_boost = validity_boost
        self.min_vector_score = min_vector_score
        self.event_min_raw_vector_score = event_min_raw_vector_score

class TemporalLayer:
    def __init__(self, config: TemporalConfig = None):
        self.config = config or TemporalConfig()

    def _classify_and_filter(self, candidates: List[Tuple[Document, float]], query_time: datetime) -> List[Tuple[Document, float, ValidityState]]:
        result = []
        for doc, score in candidates:
            state = doc.validity_state(query_time)
            if state == ValidityState.EXPIRED and self.config.enforce_validity:
                continue
            result.append((doc, score, state))
        return result

    def _filter_too_old(self, candidates: List[Tuple[Document, float, ValidityState]], query_time: datetime) -> List[Tuple[Document, float, ValidityState]]:
        if self.config.max_age_days is None:
            return candidates
        return [
            (doc, score, state) for doc, score, state in candidates
            if doc.age_in_days(query_time) <= self.config.max_age_days
        ]

    def _decay_score(self, doc: Document, query_time: datetime) -> float:
        age = doc.age_in_days(query_time)
        half_life = self.config.decay_half_life_days
        return math.pow(0.5, age / half_life)

    def _recency_score(self, doc: Document, all_docs: List[Document], query_time: datetime) -> float:
        ages = [d.age_in_days(query_time) for d in all_docs]
        if not ages: return 1.0
        min_age, max_age = min(ages), max(ages)
        if max_age == min_age: return 1.0
        doc_age = doc.age_in_days(query_time)
        return 1.0 - (doc_age - min_age) / (max_age - min_age)

    def _validity_multiplier(self, state: ValidityState) -> float:
        return {
            ValidityState.EXPIRED:  0.0,
            ValidityState.VALID:    1.0,
            ValidityState.TEMPORAL: self.config.validity_boost,
        }[state]

    def _semantic_penalty(self, normalized_vector_score: float, min_score: float) -> float:
        if min_score <= 0.0: return 1.0
        return 1.0 if normalized_vector_score >= min_score else 0.3

    def _event_relevance_multiplier(self, doc: Document, state: ValidityState, raw_vector_score: float) -> float:
        if doc.kind != DocumentKind.EVENT or state != ValidityState.TEMPORAL:
            return 1.0
        floor = self.config.event_min_raw_vector_score
        if floor <= 0.0: return 1.0
        return 1.0 if raw_vector_score >= floor else 0.5

    def rerank(self, candidates: List[Tuple[Document, float]], query_time: datetime, top_k: int = 5) -> List[ScoredDocument]:
        classified = self._classify_and_filter(candidates, query_time)
        classified = self._filter_too_old(classified, query_time)

        if not classified:
            return []

        docs_only = [doc for doc, _, _ in classified]
        raw_scores = [s for _, s, _ in classified]
        min_s, max_s = min(raw_scores), max(raw_scores)
        def norm(s):
            return (s - min_s) / (max_s - min_s) if max_s > min_s else 1.0

        w = self.config.temporal_weight
        scored = []

        for doc, raw_vector_score, state in classified:
            vs      = norm(raw_vector_score)
            ds      = self._decay_score(doc, query_time)
            rs      = self._recency_score(doc, docs_only, query_time)
            vm      = self._validity_multiplier(state)
            erm     = self._event_relevance_multiplier(doc, state, raw_vector_score)
            penalty = self._semantic_penalty(vs, self.config.min_vector_score)

            temporal_component = ds * rs * vm * erm
            final = penalty * ((1 - w) * vs + w * temporal_component)

            scored.append(ScoredDocument(
                document=doc,
                vector_score=vs,
                recency_score=rs,
                decay_score=ds,
                validity_state=state,
                validity_multiplier=vm,
                final_score=final,
                reason="Temporal Reranked",
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored[:top_k]
