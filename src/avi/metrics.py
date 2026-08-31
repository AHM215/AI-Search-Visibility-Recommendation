from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Sequence

from avi.ingest import Brand, Query
from avi.judge import RECOMMENDATION_STRENGTHS, RecommendationStrength
from avi.storage import StoredAnswer


ConsistencyBucket = Literal["always", "sometimes", "never"]


@dataclass(frozen=True)
class VisibilityRate:
    mentioned_answer_ids: tuple[int, ...]
    relevant_answer_ids: tuple[int, ...]

    @property
    def value(self) -> float | None:
        if not self.relevant_answer_ids:
            return None
        return len(self.mentioned_answer_ids) / len(self.relevant_answer_ids)


@dataclass(frozen=True)
class Consistency:
    query_id: str
    provider_mode: str
    bucket: ConsistencyBucket
    answer_ids: tuple[int, ...]
    mentioned_answer_ids: tuple[int, ...]


@dataclass(frozen=True)
class MentionSource:
    answer_id: int
    brand_name: str


@dataclass(frozen=True)
class ShareOfVoice:
    boutiqaat_mentions: tuple[MentionSource, ...]
    seed_mentions: tuple[MentionSource, ...]

    @property
    def value(self) -> float | None:
        if not self.seed_mentions:
            return None
        return len(self.boutiqaat_mentions) / len(self.seed_mentions)


@dataclass(frozen=True)
class EmergentBrand:
    name: str
    answer_ids: tuple[int, ...]
    unlocated_answer_ids: tuple[int, ...]


@dataclass(frozen=True)
class RecommendationDistribution:
    answer_ids_by_strength: dict[RecommendationStrength, tuple[int, ...]]


@dataclass(frozen=True)
class Metrics:
    answer_ids: tuple[int, ...]
    visibility_rate: VisibilityRate
    consistency: tuple[Consistency, ...]
    share_of_voice: ShareOfVoice
    emergent_brands: tuple[EmergentBrand, ...]
    recommendation_strength: RecommendationDistribution


def compute_metrics(
    answers: Sequence[StoredAnswer],
    queries: Sequence[Query],
    seed_competitors: Sequence[Brand],
    *,
    locale: str | None = None,
    intent: str | None = None,
    provider_mode: str | None = None,
) -> Metrics:
    """Compute report figures from stored Answer facts without persisting aggregates."""
    query_by_id = {query.id: query for query in queries}
    selected_answers = [
        answer
        for answer in answers
        if (query := query_by_id[answer.query_id])
        and (locale is None or query.locale == locale)
        and (intent is None or query.intent == intent)
        and (provider_mode is None or answer.provider_mode == provider_mode)
    ]
    relevant_answers = [
        answer for answer in selected_answers if query_by_id[answer.query_id].relevance == "relevant"
    ]
    mentioned_answers = [answer for answer in relevant_answers if answer.mentioned]

    consistency_by_query: dict[tuple[str, str], list[StoredAnswer]] = defaultdict(list)
    for answer in relevant_answers:
        consistency_by_query[(answer.query_id, answer.provider_mode)].append(answer)
    consistency = tuple(
        _consistency(query_id, mode, query_answers)
        for (query_id, mode), query_answers in sorted(consistency_by_query.items())
    )

    seed_names = {brand.name.casefold() for brand in seed_competitors}
    seed_names.add("boutiqaat")
    seed_mentions = tuple(
        MentionSource(answer.id, brand_name)
        for answer in selected_answers
        for brand_name in _unique_mentions(answer)
        if brand_name.casefold() in seed_names
    )
    boutiqaat_mentions = tuple(
        mention for mention in seed_mentions if mention.brand_name.casefold() == "boutiqaat"
    )

    aliases_to_seed_names = {
        alias.casefold(): brand.name.casefold()
        for brand in seed_competitors
        for alias in [brand.name, *brand.aliases]
    }
    aliases_to_seed_names["boutiqaat"] = "boutiqaat"
    emergent: dict[str, tuple[str, list[int], list[int]]] = {}
    for answer in selected_answers:
        if answer.verdict is None:
            continue
        located = set(answer.verdict.brands)
        for brand_name in [*answer.verdict.brands, *answer.verdict.unlocated_brands]:
            if brand_name.casefold() in aliases_to_seed_names:
                continue
            key = brand_name.casefold()
            if key not in emergent:
                emergent[key] = (brand_name, [], [])
            _, answer_ids, unlocated_answer_ids = emergent[key]
            answer_ids.append(answer.id)
            if brand_name not in located:
                unlocated_answer_ids.append(answer.id)

    recommendation_ids: dict[RecommendationStrength, list[int]] = {
        strength: [] for strength in RECOMMENDATION_STRENGTHS
    }
    for answer in selected_answers:
        if answer.verdict is not None:
            recommendation_ids[answer.verdict.recommendation_strength].append(answer.id)

    return Metrics(
        answer_ids=tuple(answer.id for answer in selected_answers),
        visibility_rate=VisibilityRate(
            tuple(answer.id for answer in mentioned_answers),
            tuple(answer.id for answer in relevant_answers),
        ),
        consistency=consistency,
        share_of_voice=ShareOfVoice(boutiqaat_mentions, seed_mentions),
        emergent_brands=tuple(
            EmergentBrand(name, tuple(answer_ids), tuple(unlocated_answer_ids))
            for name, answer_ids, unlocated_answer_ids in sorted(emergent.values())
        ),
        recommendation_strength=RecommendationDistribution(
            {strength: tuple(answer_ids) for strength, answer_ids in recommendation_ids.items()}
        ),
    )


def _unique_mentions(answer: StoredAnswer) -> list[str]:
    return list(dict.fromkeys(mention.brand_name for mention in answer.mentions))


def _consistency(query_id: str, provider_mode: str, answers: Sequence[StoredAnswer]) -> Consistency:
    answer_ids = tuple(answer.id for answer in answers)
    mentioned_answer_ids = tuple(answer.id for answer in answers if answer.mentioned)
    if len(mentioned_answer_ids) == len(answer_ids):
        bucket: ConsistencyBucket = "always"
    elif mentioned_answer_ids:
        bucket = "sometimes"
    else:
        bucket = "never"
    return Consistency(query_id, provider_mode, bucket, answer_ids, mentioned_answer_ids)
