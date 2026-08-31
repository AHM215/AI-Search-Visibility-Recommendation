from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from avi.detect import first_mention_index
from avi.providers import Answer, Provider

if TYPE_CHECKING:
    from avi.ingest import Query


RecommendationStrength = Literal["recommended", "listed", "passing", "dismissed"]
RECOMMENDATION_STRENGTHS: tuple[RecommendationStrength, ...] = (
    "recommended",
    "listed",
    "passing",
    "dismissed",
)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    recommendation_strength: RecommendationStrength
    brands: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def brands_identify_boutiqaat(self, info: ValidationInfo) -> Self:
        if any(not brand.strip() for brand in self.brands):
            raise ValueError("Brands must not contain empty names")
        if len(set(self.brands)) != len(self.brands):
            raise ValueError("Brands must not repeat names")
        aliases = info.context.get("boutiqaat_aliases", ["Boutiqaat"]) if info.context else ["Boutiqaat"]
        if not any(_is_boutiqaat_alias(brand, aliases) for brand in self.brands):
            raise ValueError("Brands must include Boutiqaat")
        return self


@dataclass(frozen=True)
class Verdict:
    recommendation_strength: RecommendationStrength
    brands: list[str]
    unlocated_brands: list[str]

    @property
    def rank(self) -> int:
        return self.brands.index("Boutiqaat") + 1

    @classmethod
    def from_judge_verdict(
        cls,
        judge_verdict: JudgeVerdict,
        answer_text: str,
        boutiqaat_aliases: list[str] | None = None,
    ) -> Self:
        aliases = boutiqaat_aliases or ["Boutiqaat"]
        boutiqaat_position = _first_boutiqaat_alias_index(answer_text, aliases)
        if boutiqaat_position is None:
            raise ValueError("Boutiqaat Alias is absent from Answer text")
        first_mentions: list[tuple[int, str]] = []
        unlocated_brands: list[str] = []
        boutiqaat_located = False
        for brand in judge_verdict.brands:
            if _is_boutiqaat_alias(brand, aliases):
                if not boutiqaat_located:
                    first_mentions.append((boutiqaat_position, "Boutiqaat"))
                    boutiqaat_located = True
                continue
            first_mention = first_mention_index(answer_text, brand)
            if first_mention is None:
                unlocated_brands.append(brand)
                continue
            first_mentions.append((first_mention, brand))
        brands = [brand for _, brand in sorted(first_mentions, key=lambda item: (item[0], item[1]))]
        return cls(judge_verdict.recommendation_strength, brands, unlocated_brands)


def _is_boutiqaat_alias(brand: str, aliases: list[str]) -> bool:
    return brand.casefold() in {alias.casefold() for alias in aliases}


def _first_boutiqaat_alias_index(answer_text: str, aliases: list[str]) -> int | None:
    positions = [
        position
        for alias in aliases
        if (position := first_mention_index(answer_text, alias)) is not None
    ]
    return min(positions, default=None)


JUDGE_INSTRUCTIONS = """You are the Judge for a Brand measurement Run.
Boutiqaat has already been Mentioned by deterministic Alias matching.
Read the Answer below and return only a JSON object with exactly these fields:
- recommendation_strength: one of recommended, listed, passing, dismissed
- brands: every retailer Brand named in the Answer, including Boutiqaat. Order does not matter and will be ignored.

Use recommended only when the Answer actively puts Boutiqaat forward as a place to buy.
Use listed when it names Boutiqaat in a list without actively putting it forward.
Use passing when it names Boutiqaat incidentally.
Use dismissed when it names Boutiqaat negatively or rules it out.
Do not include product brands, places, or other non-retailer names in brands.

Answer:
"""


def judge_query_for_answer(answer: Answer) -> Query:
    from avi.ingest import Query

    instructions = JUDGE_INSTRUCTIONS + answer.text
    answer_digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    return Query(
        id=f"judge-{answer_digest}",
        text=instructions,
        intent="judge",
        locale="global_en",
        specificity="narrow",
        relevance="relevant",
    )


def judge_answer(answer: Answer, provider: Provider, boutiqaat_aliases: list[str]) -> Verdict:
    judge_query = judge_query_for_answer(answer)
    verdict_answer = provider.ask(judge_query, 0)
    try:
        verdict_data = json.loads(verdict_answer.text)
    except json.JSONDecodeError as error:
        raise ValueError("Judge returned invalid JSON") from error
    return Verdict.from_judge_verdict(
        JudgeVerdict.model_validate(verdict_data, context={"boutiqaat_aliases": boutiqaat_aliases}),
        answer.text,
        boutiqaat_aliases,
    )
