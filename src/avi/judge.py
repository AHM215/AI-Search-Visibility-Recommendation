from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    def brands_identify_boutiqaat(self) -> Self:
        if any(not brand.strip() for brand in self.brands):
            raise ValueError("Brands must not contain empty names")
        if len(set(self.brands)) != len(self.brands):
            raise ValueError("Brands must not repeat names")
        if "Boutiqaat" not in self.brands:
            raise ValueError("Brands must include Boutiqaat")
        return self


@dataclass(frozen=True)
class Verdict:
    recommendation_strength: RecommendationStrength
    brands: list[str]

    @property
    def rank(self) -> int:
        return self.brands.index("Boutiqaat") + 1

    @classmethod
    def from_judge_verdict(cls, judge_verdict: JudgeVerdict, answer_text: str) -> Self:
        first_mentions: list[tuple[int, str]] = []
        for brand in judge_verdict.brands:
            first_mention = first_mention_index(answer_text, brand)
            if first_mention is None:
                raise ValueError(f"Judge named Brand {brand!r} absent from Answer text")
            first_mentions.append((first_mention, brand))
        brands = [brand for _, brand in sorted(first_mentions, key=lambda item: (item[0], item[1]))]
        return cls(judge_verdict.recommendation_strength, brands)


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


def judge_answer(answer: Answer, provider: Provider) -> Verdict:
    from avi.ingest import Query

    instructions = JUDGE_INSTRUCTIONS + answer.text
    answer_digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    judge_query = Query(
        id=f"judge-{answer_digest}",
        text=instructions,
        intent="judge",
        locale="global_en",
        specificity="narrow",
        relevance="relevant",
    )
    verdict_answer = provider.ask(judge_query, 0)
    try:
        verdict_data = json.loads(verdict_answer.text)
    except json.JSONDecodeError as error:
        raise ValueError("Judge returned invalid JSON") from error
    return Verdict.from_judge_verdict(JudgeVerdict.model_validate(verdict_data), answer.text)
