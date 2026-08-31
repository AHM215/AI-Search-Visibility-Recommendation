from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mention:
    alias: str


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped_alias = re.escape(alias)
    if re.search(r"[\u0600-\u06ff]", alias):
        return re.compile(
            rf"(?<!\w)(?:[وف](?:[بكل])?|[بكل])?{escaped_alias}(?!\w)", re.IGNORECASE
        )
    return re.compile(rf"(?<!\w){escaped_alias}(?!\w)", re.IGNORECASE)


def first_mention_index(answer_text: str, alias: str) -> int | None:
    occurrence = _alias_pattern(alias).search(answer_text)
    return occurrence.start() if occurrence is not None else None


def detect_mentions(answer_text: str, aliases: list[str]) -> list[Mention]:
    mentions: list[Mention] = []
    occupied_spans: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        occurrence = _alias_pattern(alias).search(answer_text)
        if occurrence is not None and not any(
            start <= occurrence.start() and occurrence.end() <= end
            for start, end in occupied_spans
        ):
            mentions.append(Mention(alias=alias))
            occupied_spans.append(occurrence.span())
    return mentions
