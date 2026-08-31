from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Mention:
    alias: str


def detect_mentions(answer_text: str, aliases: list[str]) -> list[Mention]:
    mentions: list[Mention] = []
    occupied_spans: list[tuple[int, int]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        escaped_alias = re.escape(alias)
        if re.search(r"[\u0600-\u06ff]", alias):
            pattern = re.compile(
                rf"(?<!\w)(?:[وف](?:[بكل])?|[بكل])?{escaped_alias}(?!\w)", re.IGNORECASE
            )
        else:
            pattern = re.compile(rf"(?<!\w){escaped_alias}(?!\w)", re.IGNORECASE)
        occurrence = pattern.search(answer_text)
        if occurrence is not None and not any(
            start <= occurrence.start() and occurrence.end() <= end
            for start, end in occupied_spans
        ):
            mentions.append(Mention(alias=alias))
            occupied_spans.append(occurrence.span())
    return mentions
