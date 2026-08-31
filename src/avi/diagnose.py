"""Evidence-constrained Diagnosis.

Every claim rests on something observed in the Run: an Answer, a Citation, or a fetched Citation
Page. A candidate cause with no supporting Evidence is dropped rather than rendered, and an
`unfetched` Citation Page never contributes to an absence claim -- turning a network error into
evidence would fabricate the finding this module exists to establish.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from avi.ingest import Brand, Query
from avi.storage import StoredAnswer


@dataclass(frozen=True)
class Finding:
    """One supported claim, and the Evidence it rests on."""

    cause: str
    statement: str
    remedy: str
    answer_ids: tuple[int, ...] = ()
    citation_urls: tuple[str, ...] = ()
    fetched_page_count: int = 0
    unfetched_page_count: int = 0
    support: int = field(default=0)


def diagnose(
    answers: Sequence[StoredAnswer],
    queries: Sequence[Query],
    seed_competitors: Sequence[Brand],
) -> tuple[Finding, ...]:
    """Return the supported Findings, most-supported first."""
    query_by_id = {query.id: query for query in queries}
    relevant = [
        answer for answer in answers if query_by_id[answer.query_id].relevance == "relevant"
    ]
    candidates = (
        _absent_from_cited_sources(relevant),
        _locale_gap(relevant, query_by_id),
        _memory_exceeds_retrieval(relevant),
        _inconsistent_across_trials(relevant),
        _mentioned_not_recommended(relevant),
        _outranked_by_competitors(relevant, seed_competitors),
    )
    supported = [finding for finding in candidates if finding is not None]
    return tuple(sorted(supported, key=lambda finding: finding.support, reverse=True))


def _absent_from_cited_sources(answers: Sequence[StoredAnswer]) -> Finding | None:
    """Boutiqaat is missing from the pages the model actually read.

    Only FETCHED pages count. If every page is unfetched there is no evidence either way, and the
    claim is not made.
    """
    answer_ids: list[int] = []
    urls: list[str] = []
    fetched = 0
    unfetched = 0
    for answer in answers:
        if answer.provider_mode != "grounded" or answer.mentioned:
            continue
        for citation in answer.citations:
            if citation.page is None:
                continue
            if citation.page.status == "unfetched":
                unfetched += 1
                continue
            fetched += 1
            if citation.page.status == "absent":
                urls.append(citation.url)
                if answer.id not in answer_ids:
                    answer_ids.append(answer.id)
    if fetched == 0 or not urls:
        return None
    return Finding(
        cause="absent_from_cited_sources",
        statement=(
            f"Boutiqaat does not appear on {len(urls)} of the {fetched} cited pages that could be "
            f"fetched for Answers where it is Absent. A further {unfetched} cited pages were "
            f"unfetched and are excluded from this claim."
        ),
        remedy=(
            "Get Boutiqaat onto the pages these Answers are assembled from: the retailer roundups, "
            "listicles and marketplace pages the model cites for this category."
        ),
        answer_ids=tuple(answer_ids),
        citation_urls=tuple(urls),
        fetched_page_count=fetched,
        unfetched_page_count=unfetched,
        support=len(urls),
    )


def _locale_gap(
    answers: Sequence[StoredAnswer], query_by_id: dict[str, Query]
) -> Finding | None:
    """Boutiqaat is invisible in one Locale while visible in another."""
    by_locale: dict[str, list[StoredAnswer]] = defaultdict(list)
    for answer in answers:
        by_locale[query_by_id[answer.query_id].locale].append(answer)
    blind: list[str] = []
    seeing: list[str] = []
    blind_answer_ids: list[int] = []
    for locale, locale_answers in sorted(by_locale.items()):
        if any(answer.mentioned for answer in locale_answers):
            seeing.append(locale)
        else:
            blind.append(locale)
            blind_answer_ids.extend(answer.id for answer in locale_answers)
    if not blind or not seeing:
        return None
    return Finding(
        cause="locale_gap",
        statement=(
            f"Boutiqaat is Mentioned in no Trial at all for {', '.join(blind)}, while appearing in "
            f"{', '.join(seeing)}. Its visibility is confined to the markets it already serves."
        ),
        remedy=(
            "Decide whether the absent Locales are a target. If they are, the content the model "
            "reads for those markets has to name Boutiqaat; if they are not, exclude them from the "
            "headline figure so it describes the market being competed for."
        ),
        answer_ids=tuple(blind_answer_ids),
        support=len(blind_answer_ids),
    )


def _memory_exceeds_retrieval(answers: Sequence[StoredAnswer]) -> Finding | None:
    """The model recalls Boutiqaat more readily than its web sources surface it."""
    rates: dict[str, tuple[int, int]] = {}
    for mode in ("ungrounded", "grounded"):
        mode_answers = [answer for answer in answers if answer.provider_mode == mode]
        if not mode_answers:
            return None
        rates[mode] = (sum(1 for a in mode_answers if a.mentioned), len(mode_answers))
    ungrounded_hits, ungrounded_total = rates["ungrounded"]
    grounded_hits, grounded_total = rates["grounded"]
    if ungrounded_hits / ungrounded_total <= grounded_hits / grounded_total:
        return None
    ids = tuple(
        answer.id for answer in answers if answer.provider_mode == "ungrounded" and answer.mentioned
    )
    return Finding(
        cause="memory_exceeds_retrieval",
        statement=(
            f"Boutiqaat is Mentioned in {ungrounded_hits} of {ungrounded_total} ungrounded Trials "
            f"but only {grounded_hits} of {grounded_total} grounded ones. The model knows Boutiqaat "
            "better than its search results show it."
        ),
        remedy=(
            "This is a retrieval problem rather than a reputation one. Improving what the web says "
            "about Boutiqaat will move the grounded figure; the model already recalls the brand."
        ),
        answer_ids=ids,
        support=ungrounded_hits - grounded_hits,
    )


def _inconsistent_across_trials(answers: Sequence[StoredAnswer]) -> Finding | None:
    """Boutiqaat appears in some Trials of a Query and not others."""
    by_query: dict[tuple[str, str], list[StoredAnswer]] = defaultdict(list)
    for answer in answers:
        by_query[(answer.query_id, answer.provider_mode)].append(answer)
    unstable_ids: list[int] = []
    unstable_queries = 0
    for trial_answers in by_query.values():
        mentioned = [answer for answer in trial_answers if answer.mentioned]
        if mentioned and len(mentioned) != len(trial_answers):
            unstable_queries += 1
            unstable_ids.extend(answer.id for answer in trial_answers)
    if not unstable_queries:
        return None
    return Finding(
        cause="inconsistent_across_trials",
        statement=(
            f"{unstable_queries} Query/mode pairs Mention Boutiqaat in some Trials but not others. "
            "A shopper asking the same question twice may or may not be shown the brand."
        ),
        remedy=(
            "Treat these as the cheapest wins: the model already considers Boutiqaat a candidate "
            "here, so a small shift in source coverage may make it consistent."
        ),
        answer_ids=tuple(unstable_ids),
        support=unstable_queries,
    )


def _mentioned_not_recommended(answers: Sequence[StoredAnswer]) -> Finding | None:
    """Boutiqaat is named but never actively put forward."""
    judged = [answer for answer in answers if answer.verdict is not None]
    if not judged:
        return None
    recommended = [
        answer
        for answer in judged
        if answer.verdict is not None
        and answer.verdict.recommendation_strength == "recommended"
    ]
    if recommended:
        return None
    ids = tuple(answer.id for answer in judged)
    return Finding(
        cause="mentioned_not_recommended",
        statement=(
            f"In all {len(judged)} Answers that name Boutiqaat, it is listed, mentioned in passing "
            "or dismissed. None actively recommends it as a place to buy."
        ),
        remedy=(
            "Presence is not the same as endorsement. The sources that do name Boutiqaat describe "
            "it neutrally; what is missing is content that gives a reason to prefer it."
        ),
        answer_ids=ids,
        support=len(judged),
    )


def _outranked_by_competitors(
    answers: Sequence[StoredAnswer], seed_competitors: Sequence[Brand]
) -> Finding | None:
    """A Seed Competitor is named far more often than Boutiqaat."""
    counts: dict[str, int] = defaultdict(int)
    for answer in answers:
        for brand_name in {mention.brand_name for mention in answer.mentions}:
            counts[brand_name] += 1
    boutiqaat = counts.get("Boutiqaat", 0)
    rivals = {
        brand.name: counts.get(brand.name, 0)
        for brand in seed_competitors
        if counts.get(brand.name, 0) > max(boutiqaat * 2, 1)
    }
    if not rivals:
        return None
    leader, leader_count = max(rivals.items(), key=lambda item: item[1])
    ids = tuple(
        answer.id
        for answer in answers
        if any(mention.brand_name == leader for mention in answer.mentions)
    )
    ranked = ", ".join(f"{name} ({count})" for name, count in sorted(rivals.items()))
    return Finding(
        cause="outranked_by_competitors",
        statement=(
            f"Boutiqaat is Mentioned in {boutiqaat} Trials against {leader} in {leader_count}. "
            f"Competitors named more than twice as often: {ranked}."
        ),
        remedy=(
            f"Study where {leader} is being cited from for these Queries. Those are the specific "
            "pages that decide who appears in the answer."
        ),
        answer_ids=ids,
        support=leader_count - boutiqaat,
    )
