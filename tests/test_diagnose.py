from __future__ import annotations

from avi.citations import CitationPage
from avi.diagnose import diagnose
from avi.ingest import Brand, Query
from avi.judge import Verdict
from avi.storage import StoredAnswer, StoredCitation, StoredMention


def query(query_id: str, locale: str = "gcc_en", relevance: str = "relevant") -> Query:
    return Query(
        id=query_id,
        text=f"text for {query_id}",
        intent="retailer_discovery",
        locale=locale,
        specificity="broad",
        relevance=relevance,
    )


def answer(
    answer_id: int,
    query_id: str,
    *,
    mode: str = "ungrounded",
    trial: int = 0,
    mentioned: bool = False,
    text: str = "Sephora and Ounass are options.",
    citations: list[StoredCitation] | None = None,
    verdict: Verdict | None = None,
) -> StoredAnswer:
    return StoredAnswer(
        id=answer_id,
        query_id=query_id,
        provider_mode=mode,
        trial_index=trial,
        model_identifier="openai/gpt-5.2-2025-12-11",
        text=text,
        mentioned=mentioned,
        search_performed=mode == "grounded",
        mentions=[StoredMention("Boutiqaat", "Boutiqaat")] if mentioned else [],
        citations=citations or [],
        verdict=verdict,
    )


def cited(url: str, status: str | None, reason: str | None = None) -> StoredCitation:
    return StoredCitation(
        url=url,
        title=url,
        source_type="retailer",
        page=None if status is None else CitationPage(status, reason),
    )


SEEDS = [Brand(name="Sephora", aliases=["Sephora"]), Brand(name="Ounass", aliases=["Ounass"])]


def test_a_cause_with_no_supporting_evidence_is_dropped() -> None:
    answers = [answer(1, "q1", mentioned=True), answer(2, "q1", trial=1, mentioned=True)]

    findings = diagnose(answers, [query("q1")], SEEDS)

    assert "absent_from_cited_sources" not in {finding.cause for finding in findings}


def test_only_unfetched_pages_produce_no_source_coverage_claim() -> None:
    answers = [
        answer(
            1,
            "q1",
            mode="grounded",
            citations=[cited("https://a.example/x", "unfetched", "HTTP status 403")],
        )
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)

    assert "absent_from_cited_sources" not in {finding.cause for finding in findings}


def test_fetched_pages_without_boutiqaat_support_a_source_coverage_claim() -> None:
    answers = [
        answer(
            1,
            "q1",
            mode="grounded",
            citations=[
                cited("https://a.example/x", "absent"),
                cited("https://b.example/y", "absent"),
                cited("https://c.example/z", "unfetched", "timeout"),
            ],
        )
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)
    finding = next(f for f in findings if f.cause == "absent_from_cited_sources")

    assert finding.fetched_page_count == 2
    assert finding.unfetched_page_count == 1
    assert finding.citation_urls == ("https://a.example/x", "https://b.example/y")
    assert finding.answer_ids == (1,)


def test_a_page_claim_states_both_counts_in_its_statement() -> None:
    answers = [
        answer(1, "q1", mode="grounded", citations=[cited("https://a.example/x", "absent")]),
        answer(
            2,
            "q1",
            mode="grounded",
            trial=1,
            citations=[cited("https://b.example/y", "unfetched", "timeout")],
        ),
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)
    finding = next(f for f in findings if f.cause == "absent_from_cited_sources")

    assert "1" in finding.statement
    assert "unfetched" in finding.statement.lower()


def test_a_locale_with_no_visibility_beside_one_with_visibility_is_reported() -> None:
    answers = [
        answer(1, "global", mentioned=False),
        answer(2, "global", trial=1, mentioned=False),
        answer(3, "gcc", mentioned=True),
    ]
    queries = [query("global", locale="global_en"), query("gcc", locale="gcc_en")]

    findings = diagnose(answers, queries, SEEDS)
    finding = next(f for f in findings if f.cause == "locale_gap")

    assert "global_en" in finding.statement
    assert finding.answer_ids == (1, 2)


def test_irrelevant_queries_never_contribute_to_a_locale_gap() -> None:
    answers = [
        answer(1, "off", mentioned=False),
        answer(2, "gcc", mentioned=True),
    ]
    queries = [
        query("off", locale="global_en", relevance="irrelevant"),
        query("gcc", locale="gcc_en"),
    ]

    findings = diagnose(answers, queries, SEEDS)

    assert "locale_gap" not in {finding.cause for finding in findings}


def test_every_rendered_finding_carries_at_least_one_answer_id_or_citation_url() -> None:
    answers = [
        answer(1, "q1", mode="grounded", citations=[cited("https://a.example/x", "absent")]),
        answer(2, "q1", mentioned=True, verdict=Verdict("listed", ["Boutiqaat"], [])),
        answer(3, "q1", trial=1, mentioned=False),
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)

    assert findings
    for finding in findings:
        assert finding.answer_ids or finding.citation_urls


def test_findings_are_ordered_by_how_much_evidence_supports_them() -> None:
    answers = [
        answer(1, "q1", mode="grounded", citations=[cited("https://a.example/x", "absent")]),
        answer(2, "q1", mentioned=True, verdict=Verdict("listed", ["Boutiqaat"], [])),
        answer(3, "q1", trial=1, mentioned=False),
        answer(4, "q1", trial=2, mentioned=False),
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)
    supports = [finding.support for finding in findings]

    assert supports == sorted(supports, reverse=True)


def test_mentioned_but_never_recommended_is_reported() -> None:
    answers = [
        answer(1, "q1", mentioned=True, verdict=Verdict("listed", ["Boutiqaat"], [])),
        answer(2, "q1", trial=1, mentioned=True, verdict=Verdict("passing", ["Boutiqaat"], [])),
    ]

    findings = diagnose(answers, [query("q1")], SEEDS)
    finding = next(f for f in findings if f.cause == "mentioned_not_recommended")

    assert finding.answer_ids == (1, 2)


def test_diagnosis_needs_no_model_call() -> None:
    answers = [answer(1, "q1", mentioned=True)]

    findings = diagnose(answers, [query("q1")], SEEDS)

    assert all(isinstance(finding.statement, str) for finding in findings)
