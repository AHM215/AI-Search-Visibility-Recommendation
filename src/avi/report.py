from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from avi.ingest import Query, load_brands, load_query_set
from avi.judge import RECOMMENDATION_STRENGTHS
from avi.metrics import MentionSource, Metrics, compute_metrics
from avi.storage import StoredAnswer, StoredCitation, open_database, read_answers, read_run


ROOT = Path(__file__).resolve().parents[2]


def render_report(
    database_path: Path,
    run_id: str,
    query_set_path: Path = ROOT / "questions.v1.yaml",
    brand_path: Path = ROOT / "brands.yaml",
) -> str:
    connection = open_database(database_path)
    try:
        stored_run = read_run(connection, run_id)
        answers = read_answers(connection, run_id)
    finally:
        connection.close()
    query_set = load_query_set(query_set_path)
    if query_set.version != stored_run.query_set_version:
        raise ValueError(
            f"Query Set version {query_set.version!r} does not match Run version "
            f"{stored_run.query_set_version!r}"
        )
    brand_file = load_brands(brand_path)
    lines = [
        "# Boutiqaat AI Search Visibility Report",
        "",
        f"Run: {stored_run.id}",
        f"Query Set version: {stored_run.query_set_version}",
        f"Run timestamp: {stored_run.run_at}",
        f"Run status: {stored_run.status}",
        "",
        "Findings describe OpenAI's models, not AI search in general.",
        "",
        "## Metrics",
        "",
        "Metrics are recomputed from stored Answers when this Report is rendered.",
        "",
        "### Overall",
        "",
    ]
    _append_metrics(lines, compute_metrics(answers, query_set.queries, brand_file.seed_competitors))
    query_by_id = {query.id: query for query in query_set.queries}
    dimensions = (
        ("Locale", sorted({query_by_id[answer.query_id].locale for answer in answers}), "locale"),
        ("Intent", sorted({query_by_id[answer.query_id].intent for answer in answers}), "intent"),
        ("Provider Mode", sorted({answer.provider_mode for answer in answers}), "provider_mode"),
    )
    for title, values, keyword in dimensions:
        lines.extend([f"### By {title}", ""])
        for value in values:
            lines.extend([f"#### {value}", ""])
            _append_metrics(
                lines,
                compute_metrics(
                    answers,
                    query_set.queries,
                    brand_file.seed_competitors,
                    **{keyword: value},
                ),
                )
    _append_citation_pages(lines, answers, query_set.queries)
    lines.extend(["## Answers", ""])
    for mode in ("ungrounded", "grounded"):
        mode_answers = [answer for answer in answers if answer.provider_mode == mode]
        if not mode_answers:
            continue
        mentioned_ids = [str(answer.id) for answer in mode_answers if answer.mentioned]
        answer_ids = ", ".join(str(answer.id) for answer in mode_answers)
        boutiqaat_statement = (
            f"Boutiqaat was Mentioned. Answer ids: {', '.join(mentioned_ids)}."
            if mentioned_ids
            else f"Boutiqaat was not Mentioned. Answer ids: {answer_ids}."
        )
        lines.extend([f"## {mode.title()} Provider", "", boutiqaat_statement, ""])
        verdicts = [answer.verdict for answer in mode_answers if answer.verdict is not None]
        if verdicts:
            distribution = ", ".join(
                f"{strength}: {sum(verdict.recommendation_strength == strength for verdict in verdicts)}"
                for strength in RECOMMENDATION_STRENGTHS
            )
            lines.extend(
                [
                    f"Recommendation Strength distribution: {distribution}. "
                    f"Answer ids: {_answer_ids(answer.id for answer in mode_answers if answer.verdict is not None)}.",
                    "",
                ]
            )
        for answer in mode_answers:
            lines.extend(
                [
                    f"### Answer {answer.id}",
                    "",
                    f"Query: {answer.query_id}",
                    f"Provider mode: {answer.provider_mode}",
                    f"Trial: {answer.trial_index}",
                    f"Model identifier: {answer.model_identifier}",
                    f"Search performed: {'yes' if answer.search_performed else 'no'}",
                    "",
                    answer.text,
                    "",
                ]
            )
            if answer.verdict is not None:
                lines.extend(
                    [
                        f"Recommendation Strength: {answer.verdict.recommendation_strength}",
                        f"Rank: {answer.verdict.rank}",
                        f"Brands: {', '.join(answer.verdict.brands)}",
                    ]
                )
                if answer.verdict.unlocated_brands:
                    lines.append(f"Unlocated Brands: {', '.join(answer.verdict.unlocated_brands)}")
                lines.append("")
            if answer.citations:
                lines.extend(["#### Citations", ""])
                lines.extend(
                    _render_citation(index, citation)
                    for index, citation in enumerate(answer.citations, start=1)
                )
                lines.append("")
    return "\n".join(lines)


def _append_metrics(lines: list[str], metrics: Metrics) -> None:
    visibility = metrics.visibility_rate
    cited_not_named_count = len(metrics.cited_not_named_answer_ids)
    cited_not_named_trial_label = "Trial" if cited_not_named_count == 1 else "Trials"
    lines.extend(
        [
            "**Visibility Rate:** "
            f"{len(visibility.mentioned_answer_ids)}/{len(visibility.relevant_answer_ids)} "
            f"({_format_rate(visibility.value)}). Relevant Answer ids: "
            f"{_answer_ids(visibility.relevant_answer_ids)}. Mentioned Answer ids: "
            f"{_answer_ids(visibility.mentioned_answer_ids)}.",
            "",
            "**Cited-not-named:** "
            f"{cited_not_named_count} grounded {cited_not_named_trial_label} cited a Boutiqaat URL "
            "without naming Boutiqaat. Answer ids: "
            f"{_answer_ids(metrics.cited_not_named_answer_ids)}.",
            "",
            "**Consistency:**",
        ]
    )
    if metrics.consistency:
        lines.extend(
            f"- {item.query_id} ({item.provider_mode}): {item.bucket}. Trial Answer ids: "
            f"{_answer_ids(item.answer_ids)}. Mentioned Answer ids: "
            f"{_answer_ids(item.mentioned_answer_ids)}."
            for item in metrics.consistency
        )
    else:
        lines.append("- No Relevant Query Trials.")
    share_of_voice = metrics.share_of_voice
    lines.extend(
        [
            "",
            "**Share of Voice:** "
            f"{len(share_of_voice.boutiqaat_mentions)}/{len(share_of_voice.seed_mentions)} "
            f"({_format_rate(share_of_voice.value)}). Boutiqaat Mention Answer ids: "
            f"{_answer_ids(source.answer_id for source in share_of_voice.boutiqaat_mentions)}. "
            f"Seed Brand Mention sources: {_mention_sources(share_of_voice.seed_mentions)}.",
            "",
            "**Emergent Brands:**",
        ]
    )
    if metrics.emergent_brands:
        lines.extend(
            f"- {brand.name}: {len(brand.answer_ids)}. Answer ids: {_answer_ids(brand.answer_ids)}. "
            f"Unlocated Answer ids: {_answer_ids(brand.unlocated_answer_ids)}."
            for brand in metrics.emergent_brands
        )
    else:
        lines.append("- None.")
    lines.extend(["", "**Recommendation Strength:"])
    for strength in RECOMMENDATION_STRENGTHS:
        answer_ids = metrics.recommendation_strength.answer_ids_by_strength[strength]
        lines.append(f"- {strength}: {len(answer_ids)}. Answer ids: {_answer_ids(answer_ids)}.")
    lines.append("")


def _append_citation_pages(
    lines: list[str], answers: Sequence[StoredAnswer], queries: Sequence[Query]
) -> None:
    query_relevance = {query.id: query.relevance for query in queries}
    absent_answers = [
        answer
        for answer in answers
        if answer.provider_mode == "grounded"
        and query_relevance.get(answer.query_id) == "relevant"
        and not answer.mentioned
    ]
    lines.extend(["## Citation Pages", ""])
    if not absent_answers:
        lines.extend(["No Relevant Grounded Answers where Boutiqaat was Absent.", ""])
        return
    lines.append("For Relevant Grounded Answers where Boutiqaat was Absent:")
    for answer in absent_answers:
        fetched = [
            citation
            for citation in answer.citations
            if citation.page is not None and citation.page.status != "unfetched"
        ]
        present = [
            citation
            for citation in fetched
            if citation.page is not None and citation.page.status == "present"
        ]
        unfetched = [
            citation
            for citation in answer.citations
            if citation.page is None or citation.page.status == "unfetched"
        ]
        if not fetched:
            lines.append(
                f"- Answer {answer.id}: no cited pages were fetched; {len(unfetched)} unfetched."
            )
            continue
        lines.append(
            f"- Answer {answer.id}: Boutiqaat appears on {len(present)} of {len(fetched)} "
            f"fetched cited pages, {len(unfetched)} unfetched."
        )
    lines.append("")


def _render_citation(index: int, citation: StoredCitation) -> str:
    rendered = (
        f"{index}. [{citation.title}]({citation.url}) (Source Type: {citation.source_type})"
    )
    page = citation.page
    if page is None:
        return rendered
    if page.status == "unfetched":
        return f"{rendered}; Citation Page: unfetched ({page.unfetched_reason})"
    return f"{rendered}; Citation Page: {page.status}"


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _answer_ids(answer_ids: Iterable[int]) -> str:
    ids = tuple(answer_ids)
    return ", ".join(str(answer_id) for answer_id in ids) if ids else "none"


def _mention_sources(sources: Iterable[MentionSource]) -> str:
    return ", ".join(f"{source.answer_id} ({source.brand_name})" for source in sources) or "none"
