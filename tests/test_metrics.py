from __future__ import annotations

from pathlib import Path

from avi.ingest import Brand, Query
from avi.judge import Verdict
from avi.metrics import compute_metrics
from avi.report import render_report
from avi.storage import (
    StoredAnswer,
    StoredMention,
    create_run,
    open_database,
    store_answer,
    store_mention,
)


SEEDS = [Brand(name="Sephora", aliases=["Sephora"])]
QUERIES = [
    Query(
        id="never",
        text="Never",
        intent="retailer_discovery",
        locale="global_en",
        specificity="broad",
        relevance="relevant",
    ),
    Query(
        id="sometimes-one",
        text="Sometimes one",
        intent="find_category",
        locale="global_en",
        specificity="broad",
        relevance="relevant",
    ),
    Query(
        id="sometimes-two",
        text="Sometimes two",
        intent="comparison",
        locale="gcc_en",
        specificity="medium",
        relevance="relevant",
    ),
    Query(
        id="always",
        text="Always",
        intent="occasion_shopping",
        locale="ar",
        specificity="narrow",
        relevance="relevant",
    ),
    Query(
        id="irrelevant",
        text="Irrelevant",
        intent="retailer_discovery",
        locale="global_en",
        specificity="narrow",
        relevance="irrelevant",
    ),
]


def _answer(
    answer_id: int,
    query_id: str,
    trial_index: int,
    mentioned: bool,
    *,
    provider_mode: str = "ungrounded",
    brands: list[str] | None = None,
    verdict: Verdict | None = None,
) -> StoredAnswer:
    mentioned_brands = brands or ([] if not mentioned else ["Boutiqaat"])
    return StoredAnswer(
        id=answer_id,
        query_id=query_id,
        provider_mode=provider_mode,
        trial_index=trial_index,
        model_identifier="test-model",
        text="Answer",
        mentioned=mentioned,
        search_performed=False,
        mentions=[StoredMention(brand_name=brand, alias=brand) for brand in mentioned_brands],
        citations=[],
        verdict=verdict,
    )


def _answers() -> list[StoredAnswer]:
    answers: list[StoredAnswer] = []
    answer_id = 1
    visibility = {
        "never": (False, False, False),
        "sometimes-one": (True, False, False),
        "sometimes-two": (True, True, False),
        "always": (True, True, True),
        "irrelevant": (True, True, True),
    }
    for mode in ("ungrounded", "grounded"):
        for query_id, trials in visibility.items():
            for trial_index, mentioned in enumerate(trials):
                verdict = None
                brands = None
                if answer_id == 4:
                    brands = ["Boutiqaat", "Sephora", "Cult Beauty"]
                    verdict = Verdict("recommended", ["Boutiqaat", "Cult Beauty"], [])
                elif answer_id == 7:
                    verdict = Verdict("listed", ["Boutiqaat"], ["Retailer X"])
                elif answer_id == 8:
                    verdict = Verdict("passing", ["Boutiqaat"], [])
                answers.append(
                    _answer(
                        answer_id,
                        query_id,
                        trial_index,
                        mentioned,
                        provider_mode=mode,
                        brands=brands,
                        verdict=verdict,
                    )
                )
                answer_id += 1
    return answers


def test_metrics_use_relevant_trials_and_keep_consistency_buckets_separate() -> None:
    metrics = compute_metrics(_answers(), QUERIES, SEEDS, provider_mode="ungrounded")

    assert metrics.visibility_rate.relevant_answer_ids == tuple(range(1, 13))
    assert metrics.visibility_rate.mentioned_answer_ids == (4, 7, 8, 10, 11, 12)
    assert metrics.visibility_rate.value == 0.5
    assert compute_metrics(_answers(), QUERIES, SEEDS, intent="find_category").visibility_rate.value == 1 / 3
    assert {item.query_id: item.bucket for item in metrics.consistency} == {
        "never": "never",
        "sometimes-one": "sometimes",
        "sometimes-two": "sometimes",
        "always": "always",
    }


def test_share_of_voice_excludes_emergent_brands_and_strength_is_a_distribution() -> None:
    metrics = compute_metrics(_answers(), QUERIES, SEEDS, provider_mode="ungrounded")

    assert [(source.answer_id, source.brand_name) for source in metrics.share_of_voice.seed_mentions] == [
        (4, "Boutiqaat"),
        (4, "Sephora"),
        (7, "Boutiqaat"),
        (8, "Boutiqaat"),
        (10, "Boutiqaat"),
        (11, "Boutiqaat"),
        (12, "Boutiqaat"),
        (13, "Boutiqaat"),
        (14, "Boutiqaat"),
        (15, "Boutiqaat"),
    ]
    assert metrics.share_of_voice.value == 0.9
    assert [(brand.name, brand.answer_ids, brand.unlocated_answer_ids) for brand in metrics.emergent_brands] == [
        ("Cult Beauty", (4,), ()),
        ("Retailer X", (7,), (7,)),
    ]
    assert metrics.recommendation_strength.answer_ids_by_strength == {
        "recommended": (4,),
        "listed": (7,),
        "passing": (8,),
        "dismissed": (),
    }


def test_locale_intent_and_mode_slices_partition_each_trial_once() -> None:
    answers = _answers()
    overall = compute_metrics(answers, QUERIES, SEEDS)

    locale_total = sum(
        len(compute_metrics(answers, QUERIES, SEEDS, locale=locale).answer_ids)
        for locale in {query.locale for query in QUERIES}
    )
    intent_total = sum(
        len(compute_metrics(answers, QUERIES, SEEDS, intent=intent).answer_ids)
        for intent in {query.intent for query in QUERIES}
    )
    mode_total = sum(
        len(compute_metrics(answers, QUERIES, SEEDS, provider_mode=mode).answer_ids)
        for mode in {answer.provider_mode for answer in answers}
    )

    assert locale_total == intent_total == mode_total == len(overall.answer_ids)


def test_report_renders_traceable_metrics_and_zero_strength_labels(tmp_path: Path) -> None:
    database_path = tmp_path / "avi.db"
    connection = open_database(database_path)
    try:
        with connection:
            create_run(connection, "metrics-run", "v1", "2026-08-31T12:00:00+00:00")
            for trial_index in range(3):
                answer_id = store_answer(
                    connection,
                    "metrics-run",
                    "korean-skincare-kuwait",
                    "ungrounded",
                    trial_index,
                    "test-model",
                    "Answer",
                    False,
                )
                if trial_index == 0:
                    store_mention(connection, answer_id, "Boutiqaat", "Boutiqaat")
                    store_mention(connection, answer_id, "Sephora", "Sephora")
    finally:
        connection.close()

    report = render_report(database_path, "metrics-run")

    assert "## Metrics" in report
    assert "### By Locale" in report
    assert "### By Intent" in report
    assert "### By Provider Mode" in report
    assert "**Visibility Rate:** 1/3 (33.3%). Relevant Answer ids: 1, 2, 3." in report
    assert "Seed Brand Mention sources: 1 (Boutiqaat), 1 (Sephora)." in report
    assert "- recommended: 0. Answer ids: none." in report
    assert "- dismissed: 0. Answer ids: none." in report
    assert "mean" not in report.casefold()
