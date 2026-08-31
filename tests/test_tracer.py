from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from avi.cli import main
from avi.citations import Citation, classify_source_type
from avi.detect import detect_mentions
from avi.ingest import Query, execute_one_query, execute_run, load_query_set
from avi.judge import JudgeVerdict, Verdict
from avi.providers import (
    Answer,
    CachingProvider,
    FixtureProvider,
    GroundedOpenAIProvider,
    configured_openai_base_url,
    configured_model_identifier,
)
from avi.report import render_report
from avi.storage import open_database, read_answers, read_run, store_answer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def boutiqaat() -> tuple[str, list[str]]:
    return (
        "Boutiqaat",
        [
            "Boutiqaat",
            "Boutiqat",
            "Boutiquaat",
            "بوتيكات",
            "boutiqaat.com",
            "boutiqat.com",
        ],
    )


@pytest.mark.parametrize(
    ("answer_text", "alias"),
    [
        ("Boutiqaat ships beauty products in Kuwait.", "Boutiqaat"),
        ("Try Boutiquaat for beauty products.", "Boutiquaat"),
        ("بوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("وبوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("لبوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("البوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("والبوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("للبوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("وللبوتيكات متجر تجميل كويتي.", "بوتيكات"),
        ("Visit https://www.boutiqaat.com/beauty for its catalogue.", "boutiqaat.com"),
        ("Visit https://www.boutiqat.com/beauty for its catalogue.", "boutiqat.com"),
        ("Boutiqaat is a poor choice because it only serves Kuwait.", "Boutiqaat"),
    ],
)
def test_alias_detection_covers_required_surface_forms(
    boutiqaat: tuple[str, list[str]], answer_text: str, alias: str
) -> None:
    _, aliases = boutiqaat

    mentions = detect_mentions(answer_text, aliases)

    assert [mention.alias for mention in mentions] == [alias]


def test_alias_detection_rejects_text_inside_another_brand_name(
    boutiqaat: tuple[str, list[str]]
) -> None:
    _, aliases = boutiqaat

    mentions = detect_mentions("Boutiqaati is another beauty retailer.", aliases)

    assert mentions == []


def test_default_model_identifier_is_current(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVI_OPENAI_MODEL", raising=False)

    assert configured_model_identifier() == "openai/gpt-5.2-2025-12-11"


def test_openai_base_url_defaults_to_lightning_and_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AVI_OPENAI_BASE_URL", raising=False)
    assert configured_openai_base_url() == "https://lightning.ai/api/v1"

    monkeypatch.setenv("AVI_OPENAI_BASE_URL", "https://provider.example/api/v1")
    assert configured_openai_base_url() == "https://provider.example/api/v1"


@pytest.mark.parametrize(
    ("url", "source_type"),
    [
        ("https://www.boutiqat.com/beauty", "retailer"),
        ("https://www.noon.com/beauty", "marketplace"),
        ("https://www.vogue.com/beauty", "editorial/listicle"),
        ("https://www.trustpilot.com/review/example.com", "review site"),
        ("https://example.com/article", "other"),
    ],
)
def test_citations_are_classified_from_their_domains(url: str, source_type: str) -> None:
    assert classify_source_type(url) == source_type


def test_caching_provider_reuses_a_recorded_answer_without_a_second_call(
    tmp_path: Path,
) -> None:
    query = Query(
        id="korean-skincare-kuwait",
        text="Where can I buy Korean skincare in Kuwait?",
        intent="find_category",
        locale="gcc_en",
        specificity="narrow",
        relevance="relevant",
    )

    class FirstProvider:
        mode = "grounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert supplied_query == query
            assert trial_index == 0
            return Answer(
                "Boutiqaat is one option.",
                [Citation("https://www.boutiqat.com", "Boutiqaat")],
                True,
            )

    class NoNetworkProvider:
        mode = "grounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            raise AssertionError("the cache should prevent a network call")

    first_answer = CachingProvider(FirstProvider(), tmp_path).ask(query, 0)
    replayed_answer = CachingProvider(NoNetworkProvider(), tmp_path).ask(query, 0)

    assert first_answer == Answer(
        "Boutiqaat is one option.",
        [Citation("https://www.boutiqat.com", "Boutiqaat")],
        True,
    )
    assert replayed_answer == first_answer


def test_grounded_provider_requests_web_search_and_extracts_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = Query(
        id="korean-skincare-kuwait",
        text="Where can I buy Korean skincare in Kuwait?",
        intent="find_category",
        locale="gcc_en",
        specificity="narrow",
        relevance="relevant",
    )
    calls: list[dict[str, object]] = []

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                output=[
                    SimpleNamespace(type="web_search_call"),
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text="Boutiqaat sells beauty products.",
                                annotations=[
                                    SimpleNamespace(
                                        type="url_citation",
                                        url="https://www.boutiqat.com/beauty",
                                        title="Boutiqaat Beauty",
                                    )
                                ],
                            )
                        ],
                    ),
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["base_url"] == "https://lightning.ai/api/v1"
            self.responses = FakeResponses()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-lit-test")
    monkeypatch.setattr("avi.providers.OpenAI", FakeOpenAI)

    answer = GroundedOpenAIProvider().ask(query, 0)

    assert calls == [
        {
            "model": "openai/gpt-5.2-2025-12-11",
            "input": "Where can I buy Korean skincare in Kuwait?",
            "tools": [{"type": "web_search"}],
        }
    ]
    assert answer == Answer(
        text="Boutiqaat sells beauty products.",
        citations=[Citation("https://www.boutiqat.com/beauty", "Boutiqaat Beauty")],
        search_performed=True,
    )


def test_grounded_answer_stores_citations_and_renders_modes_separately(tmp_path: Path) -> None:
    database_path = tmp_path / "avi.db"

    class GroundedProvider:
        mode = "grounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer(
                    text="Here are places to buy Korean skincare, including Boutiqaat.",
                    citations=[
                        Citation("https://www.boutiqat.com/beauty", "Boutiqaat Beauty"),
                        Citation("https://www.trustpilot.com/review/example.com", "Reviews"),
                    ],
                    search_performed=True,
                )
            assert supplied_query.id.startswith("judge-")
            return Answer(
                '{"recommendation_strength":"listed","brands":["Boutiqaat"]}',
                [],
                False,
            )

    run_id = execute_one_query(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        GroundedProvider(),
        "grounded-run",
        "2026-08-31T12:00:00+00:00",
    )
    connection = open_database(database_path)
    try:
        with connection:
            store_answer(
                connection,
                run_id,
                "korean-skincare-kuwait",
                "ungrounded",
                0,
                "openai/gpt-5.2-2025-12-11",
                "An ungrounded Answer.",
                False,
            )
        answers = read_answers(connection, run_id)
    finally:
        connection.close()

    grounded_answer = next(answer for answer in answers if answer.provider_mode == "grounded")
    assert grounded_answer.search_performed is True
    assert grounded_answer.mentioned is True
    assert [(citation.url, citation.title, citation.source_type) for citation in grounded_answer.citations] == [
        ("https://www.boutiqat.com/beauty", "Boutiqaat Beauty", "retailer"),
        ("https://www.trustpilot.com/review/example.com", "Reviews", "review site"),
    ]

    report_text = render_report(database_path, run_id)
    assert "## Ungrounded Provider" in report_text
    assert "## Grounded Provider" in report_text
    assert "Search performed: yes" in report_text
    assert "[Boutiqaat Beauty](https://www.boutiqat.com/beauty) (Source Type: retailer)" in report_text
    assert "[Reviews](https://www.trustpilot.com/review/example.com) (Source Type: review site)" in report_text


def test_grounded_answer_without_citations_stores_successfully(tmp_path: Path) -> None:
    database_path = tmp_path / "avi.db"

    class GroundedProvider:
        mode = "grounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer("An Answer from memory.", [], False)

    execute_one_query(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        GroundedProvider(),
        "no-citation-run",
        "2026-08-31T12:00:00+00:00",
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, "no-citation-run")
    finally:
        connection.close()

    assert len(answers) == 1
    assert answers[0].citations == []
    assert answers[0].search_performed is False


def test_fixture_replay_drives_cli_to_a_traceable_markdown_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = tmp_path / "avi.db"
    fixture_provider = FixtureProvider(ROOT / "cache", configured_model_identifier())
    try:
        run_status = main(
            [
                "run",
                "korean-skincare-kuwait",
                "--database",
                str(database_path),
                "--query-set",
                str(ROOT / "questions.v1.yaml"),
                "--brands",
                str(ROOT / "brands.yaml"),
            ],
            provider=fixture_provider,
            run_at="2026-08-31T12:00:00+00:00",
            run_id="fixture-run",
        )
    except FileNotFoundError:
        pytest.skip(
            "No ungrounded fixture recording for an Answer or Judge verdict exists. Set OPENAI_API_KEY "
            "to the Lightning AI key and run "
            "`python -m avi.cli run korean-skincare-kuwait` to record one."
        )

    assert run_status == 0
    assert capsys.readouterr().out == "fixture-run\n"

    report_status = main(["report", "fixture-run", "--database", str(database_path), "--full"])

    assert report_status == 0
    report_text = capsys.readouterr().out
    assert "# Boutiqaat AI Search Visibility Report" in report_text
    assert "Run: fixture-run" in report_text
    assert "Query Set version: v1" in report_text
    assert f"Model identifier: {fixture_provider.model_identifier}" in report_text
    assert "OpenAI" in report_text

    connection = open_database(database_path)
    try:
        answers = read_answers(connection, "fixture-run")
    finally:
        connection.close()
    assert answers
    assert all(answer.verdict is not None for answer in answers if answer.mentioned)
    answer_ids = ", ".join(str(answer.id) for answer in answers)
    mentioned_ids = ", ".join(str(answer.id) for answer in answers if answer.mentioned)
    statement = (
        f"Boutiqaat was Mentioned. Answer ids: {mentioned_ids}."
        if mentioned_ids
        else f"Boutiqaat was not Mentioned. Answer ids: {answer_ids}."
    )
    assert statement in report_text
    for answer in answers:
        assert f"### Answer {answer.id}" in report_text
        assert answer.text in report_text


def test_grounded_fixture_replay_stores_and_renders_citations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database_path = tmp_path / "avi.db"
    fixture_provider = FixtureProvider(
        ROOT / "cache", configured_model_identifier(), mode="grounded"
    )
    try:
        run_status = main(
            [
                "run",
                "korean-skincare-kuwait",
                "--database",
                str(database_path),
                "--query-set",
                str(ROOT / "questions.v1.yaml"),
                "--brands",
                str(ROOT / "brands.yaml"),
            ],
            provider=fixture_provider,
            run_at="2026-08-31T12:00:00+00:00",
            run_id="grounded-fixture-run",
        )
    except FileNotFoundError:
        pytest.skip(
            "No grounded fixture recording exists. Set OPENAI_API_KEY to the Lightning AI key and run "
            "`python -m avi.cli run korean-skincare-kuwait --mode grounded` to record one."
        )

    assert run_status == 0
    assert capsys.readouterr().out == "grounded-fixture-run\n"
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, "grounded-fixture-run")
    finally:
        connection.close()
    assert len(answers) == 3
    assert [answer.trial_index for answer in answers] == [0, 1, 2]
    assert {answer.provider_mode for answer in answers} == {"grounded"}
    assert all(answer.citations for answer in answers)
    assert all(
        citation.title and citation.url for answer in answers for citation in answer.citations
    )
    assert any(
        answer.verdict is not None
        and "K-Beauty Kuwait (Almail Al Thahbi)" in answer.verdict.unlocated_brands
        for answer in answers
    )

    report_status = main(["report", "grounded-fixture-run", "--database", str(database_path), "--full"])

    assert report_status == 0
    report_text = capsys.readouterr().out
    assert "## Grounded Provider" in report_text
    assert "#### Citations" in report_text
    assert "Unlocated Brands: K-Beauty Kuwait (Almail Al Thahbi)" in report_text


def test_dismissed_boutiqaat_is_mentioned_and_has_a_dismissed_verdict(tmp_path: Path) -> None:
    database_path = tmp_path / "avi.db"

    class DismissalProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer(
                    "Do not buy from Boutiqaat because it only serves Kuwait.", [], False
                )
            assert supplied_query.id.startswith("judge-")
            return Answer(
                '{"recommendation_strength":"dismissed","brands":["Boutiqaat"]}',
                [],
                False,
            )

    run_id = execute_one_query(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        DismissalProvider(),
        "dismissal-run",
        "2026-08-31T12:00:00+00:00",
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, run_id)
    finally:
        connection.close()

    assert len(answers) == 1
    assert answers[0].mentioned is True
    assert answers[0].verdict is not None
    assert answers[0].verdict.recommendation_strength == "dismissed"
    assert answers[0].verdict.recommendation_strength != "recommended"
    assert answers[0].verdict.rank == 1
    assert answers[0].verdict.brands == ["Boutiqaat"]
    report_text = render_report(database_path, run_id)
    assert "Recommendation Strength distribution: recommended: 0, listed: 0, passing: 0, dismissed: 1" in report_text
    assert "Recommendation Strength: dismissed" in report_text


def test_answer_without_a_mention_never_reaches_the_judge(tmp_path: Path) -> None:
    database_path = tmp_path / "avi.db"

    class NoMentionProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer("Try a local beauty retailer.", [], False)
            raise AssertionError("the Judge must not receive an Answer without a Mention")

    run_id = execute_one_query(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        NoMentionProvider(),
        "no-mention-run",
        "2026-08-31T12:00:00+00:00",
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, run_id)
    finally:
        connection.close()

    assert answers[0].mentioned is False
    assert answers[0].verdict is None


def test_judge_response_with_rank_is_rejected_as_an_extra_field(tmp_path: Path) -> None:
    class MalformedVerdictProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer("Boutiqaat is one option.", [], False)
            return Answer(
                '{"recommendation_strength":"recommended","rank":1,"brands":["Boutiqaat"]}',
                [],
                False,
            )

    with pytest.raises(ValidationError) as error_info:
        execute_one_query(
            tmp_path / "avi.db",
            ROOT / "questions.v1.yaml",
            ROOT / "brands.yaml",
            "korean-skincare-kuwait",
            MalformedVerdictProvider(),
            "unexpected-rank-run",
            "2026-08-31T12:00:00+00:00",
        )
    assert error_info.value.errors()[0]["loc"] == ("rank",)


def test_judge_response_without_boutiqaat_in_brands_raises(tmp_path: Path) -> None:
    class MissingBoutiqaatProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer("Boutiqaat is one option.", [], False)
            return Answer(
                '{"recommendation_strength":"recommended","brands":["Boots"]}',
                [],
                False,
            )

    with pytest.raises(ValidationError):
        execute_one_query(
            tmp_path / "avi.db",
            ROOT / "questions.v1.yaml",
            ROOT / "brands.yaml",
            "korean-skincare-kuwait",
            MissingBoutiqaatProvider(),
            "missing-boutiqaat-run",
            "2026-08-31T12:00:00+00:00",
        )


def test_judged_brand_absent_from_answer_text_is_stored_as_unlocated() -> None:
    judge_verdict = JudgeVerdict.model_validate(
        {"recommendation_strength": "recommended", "brands": ["Boutiqaat", "Boots"]}
    )

    verdict = Verdict.from_judge_verdict(judge_verdict, "Boutiqaat is one option.")

    assert verdict.brands == ["Boutiqaat"]
    assert verdict.unlocated_brands == ["Boots"]
    assert verdict.rank == 1


def test_rank_uses_boutiqaat_alias_and_excludes_unlocated_brands() -> None:
    aliases = ["Boutiqaat", "Boutiqat", "Boutiquaat", "بوتيكات"]
    judge_verdict = JudgeVerdict.model_validate(
        {
            "recommendation_strength": "listed",
            "brands": ["Boutiqat", "K-Beauty Kuwait (Almail Al Thahbi)", "Boots"],
        },
        context={"boutiqaat_aliases": aliases},
    )

    verdict = Verdict.from_judge_verdict(
        judge_verdict,
        "Boots is first, then بوتيكات.",
        aliases,
    )

    assert verdict.brands == ["Boots", "Boutiqaat"]
    assert verdict.unlocated_brands == ["K-Beauty Kuwait (Almail Al Thahbi)"]
    assert verdict.rank == 2


def test_brands_and_rank_follow_answer_order_not_judge_order() -> None:
    answer_text = "YesStyle is first, then Boutiqaat, then Boots."
    alphabetical_verdict = JudgeVerdict.model_validate(
        {"recommendation_strength": "listed", "brands": ["Boots", "Boutiqaat", "YesStyle"]}
    )
    reverse_verdict = JudgeVerdict.model_validate(
        {"recommendation_strength": "listed", "brands": ["YesStyle", "Boutiqaat", "Boots"]}
    )

    alphabetical_order = Verdict.from_judge_verdict(alphabetical_verdict, answer_text)
    reverse_order = Verdict.from_judge_verdict(reverse_verdict, answer_text)

    assert alphabetical_order.brands == ["YesStyle", "Boutiqaat", "Boots"]
    assert reverse_order.brands == alphabetical_order.brands
    assert alphabetical_order.rank == 2


def test_fixture_provider_replays_an_answer_and_judge_verdict_without_network(
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"

    class RecordingProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert trial_index == 0
            if supplied_query.id == "korean-skincare-kuwait":
                return Answer("Boots, YesStyle, and Boutiqaat are options.", [], False)
            assert supplied_query.id.startswith("judge-")
            return Answer(
                '{"recommendation_strength":"recommended","brands":["Boots","YesStyle","Boutiqaat"]}',
                [],
                False,
            )

    execute_one_query(
        tmp_path / "recorded.db",
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        CachingProvider(RecordingProvider(), cache_directory),
        "recorded-run",
        "2026-08-31T12:00:00+00:00",
    )
    replay_database_path = tmp_path / "replayed.db"
    execute_one_query(
        replay_database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        FixtureProvider(cache_directory, "openai/gpt-5.2-2025-12-11"),
        "replayed-run",
        "2026-08-31T12:00:00+00:00",
    )
    connection = open_database(replay_database_path)
    try:
        answers = read_answers(connection, "replayed-run")
    finally:
        connection.close()

    assert answers[0].verdict is not None
    assert answers[0].verdict.recommendation_strength == "recommended"
    assert answers[0].verdict.rank == 3


def test_query_set_has_three_locale_tiers_and_relevance_labels() -> None:
    query_set = load_query_set(ROOT / "questions.v1.yaml")

    assert len(query_set.queries) == 24
    assert {query.locale for query in query_set.queries} == {"global_en", "gcc_en", "ar"}
    assert all(sum(query.locale == locale for query in query_set.queries) == 8 for locale in {
        "global_en",
        "gcc_en",
        "ar",
    })
    assert {query.relevance for query in query_set.queries} == {"relevant", "irrelevant"}


@pytest.mark.parametrize("mode", ["ungrounded", "grounded"])
def test_run_stores_three_distinct_trials_per_query_per_mode(tmp_path: Path, mode: str) -> None:
    class ThreeTrialProvider:
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def __init__(self, provider_mode: str) -> None:
            self.mode = provider_mode

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert supplied_query.id == "korean-skincare-kuwait"
            return Answer(f"Trial {trial_index} Answer.", [], False)

    database_path = tmp_path / f"{mode}.db"
    result = execute_run(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        ThreeTrialProvider(mode),  # type: ignore[arg-type]
        f"{mode}-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, result.run_id)
    finally:
        connection.close()

    assert result.status == "completed"
    assert [(answer.provider_mode, answer.trial_index) for answer in answers] == [
        (mode, 0),
        (mode, 1),
        (mode, 2),
    ]


def test_rerun_uses_cached_trials_without_calling_the_provider(tmp_path: Path) -> None:
    class RecordingProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer(f"Recorded trial {trial_index}.", [], False)

    class NoNetworkProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            raise AssertionError("the cache should prevent every Provider call")

    cache_directory = tmp_path / "cache"
    first_result = execute_run(
        tmp_path / "first.db",
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        CachingProvider(RecordingProvider(), cache_directory),
        "first-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
    )
    replay_result = execute_run(
        tmp_path / "replay.db",
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        CachingProvider(NoNetworkProvider(), cache_directory),
        "replay-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
    )

    assert first_result.live_calls == 3
    assert replay_result.status == "completed"
    assert replay_result.live_calls == 0


def test_dry_run_reports_mixed_cache_without_executing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    query = Query(
        id="korean-skincare-kuwait",
        text="Where can I buy Korean skincare in Kuwait?",
        intent="find_category",
        locale="gcc_en",
        specificity="narrow",
        relevance="relevant",
    )

    class OneRecordingProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert supplied_query == query
            assert trial_index == 0
            return Answer("A cached Answer without a Mention.", [], False)

    class NoNetworkProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            raise AssertionError("dry-run must not call the Provider")

    cache_directory = tmp_path / "cache"
    CachingProvider(OneRecordingProvider(), cache_directory).ask(query, 0)
    database_path = tmp_path / "dry-run.db"

    status = main(
        [
            "run",
            "korean-skincare-kuwait",
            "--dry-run",
            "--database",
            str(database_path),
        ],
        provider=CachingProvider(NoNetworkProvider(), cache_directory),
    )

    assert status == 0
    assert capsys.readouterr().out == (
        "Dry run: 1 cached calls, up to 4 live calls (2 Answer calls, 0 known Judge calls, "
        "2 possible Judge calls); estimated cost $0.04\n"
    )
    assert not database_path.exists()


def test_call_budget_aborts_and_keeps_stored_trials(tmp_path: Path) -> None:
    class CountingProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer(f"Trial {trial_index} Answer.", [], False)

    database_path = tmp_path / "aborted.db"
    result = execute_run(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        CountingProvider(),
        "aborted-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
        call_budget=2,
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, result.run_id)
        stored_run = read_run(connection, result.run_id)
    finally:
        connection.close()

    assert result.status == "aborted"
    assert result.live_calls == 2
    assert [answer.trial_index for answer in answers] == [0, 1]
    assert stored_run.status == "aborted"


def test_cli_runs_the_full_query_set_when_query_id_is_omitted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FullSetProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert 0 <= trial_index < 3
            return Answer(f"Answer for {supplied_query.id}, trial {trial_index}.", [], False)

    database_path = tmp_path / "full-set.db"
    status = main(
        ["run", "--database", str(database_path)],
        provider=FullSetProvider(),
        run_at="2026-08-31T12:00:00+00:00",
        run_id="full-set-run",
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, "full-set-run")
    finally:
        connection.close()

    assert status == 0
    assert capsys.readouterr().out == "full-set-run\n"
    assert len(answers) == 72
    assert {answer.trial_index for answer in answers} == {0, 1, 2}


def test_run_stores_both_provider_modes_under_one_run(tmp_path: Path) -> None:
    class ModeProvider:
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def __init__(self, mode: str) -> None:
            self.mode = mode

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer(f"{self.mode} {supplied_query.id} {trial_index}", [], False)

    database_path = tmp_path / "both-modes.db"
    result = execute_run(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        [ModeProvider("ungrounded"), ModeProvider("grounded")],  # type: ignore[list-item]
        "both-modes-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["global-best-beauty-websites", "korean-skincare-kuwait"],
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, result.run_id)
    finally:
        connection.close()

    assert result.status == "completed"
    assert len(answers) == 12
    assert {
        (answer.query_id, answer.provider_mode, answer.trial_index) for answer in answers
    } == {
        (query_id, mode, trial_index)
        for query_id in ("global-best-beauty-websites", "korean-skincare-kuwait")
        for mode in ("ungrounded", "grounded")
        for trial_index in range(3)
    }


def test_dry_run_counts_cached_and_live_calls_for_both_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    query = Query(
        id="korean-skincare-kuwait",
        text="Where can I buy Korean skincare in Kuwait?",
        intent="find_category",
        locale="gcc_en",
        specificity="narrow",
        relevance="relevant",
    )

    class RecordingProvider:
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def __init__(self, mode: str) -> None:
            self.mode = mode

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            assert supplied_query == query
            assert trial_index == 0
            return Answer("Cached Answer without a Mention.", [], False)

    cache_directory = tmp_path / "cache"
    for mode in ("ungrounded", "grounded"):
        CachingProvider(RecordingProvider(mode), cache_directory).ask(query, 0)  # type: ignore[arg-type]

    status = main(
        ["run", "korean-skincare-kuwait", "--dry-run", "--cache", str(cache_directory)]
    )

    assert status == 0
    assert capsys.readouterr().out == (
        "Dry run: 2 cached calls, up to 8 live calls (4 Answer calls, 0 known Judge calls, "
        "4 possible Judge calls); estimated cost $0.08\n"
    )


def test_mode_filter_restricts_cli_run_to_one_provider_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UngroundedProvider:
        mode = "ungrounded"
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer("An ungrounded Answer.", [], False)

    class UnexpectedGroundedProvider:
        def __init__(self) -> None:
            raise AssertionError("--mode ungrounded must not construct the grounded Provider")

    monkeypatch.setattr("avi.cli.UngroundedOpenAIProvider", UngroundedProvider)
    monkeypatch.setattr("avi.cli.GroundedOpenAIProvider", UnexpectedGroundedProvider)
    database_path = tmp_path / "ungrounded-only.db"

    status = main(
        [
            "run",
            "korean-skincare-kuwait",
            "--mode",
            "ungrounded",
            "--database",
            str(database_path),
            "--cache",
            str(tmp_path / "cache"),
        ],
        run_at="2026-08-31T12:00:00+00:00",
        run_id="ungrounded-only-run",
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, "ungrounded-only-run")
    finally:
        connection.close()

    assert status == 0
    assert len(answers) == 3
    assert {answer.provider_mode for answer in answers} == {"ungrounded"}


def test_call_budget_is_shared_between_provider_modes(tmp_path: Path) -> None:
    class ModeProvider:
        model_identifier = "openai/gpt-5.2-2025-12-11"

        def __init__(self, mode: str) -> None:
            self.mode = mode

        def ask(self, supplied_query: Query, trial_index: int) -> Answer:
            return Answer(f"{self.mode} {trial_index}", [], False)

    database_path = tmp_path / "shared-budget.db"
    result = execute_run(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        [ModeProvider("ungrounded"), ModeProvider("grounded")],  # type: ignore[list-item]
        "shared-budget-run",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
        call_budget=4,
    )
    connection = open_database(database_path)
    try:
        answers = read_answers(connection, result.run_id)
        stored_run = read_run(connection, result.run_id)
    finally:
        connection.close()

    assert result.status == "aborted"
    assert result.live_calls == 4
    assert len(answers) == 4
    assert {answer.provider_mode for answer in answers} == {"ungrounded", "grounded"}
    assert stored_run.status == "aborted"
