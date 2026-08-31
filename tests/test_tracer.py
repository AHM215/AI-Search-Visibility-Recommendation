from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from avi.cli import main
from avi.citations import Citation, classify_source_type
from avi.detect import detect_mentions
from avi.ingest import Query, execute_one_query
from avi.providers import (
    Answer,
    CachingProvider,
    FixtureProvider,
    GroundedOpenAIProvider,
    configured_openai_base_url,
    configured_model_identifier,
)
from avi.report import render_report
from avi.storage import open_database, read_answers, store_answer


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
            assert supplied_query.id == "korean-skincare-kuwait"
            assert trial_index == 0
            return Answer(
                text="Here are places to buy Korean skincare.",
                citations=[
                    Citation("https://www.boutiqat.com/beauty", "Boutiqaat Beauty"),
                    Citation("https://www.trustpilot.com/review/example.com", "Reviews"),
                ],
                search_performed=True,
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
            "No ungrounded fixture recording exists. Set OPENAI_API_KEY to the Lightning AI key and run "
            "`python -m avi.cli run korean-skincare-kuwait` to record one."
        )

    assert run_status == 0
    assert capsys.readouterr().out == "fixture-run\n"

    report_status = main(["report", "fixture-run", "--database", str(database_path)])

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
    assert len(answers) == 1
    assert answers[0].provider_mode == "grounded"
    assert answers[0].citations
    assert all(citation.title and citation.url for citation in answers[0].citations)

    report_status = main(["report", "grounded-fixture-run", "--database", str(database_path)])

    assert report_status == 0
    report_text = capsys.readouterr().out
    assert "## Grounded Provider" in report_text
    assert "#### Citations" in report_text
