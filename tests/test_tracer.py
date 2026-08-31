from __future__ import annotations

from pathlib import Path

import pytest

from avi.cli import main
from avi.detect import detect_mentions
from avi.ingest import Query
from avi.providers import CachingProvider, FixtureProvider, configured_model_identifier
from avi.storage import open_database, read_answers


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def boutiqaat() -> tuple[str, list[str]]:
    return (
        "Boutiqaat",
        ["Boutiqaat", "Boutiqat", "Boutiquaat", "بوتيكات", "boutiqaat.com"],
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

    assert configured_model_identifier() == "gpt-5.5-2026-04-23"


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
        mode = "ungrounded"
        model_identifier = "gpt-5.5-2026-04-23"

        def ask(self, supplied_query: Query, trial_index: int) -> str:
            assert supplied_query == query
            assert trial_index == 0
            return "Boutiqaat is one option."

    class NoNetworkProvider:
        mode = "ungrounded"
        model_identifier = "gpt-5.5-2026-04-23"

        def ask(self, supplied_query: Query, trial_index: int) -> str:
            raise AssertionError("the cache should prevent a network call")

    first_answer = CachingProvider(FirstProvider(), tmp_path).ask(query, 0)
    replayed_answer = CachingProvider(NoNetworkProvider(), tmp_path).ask(query, 0)

    assert first_answer == "Boutiqaat is one option."
    assert replayed_answer == first_answer


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
            "No real OpenAI recording exists. Set OPENAI_API_KEY and run "
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
