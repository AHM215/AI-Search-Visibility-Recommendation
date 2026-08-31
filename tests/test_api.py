from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from avi.api import build_app
from avi.ingest import Query
from avi.providers import Answer
from avi.storage import create_run, open_database, store_answer, store_mention


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seeded_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "seeded.db"
    connection = open_database(database_path)
    with connection:
        create_run(connection, "run-1", "v1", "2026-08-31T00:00:00+00:00")
        first = store_answer(
            connection,
            "run-1",
            "korean-skincare-kuwait",
            "ungrounded",
            0,
            "openai/gpt-5.2-2025-12-11",
            "Boutiqaat is one option in Kuwait.",
            False,
        )
        store_mention(connection, first, "Boutiqaat", "Boutiqaat")
        store_answer(
            connection,
            "run-1",
            "korean-skincare-kuwait",
            "ungrounded",
            1,
            "openai/gpt-5.2-2025-12-11",
            "Sephora and iHerb are options.",
            False,
        )
    connection.close()
    return database_path


class StubProvider:
    """Stands in for the Provider seam; the ad-hoc endpoint must never reach the network."""

    mode = "ungrounded"
    model_identifier = "openai/gpt-5.2-2025-12-11"

    def __init__(self, relevance: str = "relevant", text: str = "Boutiqaat sells beauty.") -> None:
        self.relevance = relevance
        self.text = text
        self.asked: list[str] = []

    def ask(self, query: Query, trial_index: int) -> Answer:
        self.asked.append(query.text)
        if "relevant" in query.text.lower() or query.intent == "relevance_gate":
            return Answer(text=self.relevance, citations=[], search_performed=False)
        return Answer(text=self.text, citations=[], search_performed=False)


def client(database_path: Path, provider: StubProvider | None = None) -> TestClient:
    return TestClient(build_app(database_path, provider=provider or StubProvider()))


def test_runs_are_listed(seeded_database: Path) -> None:
    response = client(seeded_database).get("/runs")

    assert response.status_code == 200
    assert [run["id"] for run in response.json()] == ["run-1"]


def test_a_run_returns_its_metrics(seeded_database: Path) -> None:
    response = client(seeded_database).get("/runs/run-1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["visibility_rate"]["mentioned"] == 1
    assert body["visibility_rate"]["relevant_trials"] == 2


def test_a_query_returns_its_trials_with_raw_text(seeded_database: Path) -> None:
    response = client(seeded_database).get("/runs/run-1/queries/korean-skincare-kuwait")

    assert response.status_code == 200
    trials = response.json()["trials"]
    assert len(trials) == 2
    assert trials[0]["text"] == "Boutiqaat is one option in Kuwait."
    assert trials[0]["mentioned"] is True
    assert trials[1]["mentioned"] is False


def test_an_unknown_run_is_not_found(seeded_database: Path) -> None:
    assert client(seeded_database).get("/runs/nope/metrics").status_code == 404


def test_no_endpoint_starts_a_full_run(seeded_database: Path) -> None:
    app_client = client(seeded_database)

    paths = {route.path for route in app_client.app.routes}  # type: ignore[attr-defined]

    assert not any(path.rstrip("/").endswith("/run") for path in paths)


def test_an_adhoc_query_is_classified_for_relevance_before_it_is_scored(
    seeded_database: Path,
) -> None:
    provider = StubProvider(relevance="irrelevant")

    response = client(seeded_database, provider).post(
        "/adhoc", json={"text": "What is a good vegetarian dinner recipe?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["relevance"] == "irrelevant"
    assert body["scored"] is False
    assert body["mentioned"] is None


def test_a_relevant_adhoc_query_is_scored(seeded_database: Path) -> None:
    provider = StubProvider(relevance="relevant", text="Boutiqaat is a Kuwaiti beauty retailer.")

    response = client(seeded_database, provider).post(
        "/adhoc", json={"text": "Where can I buy makeup in Kuwait?"}
    )

    body = response.json()
    assert body["relevance"] == "relevant"
    assert body["scored"] is True
    assert body["mentioned"] is True
    assert "Boutiqaat" in body["text"]


def test_slices_break_visibility_down_by_locale_intent_and_mode(seeded_database: Path) -> None:
    response = client(seeded_database).get("/runs/run-1/slices")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"locale", "intent", "provider_mode"}
    assert body["provider_mode"][0]["relevant_trials"] == 2


def test_diagnosis_is_served_and_every_finding_carries_evidence(seeded_database: Path) -> None:
    response = client(seeded_database).get("/runs/run-1/diagnosis")

    assert response.status_code == 200
    for finding in response.json():
        assert finding["answer_ids"] or finding["citation_urls"]
