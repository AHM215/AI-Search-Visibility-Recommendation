from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import pytest

from avi.citations import (
    MAX_PAGES_PER_ANSWER,
    CachingPageFetcher,
    Citation,
    FixturePageFetcher,
    HttpPageFetcher,
    PageFetch,
    fetch_citation_pages,
)
from avi.ingest import Query, execute_one_query, execute_run, load_brands, load_query_set
from avi.metrics import compute_metrics
from avi.providers import Answer, CachingProvider, FixtureProvider
from avi.report import render_report
from avi.storage import open_database, read_answers


ROOT = Path(__file__).resolve().parents[1]
ALIASES = ["Boutiqaat", "Boutiqat", "Boutiquaat", "بوتيكات", "boutiqaat.com"]


class StaticPageFetcher:
    def __init__(self, pages: dict[str, PageFetch]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def fetch(self, url: str) -> PageFetch:
        self.calls.append(url)
        return self.pages[url]


class AnswerProvider:
    model_identifier = "test-model"

    def __init__(self, mode: str, answer: Answer) -> None:
        self.mode = mode
        self._answer = answer

    def ask(self, query: Query, trial_index: int) -> Answer:
        assert trial_index == 0
        if query.id.startswith("judge-"):
            return Answer(
                '{"recommendation_strength":"listed","brands":["Boutiqaat"]}', [], False
            )
        return self._answer


def test_fetching_only_runs_for_relevant_grounded_absent_answers(tmp_path: Path) -> None:
    page_fetcher = StaticPageFetcher({"https://example.com/page": PageFetch("Boutiqaat")})
    absent = Answer("Try another retailer.", [Citation("https://example.com/page", "Page")], True)
    mentioned = Answer("Try Boutiqaat.", [Citation("https://example.com/page", "Page")], True)

    for run_id, query_id, provider, fetches_page in [
        ("grounded-absent", "korean-skincare-kuwait", AnswerProvider("grounded", absent), True),
        ("ungrounded-absent", "korean-skincare-kuwait", AnswerProvider("ungrounded", absent), False),
        ("irrelevant", "global-buy-sports-equipment", AnswerProvider("grounded", absent), False),
        ("mentioned", "korean-skincare-kuwait", AnswerProvider("grounded", mentioned), False),
    ]:
        before = len(page_fetcher.calls)
        execute_one_query(
            tmp_path / f"{run_id}.db",
            ROOT / "questions.v1.yaml",
            ROOT / "brands.yaml",
            query_id,
            provider,  # type: ignore[arg-type]
            run_id,
            "2026-08-31T12:00:00+00:00",
            page_fetcher=page_fetcher,
        )
        assert (len(page_fetcher.calls) == before + 1) is fetches_page


def test_pages_record_alias_presence_and_absence(tmp_path: Path) -> None:
    database_path = tmp_path / "pages.db"
    citations = [
        Citation("https://example.com/arabic", "Arabic"),
        Citation("https://example.com/absent", "Absent"),
    ]
    page_fetcher = StaticPageFetcher(
        {
            citations[0].url: PageFetch("متجر بوتيكات للتجميل"),
            citations[1].url: PageFetch("A different beauty retailer"),
        }
    )

    execute_one_query(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        AnswerProvider("grounded", Answer("Try another retailer.", citations, True)),  # type: ignore[arg-type]
        "page-statuses",
        "2026-08-31T12:00:00+00:00",
        page_fetcher=page_fetcher,
    )
    connection = open_database(database_path)
    try:
        answer = read_answers(connection, "page-statuses")[0]
    finally:
        connection.close()

    assert [citation.page.status if citation.page else None for citation in answer.citations] == [
        "present",
        "absent",
    ]


@pytest.mark.parametrize(
    ("urlopen_result", "expected_reason"),
    [
        (TimeoutError(), "timeout"),
        (URLError("connection refused"), "connection error: connection refused"),
        ("non-200", "HTTP status 404"),
        ("server-error", "HTTP status 500"),
        ("non-html", "non-HTML content type: application/pdf"),
        ("empty", "no extractable text"),
    ],
)
def test_fetch_failures_are_unfetched_not_absent(
    monkeypatch: pytest.MonkeyPatch, urlopen_result: object, expected_reason: str
) -> None:
    class Response:
        def __init__(self, status: int, content_type: str, body: bytes = b"Boutiqaat") -> None:
            self._status = status
            self.headers = {"Content-Type": content_type}
            self._body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return self._status

        def read(self) -> bytes:
            return self._body

    def fake_urlopen(*args: object, **kwargs: object) -> Response:
        if isinstance(urlopen_result, BaseException):
            raise urlopen_result
        if urlopen_result == "non-200":
            return Response(404, "text/html")
        if urlopen_result == "server-error":
            return Response(500, "text/html")
        if urlopen_result == "empty":
            return Response(200, "text/html", b"")
        return Response(200, "application/pdf")

    monkeypatch.setattr("avi.citations.urlopen", fake_urlopen)

    pages = fetch_citation_pages(
        [Citation("https://example.com/page", "Page")], HttpPageFetcher(), ALIASES
    )

    assert pages == [type(pages[0])("unfetched", expected_reason)]
    assert pages[0].status != "absent"


def test_page_fetch_cap_marks_unselected_citations_unfetched() -> None:
    citations = [Citation(f"https://example.com/{index}", str(index)) for index in range(10)]
    page_fetcher = StaticPageFetcher({citation.url: PageFetch("No match") for citation in citations})

    pages = fetch_citation_pages(citations, page_fetcher, ALIASES)

    assert len(page_fetcher.calls) == MAX_PAGES_PER_ANSWER
    assert [page.status for page in pages[:MAX_PAGES_PER_ANSWER]] == [
        "absent"
    ] * MAX_PAGES_PER_ANSWER
    assert [page.status for page in pages[MAX_PAGES_PER_ANSWER:]] == ["unfetched", "unfetched"]
    assert all(page.unfetched_reason == "per-Answer page cap reached" for page in pages[8:])


def test_page_fixture_replay_uses_cached_fetched_and_unfetched_pages_without_network(
    tmp_path: Path,
) -> None:
    cache_directory = tmp_path / "cache"
    citations = [
        Citation("https://example.com/contains", "Contains"),
        Citation("https://example.com/absent", "Absent"),
        Citation("https://example.com/unfetched", "Unfetched"),
    ]
    answer = Answer("Try another retailer.", citations, True)
    page_fetcher = StaticPageFetcher(
        {
            citations[0].url: PageFetch("Boutiquaat is available here."),
            citations[1].url: PageFetch("No retailer name here."),
            citations[2].url: PageFetch(None, "HTTP status 403"),
        }
    )

    execute_one_query(
        tmp_path / "recorded.db",
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        CachingProvider(AnswerProvider("grounded", answer), cache_directory),  # type: ignore[arg-type]
        "recorded",
        "2026-08-31T12:00:00+00:00",
        page_fetcher=CachingPageFetcher(page_fetcher, cache_directory),
    )
    replay_database = tmp_path / "replayed.db"
    execute_one_query(
        replay_database,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        FixtureProvider(cache_directory, "test-model", mode="grounded"),
        "replayed",
        "2026-08-31T12:00:00+00:00",
        page_fetcher=FixturePageFetcher(cache_directory),
    )
    connection = open_database(replay_database)
    try:
        answer = read_answers(connection, "replayed")[0]
    finally:
        connection.close()

    assert [citation.page.status if citation.page else None for citation in answer.citations] == [
        "present",
        "absent",
        "unfetched",
    ]
    assert answer.citations[2].page is not None
    assert answer.citations[2].page.unfetched_reason == "HTTP status 403"
    report = render_report(replay_database, "replayed")
    assert "Boutiqaat appears on 1 of 2 fetched cited pages, 1 unfetched" in report


def test_page_text_extraction_ignores_markup_and_scripts() -> None:
    citations = [Citation("https://example.com/page", "Page")]
    page_fetcher = StaticPageFetcher(
        {
            citations[0].url: PageFetch(
                '<html><head><meta content="Boutiqaat"><script>Boutiqaat</script></head>'
                "<body>Another beauty retailer</body></html>"
            )
        }
    )

    pages = fetch_citation_pages(citations, page_fetcher, ALIASES)

    assert [page.status for page in pages] == ["absent"]


def test_all_unfetched_pages_do_not_produce_an_absence_claim(tmp_path: Path) -> None:
    citation = Citation("https://example.com/unavailable", "Unavailable")
    execute_one_query(
        tmp_path / "unfetched.db",
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        "korean-skincare-kuwait",
        AnswerProvider("grounded", Answer("Try another retailer.", [citation], True)),  # type: ignore[arg-type]
        "all-unfetched",
        "2026-08-31T12:00:00+00:00",
        page_fetcher=StaticPageFetcher({citation.url: PageFetch(None, "timeout")}),
    )

    report = render_report(tmp_path / "unfetched.db", "all-unfetched")

    assert "no cited pages were fetched; 1 unfetched" in report
    assert "Boutiqaat appears on" not in report


def test_cited_not_named_is_stored_without_changing_text_mention_metrics(tmp_path: Path) -> None:
    own_site = Citation("https://www.boutiqat.com/beauty", "Boutiqaat")
    other_site = Citation("https://example.com/beauty", "Other retailer")
    answers = [
        Answer("Try another retailer.", [own_site], True),
        Answer("Try Boutiqaat.", [own_site], True),
        Answer("Try another retailer.", [other_site], True),
    ]

    class ThreeTrialProvider:
        mode = "grounded"
        model_identifier = "test-model"

        def ask(self, query: Query, trial_index: int) -> Answer:
            if query.id.startswith("judge-"):
                return Answer(
                    '{"recommendation_strength":"listed","brands":["Boutiqaat"]}', [], False
                )
            return answers[trial_index]

    database_path = tmp_path / "cited-not-named.db"
    execute_run(
        database_path,
        ROOT / "questions.v1.yaml",
        ROOT / "brands.yaml",
        ThreeTrialProvider(),
        "cited-not-named",
        "2026-08-31T12:00:00+00:00",
        query_ids=["korean-skincare-kuwait"],
        page_fetcher=StaticPageFetcher(
            {
                own_site.url: PageFetch("Boutiqaat is available here."),
                other_site.url: PageFetch("Another retailer is available here."),
            }
        ),
    )
    connection = open_database(database_path)
    try:
        stored_answers = read_answers(connection, "cited-not-named")
    finally:
        connection.close()

    assert [answer.mentioned for answer in stored_answers] == [False, True, False]
    assert [answer.cited_not_named for answer in stored_answers] == [True, False, False]
    assert stored_answers[0].citations[0].page is not None
    assert stored_answers[0].citations[0].page.status == "present"
    metrics = compute_metrics(
        stored_answers,
        load_query_set(ROOT / "questions.v1.yaml").queries,
        load_brands(ROOT / "brands.yaml").seed_competitors,
    )
    assert metrics.visibility_rate.mentioned_answer_ids == (2,)
    assert metrics.visibility_rate.relevant_answer_ids == (1, 2, 3)
    assert metrics.cited_not_named_answer_ids == (1,)
    report = render_report(database_path, "cited-not-named")
    assert "**Cited-not-named:** 1 grounded Trial cited a Boutiqaat URL without naming Boutiqaat. Answer ids: 1." in report
