from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import yaml
from pydantic import BaseModel

from avi.citations import PageFetcher, fetch_citation_pages
from avi.detect import detect_mentions
from avi.judge import judge_answer, judge_query_for_answer
from avi.providers import Answer, CachingProvider, FixtureProvider, Provider
from avi.storage import (
    create_run,
    open_database,
    set_run_status,
    store_answer,
    store_citation,
    store_citation_page,
    store_mention,
    store_verdict,
)


TRIALS_PER_QUERY = 3
DEFAULT_CALL_BUDGET = 300
ESTIMATED_COST_PER_CALL_USD = 0.01


class Query(BaseModel):
    id: str
    text: str
    intent: str
    locale: str
    specificity: str
    relevance: Literal["relevant", "irrelevant"]


class QuerySet(BaseModel):
    version: str
    queries: list[Query]


class Brand(BaseModel):
    name: str
    aliases: list[str]


class BrandFile(BaseModel):
    brands: list[Brand]
    seed_competitors: list[Brand]


@dataclass(frozen=True)
class RunPlan:
    cached_calls: int
    live_answer_calls: int
    live_judge_calls: int
    potential_judge_calls: int

    @property
    def projected_live_calls(self) -> int:
        return self.live_answer_calls + self.live_judge_calls + self.potential_judge_calls

    @property
    def estimated_cost_usd(self) -> float:
        return self.projected_live_calls * ESTIMATED_COST_PER_CALL_USD


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: Literal["completed", "aborted"]
    live_calls: int


def load_query_set(path: Path) -> QuerySet:
    return QuerySet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_brands(path: Path) -> BrandFile:
    return BrandFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def select_query(query_set: QuerySet, query_id: str) -> Query:
    for query in query_set.queries:
        if query.id == query_id:
            return query
    raise ValueError(f"Query {query_id!r} is not in Query Set {query_set.version}")


def select_brand(brand_file: BrandFile, brand_name: str) -> Brand:
    for brand in brand_file.brands:
        if brand.name == brand_name:
            return brand
    raise ValueError(f"Brand {brand_name!r} is not in the Brand file")


def _recorded_answer(provider: Provider, query: Query, trial_index: int) -> Answer | None:
    if isinstance(provider, (CachingProvider, FixtureProvider)):
        return provider.recorded_answer(query, trial_index)
    return None


def _provider_list(providers: Provider | Sequence[Provider]) -> list[Provider]:
    if isinstance(providers, Sequence):
        return list(providers)
    return [providers]


def _mentions_boutiqaat(answer: Answer, boutiqaat: Brand) -> bool:
    return bool(detect_mentions(answer.text, boutiqaat.aliases))


def _all_brands(brand_file: BrandFile) -> list[Brand]:
    return [*brand_file.brands, *brand_file.seed_competitors]


def plan_run(
    query_set_path: Path,
    brand_path: Path,
    providers: Provider | Sequence[Provider],
    query_ids: Sequence[str] | None = None,
    trials_per_query: int = TRIALS_PER_QUERY,
) -> RunPlan:
    query_set = load_query_set(query_set_path)
    queries = (
        [select_query(query_set, query_id) for query_id in query_ids]
        if query_ids is not None
        else query_set.queries
    )
    active_providers = _provider_list(providers)
    if not active_providers:
        raise ValueError("A Run requires at least one Provider")
    brand_file = load_brands(brand_path)
    boutiqaat = select_brand(brand_file, "Boutiqaat")
    cached_calls = 0
    live_answer_calls = 0
    live_judge_calls = 0
    potential_judge_calls = 0
    for query in queries:
        for trial_index in range(trials_per_query):
            for provider in active_providers:
                answer = _recorded_answer(provider, query, trial_index)
                if answer is None:
                    live_answer_calls += 1
                    # An unrecorded Answer might mention Boutiqaat and require the Judge.
                    potential_judge_calls += 1
                    continue
                cached_calls += 1
                if not _mentions_boutiqaat(answer, boutiqaat):
                    continue
                judge_recording = _recorded_answer(provider, judge_query_for_answer(answer), 0)
                if judge_recording is None:
                    live_judge_calls += 1
                else:
                    cached_calls += 1
    return RunPlan(
        cached_calls=cached_calls,
        live_answer_calls=live_answer_calls,
        live_judge_calls=live_judge_calls,
        potential_judge_calls=potential_judge_calls,
    )


def _call_within_budget(
    provider: Provider, query: Query, trial_index: int, live_calls: int, call_budget: int
) -> tuple[bool, int]:
    if _recorded_answer(provider, query, trial_index) is not None:
        return True, live_calls
    if live_calls >= call_budget:
        return False, live_calls
    return True, live_calls + 1


def execute_run(
    database_path: Path,
    query_set_path: Path,
    brand_path: Path,
    providers: Provider | Sequence[Provider],
    run_id: str,
    run_at: str,
    query_ids: Sequence[str] | None = None,
    trials_per_query: int = TRIALS_PER_QUERY,
    call_budget: int = DEFAULT_CALL_BUDGET,
    page_fetcher: PageFetcher | None = None,
) -> RunResult:
    if call_budget < 0:
        raise ValueError("Call budget must not be negative")
    active_providers = _provider_list(providers)
    if not active_providers:
        raise ValueError("A Run requires at least one Provider")
    query_set = load_query_set(query_set_path)
    queries = (
        [select_query(query_set, query_id) for query_id in query_ids]
        if query_ids is not None
        else query_set.queries
    )
    brand_file = load_brands(brand_path)
    boutiqaat = select_brand(brand_file, "Boutiqaat")
    connection: sqlite3.Connection = open_database(database_path)
    live_calls = 0
    aborted = False
    try:
        with connection:
            create_run(connection, run_id, query_set.version, run_at)
        for query in queries:
            for trial_index in range(trials_per_query):
                for provider in active_providers:
                    allowed, live_calls = _call_within_budget(
                        provider, query, trial_index, live_calls, call_budget
                    )
                    if not allowed:
                        aborted = True
                        break
                    answer = provider.ask(query, trial_index)
                    boutiqaat_mentions = detect_mentions(answer.text, boutiqaat.aliases)
                    citation_urls = "\n".join(citation.url for citation in answer.citations)
                    cited_not_named = (
                        provider.mode == "grounded"
                        and not boutiqaat_mentions
                        and bool(detect_mentions(citation_urls, boutiqaat.aliases))
                    )
                    with connection:
                        answer_id = store_answer(
                            connection,
                            run_id,
                            query.id,
                            provider.mode,
                            trial_index,
                            provider.model_identifier,
                            answer.text,
                            answer.search_performed,
                            cited_not_named,
                        )
                        for citation_index, citation in enumerate(answer.citations):
                            store_citation(connection, answer_id, citation, citation_index)
                        for brand in _all_brands(brand_file):
                            for mention in detect_mentions(answer.text, brand.aliases):
                                store_mention(connection, answer_id, brand.name, mention.alias)
                    if (
                        provider.mode == "grounded"
                        and query.relevance == "relevant"
                        and not boutiqaat_mentions
                        and answer.citations
                    ):
                        pages = fetch_citation_pages(
                            answer.citations, page_fetcher, boutiqaat.aliases
                        )
                        with connection:
                            for citation_index, page in enumerate(pages):
                                store_citation_page(connection, answer_id, citation_index, page)
                    if not boutiqaat_mentions:
                        continue
                    judge_query = judge_query_for_answer(answer)
                    allowed, live_calls = _call_within_budget(
                        provider, judge_query, 0, live_calls, call_budget
                    )
                    if not allowed:
                        aborted = True
                        break
                    verdict = judge_answer(answer, provider, boutiqaat.aliases)
                    with connection:
                        store_verdict(connection, answer_id, verdict)
                if aborted:
                    break
            if aborted:
                break
        with connection:
            set_run_status(connection, run_id, "aborted" if aborted else "completed")
    finally:
        connection.close()
    return RunResult(run_id, "aborted" if aborted else "completed", live_calls)


def execute_one_query(
    database_path: Path,
    query_set_path: Path,
    brand_path: Path,
    query_id: str,
    provider: Provider,
    run_id: str,
    run_at: str,
    page_fetcher: PageFetcher | None = None,
) -> str:
    result = execute_run(
        database_path,
        query_set_path,
        brand_path,
        provider,
        run_id,
        run_at,
        query_ids=[query_id],
        trials_per_query=1,
        page_fetcher=page_fetcher,
    )
    return result.run_id
