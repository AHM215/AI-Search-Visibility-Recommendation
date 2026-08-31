from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from avi.detect import detect_mentions
from avi.providers import Provider
from avi.storage import create_run, open_database, store_answer, store_citation, store_mention


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


def execute_one_query(
    database_path: Path,
    query_set_path: Path,
    brand_path: Path,
    query_id: str,
    provider: Provider,
    run_id: str,
    run_at: str,
) -> str:
    query_set = load_query_set(query_set_path)
    query = select_query(query_set, query_id)
    boutiqaat = select_brand(load_brands(brand_path), "Boutiqaat")
    answer = provider.ask(query, 0)

    connection: sqlite3.Connection = open_database(database_path)
    try:
        with connection:
            create_run(connection, run_id, query_set.version, run_at)
            answer_id = store_answer(
                connection,
                run_id,
                query.id,
                provider.mode,
                0,
                provider.model_identifier,
                answer.text,
                answer.search_performed,
            )
            for citation_index, citation in enumerate(answer.citations):
                store_citation(connection, answer_id, citation, citation_index)
            citation_urls = "\n".join(citation.url for citation in answer.citations)
            for mention in detect_mentions(answer.text + "\n" + citation_urls, boutiqaat.aliases):
                store_mention(connection, answer_id, boutiqaat.name, mention.alias)
    finally:
        connection.close()
    return run_id
