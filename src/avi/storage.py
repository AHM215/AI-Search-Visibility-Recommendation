from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from avi.citations import Citation, SourceType, classify_source_type
from avi.judge import RecommendationStrength, Verdict


@dataclass(frozen=True)
class StoredRun:
    id: str
    query_set_version: str
    run_at: str
    status: str


@dataclass(frozen=True)
class StoredAnswer:
    id: int
    query_id: str
    provider_mode: str
    trial_index: int
    model_identifier: str
    text: str
    mentioned: bool
    search_performed: bool
    citations: list[StoredCitation]
    verdict: Verdict | None


@dataclass(frozen=True)
class StoredCitation:
    url: str
    title: str
    source_type: SourceType


def open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            query_set_version TEXT NOT NULL,
            run_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'completed'
        );
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id),
            query_id TEXT NOT NULL,
            provider_mode TEXT NOT NULL,
            trial_index INTEGER NOT NULL,
            model_identifier TEXT NOT NULL,
            text TEXT NOT NULL,
            search_performed INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS mentions (
            answer_id INTEGER NOT NULL REFERENCES answers(id),
            brand_name TEXT NOT NULL,
            alias TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS citations (
            answer_id INTEGER NOT NULL REFERENCES answers(id),
            citation_index INTEGER NOT NULL,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            PRIMARY KEY (answer_id, citation_index)
        );
        CREATE TABLE IF NOT EXISTS verdicts (
            answer_id INTEGER PRIMARY KEY REFERENCES answers(id),
            recommendation_strength TEXT NOT NULL,
            rank INTEGER NOT NULL,
            brands TEXT NOT NULL,
            unlocated_brands TEXT NOT NULL DEFAULT '[]'
        );
        """
    )
    answer_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(answers)")}
    if "search_performed" not in answer_columns:
        connection.execute(
            "ALTER TABLE answers ADD COLUMN search_performed INTEGER NOT NULL DEFAULT 0"
        )
    run_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")}
    if "status" not in run_columns:
        connection.execute("ALTER TABLE runs ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'")
    verdict_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(verdicts)")}
    if "unlocated_brands" not in verdict_columns:
        connection.execute(
            "ALTER TABLE verdicts ADD COLUMN unlocated_brands TEXT NOT NULL DEFAULT '[]'"
        )
    return connection


def create_run(
    connection: sqlite3.Connection, run_id: str, query_set_version: str, run_at: str
) -> None:
    connection.execute(
        "INSERT INTO runs (id, query_set_version, run_at, status) VALUES (?, ?, ?, 'running')",
        (run_id, query_set_version, run_at),
    )


def set_run_status(connection: sqlite3.Connection, run_id: str, status: str) -> None:
    connection.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


def store_answer(
    connection: sqlite3.Connection,
    run_id: str,
    query_id: str,
    provider_mode: str,
    trial_index: int,
    model_identifier: str,
    answer_text: str,
    search_performed: bool,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO answers
            (run_id, query_id, provider_mode, trial_index, model_identifier, text, search_performed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            query_id,
            provider_mode,
            trial_index,
            model_identifier,
            answer_text,
            search_performed,
        ),
    )
    answer_id = cursor.lastrowid
    if answer_id is None:
        raise RuntimeError("SQLite did not return an Answer id")
    return answer_id


def store_citation(
    connection: sqlite3.Connection, answer_id: int, citation: Citation, citation_index: int
) -> None:
    connection.execute(
        """
        INSERT INTO citations (answer_id, citation_index, url, title, source_type)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            answer_id,
            citation_index,
            citation.url,
            citation.title,
            classify_source_type(citation.url),
        ),
    )


def store_mention(
    connection: sqlite3.Connection, answer_id: int, brand_name: str, alias: str
) -> None:
    connection.execute(
        "INSERT INTO mentions (answer_id, brand_name, alias) VALUES (?, ?, ?)",
        (answer_id, brand_name, alias),
    )


def store_verdict(connection: sqlite3.Connection, answer_id: int, verdict: Verdict) -> None:
    connection.execute(
        """
        INSERT INTO verdicts (answer_id, recommendation_strength, rank, brands, unlocated_brands)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            answer_id,
            verdict.recommendation_strength,
            verdict.rank,
            json.dumps(verdict.brands, ensure_ascii=False),
            json.dumps(verdict.unlocated_brands, ensure_ascii=False),
        ),
    )


def read_run(connection: sqlite3.Connection, run_id: str) -> StoredRun:
    row = connection.execute(
        "SELECT id, query_set_version, run_at, status FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Run {run_id!r} does not exist")
    return StoredRun(
        id=str(row[0]), query_set_version=str(row[1]), run_at=str(row[2]), status=str(row[3])
    )


def read_answers(connection: sqlite3.Connection, run_id: str) -> list[StoredAnswer]:
    rows = connection.execute(
        """
        SELECT a.id, a.query_id, a.provider_mode, a.trial_index, a.model_identifier, a.text,
               EXISTS(
                    SELECT 1 FROM mentions AS m
                    WHERE m.answer_id = a.id AND m.brand_name = 'Boutiqaat'
               ), a.search_performed, v.recommendation_strength, v.brands, v.unlocated_brands
        FROM answers AS a
        LEFT JOIN verdicts AS v ON v.answer_id = a.id
        WHERE a.run_id = ?
        ORDER BY a.id
        """,
        (run_id,),
    ).fetchall()
    answers: list[StoredAnswer] = []
    for row in rows:
        citation_rows = connection.execute(
            """
            SELECT url, title, source_type FROM citations
            WHERE answer_id = ? ORDER BY citation_index
            """,
            (row[0],),
        ).fetchall()
        citations = [
            StoredCitation(
                url=str(citation_row[0]),
                title=str(citation_row[1]),
                source_type=cast(SourceType, str(citation_row[2])),
            )
            for citation_row in citation_rows
        ]
        verdict = (
            Verdict(
                recommendation_strength=cast(RecommendationStrength, str(row[8])),
                brands=json.loads(str(row[9])),
                unlocated_brands=json.loads(str(row[10])),
            )
            if row[8] is not None
            else None
        )
        answers.append(
            StoredAnswer(
                id=int(row[0]),
                query_id=str(row[1]),
                provider_mode=str(row[2]),
                trial_index=int(row[3]),
                model_identifier=str(row[4]),
                text=str(row[5]),
                mentioned=bool(row[6]),
                search_performed=bool(row[7]),
                citations=citations,
                verdict=verdict,
            )
        )
    return answers
