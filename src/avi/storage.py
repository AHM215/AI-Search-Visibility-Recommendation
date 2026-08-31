from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredRun:
    id: str
    query_set_version: str
    run_at: str


@dataclass(frozen=True)
class StoredAnswer:
    id: int
    query_id: str
    provider_mode: str
    trial_index: int
    model_identifier: str
    text: str
    mentioned: bool


def open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            query_set_version TEXT NOT NULL,
            run_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id),
            query_id TEXT NOT NULL,
            provider_mode TEXT NOT NULL,
            trial_index INTEGER NOT NULL,
            model_identifier TEXT NOT NULL,
            text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mentions (
            answer_id INTEGER NOT NULL REFERENCES answers(id),
            brand_name TEXT NOT NULL,
            alias TEXT NOT NULL
        );
        """
    )
    return connection


def create_run(
    connection: sqlite3.Connection, run_id: str, query_set_version: str, run_at: str
) -> None:
    connection.execute(
        "INSERT INTO runs (id, query_set_version, run_at) VALUES (?, ?, ?)",
        (run_id, query_set_version, run_at),
    )


def store_answer(
    connection: sqlite3.Connection,
    run_id: str,
    query_id: str,
    provider_mode: str,
    trial_index: int,
    model_identifier: str,
    answer_text: str,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO answers
            (run_id, query_id, provider_mode, trial_index, model_identifier, text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, query_id, provider_mode, trial_index, model_identifier, answer_text),
    )
    answer_id = cursor.lastrowid
    if answer_id is None:
        raise RuntimeError("SQLite did not return an Answer id")
    return answer_id


def store_mention(
    connection: sqlite3.Connection, answer_id: int, brand_name: str, alias: str
) -> None:
    connection.execute(
        "INSERT INTO mentions (answer_id, brand_name, alias) VALUES (?, ?, ?)",
        (answer_id, brand_name, alias),
    )


def read_run(connection: sqlite3.Connection, run_id: str) -> StoredRun:
    row = connection.execute(
        "SELECT id, query_set_version, run_at FROM runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Run {run_id!r} does not exist")
    return StoredRun(id=str(row[0]), query_set_version=str(row[1]), run_at=str(row[2]))


def read_answers(connection: sqlite3.Connection, run_id: str) -> list[StoredAnswer]:
    rows = connection.execute(
        """
        SELECT a.id, a.query_id, a.provider_mode, a.trial_index, a.model_identifier, a.text,
               EXISTS(
                   SELECT 1 FROM mentions AS m
                   WHERE m.answer_id = a.id AND m.brand_name = 'Boutiqaat'
               )
        FROM answers AS a
        WHERE a.run_id = ?
        ORDER BY a.id
        """,
        (run_id,),
    ).fetchall()
    return [
        StoredAnswer(
            id=int(row[0]),
            query_id=str(row[1]),
            provider_mode=str(row[2]),
            trial_index=int(row[3]),
            model_identifier=str(row[4]),
            text=str(row[5]),
            mentioned=bool(row[6]),
        )
        for row in rows
    ]
