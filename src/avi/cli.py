from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from avi.ingest import execute_one_query
from avi.providers import (
    CachingProvider,
    GroundedOpenAIProvider,
    Provider,
    ProviderMode,
    UngroundedOpenAIProvider,
)
from avi.report import render_report


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avi")
    commands = parser.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run")
    run_command.add_argument("query_id")
    run_command.add_argument("--database", type=Path, default=ROOT / "avi.db")
    run_command.add_argument("--query-set", type=Path, default=ROOT / "questions.v1.yaml")
    run_command.add_argument("--brands", type=Path, default=ROOT / "brands.yaml")
    run_command.add_argument("--cache", type=Path, default=ROOT / "cache")
    run_command.add_argument("--mode", choices=("ungrounded", "grounded"), default="ungrounded")

    report_command = commands.add_parser("report")
    report_command.add_argument("run_id")
    report_command.add_argument("--database", type=Path, default=ROOT / "avi.db")
    return parser


def main(
    arguments: Sequence[str] | None = None,
    *,
    provider: Provider | None = None,
    run_at: str | None = None,
    run_id: str | None = None,
) -> int:
    arguments_namespace = _parser().parse_args(arguments)
    if arguments_namespace.command == "run":
        active_provider = provider
        if active_provider is None:
            mode: ProviderMode = arguments_namespace.mode
            inner_provider = (
                GroundedOpenAIProvider() if mode == "grounded" else UngroundedOpenAIProvider()
            )
            active_provider = CachingProvider(inner_provider, arguments_namespace.cache)
        created_run_id = execute_one_query(
            arguments_namespace.database,
            arguments_namespace.query_set,
            arguments_namespace.brands,
            arguments_namespace.query_id,
            active_provider,
            run_id or str(uuid4()),
            run_at or datetime.now(timezone.utc).isoformat(),
        )
        print(created_run_id)
        return 0

    print(render_report(arguments_namespace.database, arguments_namespace.run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
