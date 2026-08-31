from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from avi.ingest import DEFAULT_CALL_BUDGET, execute_run, plan_run
from avi.providers import (
    CachingProvider,
    FixtureProvider,
    GroundedOpenAIProvider,
    Provider,
    ProviderMode,
    UngroundedOpenAIProvider,
    configured_model_identifier,
)
from avi.report import render_report


ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avi")
    commands = parser.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run")
    run_command.add_argument("query_id", nargs="?")
    run_command.add_argument("--database", type=Path, default=ROOT / "avi.db")
    run_command.add_argument("--query-set", type=Path, default=ROOT / "questions.v1.yaml")
    run_command.add_argument("--brands", type=Path, default=ROOT / "brands.yaml")
    run_command.add_argument("--cache", type=Path, default=ROOT / "cache")
    run_command.add_argument("--mode", choices=("ungrounded", "grounded"))
    run_command.add_argument("--dry-run", action="store_true")
    run_command.add_argument("--call-budget", type=int, default=DEFAULT_CALL_BUDGET)

    report_command = commands.add_parser("report")
    report_command.add_argument("run_id")
    report_command.add_argument("--database", type=Path, default=ROOT / "avi.db")
    report_command.add_argument("--query-set", type=Path, default=ROOT / "questions.v1.yaml")
    report_command.add_argument("--brands", type=Path, default=ROOT / "brands.yaml")
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
        if provider is not None:
            active_providers = [provider]
        else:
            modes: tuple[ProviderMode, ...] = (
                (arguments_namespace.mode,)
                if arguments_namespace.mode is not None
                else ("ungrounded", "grounded")
            )
            active_providers = [
                (
                    FixtureProvider(arguments_namespace.cache, configured_model_identifier(), mode)
                    if arguments_namespace.dry_run
                    else CachingProvider(
                        GroundedOpenAIProvider() if mode == "grounded" else UngroundedOpenAIProvider(),
                        arguments_namespace.cache,
                    )
                )
                for mode in modes
            ]
        query_ids = [arguments_namespace.query_id] if arguments_namespace.query_id else None
        if arguments_namespace.dry_run:
            plan = plan_run(
                arguments_namespace.query_set,
                arguments_namespace.brands,
                active_providers,
                query_ids,
            )
            print(
                f"Dry run: {plan.cached_calls} cached calls, up to {plan.projected_live_calls} live calls "
                f"({plan.live_answer_calls} Answer calls, {plan.live_judge_calls} known Judge calls, "
                f"{plan.potential_judge_calls} possible Judge calls); "
                f"estimated cost ${plan.estimated_cost_usd:.2f}"
            )
            return 0
        result = execute_run(
            arguments_namespace.database,
            arguments_namespace.query_set,
            arguments_namespace.brands,
            active_providers,
            run_id or str(uuid4()),
            run_at or datetime.now(timezone.utc).isoformat(),
            query_ids=query_ids,
            call_budget=arguments_namespace.call_budget,
        )
        print(result.run_id)
        return 1 if result.status == "aborted" else 0

    print(
        render_report(
            arguments_namespace.database,
            arguments_namespace.run_id,
            arguments_namespace.query_set,
            arguments_namespace.brands,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
