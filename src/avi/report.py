from __future__ import annotations

from pathlib import Path

from avi.storage import open_database, read_answers, read_run


def render_report(database_path: Path, run_id: str) -> str:
    connection = open_database(database_path)
    try:
        stored_run = read_run(connection, run_id)
        answers = read_answers(connection, run_id)
    finally:
        connection.close()

    lines = [
        "# Boutiqaat AI Search Visibility Report",
        "",
        f"Run: {stored_run.id}",
        f"Query Set version: {stored_run.query_set_version}",
        f"Run timestamp: {stored_run.run_at}",
        "",
        "Findings describe OpenAI's models, not AI search in general.",
        "",
        "## Answers",
        "",
    ]
    for mode in ("ungrounded", "grounded"):
        mode_answers = [answer for answer in answers if answer.provider_mode == mode]
        if not mode_answers:
            continue
        mentioned_ids = [str(answer.id) for answer in mode_answers if answer.mentioned]
        answer_ids = ", ".join(str(answer.id) for answer in mode_answers)
        boutiqaat_statement = (
            f"Boutiqaat was Mentioned. Answer ids: {', '.join(mentioned_ids)}."
            if mentioned_ids
            else f"Boutiqaat was not Mentioned. Answer ids: {answer_ids}."
        )
        lines.extend([f"## {mode.title()} Provider", "", boutiqaat_statement, ""])
        for answer in mode_answers:
            lines.extend(
                [
                    f"### Answer {answer.id}",
                    "",
                    f"Query: {answer.query_id}",
                    f"Provider mode: {answer.provider_mode}",
                    f"Trial: {answer.trial_index}",
                    f"Model identifier: {answer.model_identifier}",
                    f"Search performed: {'yes' if answer.search_performed else 'no'}",
                    "",
                    answer.text,
                    "",
                ]
            )
            if answer.citations:
                lines.extend(["#### Citations", ""])
                lines.extend(
                    f"{index}. [{citation.title}]({citation.url}) "
                    f"(Source Type: {citation.source_type})"
                    for index, citation in enumerate(answer.citations, start=1)
                )
                lines.append("")
    return "\n".join(lines)
