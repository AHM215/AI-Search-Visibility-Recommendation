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

    mentioned_ids = [str(answer.id) for answer in answers if answer.mentioned]
    if mentioned_ids:
        boutiqaat_statement = f"Boutiqaat was Mentioned. Answer ids: {', '.join(mentioned_ids)}."
    else:
        answer_ids = ", ".join(str(answer.id) for answer in answers)
        boutiqaat_statement = f"Boutiqaat was not Mentioned. Answer ids: {answer_ids}."

    lines = [
        "# Boutiqaat AI Search Visibility Report",
        "",
        f"Run: {stored_run.id}",
        f"Query Set version: {stored_run.query_set_version}",
        f"Run timestamp: {stored_run.run_at}",
        "",
        "Findings describe OpenAI's models, not AI search in general.",
        "",
        "## Boutiqaat",
        "",
        boutiqaat_statement,
        "",
        "## Answers",
        "",
    ]
    for answer in answers:
        lines.extend(
            [
                f"### Answer {answer.id}",
                "",
                f"Query: {answer.query_id}",
                f"Provider mode: {answer.provider_mode}",
                f"Trial: {answer.trial_index}",
                f"Model identifier: {answer.model_identifier}",
                "",
                answer.text,
                "",
            ]
        )
    return "\n".join(lines)
