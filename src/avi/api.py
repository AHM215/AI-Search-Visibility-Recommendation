"""Read-only HTTP surface over stored Runs.

Runs are minutes of paid calls and are started from the CLI (ADR-0003); this layer only reads what
is already stored. The single exception is one ad-hoc Query, which passes the Relevance gate before
anything is scored, so an off-topic question never produces a meaningless verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from avi.detect import detect_mentions
from avi.diagnose import diagnose
from avi.ingest import Query, load_brands, load_query_set, select_brand
from avi.metrics import compute_metrics
from avi.providers import Provider
from avi.storage import open_database, read_answers, read_run


ROOT = Path(__file__).resolve().parents[2]
RELEVANCE_INSTRUCTIONS = (
    "Boutiqaat is a Kuwait-based online retailer selling beauty, skincare, fragrance and "
    "cosmetics, shipping within the GCC. Could Boutiqaat legitimately be a good answer to the "
    "shopper question below? Reply with exactly one word: relevant or irrelevant.\n\nQuestion: "
)


class AdhocRequest(BaseModel):
    text: str


class AdhocResponse(BaseModel):
    text: str | None
    relevance: Literal["relevant", "irrelevant"]
    scored: bool
    mentioned: bool | None


def build_app(
    database_path: Path,
    *,
    provider: Provider | None = None,
    query_set_path: Path = ROOT / "questions.v1.yaml",
    brand_path: Path = ROOT / "brands.yaml",
) -> FastAPI:
    app = FastAPI(title="Boutiqaat AI Search Visibility")

    def _answers(run_id: str) -> Any:
        connection = open_database(database_path)
        try:
            read_run(connection, run_id)
            return read_answers(connection, run_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        finally:
            connection.close()

    @app.get("/runs")
    def list_runs() -> list[dict[str, str]]:
        connection = open_database(database_path)
        try:
            rows = connection.execute(
                "SELECT id, query_set_version, run_at, status FROM runs ORDER BY run_at DESC"
            ).fetchall()
        finally:
            connection.close()
        return [
            {"id": row[0], "query_set_version": row[1], "run_at": row[2], "status": row[3]}
            for row in rows
        ]

    @app.get("/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> dict[str, Any]:
        answers = _answers(run_id)
        query_set = load_query_set(query_set_path)
        brand_file = load_brands(brand_path)
        metrics = compute_metrics(answers, query_set.queries, brand_file.seed_competitors)
        rate = metrics.visibility_rate
        share = metrics.share_of_voice
        return {
            "run_id": run_id,
            "visibility_rate": {
                "mentioned": len(rate.mentioned_answer_ids),
                "relevant_trials": len(rate.relevant_answer_ids),
                "value": rate.value,
                "mentioned_answer_ids": list(rate.mentioned_answer_ids),
            },
            "share_of_voice": {
                "boutiqaat_mentions": len(share.boutiqaat_mentions),
                "seed_mentions": len(share.seed_mentions),
                "value": share.value,
            },
            "consistency": [
                {
                    "query_id": item.query_id,
                    "provider_mode": item.provider_mode,
                    "bucket": item.bucket,
                    "answer_ids": list(item.answer_ids),
                }
                for item in metrics.consistency
            ],
            "recommendation_strength": {
                strength: len(answer_ids)
                for strength, answer_ids in
                metrics.recommendation_strength.answer_ids_by_strength.items()
            },
        }

    @app.get("/runs/{run_id}/slices")
    def run_slices(run_id: str) -> dict[str, Any]:
        """Visibility Rate broken down by Locale, Intent and Provider mode."""
        answers = _answers(run_id)
        query_set = load_query_set(query_set_path)
        brand_file = load_brands(brand_path)
        query_by_id = {query.id: query for query in query_set.queries}
        dimensions = {
            "locale": sorted({query_by_id[a.query_id].locale for a in answers}),
            "intent": sorted({query_by_id[a.query_id].intent for a in answers}),
            "provider_mode": sorted({a.provider_mode for a in answers}),
        }
        result: dict[str, Any] = {}
        for keyword, values in dimensions.items():
            rows = []
            for value in values:
                sliced = compute_metrics(
                    answers, query_set.queries, brand_file.seed_competitors, **{keyword: value}
                )
                rate = sliced.visibility_rate
                rows.append(
                    {
                        "value": value,
                        "mentioned": len(rate.mentioned_answer_ids),
                        "relevant_trials": len(rate.relevant_answer_ids),
                        "visibility_rate": rate.value,
                    }
                )
            result[keyword] = rows
        return result

    @app.get("/runs/{run_id}/diagnosis")
    def run_diagnosis(run_id: str) -> list[dict[str, Any]]:
        """Evidence-backed Findings. A cause with no supporting Evidence is never returned."""
        answers = _answers(run_id)
        query_set = load_query_set(query_set_path)
        brand_file = load_brands(brand_path)
        return [
            {
                "cause": finding.cause,
                "statement": finding.statement,
                "remedy": finding.remedy,
                "answer_ids": list(finding.answer_ids),
                "citation_urls": list(finding.citation_urls),
                "fetched_page_count": finding.fetched_page_count,
                "unfetched_page_count": finding.unfetched_page_count,
            }
            for finding in diagnose(answers, query_set.queries, brand_file.seed_competitors)
        ]

    @app.get("/runs/{run_id}/queries/{query_id}")
    def run_query(run_id: str, query_id: str) -> dict[str, Any]:
        answers = [answer for answer in _answers(run_id) if answer.query_id == query_id]
        if not answers:
            raise HTTPException(status_code=404, detail=f"Query {query_id!r} is not in this Run")
        return {
            "run_id": run_id,
            "query_id": query_id,
            "trials": [
                {
                    "answer_id": answer.id,
                    "provider_mode": answer.provider_mode,
                    "trial_index": answer.trial_index,
                    "model_identifier": answer.model_identifier,
                    "text": answer.text,
                    "mentioned": answer.mentioned,
                    "search_performed": answer.search_performed,
                    "citations": [
                        {
                            "url": citation.url,
                            "title": citation.title,
                            "source_type": citation.source_type,
                            "page_status": None
                            if citation.page is None
                            else citation.page.status,
                        }
                        for citation in answer.citations
                    ],
                    "recommendation_strength": None
                    if answer.verdict is None
                    else answer.verdict.recommendation_strength,
                }
                for answer in answers
            ],
        }

    @app.post("/adhoc")
    def adhoc(request: AdhocRequest) -> AdhocResponse:
        if provider is None:
            raise HTTPException(status_code=503, detail="No Provider is configured")
        gate = Query(
            id="adhoc-relevance",
            text=RELEVANCE_INSTRUCTIONS + request.text,
            intent="relevance_gate",
            locale="global_en",
            specificity="narrow",
            relevance="relevant",
        )
        verdict = provider.ask(gate, 0).text.strip().casefold()
        if not verdict.startswith("relevant"):
            return AdhocResponse(text=None, relevance="irrelevant", scored=False, mentioned=None)
        query = Query(
            id="adhoc",
            text=request.text,
            intent="adhoc",
            locale="global_en",
            specificity="narrow",
            relevance="relevant",
        )
        answer = provider.ask(query, 0)
        boutiqaat = select_brand(load_brands(brand_path), "Boutiqaat")
        mentioned = bool(detect_mentions(answer.text, boutiqaat.aliases))
        return AdhocResponse(
            text=answer.text, relevance="relevant", scored=True, mentioned=mentioned
        )

    return app
