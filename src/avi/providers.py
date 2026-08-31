from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from openai import OpenAI

from avi.citations import Citation

if TYPE_CHECKING:
    from avi.ingest import Query


DEFAULT_MODEL_IDENTIFIER = "openai/gpt-5.2-2025-12-11"
DEFAULT_OPENAI_BASE_URL = "https://lightning.ai/api/v1"
ProviderMode = Literal["ungrounded", "grounded"]


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    search_performed: bool


class Provider(Protocol):
    mode: ProviderMode
    model_identifier: str

    def ask(self, query: Query, trial_index: int) -> Answer: ...


def configured_model_identifier() -> str:
    return os.environ.get("AVI_OPENAI_MODEL", DEFAULT_MODEL_IDENTIFIER)


def configured_openai_base_url() -> str:
    return os.environ.get("AVI_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def _recording_path(
    cache_directory: Path,
    query: Query,
    mode: ProviderMode,
    trial_index: int,
    model_identifier: str,
) -> Path:
    call = {
        "model_identifier": model_identifier,
        "provider_mode": mode,
        "query_id": query.id,
        "query_text": query.text,
        "trial_index": trial_index,
    }
    digest = hashlib.sha256(
        json.dumps(call, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_directory / f"{digest}.json"


class UngroundedOpenAIProvider:
    mode: ProviderMode = "ungrounded"

    def __init__(self, model_identifier: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set to call the Provider")
        self.model_identifier = model_identifier or configured_model_identifier()
        self._client = OpenAI(api_key=api_key, base_url=configured_openai_base_url())

    def ask(self, query: Query, trial_index: int) -> Answer:
        response = self._client.chat.completions.create(
            model=self.model_identifier,
            messages=[{"role": "user", "content": query.text}],
        )
        answer_text = response.choices[0].message.content
        if answer_text is None:
            raise RuntimeError("Provider returned an empty Answer")
        return Answer(text=answer_text, citations=[], search_performed=False)


class GroundedOpenAIProvider:
    mode: ProviderMode = "grounded"

    def __init__(self, model_identifier: str | None = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY must be set to call the Provider")
        self.model_identifier = model_identifier or configured_model_identifier()
        self._client = OpenAI(api_key=api_key, base_url=configured_openai_base_url())

    def ask(self, query: Query, trial_index: int) -> Answer:
        response = self._client.responses.create(
            model=self.model_identifier,
            input=query.text,
            tools=[{"type": "web_search"}],
        )
        answer_parts: list[str] = []
        citations: list[Citation] = []
        search_performed = False
        for item in response.output:
            if item.type == "web_search_call":
                search_performed = True
            elif item.type == "message":
                for content in item.content:
                    if content.type != "output_text":
                        continue
                    answer_parts.append(content.text)
                    for annotation in content.annotations:
                        if annotation.type == "url_citation":
                            citations.append(Citation(url=annotation.url, title=annotation.title))
        answer_text = "\n".join(answer_parts)
        if not answer_text:
            raise RuntimeError("Provider returned an empty Answer")
        return Answer(
            text=answer_text,
            citations=citations,
            search_performed=search_performed,
        )


class CachingProvider:
    def __init__(self, provider: Provider, cache_directory: Path) -> None:
        self._provider = provider
        self._cache_directory = cache_directory
        self.mode = provider.mode
        self.model_identifier = provider.model_identifier

    def ask(self, query: Query, trial_index: int) -> Answer:
        recording_path = _recording_path(
            self._cache_directory,
            query,
            self.mode,
            trial_index,
            self.model_identifier,
        )
        if recording_path.exists():
            recording = json.loads(recording_path.read_text(encoding="utf-8"))
            return _answer_from_recording(recording)

        answer = self._provider.ask(query, trial_index)
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        recording = {
            "answer_text": answer.text,
            "citations": [
                {"title": citation.title, "url": citation.url} for citation in answer.citations
            ],
            "model_identifier": self.model_identifier,
            "provider_mode": self.mode,
            "query_id": query.id,
            "query_text": query.text,
            "search_performed": answer.search_performed,
            "trial_index": trial_index,
        }
        recording_path.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return answer


class FixtureProvider:
    mode: ProviderMode = "ungrounded"

    def __init__(
        self, cache_directory: Path, model_identifier: str, mode: ProviderMode = "ungrounded"
    ) -> None:
        self._cache_directory = cache_directory
        self.model_identifier = model_identifier
        self.mode = mode

    def ask(self, query: Query, trial_index: int) -> Answer:
        recording_path = _recording_path(
            self._cache_directory,
            query,
            self.mode,
            trial_index,
            self.model_identifier,
        )
        if not recording_path.exists():
            raise FileNotFoundError(f"No fixture recording for Query {query.id}")
        recording = json.loads(recording_path.read_text(encoding="utf-8"))
        return _answer_from_recording(recording)


def _answer_from_recording(recording: object) -> Answer:
    if not isinstance(recording, dict):
        raise ValueError("Fixture recording must be a JSON object")
    raw_citations = recording.get("citations", [])
    if not isinstance(raw_citations, list):
        raise ValueError("Fixture recording Citations must be a list")
    citations = [
        Citation(url=str(citation["url"]), title=str(citation["title"]))
        for citation in raw_citations
        if isinstance(citation, dict)
    ]
    return Answer(
        text=str(recording["answer_text"]),
        citations=citations,
        search_performed=bool(recording.get("search_performed", False)),
    )
