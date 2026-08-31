from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from openai import OpenAI

if TYPE_CHECKING:
    from avi.ingest import Query


DEFAULT_MODEL_IDENTIFIER = "gpt-5.5-2026-04-23"
ProviderMode = Literal["ungrounded"]


class Provider(Protocol):
    mode: ProviderMode
    model_identifier: str

    def ask(self, query: Query, trial_index: int) -> str: ...


def configured_model_identifier() -> str:
    return os.environ.get("AVI_OPENAI_MODEL", DEFAULT_MODEL_IDENTIFIER)


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
            raise RuntimeError("OPENAI_API_KEY must be set to call OpenAI")
        self.model_identifier = model_identifier or configured_model_identifier()
        self._client = OpenAI(api_key=api_key)

    def ask(self, query: Query, trial_index: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model_identifier,
            messages=[{"role": "user", "content": query.text}],
        )
        answer_text = response.choices[0].message.content
        if answer_text is None:
            raise RuntimeError("OpenAI returned an empty Answer")
        return answer_text


class CachingProvider:
    def __init__(self, provider: Provider, cache_directory: Path) -> None:
        self._provider = provider
        self._cache_directory = cache_directory
        self.mode = provider.mode
        self.model_identifier = provider.model_identifier

    def ask(self, query: Query, trial_index: int) -> str:
        recording_path = _recording_path(
            self._cache_directory,
            query,
            self.mode,
            trial_index,
            self.model_identifier,
        )
        if recording_path.exists():
            recording = json.loads(recording_path.read_text(encoding="utf-8"))
            return str(recording["answer_text"])

        answer_text = self._provider.ask(query, trial_index)
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        recording = {
            "answer_text": answer_text,
            "model_identifier": self.model_identifier,
            "provider_mode": self.mode,
            "query_id": query.id,
            "query_text": query.text,
            "trial_index": trial_index,
        }
        recording_path.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return answer_text


class FixtureProvider:
    mode: ProviderMode = "ungrounded"

    def __init__(self, cache_directory: Path, model_identifier: str) -> None:
        self._cache_directory = cache_directory
        self.model_identifier = model_identifier

    def ask(self, query: Query, trial_index: int) -> str:
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
        return str(recording["answer_text"])
