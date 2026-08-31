from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import socket
from pathlib import Path
from typing import Literal, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from avi.detect import detect_mentions


SourceType = Literal[
    "retailer", "marketplace", "editorial/listicle", "review site", "other"
]
CitationPageStatus = Literal["present", "absent", "unfetched"]

PAGE_FETCH_TIMEOUT_SECONDS = 5.0
MAX_PAGES_PER_ANSWER = 8
PAGE_FETCH_CONCURRENCY = 3


@dataclass(frozen=True)
class Citation:
    url: str
    title: str


@dataclass(frozen=True)
class PageFetch:
    html: str | None
    unfetched_reason: str | None = None


@dataclass(frozen=True)
class CitationPage:
    status: CitationPageStatus
    unfetched_reason: str | None = None


class PageFetcher(Protocol):
    def fetch(self, url: str) -> PageFetch: ...


class HttpPageFetcher:
    def __init__(self, timeout_seconds: float = PAGE_FETCH_TIMEOUT_SECONDS) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Page fetch timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def fetch(self, url: str) -> PageFetch:
        request = Request(
            url,
            headers={
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "avi-citation-fetcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                status = response.getcode()
                if not 200 <= status < 300:
                    return PageFetch(None, f"HTTP status {status}")
                content_type = response.headers.get("Content-Type", "")
                if not _is_html_content_type(content_type):
                    return PageFetch(None, f"non-HTML content type: {content_type or 'missing'}")
                charset = _charset_from_content_type(content_type)
                return PageFetch(response.read().decode(charset, errors="replace"))
        except HTTPError as error:
            return PageFetch(None, f"HTTP status {error.code}")
        except (TimeoutError, socket.timeout):
            return PageFetch(None, "timeout")
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)) or "timed out" in str(
                error.reason
            ).casefold():
                return PageFetch(None, "timeout")
            return PageFetch(None, f"connection error: {error.reason}")
        except OSError as error:
            return PageFetch(None, f"connection error: {error}")
        except (LookupError, UnicodeError) as error:
            return PageFetch(None, f"text decoding error: {error}")


class CachingPageFetcher:
    def __init__(self, page_fetcher: PageFetcher, cache_directory: Path) -> None:
        self._page_fetcher = page_fetcher
        self._cache_directory = cache_directory

    def fetch(self, url: str) -> PageFetch:
        recording_path = self._recording_path(url)
        if recording_path.exists():
            return _page_fetch_from_recording(json.loads(recording_path.read_text(encoding="utf-8")))
        try:
            page_fetch = self._page_fetcher.fetch(url)
        except (TimeoutError, socket.timeout):
            page_fetch = PageFetch(None, "timeout")
        except Exception as error:
            page_fetch = PageFetch(None, f"fetch error: {error}")
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        recording = {
            "citation_url": url,
            "html": page_fetch.html,
            "outcome": "fetched" if page_fetch.html is not None else "unfetched",
            "recording_type": "citation_page",
            "unfetched_reason": page_fetch.unfetched_reason,
        }
        recording_path.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return page_fetch

    def _recording_path(self, url: str) -> Path:
        return _page_recording_path(self._cache_directory, url)


class FixturePageFetcher:
    def __init__(self, cache_directory: Path) -> None:
        self._cache_directory = cache_directory

    def fetch(self, url: str) -> PageFetch:
        recording_path = _page_recording_path(self._cache_directory, url)
        if not recording_path.exists():
            return PageFetch(None, "no fixture recording")
        return _page_fetch_from_recording(json.loads(recording_path.read_text(encoding="utf-8")))


def fetch_citation_pages(
    citations: Sequence[Citation],
    page_fetcher: PageFetcher | None,
    boutiqaat_aliases: list[str],
    *,
    maximum_pages: int = MAX_PAGES_PER_ANSWER,
    concurrency: int = PAGE_FETCH_CONCURRENCY,
) -> list[CitationPage]:
    if maximum_pages < 0:
        raise ValueError("Page fetch cap must not be negative")
    if concurrency <= 0:
        raise ValueError("Page fetch concurrency must be positive")
    selected_citations = list(citations[:maximum_pages])
    capped_citations = citations[maximum_pages:]
    if page_fetcher is None:
        return [CitationPage("unfetched", "no PageFetcher configured") for _ in citations]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(selected_citations) or 1)) as executor:
        selected_pages = list(
            executor.map(
                lambda citation: _fetch_citation_page(citation.url, page_fetcher, boutiqaat_aliases),
                selected_citations,
            )
        )
    return [
        *selected_pages,
        *(CitationPage("unfetched", "per-Answer page cap reached") for _ in capped_citations),
    ]


def _fetch_citation_page(
    url: str, page_fetcher: PageFetcher, boutiqaat_aliases: list[str]
) -> CitationPage:
    try:
        page_fetch = page_fetcher.fetch(url)
    except (TimeoutError, socket.timeout):
        return CitationPage("unfetched", "timeout")
    except Exception as error:
        return CitationPage("unfetched", f"fetch error: {error}")
    if page_fetch.html is None:
        return CitationPage("unfetched", page_fetch.unfetched_reason or "fetch returned no HTML")
    text = _extract_page_text(page_fetch.html)
    if not text:
        return CitationPage("unfetched", "no extractable text")
    status: CitationPageStatus = (
        "present" if detect_mentions(text, boutiqaat_aliases) else "absent"
    )
    return CitationPage(status)


def _page_recording_path(cache_directory: Path, url: str) -> Path:
    call = {"citation_url": url, "recording_type": "citation_page"}
    digest = hashlib.sha256(
        json.dumps(call, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_directory / f"page-{digest}.json"


def _page_fetch_from_recording(recording: object) -> PageFetch:
    if not isinstance(recording, dict) or recording.get("recording_type") != "citation_page":
        raise ValueError("Page fixture recording must be a citation_page JSON object")
    if recording.get("outcome") == "fetched":
        html = recording.get("html")
        if not isinstance(html, str):
            raise ValueError("Fetched page fixture recording must include HTML")
        return PageFetch(html)
    if recording.get("outcome") == "unfetched":
        reason = recording.get("unfetched_reason")
        return PageFetch(None, str(reason) if reason else "fixture recorded as unfetched")
    raise ValueError("Page fixture recording has an invalid outcome")


def _charset_from_content_type(content_type: str) -> str:
    for parameter in content_type.split(";")[1:]:
        key, separator, value = parameter.partition("=")
        if separator and key.strip().casefold() == "charset":
            return value.strip().strip('"') or "utf-8"
    return "utf-8"


def _is_html_content_type(content_type: str) -> bool:
    media_type = content_type.partition(";")[0].strip().casefold()
    return media_type in {"text/html", "application/xhtml+xml"}


class _PageTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_tag_depth = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "template"}:
            self._ignored_tag_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "template"} and self._ignored_tag_depth:
            self._ignored_tag_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_tag_depth:
            self.text.append(data)


def _extract_page_text(html: str) -> str:
    parser = _PageTextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(" ".join(parser.text).split())


RETAILER_DOMAINS = {
    "boutiqat.com",
    "boutiqaat.com",
    "boots.com",
    "sephora.com",
    "thefaceshop.com",
    "yesstyle.com",
    "stylekorean.com",
    "jolse.com",
    "oliveyoung.com",
    "iherb.com",
}
MARKETPLACE_DOMAINS = {"noon.com", "ubuy.com", "aliexpress.com", "ebay.com"}
EDITORIAL_LISTICLE_DOMAINS = {
    "allure.com",
    "byrdie.com",
    "cosmopolitan.com",
    "elle.com",
    "glamour.com",
    "goodhousekeeping.com",
    "harpersbazaar.com",
    "vogue.com",
}
REVIEW_SITE_DOMAINS = {"g2.com", "reviews.io", "sitejabber.com", "trustpilot.com"}


def classify_source_type(url: str) -> SourceType:
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    if host in RETAILER_DOMAINS:
        return "retailer"
    if host in MARKETPLACE_DOMAINS or host == "amazon.com" or host.startswith("amazon."):
        return "marketplace"
    if host in EDITORIAL_LISTICLE_DOMAINS:
        return "editorial/listicle"
    if host in REVIEW_SITE_DOMAINS:
        return "review site"
    return "other"
