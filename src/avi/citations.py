from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit


SourceType = Literal[
    "retailer", "marketplace", "editorial/listicle", "review site", "other"
]


@dataclass(frozen=True)
class Citation:
    url: str
    title: str


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
