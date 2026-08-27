"""Pass-1 news quality gates shared by the pipeline and hide script.

No published_at → not news. Hidden rows use source='hidden' (is_hidden equivalent
until the SQL migration is applied).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Tavily: Monday = week, Tue–Sun = "d" (never "month").
TAVILY_RANGE_WEEK = "week"
TAVILY_RANGE_48H = "d"
UPSERT_MAX_AGE_DAYS = 14
HIDDEN_SOURCE = "hidden"
FORASOFT_VISIBLE_CAP = 2

_ARXIV_ID_RE = re.compile(
    r"(?:export\.)?arxiv\.org/(?:abs|pdf|html|ftp/[^/]+/papers/[^/]+)/"
    r"(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?",
    re.IGNORECASE,
)

_HEADING_TITLES = {
    "introduction",
    "abstract",
    "introduction - vss",
    "introduction — vss",
    "introduction – vss",
}

_CFP_TITLE_RE = re.compile(
    r"(premier conference for innovators|call for papers|call for proposals|"
    r"call-proposals|submit a paper|submit your paper)",
    re.IGNORECASE,
)

_SEO_MILL_TITLE_RE = re.compile(
    r"\b(20\d{2}\s+)?((buyer'?s?\s+guide)|playbook|top\s+\d+)\b",
    re.IGNORECASE,
)

_SIZE_IN_PATH_RE = re.compile(
    r"(?:^|[^\d])(\d{2,3})x(\d{2,3})(?:[^\d]|$)",
    re.IGNORECASE,
)

_BAD_IMAGE_TOKENS = (
    "apple-touch-icon",
    "favicon",
    "msapplication",
    "mstile",
    "android-chrome",
    "safari-pinned-tab",
    "apple-icon",
    "site-icon",
)

_LANDING_FIRST_SEGMENTS = {
    "solutions",
    "products",
    "product",
    "platform",
    "cameras",
}

_ARTICLE_PATH_HINTS = (
    "/blog",
    "/news",
    "/article",
    "/press",
    "/insights",
    "/resources",
    "/post",
)


def hostname(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def extract_arxiv_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _ARXIV_ID_RE.search(url)
    return match.group(1) if match else None


def arxiv_abs_url(url: str | None) -> str | None:
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        return None
    return f"https://arxiv.org/abs/{arxiv_id}"


def canonicalize_article_url(url: str | None) -> str:
    """Strip www, trailing slash, utm_*; rewrite arXiv html/pdf to /abs/."""
    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""

    abs_url = arxiv_abs_url(raw)
    if abs_url:
        return abs_url

    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.rstrip("/")

    netloc = (parsed.netloc or "").lower()
    if "@" in netloc:
        netloc = netloc.split("@", 1)[-1]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = (parsed.path or "").rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    query = urlencode(query_pairs)
    scheme = "https" if parsed.scheme in {"http", "https", ""} else parsed.scheme
    return urlunparse((scheme or "https", netloc, path, "", query, ""))


def canonical_url_key(url: str | None) -> str:
    canonical = canonicalize_article_url(url)
    if not canonical:
        return ""
    arxiv_id = extract_arxiv_id(canonical)
    if arxiv_id:
        return f"arxiv:{arxiv_id.lower()}"
    return canonical.lower()


def tavily_time_range(now: datetime | None = None) -> str:
    """Monday → week; Tue–Sun → d. Never month."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    if current.weekday() == 0:
        return TAVILY_RANGE_WEEK
    return TAVILY_RANGE_48H


def _normalize_title(title: str) -> str:
    text = (title or "").strip().lower()
    text = text.replace("—", "-").replace("–", "-").replace("|", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def is_heading_title(title: str | None) -> bool:
    text = _normalize_title(title or "")
    if not text:
        return True
    if text in _HEADING_TITLES:
        return True
    if text.startswith("introduction -") or text.startswith("introduction |"):
        return True
    if text == "introduction" or text.startswith("introduction "):
        # "Introduction to the EDGE AI FOUNDATION Taxonomy" is a real page title;
        # still a docs-heading, not news.
        if text.startswith("introduction to ") or text.startswith("introduction —"):
            return True
        if text.startswith("introduction"):
            rest = text[len("introduction") :].strip(" -|:")
            if not rest or rest.startswith("vss") or rest.startswith("the "):
                return True
    if text == "abstract" or text.startswith("abstract -"):
        return True
    return False


def is_cfp_or_tagline(title: str | None, url: str | None = None) -> bool:
    text = title or ""
    href = (url or "").lower()
    if _CFP_TITLE_RE.search(text):
        return True
    if "call-proposals" in href or "call-for-papers" in href or "callforpapers" in href:
        return True
    return False


def is_docs_homepage(url: str | None, title: str | None = None) -> bool:
    href = (url or "").lower()
    host = hostname(url)
    path = urlparse(url or "").path.lower()
    if host == "docs.nvidia.com":
        return True
    if "/docs/" in href and is_heading_title(title or ""):
        return True
    if path.rstrip("/") in {"", "/latest", "/docs"} and "docs." in host:
        return True
    return False


def is_forasoft_url(url: str | None) -> bool:
    return hostname(url) == "forasoft.com"


def is_generic_seo_mill(title: str | None, url: str | None) -> bool:
    """Playbook / Buyer's Guide / Top N from unknown mills. Fora Soft is allowed."""
    if is_forasoft_url(url):
        return False
    return bool(_SEO_MILL_TITLE_RE.search(title or ""))


def is_product_landing(title: str | None, url: str | None) -> bool:
    host = hostname(url)
    path = (urlparse(url or "").path or "/").lower()
    title_l = (title or "").lower()

    if host == "metrolla.com":
        return True
    if host == "appther.com":
        return True
    if "pulsevi" in title_l or "pulsevi" in path:
        return True

    if any(hint in path for hint in _ARTICLE_PATH_HINTS):
        return False

    segments = [seg for seg in path.split("/") if seg]
    if segments and segments[0] in _LANDING_FIRST_SEGMENTS:
        return True
    if not segments or path in {"/", ""}:
        # Bare company homepage.
        if title and " | " in title:
            return True
    return False


def is_hashtag_title(title: str | None) -> bool:
    return (title or "").lstrip().startswith("#")


def reject_reason(
    title: str | None,
    url: str | None,
    published_at: str | None = None,
    *,
    require_published_at: bool = False,
) -> str | None:
    """Deterministic reject. Date-required is opt-in after HTML/arXiv enrichment."""
    if is_hashtag_title(title):
        return "hashtag title"
    if is_heading_title(title):
        return "heading title"
    if is_cfp_or_tagline(title, url):
        return "CFP/tagline"
    if is_docs_homepage(url, title):
        return "docs homepage"
    if is_product_landing(title, url):
        return "product landing page"
    if is_generic_seo_mill(title, url):
        return "SEO mill playbook/guide"
    if require_published_at and not (published_at or "").strip():
        return "no published_at"
    return None


def is_quality_article_image(url: str | None) -> bool:
    """og/twitter/schema images only after callers extract them; this rejects icons/logos."""
    if not url or not isinstance(url, str):
        return False
    candidate = url.strip()
    if not candidate or candidate.startswith(("data:", "javascript:")):
        return False
    lowered = candidate.lower()
    if any(token in lowered for token in _BAD_IMAGE_TOKENS):
        return False
    try:
        parsed = urlparse(candidate)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    path = parsed.path.lower()
    if path.endswith(".ico"):
        return False
    if "/logo/" in path or "/logos/" in path or path.rstrip("/").endswith("/logo"):
        return False
    if re.search(r"(^|/)logo[-_.]", path):
        return False
    match = _SIZE_IN_PATH_RE.search(path)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        if width <= 180 and height <= 180:
            return False
    return True


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def published_at_older_than(published_at: str | None, days: int) -> bool:
    parsed = parse_iso_datetime(published_at)
    if parsed is None:
        return True
    return datetime.now(timezone.utc) - parsed > timedelta(days=days)


def is_hidden_row(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    if row.get("is_hidden") is True:
        return True
    return (row.get("source") or "").strip().lower() == HIDDEN_SOURCE
