
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, NotRequired, TypedDict
from urllib.parse import urljoin, urlparse

import arxiv
import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import Client, create_client
from tavily import TavilyClient

from news_quality import (
    HIDDEN_SOURCE,
    UPSERT_MAX_AGE_DAYS,
    arxiv_abs_url,
    canonicalize_article_url,
    canonical_url_key,
    extract_arxiv_id,
    is_heading_title,
    is_hidden_row,
    is_quality_article_image,
    published_at_older_than,
    reject_reason,
    tavily_time_range,
)
from prompts import IVS_JUDGE_PROMPT, SYSTEM_PROMPT

load_dotenv(override=True)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
openai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if supabase_url and supabase_key:
    supabase: Client | None = create_client(supabase_url, supabase_key)
    supabase.postgrest.session.headers.update(
        {"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"}
    )
    print("✅ Supabase client initialized with service_role key")
else:
    supabase = None
    print("⚠️ Supabase credentials missing in .env")

DEFAULT_QUERIES = [
    "intelligent video surveillance AI 2026 OR edge AI",
    "AI video analytics customer case study deployment OR implementation 2025 OR 2026",
    "new computer vision techniques edge AI surveillance OR anomaly detection OR tracking",
    "video analytics marketplace news product launch OR ISC West 2026 OR Embedded Vision Summit",
    "intelligent video surveillance trends technical edge AI OR spatial intelligence OR vision language models",
    "LiDAR OR event cameras OR sparse cameras video surveillance OR security",
    "Nvidia OR Qualcomm OR Axis OR Hanwha OR Nutanix OR Cisco video surveillance AI OR edge",
    "managed service provider MSP OR VSaaS OR PhySec InfoSec convergence video surveillance",
    "VSaaS OR video surveillance as a service MSP managed platform OR multi-tenant",
    "IoT sensor fusion OR physical security convergence OR enterprise application integration video analytics",
]

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_IMAGE_CACHE_TTL_DAYS = 30
_IMAGE_CACHE_PATH = os.getenv("IMAGE_CACHE_PATH", "image_cache.json")
_IMAGE_REQUEST_TIMEOUT_SECONDS = 12
_IMAGE_REQUEST_RETRIES = 4
_TAVILY_MAX_RESULTS = 12
_RSS_ENTRIES_PER_FEED = 12
_SEARCH_RESULTS_CAP = 65
_PAGE_FETCH_CACHE: Dict[str, Dict[str, Any]] = {}
_PAGE_FETCH_LOCK = threading.RLock()
_image_cache_lock = threading.RLock()
_image_stats_lock = threading.Lock()
_image_runtime_stats: Dict[str, int] = {
    "cache_hits": 0,
    "http_requests": 0,
    "request_errors": 0,
    "rate_limit_errors": 0,
}


def _reset_image_runtime_stats() -> None:
    with _image_stats_lock:
        for key in _image_runtime_stats:
            _image_runtime_stats[key] = 0


def _increment_image_stat(name: str, amount: int = 1) -> None:
    with _image_stats_lock:
        _image_runtime_stats[name] = _image_runtime_stats.get(name, 0) + amount


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ["429", "rate limit", "too many requests"])


def _load_image_cache() -> Dict[str, Dict[str, Any]]:
    try:
        with open(_IMAGE_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_image_cache: Dict[str, Dict[str, Any]] = _load_image_cache()


def _persist_image_cache() -> None:
    with _image_cache_lock:
        try:
            with open(_IMAGE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(_image_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def _get_cached_image(article_url: str | None) -> str | None | object:
    normalized_url = normalize_article_url_for_image(article_url)
    if not normalized_url:
        return None

    key = canonical_url_key(normalized_url)
    if not key:
        return None

    with _image_cache_lock:
        entry = _image_cache.get(key)

    if not entry:
        return _CACHE_MISS

    fetched_at_raw = entry.get("fetched_at")
    if not fetched_at_raw:
        return _CACHE_MISS

    try:
        fetched_at = datetime.fromisoformat(fetched_at_raw)
    except Exception:
        return _CACHE_MISS

    if datetime.utcnow() - fetched_at > timedelta(days=_IMAGE_CACHE_TTL_DAYS):
        return _CACHE_MISS

    cached_image = entry.get("image")
    return cached_image if isinstance(cached_image, str) and cached_image else None


def _set_cached_image(article_url: str | None, image_url: str | None) -> None:
    normalized_url = normalize_article_url_for_image(article_url)
    if not normalized_url:
        return

    key = canonical_url_key(normalized_url)
    if not key:
        return

    with _image_cache_lock:
        _image_cache[key] = {
            "url": normalized_url,
            "image": image_url,
            "fetched_at": datetime.utcnow().isoformat(),
        }
    _persist_image_cache()


_CACHE_MISS = object()


# canonical_url_key imported from news_quality (www strip, utm strip, arXiv /abs/).


def dedupe_items_by_url(items: List[Dict]) -> List[Dict]:
    seen: set[str] = set()
    out: List[Dict] = []
    for item in items:
        key = canonical_url_key(item.get("url"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_title_key(title: str | None) -> str:
    if not title or not isinstance(title, str):
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def dedupe_items_by_title(items: List[Dict]) -> List[Dict]:
    """Drop near-duplicate headlines after URL deduplication."""
    seen: set[str] = set()
    out: List[Dict] = []
    for item in items:
        key = _normalize_title_key(item.get("title"))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _parse_datetime_string(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None

    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return _to_utc_iso(dt)
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(raw)
        return _to_utc_iso(dt)
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d %b %Y",
        "%B %d, %Y",
    ):
        try:
            dt = datetime.strptime(raw[:19] if "T" in fmt else raw[:10], fmt)
            return _to_utc_iso(dt.replace(tzinfo=timezone.utc))
        except Exception:
            continue

    return None


def _struct_time_to_iso(st: Any) -> str | None:
    if not st:
        return None
    try:
        dt = datetime(*st[:6], tzinfo=timezone.utc)
        return _to_utc_iso(dt)
    except Exception:
        return None


def _parse_feed_entry_date(entry: Any) -> str | None:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, attr, None)
        iso = _struct_time_to_iso(parsed)
        if iso:
            return iso

    for attr in ("published", "updated", "created"):
        iso = _parse_datetime_string(entry.get(attr))
        if iso:
            return iso

    return None


def _parse_feed_entry_image(entry: Any) -> str | None:
    thumbs = entry.get("media_thumbnail") or entry.get("media_thumbnails")
    if isinstance(thumbs, list) and thumbs:
        url = thumbs[0].get("url") if isinstance(thumbs[0], dict) else None
        if url:
            return url.strip()

    media = entry.get("media_content") or entry.get("enclosures")
    if isinstance(media, list):
        for item in media:
            if not isinstance(item, dict):
                continue
            media_type = (item.get("type") or item.get("medium") or "").lower()
            url = item.get("url") or item.get("href")
            if url and ("image" in media_type or not media_type):
                return url.strip()

    image_block = entry.get("image")
    if isinstance(image_block, dict):
        href = image_block.get("href") or image_block.get("url")
        if isinstance(href, str) and href.strip():
            return href.strip()

    return None


def _extract_schema_date(value: Any) -> str | None:
    if isinstance(value, str):
        return _parse_datetime_string(value)
    if isinstance(value, list):
        for item in value:
            parsed = _extract_schema_date(item)
            if parsed:
                return parsed
    if isinstance(value, dict):
        for key in ("datePublished", "dateModified", "uploadDate", "dateCreated"):
            if key in value:
                parsed = _extract_schema_date(value[key])
                if parsed:
                    return parsed
    return None


def _find_schema_published(soup: BeautifulSoup) -> str | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue

        nodes = parsed if isinstance(parsed, list) else [parsed]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            for key in ("datePublished", "dateModified", "uploadDate"):
                candidate = _extract_schema_date(node.get(key))
                if candidate:
                    return candidate
            graph = node.get("@graph")
            if isinstance(graph, list):
                for graph_node in graph:
                    if isinstance(graph_node, dict):
                        nested = _find_schema_published_from_node(graph_node)
                        if nested:
                            return nested
    return None


def _find_schema_published_from_node(node: Dict[str, Any]) -> str | None:
    for key in ("datePublished", "dateModified", "uploadDate"):
        candidate = _extract_schema_date(node.get(key))
        if candidate:
            return candidate
    return None


def _find_html_published(soup: BeautifulSoup) -> str | None:
    meta_specs = [
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"property": "og:published_time"}, "content"),
        ("meta", {"name": "article:published_time"}, "content"),
        ("meta", {"name": "pubdate"}, "content"),
        ("meta", {"name": "publish-date"}, "content"),
        ("meta", {"name": "date"}, "content"),
        ("meta", {"itemprop": "datePublished"}, "content"),
        ("meta", {"property": "article:modified_time"}, "content"),
        ("time", {"pubdate": True}, "datetime"),
        ("time", {"itemprop": "datePublished"}, "datetime"),
    ]
    for tag_name, attrs, attr_key in meta_specs:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get(attr_key):
            parsed = _parse_datetime_string(tag[attr_key])
            if parsed:
                return parsed

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag and time_tag.get("datetime"):
        parsed = _parse_datetime_string(time_tag["datetime"])
        if parsed:
            return parsed

    return _find_schema_published(soup)


def _fetch_article_page(normalized_url: str) -> BeautifulSoup | None:
    with _PAGE_FETCH_LOCK:
        cached = _PAGE_FETCH_CACHE.get(normalized_url)
        if cached and cached.get("soup") is not None:
            return cached["soup"]

    last_error: Exception | None = None
    for attempt in range(1, _IMAGE_REQUEST_RETRIES + 1):
        try:
            _increment_image_stat("http_requests")
            headers = {
                **_DEFAULT_HEADERS,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            if attempt > 1:
                headers["Cache-Control"] = "no-cache"

            response = requests.get(
                normalized_url,
                headers=headers,
                timeout=_IMAGE_REQUEST_TIMEOUT_SECONDS,
                allow_redirects=True,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            with _PAGE_FETCH_LOCK:
                _PAGE_FETCH_CACHE[normalized_url] = {
                    "soup": soup,
                    "final_url": response.url,
                }
            return soup
        except Exception as e:
            last_error = e
            _increment_image_stat("request_errors")
            if _is_rate_limit_error(e):
                _increment_image_stat("rate_limit_errors")
            if attempt < _IMAGE_REQUEST_RETRIES:
                time.sleep(attempt * 0.75)

    with _PAGE_FETCH_LOCK:
        _PAGE_FETCH_CACHE[normalized_url] = {"soup": None, "error": str(last_error)}
    return None


def extract_published_at(
    article_url: str | None, article: Dict | None = None
) -> str | None:
    article = article or {}
    existing = article.get("published_at")
    if isinstance(existing, str) and existing.strip():
        return _parse_datetime_string(existing) or existing.strip()

    normalized_url = normalize_article_url_for_image(article_url)
    if normalized_url:
        soup = _fetch_article_page(normalized_url)
        if soup:
            html_date = _find_html_published(soup)
            if html_date:
                return html_date

    return None


def normalize_article_url_for_image(url: str | None) -> str | None:
    """Convert article URL to an HTML page URL suitable for meta-tag scraping."""
    if not url:
        return None

    url = url.strip()
    abs_url = arxiv_abs_url(url)
    if abs_url:
        return abs_url
    return url


def _extract_schema_image(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, list):
        for item in value:
            candidate = _extract_schema_image(item)
            if candidate:
                return candidate

    if isinstance(value, dict):
        if isinstance(value.get("url"), str) and value["url"].strip():
            return value["url"].strip()
        if isinstance(value.get("contentUrl"), str) and value["contentUrl"].strip():
            return value["contentUrl"].strip()

    return None


def _find_schema_image(soup: BeautifulSoup) -> str | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue

        try:
            parsed = json.loads(raw)
        except Exception:
            continue

        nodes = parsed if isinstance(parsed, list) else [parsed]
        for node in nodes:
            if isinstance(node, dict):
                candidate = _extract_schema_image(node.get("image"))
                if candidate:
                    return candidate
                graph = node.get("@graph")
                if isinstance(graph, list):
                    for graph_node in graph:
                        if isinstance(graph_node, dict):
                            nested = _extract_schema_image(graph_node.get("image"))
                            if nested:
                                return nested

    return None


def _normalize_image_candidate(
    base_url: str,
    candidate: str | None,
    *,
    strict_logo_filter: bool = True,
) -> str | None:
    if not candidate:
        return None

    candidate = candidate.strip()
    if not candidate or candidate.startswith(("data:", "javascript:")):
        return None

    resolved = urljoin(base_url, candidate)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"}:
        return None

    if not is_quality_article_image(resolved):
        return None
    return resolved


def _candidate_score(tag) -> int:
    score = 0
    try:
        width = int(float(tag.get("width", 0) or 0))
        height = int(float(tag.get("height", 0) or 0))
        score += width * height
        if width >= 320:
            score += 20000
        if height >= 180:
            score += 15000
    except Exception:
        pass

    classes = " ".join(tag.get("class", [])) if isinstance(tag.get("class"), list) else str(tag.get("class", ""))
    attrs_text = f"{classes} {tag.get('alt', '')} {tag.get('src', '')}".lower()
    if any(token in attrs_text for token in ["hero", "featured", "lead", "main"]):
        score += 30000
    if any(token in attrs_text for token in ["logo", "icon", "avatar", "sprite"]):
        score -= 50000
    return score


def _find_best_img_tag(soup: BeautifulSoup, base_url: str) -> str | None:
    candidates: List[tuple[int, str]] = []
    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-original")
            or img.get("data-lazy-src")
        )
        candidate = _normalize_image_candidate(base_url, src)
        if candidate:
            candidates.append((_candidate_score(img), candidate))

        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            for item in srcset.split(","):
                part = item.strip().split(" ")[0]
                normalized = _normalize_image_candidate(base_url, part)
                if normalized:
                    candidates.append((_candidate_score(img) + 5000, normalized))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _extract_image_from_soup(soup: BeautifulSoup, base_url: str) -> str | None:
    """og:image, twitter:image, and schema.org article image only. Never icons/logos."""
    selectors = [
        ("meta", {"property": "og:image:secure_url"}, "content"),
        ("meta", {"property": "og:image"}, "content"),
        ("meta", {"name": "og:image"}, "content"),
        ("meta", {"property": "og:image:url"}, "content"),
        ("meta", {"name": "twitter:image"}, "content"),
        ("meta", {"property": "twitter:image"}, "content"),
        ("meta", {"name": "twitter:image:src"}, "content"),
    ]

    for tag_name, attrs, attr_key in selectors:
        tag = soup.find(tag_name, attrs=attrs)
        if tag and tag.get(attr_key):
            image_url = _normalize_image_candidate(
                base_url,
                tag[attr_key],
                strict_logo_filter=True,
            )
            if image_url:
                return image_url

    schema_image = _find_schema_image(soup)
    image_url = _normalize_image_candidate(
        base_url, schema_image, strict_logo_filter=True
    )
    if image_url:
        return image_url

    return None


def extract_image_url(
    article_url: str | None, article: Dict | None = None
) -> str | None:
    """Fetch article thumbnails with retries, fallbacks, and caching."""
    article = article or {}
    existing = article.get("image")
    if isinstance(existing, str) and is_quality_article_image(existing):
        return existing.strip()

    normalized_url = normalize_article_url_for_image(article_url)
    if not normalized_url:
        return None

    cached = _get_cached_image(normalized_url)
    if cached is not _CACHE_MISS:
        _increment_image_stat("cache_hits")
        if cached is None or is_quality_article_image(cached):
            return cached
        cached = None

    image_url: str | None = None
    soup = _fetch_article_page(normalized_url)
    if soup:
        image_url = _extract_image_from_soup(soup, normalized_url)
    if image_url and not is_quality_article_image(image_url):
        image_url = None

    _set_cached_image(normalized_url, image_url)
    return image_url


def _enrich_article_metadata(article: Dict) -> Dict:
    canonical = canonicalize_article_url(article.get("url"))
    if canonical:
        article["url"] = canonical
    if not article.get("published_at"):
        article["published_at"] = extract_published_at(article.get("url"), article)
    if not is_quality_article_image(article.get("image")):
        article["image"] = extract_image_url(article.get("url"), article)
    elif article.get("image") and not is_quality_article_image(article.get("image")):
        article["image"] = None
    return article


def populate_article_metadata(articles: List[Dict]) -> None:
    if not articles:
        return

    _PAGE_FETCH_CACHE.clear()
    with ThreadPoolExecutor(max_workers=min(8, len(articles))) as executor:
        futures = [
            executor.submit(_enrich_article_metadata, article) for article in articles
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                continue
    _PAGE_FETCH_CACHE.clear()


def _score_value(judgment: Dict[str, Any], key: str) -> float:
    try:
        return float(judgment.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def article_passes_judge(judgment: Dict[str, Any]) -> bool:
    """Apply keep rules with a softer bar for Market_Trend content."""
    relevance = _score_value(judgment, "relevance")
    technical_depth = _score_value(judgment, "technical_depth")
    compellingness = _score_value(judgment, "compellingness")
    category = judgment.get("category") or "Other"
    keep_flag = bool(judgment.get("keep", False))

    if category == "Market_Trend":
        if keep_flag and relevance >= 6 and compellingness >= 6:
            return True
        if relevance >= 6 and compellingness >= 6 and technical_depth >= 4:
            return True
        if keep_flag and relevance >= 5.5 and compellingness >= 5.5:
            return True
        return False

    if keep_flag and relevance >= 7 and technical_depth >= 5.5:
        return True
    if keep_flag and relevance >= 6.5 and technical_depth >= 6:
        return True
    return False


def fetch_arxiv_papers() -> List[Dict]:
    """Rate-limit-friendly arXiv fetch with simpler-query fallbacks; returns [] on total failure."""
    attempts = [
        {
            "label": "primary",
            "query": (
                'cat:cs.CV AND (video surveillance OR "video analytics" OR '
                '"intelligent video" OR "edge AI" OR "anomaly detection" OR tracking)'
            ),
            "max_results": 30,
            "delay_seconds": 3.0,
            "initial_sleep": 2.0,
        },
        {
            "label": "fallback_surveillance",
            "query": 'cat:cs.CV AND (surveillance OR "video analytics" OR detection)',
            "max_results": 18,
            "delay_seconds": 4.0,
            "initial_sleep": 6.0,
        },
        {
            "label": "fallback_cs_cv",
            "query": "cat:cs.CV",
            "max_results": 12,
            "delay_seconds": 5.0,
            "initial_sleep": 10.0,
        },
    ]

    papers: List[Dict] = []
    for i, cfg in enumerate(attempts):
        try:
            print(
                f"   Fetching arXiv ({cfg['label']})..."
                if i == 0
                else f"   arXiv retry ({cfg['label']}) after backoff..."
            )
            time.sleep(cfg["initial_sleep"])

            search = arxiv.Search(
                query=cfg["query"],
                max_results=cfg["max_results"],
                sort_by=arxiv.SortCriterion.LastUpdatedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            client = arxiv.Client(
                page_size=min(20, cfg["max_results"]),
                delay_seconds=cfg["delay_seconds"],
                num_retries=3,
            )

            batch = []
            for paper in client.results(search):
                short_id = getattr(paper, "get_short_id", lambda: "")() or ""
                abs_url = (
                    arxiv_abs_url(paper.entry_id)
                    or (f"https://arxiv.org/abs/{short_id}" if short_id else None)
                    or paper.entry_id
                )
                batch.append(
                    {
                        "title": paper.title,
                        "content": paper.summary,
                        "url": abs_url,
                        "source": "arxiv",
                        "published_at": paper.published.isoformat()
                        if paper.published
                        else None,
                    }
                )

            if batch:
                print(f"✅ Added {len(batch)} arXiv papers ({cfg['label']})")
                papers.extend(batch)
                break

            print(f"   arXiv ({cfg['label']}) returned no papers; trying fallback...")
        except Exception as e:
            print(f"⚠️  arXiv ({cfg['label']}) failed: {e}")
            if i + 1 < len(attempts):
                print("   Continuing with simpler arXiv query...")
            else:
                print("Continuing without arXiv (Tavily/RSS only).")

    return papers


def _enrich_arxiv_result(item: Dict) -> Dict:
    """Rewrite /html/ or /pdf/ to /abs/ and take title+date from the arXiv API."""
    url = item.get("url") or ""
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        return item
    item["url"] = arxiv_abs_url(url) or item["url"]
    try:
        client = arxiv.Client(delay_seconds=1.0, num_retries=2)
        paper = next(client.results(arxiv.Search(id_list=[arxiv_id])), None)
        if not paper:
            return item
        item["title"] = paper.title
        if paper.summary:
            item["content"] = paper.summary
        if paper.published:
            item["published_at"] = paper.published.isoformat()
        item["source"] = item.get("source") or "arxiv"
    except Exception as exc:
        print(f"   arXiv enrich skipped for {arxiv_id}: {exc}")
    return item


def _prepare_search_item(item: Dict) -> Dict:
    canonical = canonicalize_article_url(item.get("url"))
    if canonical:
        item["url"] = canonical
    href = item.get("url") or ""
    title = item.get("title") or ""
    if extract_arxiv_id(href) and (
        is_heading_title(title) or "/html/" in href or "/pdf/" in href
    ):
        item = _enrich_arxiv_result(item)
    image = item.get("image")
    if image and not is_quality_article_image(image):
        item["image"] = None
    return item


class AgentState(TypedDict):
    queries: List[str]
    judge_model: NotRequired[str]
    search_results: List[Dict]
    candidates: List[Dict]
    final_articles: List[Dict]
    metrics: Dict[str, int]


def search_node(state: AgentState) -> AgentState:
    print("🔍 Step 1/3: Tavily + RSS + arXiv...")
    queries = state["queries"]
    results: List[Dict] = []
    metrics = state.setdefault(
        "metrics",
        {
            "fetched_web_results": 0,
            "fetched_rss_results": 0,
            "fetched_arxiv_results": 0,
            "search_errors": 0,
            "search_rate_limit_errors": 0,
            "judge_errors": 0,
            "judge_rate_limit_errors": 0,
        },
    )

    search_time_range = tavily_time_range()
    print(f"   Tavily time_range={search_time_range!r} (week on Monday, d Tue–Fri)")

    for i, q in enumerate(queries, 1):
        print(f"   Searching query {i}/{len(queries)}: {q}")
        for attempt in range(1, 3):
            try:
                resp = tavily_client.search(
                    query=q,
                    search_depth="advanced",
                    max_results=_TAVILY_MAX_RESULTS,
                    include_answer=True,
                    time_range=search_time_range,
                )
                web_results = resp.get("results", [])
                metrics["fetched_web_results"] += len(web_results)
                for res in web_results:
                    published = (
                        res.get("published_date")
                        or res.get("published_time")
                        or res.get("date")
                    )
                    image = res.get("image") or res.get("thumbnail")
                    results.append(
                        {
                            "url": canonicalize_article_url(res["url"]) or res["url"],
                            "title": res["title"],
                            "content": res.get("content") or res.get("snippet", ""),
                            "source": "web",
                            "published_at": _parse_datetime_string(published)
                            if published
                            else None,
                            "image": image if is_quality_article_image(image) else None,
                        }
                    )
                break
            except Exception as e:
                metrics["search_errors"] += 1
                if _is_rate_limit_error(e):
                    metrics["search_rate_limit_errors"] += 1
                print(f"   Tavily warning for '{q}' (attempt {attempt}/2): {e}")
                if attempt == 1:
                    time.sleep(1.2)

    rss_feeds = [
        "https://rss.arxiv.org/rss/cs.CV",
        "https://learnopencv.com/feed/",
        "https://viso.ai/feed/",
        "https://blog.roboflow.com/rss/",
        "https://cctvbuyersguide.com/feed/",
        "https://opencv.org/feed/",
        "https://www.edge-ai-vision.com/feed/",
    ]

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            feed_entries = feed.entries[:_RSS_ENTRIES_PER_FEED]
            metrics["fetched_rss_results"] += len(feed_entries)
            for entry in feed_entries:
                rss_image = _parse_feed_entry_image(entry)
                results.append(
                    {
                        "url": canonicalize_article_url(entry.link) or entry.link,
                        "title": entry.title,
                        "content": entry.get(
                            "summary", entry.get("description", "")
                        ),
                        "source": "rss",
                        "published_at": _parse_feed_entry_date(entry),
                        "image": rss_image
                        if is_quality_article_image(rss_image)
                        else None,
                    }
                )
        except Exception as e:
            metrics["search_errors"] += 1
            if _is_rate_limit_error(e):
                metrics["search_rate_limit_errors"] += 1

    print("Fetching recent computer vision papers from arXiv...")
    arxiv_results = fetch_arxiv_papers()
    metrics["fetched_arxiv_results"] += len(arxiv_results)
    results.extend(arxiv_results)

    results = [_prepare_search_item(item) for item in results]
    results = [
        item
        for item in results
        if not reject_reason(item.get("title"), item.get("url"))
    ]
    results = dedupe_items_by_url(results)
    state["search_results"] = results[:_SEARCH_RESULTS_CAP]
    print(f"✅ Search complete — {len(state['search_results'])} candidates (URL-deduped).\n")
    return state


def judge_node(state: AgentState) -> AgentState:
    print(
        "🤖 Step 2/3: LLM judging articles for technical relevance & depth "
        "(this may take 1–4 minutes)..."
    )
    model = state.get("judge_model") or "grok-4-1-fast-reasoning"
    candidates = []
    metrics = state.setdefault("metrics", {})
    total = len(state["search_results"])

    for i, item in enumerate(state["search_results"], 1):
        title_short = (
            item["title"][:80] + "..." if len(item["title"]) > 80 else item["title"]
        )
        print(f"   Judging {i}/{total}: {title_short}")

        cheap_reject = reject_reason(item.get("title"), item.get("url"))
        if cheap_reject:
            print(f"      → Skipped ({cheap_reject})")
            continue

        snippet = item["content"][:5500]

        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"{IVS_JUDGE_PROMPT}\n\nTitle: {item['title']}\n"
                            f"URL: {item['url']}\nContent:\n{snippet}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=700,
                response_format={"type": "json_object"},
                timeout=35,
            )
            judgment = json.loads(response.choices[0].message.content)

            if article_passes_judge(judgment):
                item.update(
                    {
                        "score_relevance": judgment.get("relevance", 0),
                        "score_technical": judgment.get("technical_depth", 0),
                        "score_compelling": judgment.get("compellingness", 0),
                        "category": judgment.get("category", "Other"),
                        "summary": judgment.get("short_summary", ""),
                        "takeaways": judgment.get("key_takeaways", []),
                        "entities": judgment.get("entities", []),
                        "why_keep": judgment.get("why_keep", ""),
                    }
                )
                candidates.append(item)
                print(
                    f"      → KEPT [{judgment.get('category', 'Other')}] "
                    f"(Rel:{judgment.get('relevance')} "
                    f"Tech:{judgment.get('technical_depth')})"
                )
            else:
                skipped_cat = judgment.get("category", "Other")
                print(
                    f"      → Skipped ({skipped_cat}, "
                    f"Rel:{judgment.get('relevance')}, "
                    f"Tech:{judgment.get('technical_depth')})"
                )

        except Exception as e:
            metrics["judge_errors"] = metrics.get("judge_errors", 0) + 1
            if _is_rate_limit_error(e):
                metrics["judge_rate_limit_errors"] = (
                    metrics.get("judge_rate_limit_errors", 0) + 1
                )
            print(f"      → Skipped (error: {type(e).__name__})")
            continue

        if i % 5 == 0 and i < total:
            time.sleep(0.8)

    state["candidates"] = candidates
    print(f"✅ Judging complete — {len(candidates)} articles passed the filter.\n")
    return state


def dedup_node(state: AgentState) -> AgentState:
    print("🧹 Step 3/3: Deduplicating (URL + semantic similarity)...")
    if not state["candidates"]:
        state["final_articles"] = []
        print("   No articles to deduplicate.\n")
        return state

    url_ordered = dedupe_items_by_url(state["candidates"])
    url_ordered = dedupe_items_by_title(url_ordered)

    texts = [f"{a['title']} {a.get('summary', '')[:500]}" for a in url_ordered]
    embeddings = embedder.encode(texts)
    sim_matrix = cosine_similarity(embeddings)

    similarity_threshold = 0.88
    kept: List[Dict] = []
    for i, item in enumerate(url_ordered):
        if all(sim_matrix[i][j] < similarity_threshold for j in range(i)):
            kept.append(item)

    state["final_articles"] = kept
    print(
        f"✅ Deduplication complete — {len(kept)} unique articles "
        f"(threshold={similarity_threshold}).\n"
    )
    return state


workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("judge", judge_node)
workflow.add_node("dedup", dedup_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "judge")
workflow.add_edge("judge", "dedup")
workflow.add_edge("dedup", END)

app = workflow.compile()


def run_pipeline(
    custom_queries: List[str] | None = None,
    model: str | None = None,
) -> List[Dict]:
    queries = custom_queries if custom_queries else DEFAULT_QUERIES
    judge_model = model or "grok-4-1-fast-reasoning"
    _reset_image_runtime_stats()

    initial_state: AgentState = {
        "queries": queries,
        "judge_model": judge_model,
        "search_results": [],
        "candidates": [],
        "final_articles": [],
        "metrics": {
            "fetched_web_results": 0,
            "fetched_rss_results": 0,
            "fetched_arxiv_results": 0,
            "search_errors": 0,
            "search_rate_limit_errors": 0,
            "judge_errors": 0,
            "judge_rate_limit_errors": 0,
        },
    }

    print("🚀 Starting IVS News Pipeline...\n")
    start_total = time.time()

    result = app.invoke(initial_state)
    populate_article_metadata(result["final_articles"])
    kept_dated = []
    for art in result["final_articles"]:
        reason = reject_reason(
            art.get("title"),
            art.get("url"),
            art.get("published_at"),
            require_published_at=True,
        )
        if reason:
            print(f"   Dropping before save ({reason}): {art.get('title', '')[:80]}")
            continue
        if not is_quality_article_image(art.get("image")):
            art["image"] = None
        kept_dated.append(art)
    result["final_articles"] = kept_dated
    result["final_articles"].sort(
        key=lambda art: art.get("published_at") or "",
        reverse=True,
    )
    metrics = result.get("metrics", {})
    fetched_total = (
        metrics.get("fetched_web_results", 0)
        + metrics.get("fetched_rss_results", 0)
        + metrics.get("fetched_arxiv_results", 0)
    )
    successful_images = sum(
        1 for article in result["final_articles"] if article.get("image")
    )
    with _image_stats_lock:
        image_stats = dict(_image_runtime_stats)

    duration = time.time() - start_total
    timestamp = datetime.now().isoformat()
    output = {
        "run_at": timestamp,
        "duration_seconds": round(duration, 1),
        "article_count": len(result["final_articles"]),
        "articles": result["final_articles"],
    }

    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"🎉 Pipeline finished in {duration:.1f} seconds!")
    print(
        f"✅ {len(result['final_articles'])} high-signal articles saved to articles.json"
    )
    print("\n📊 Run summary")
    print(f"   - Articles fetched (pre-dedup): {fetched_total}")
    print(f"   - Final articles: {len(result['final_articles'])}")
    print(f"   - Successful images: {successful_images}/{len(result['final_articles'])}")
    print(
        "   - Image extraction: "
        f"{image_stats.get('http_requests', 0)} HTTP requests, "
        f"{image_stats.get('cache_hits', 0)} cache hits, "
        f"{image_stats.get('request_errors', 0)} request errors"
    )
    total_rate_limits = (
        metrics.get("search_rate_limit_errors", 0)
        + metrics.get("judge_rate_limit_errors", 0)
        + image_stats.get("rate_limit_errors", 0)
    )
    total_errors = (
        metrics.get("search_errors", 0)
        + metrics.get("judge_errors", 0)
        + image_stats.get("request_errors", 0)
    )
    if total_rate_limits > 0 or total_errors > 0:
        print(
            "   - Issues: "
            f"{total_rate_limits} rate-limit events, {total_errors} total recoverable errors"
        )
    else:
        print("   - Issues: none detected")

    if supabase and result["final_articles"]:
        print("\n💾 Saving to Supabase...")
        existing_by_key: Dict[str, Dict[str, Any]] = {}
        try:
            existing_rows = (
                supabase.table("ivs_articles")
                .select("id,url,published_at,source")
                .execute()
                .data
                or []
            )
            for row in existing_rows:
                key = canonical_url_key(row.get("url"))
                if key and key not in existing_by_key:
                    existing_by_key[key] = row
        except Exception as exc:
            print(f"   Warning: could not load existing articles ({exc})")

        saved_count = 0
        skipped_count = 0
        write_errors = 0
        for art in result["final_articles"]:
            canonical = canonicalize_article_url(art.get("url")) or art.get("url")
            art["url"] = canonical
            if not art.get("published_at"):
                print(f"   Skip (no published_at): {art.get('title', '')[:60]}")
                skipped_count += 1
                continue

            existing = existing_by_key.get(canonical_url_key(canonical))
            if existing and is_hidden_row(existing):
                print(f"   Skip (hidden): {art.get('title', '')[:60]}")
                skipped_count += 1
                continue
            if existing and published_at_older_than(
                existing.get("published_at"), UPSERT_MAX_AGE_DAYS
            ):
                print(
                    f"   Skip (existing published_at older than {UPSERT_MAX_AGE_DAYS}d): "
                    f"{art.get('title', '')[:60]}"
                )
                skipped_count += 1
                continue

            text_for_embedding = f"{art['title']} {art.get('summary', '')}"
            embedding = embedder.encode([text_for_embedding])[0].tolist()
            source = art.get("source") or "web"
            if source == HIDDEN_SOURCE:
                source = "web"

            data = {
                "url": existing["url"] if existing else canonical,
                "title": art["title"],
                "summary": art.get("summary"),
                "content_snippet": art.get("content", "")[:2000],
                "published_at": art.get("published_at"),
                "source": source,
                "category": art.get("category", "Other"),
                "score_relevance": art.get("score_relevance"),
                "score_technical": art.get("score_technical"),
                "score_compelling": art.get("score_compelling"),
                "entities": art.get("entities", []),
                "takeaways": art.get("takeaways", []),
                "image": art.get("image")
                if is_quality_article_image(art.get("image"))
                else None,
                "embedding": embedding,
                "run_at": timestamp,
            }

            try:
                supabase.table("ivs_articles").upsert(
                    data, on_conflict="url"
                ).execute()
                saved_count += 1
            except Exception as e:
                write_errors += 1
                print(f"   Supabase upsert failed for {art['title'][:60]}...: {e}")

        print(
            f"✅ Saved/updated {saved_count} articles in Supabase "
            f"({skipped_count} skipped, {write_errors} errors).\n"
        )

        if saved_count == 0 and write_errors > 0:
            raise SystemExit(
                f"ERROR: Pipeline produced {len(result['final_articles'])} article(s) "
                "but saved 0 to Supabase. Check SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY "
                "and upsert errors above."
            )

    elif result["final_articles"] and not supabase:
        raise SystemExit(
            "ERROR: Pipeline produced articles but Supabase client is not configured "
            "(missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)."
        )

    for i, art in enumerate(result["final_articles"], 1):
        print(f"{i}. [{art.get('category', 'Other')}] {art['title']}")
        print(
            f"   Scores → Rel: {art.get('score_relevance')} | "
            f"Tech: {art.get('score_technical')} | Comp: {art.get('score_compelling')}"
        )
        print(f"   Summary: {art.get('summary', '')[:180]}...")
        print(f"   URL: {art['url']}\n")

    return result["final_articles"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVS News Pipeline")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Override default search queries",
    )
    parser.add_argument(
        "--model",
        default="grok-4-1-fast-reasoning",
        help="Grok model to use for judging",
    )
    args = parser.parse_args()

    if args.run:
        q = args.queries if args.queries else None
        run_pipeline(custom_queries=q, model=args.model)
    else:
        print("Usage: python pipeline.py --run")
