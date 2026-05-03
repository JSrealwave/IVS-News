
import argparse
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, NotRequired, TypedDict
from urllib.parse import urlparse, urlunparse

import arxiv
import feedparser
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import Client, create_client
from tavily import TavilyClient

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

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([\w.-]+)(?:\.pdf)?", re.IGNORECASE
)


def canonical_url_key(url: str | None) -> str:
    """Stable key for deduplication (handles http/https, trailing slashes, arXiv abs/pdf)."""
    if not url or not isinstance(url, str):
        return ""
    raw = url.strip()
    m = _ARXIV_ID_RE.search(raw)
    if m:
        return f"arxiv:{m.group(1).lower()}"

    try:
        p = urlparse(raw)
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = (p.path or "").rstrip("/") or "/"
        scheme = "https"
        return urlunparse((scheme, netloc, path, "", "", "")).lower()
    except Exception:
        return raw.lower().rstrip("/")


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
                url = paper.pdf_url or paper.entry_id
                batch.append(
                    {
                        "title": paper.title,
                        "content": paper.summary,
                        "url": url,
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


class AgentState(TypedDict):
    queries: List[str]
    judge_model: NotRequired[str]
    search_results: List[Dict]
    candidates: List[Dict]
    final_articles: List[Dict]


def search_node(state: AgentState) -> AgentState:
    print("🔍 Step 1/3: Tavily + RSS + arXiv...")
    queries = state["queries"]
    results: List[Dict] = []

    for i, q in enumerate(queries, 1):
        print(f"   Searching query {i}/{len(queries)}: {q}")
        try:
            resp = tavily_client.search(
                query=q,
                search_depth="advanced",
                max_results=8,
                include_answer=True,
                time_range="month",
            )
            for res in resp.get("results", []):
                results.append(
                    {
                        "url": res["url"],
                        "title": res["title"],
                        "content": res.get("content") or res.get("snippet", ""),
                        "source": "web",
                    }
                )
        except Exception as e:
            print(f"   Tavily warning for '{q}': {e}")

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
            for entry in feed.entries[:8]:
                results.append(
                    {
                        "url": entry.link,
                        "title": entry.title,
                        "content": entry.get(
                            "summary", entry.get("description", "")
                        ),
                        "source": "rss",
                    }
                )
        except Exception:
            pass

    print("Fetching recent computer vision papers from arXiv...")
    results.extend(fetch_arxiv_papers())

    results = dedupe_items_by_url(results)
    state["search_results"] = results[:35]
    print(f"✅ Search complete — {len(state['search_results'])} candidates (URL-deduped).\n")
    return state


def judge_node(state: AgentState) -> AgentState:
    print(
        "🤖 Step 2/3: LLM judging articles for technical relevance & depth "
        "(this may take 1–4 minutes)..."
    )
    model = state.get("judge_model") or "grok-4-1-fast-reasoning"
    candidates = []
    total = len(state["search_results"])

    for i, item in enumerate(state["search_results"], 1):
        title_short = (
            item["title"][:80] + "..." if len(item["title"]) > 80 else item["title"]
        )
        print(f"   Judging {i}/{total}: {title_short}")

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

            if judgment.get("keep", False):
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
                    f"      → KEPT (Rel:{judgment.get('relevance')} "
                    f"Tech:{judgment.get('technical_depth')})"
                )
            else:
                print("      → Skipped")

        except Exception as e:
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

    texts = [f"{a['title']} {a.get('summary', '')[:500]}" for a in url_ordered]
    embeddings = embedder.encode(texts)
    sim_matrix = cosine_similarity(embeddings)

    similarity_threshold = 0.85
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

    initial_state: AgentState = {
        "queries": queries,
        "judge_model": judge_model,
        "search_results": [],
        "candidates": [],
        "final_articles": [],
    }

    print("🚀 Starting IVS News Pipeline...\n")
    start_total = time.time()

    result = app.invoke(initial_state)

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

    if supabase and result["final_articles"]:
        print("\n💾 Saving to Supabase...")
        saved_count = 0
        for art in result["final_articles"]:
            text_for_embedding = f"{art['title']} {art.get('summary', '')}"
            embedding = embedder.encode([text_for_embedding])[0].tolist()

            data = {
                "url": art["url"],
                "title": art["title"],
                "summary": art.get("summary"),
                "content_snippet": art.get("content", "")[:2000],
                "published_at": art.get("published_at"),
                "source": art.get("source", "web"),
                "category": art.get("category", "Other"),
                "score_relevance": art.get("score_relevance"),
                "score_technical": art.get("score_technical"),
                "score_compelling": art.get("score_compelling"),
                "entities": art.get("entities", []),
                "takeaways": art.get("takeaways", []),
                "embedding": embedding,
                "run_at": timestamp,
            }

            try:
                response = supabase.table("ivs_articles").upsert(
                    data, on_conflict="url"
                ).execute()
                if response.data:
                    saved_count += 1
            except Exception as e:
                print(f"   Supabase upsert failed for {art['title'][:60]}...: {e}")

        print(f"✅ Saved/updated {saved_count} articles in Supabase.\n")

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
