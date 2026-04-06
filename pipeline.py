
import os
import json
import argparse
import time
import feedparser
from datetime import datetime
from typing import List, Dict, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import arxiv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from supabase import create_client, Client
import uuid

load_dotenv(override=True)

# Clients
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
openai_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Supabase client - use service_role key explicitly for bypassing RLS
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
    # Force service role mode (helps in some versions)
    supabase.postgrest.session.headers.update({
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    })
    print("✅ Supabase client initialized with service_role key")
else:
    supabase = None
    print("⚠️ Supabase credentials missing in .env")

from prompts import IVS_JUDGE_PROMPT, SYSTEM_PROMPT

class AgentState(TypedDict):
    queries: List[str]
    search_results: List[Dict]
    candidates: List[Dict]
    final_articles: List[Dict]

def search_node(state: AgentState) -> AgentState:
    print("🔍 Step 1/3: Performing semantic web search (Tavily) + recent arXiv papers...")
    queries = state["queries"]
    results = []
    
    for i, q in enumerate(queries, 1):
        print(f"   Searching query {i}/{len(queries)}: {q}")
        try:
            resp = tavily_client.search(
                query=q,
                search_depth="advanced",
                max_results=8,
                include_answer=True,
                time_range="month"
            )
            for res in resp.get("results", []):
                results.append({
                    "url": res["url"],
                    "title": res["title"],
                    "content": res.get("content") or res.get("snippet", ""),
                    "source": "web"
                })
        except Exception as e:
            print(f"   Tavily warning for '{q}': {e}")

                # Optional RSS boost for known reliable sources
    rss_feeds = [
        'https://rss.arxiv.org/rss/cs.CV',  # already in arXiv
        'https://learnopencv.com/feed/',
        'https://viso.ai/feed/',
        'https://blog.roboflow.com/rss/',
        'https://cctvbuyersguide.com/feed/',
        'https://opencv.org/feed/',
        'https://www.edge-ai-vision.com/feed/'  # great for Embedded Vision Summit / edge AI content
    ]

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)  # add: import feedparser at top
            for entry in feed.entries[:8]:
                results.append({
                    "url": entry.link,
                    "title": entry.title,
                    "content": entry.get('summary', entry.get('description', '')),
                    "source": "rss"
                })
        except:
            pass
    
    # arXiv
    print("   Fetching recent computer vision papers from arXiv...")
    client = arxiv.Client()
    search = arxiv.Search(
        query='cat:cs.CV AND (video surveillance OR "video analytics" OR "intelligent video" OR "anomaly detection" OR "edge AI")',
        max_results=12,
        sort_by=arxiv.SortCriterion.LastUpdatedDate
    )
    for paper in client.results(search):
        results.append({
            "url": paper.pdf_url or paper.entry_id,
            "title": paper.title,
            "content": paper.summary,
            "source": "arxiv"
        })
    
    state["search_results"] = results[:35]
    print(f"✅ Search complete — {len(state['search_results'])} candidates found.\n")
    return state

def judge_node(state: AgentState) -> AgentState:
    print("🤖 Step 2/3: LLM judging articles for technical relevance & depth (this may take 1–4 minutes)...")
    candidates = []
    total = len(state["search_results"])
    
    for i, item in enumerate(state["search_results"], 1):
        title_short = item["title"][:80] + "..." if len(item["title"]) > 80 else item["title"]
        print(f"   Judging {i}/{total}: {title_short}")
        
        snippet = item["content"][:5500]  # keep it small & fast
        
        try:
            start_time = time.time()
            response = openai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",   # Fast & cheap for this task — change to "grok-4-1-fast-reasoning" if you want deeper thinking
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{IVS_JUDGE_PROMPT}\n\nTitle: {item['title']}\nURL: {item['url']}\nContent:\n{snippet}"}
                ],
                temperature=0.1,
                max_tokens=700,
                response_format={"type": "json_object"},
                timeout=35
            )
            judgment = json.loads(response.choices[0].message.content)
            
            if judgment.get("keep", False):
                item.update({
                    "score_relevance": judgment.get("relevance", 0),
                    "score_technical": judgment.get("technical_depth", 0),
                    "score_compelling": judgment.get("compellingness", 0),
                    "category": judgment.get("category", "Other"),
                    "summary": judgment.get("short_summary", ""),
                    "takeaways": judgment.get("key_takeaways", []),
                    "entities": judgment.get("entities", []),
                    "why_keep": judgment.get("why_keep", "")
                })
                candidates.append(item)
                print(f"      → KEPT (Rel:{judgment.get('relevance')} Tech:{judgment.get('technical_depth')})")
            else:
                print(f"      → Skipped")
                
        except Exception as e:
            print(f"      → Skipped (error: {type(e).__name__})")
            continue
        
        # Small pause to be gentle on rate limits
        if i % 5 == 0 and i < total:
            time.sleep(0.8)
    
    state["candidates"] = candidates
    print(f"✅ Judging complete — {len(candidates)} articles passed the filter.\n")
    return state

def dedup_node(state: AgentState) -> AgentState:
    print("🧹 Step 3/3: Removing duplicate articles...")
    if not state["candidates"]:
        state["final_articles"] = []
        print("   No articles to deduplicate.\n")
        return state
    
    texts = [f"{a['title']} {a.get('summary', '')[:500]}" for a in state["candidates"]]
    embeddings = embedder.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    
    kept = []
    for i, item in enumerate(state["candidates"]):
        if all(sim_matrix[i][j] < 0.82 for j in range(i)):
            kept.append(item)
    
    state["final_articles"] = kept
    print(f"✅ Deduplication complete — {len(kept)} unique high-signal articles remaining.\n")
    return state

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("search", search_node)
workflow.add_node("judge", judge_node)
workflow.add_node("dedup", dedup_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "judge")
workflow.add_edge("judge", "dedup")
workflow.add_edge("dedup", END)

app = workflow.compile()

def run_pipeline(custom_queries: List[str] = None):
    queries = [
        "intelligent video surveillance AI 2026 OR edge AI",
        "AI video analytics customer case study deployment OR implementation 2025 OR 2026",
        "new computer vision techniques edge AI surveillance OR anomaly detection OR tracking",
        "video analytics marketplace news product launch OR ISC West 2026 OR Embedded Vision Summit",
        "intelligent video surveillance trends technical edge AI OR spatial intelligence OR vision language models",
        "LiDAR OR event cameras OR sparse cameras video surveillance OR security",
        "Nvidia OR Qualcomm OR Axis OR Hanwha OR Nutanix OR Cisco video surveillance AI OR edge",
        "managed service provider MSP OR VSaaS OR PhySec InfoSec convergence video surveillance",
        "VSaaS OR video surveillance as a service MSP managed platform OR multi-tenant",
        "IoT sensor fusion OR physical security convergence OR enterprise application integration video analytics"
    ]
    
    initial_state: AgentState = {
        "queries": queries,
        "search_results": [],
        "candidates": [],
        "final_articles": []
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
        "articles": result["final_articles"]
    }
    
    # Save local backup
    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"🎉 Pipeline finished in {duration:.1f} seconds!")
    print(f"✅ {len(result['final_articles'])} high-signal articles saved to articles.json")
    
    # === NEW: Persist to Supabase ===
    if supabase and result["final_articles"]:
        print("\n💾 Saving to Supabase...")
        saved_count = 0
        for art in result["final_articles"]:
            # Generate embedding for semantic search (using same model as dedup)
            text_for_embedding = f"{art['title']} {art.get('summary', '')}"
            embedding = embedder.encode([text_for_embedding])[0].tolist()
            
            data = {
                "url": art["url"],
                "title": art["title"],
                "summary": art.get("summary"),
                "content_snippet": art.get("content", "")[:2000],
                "published_at": art.get("published_at"),  # add if available
                "source": art.get("source", "web"),
                "category": art.get("category", "Other"),
                "score_relevance": art.get("score_relevance"),
                "score_technical": art.get("score_technical"),
                "score_compelling": art.get("score_compelling"),
                "entities": art.get("entities", []),
                "takeaways": art.get("takeaways", []),
                "embedding": embedding,
                "run_at": timestamp
            }
            
            # Upsert (insert or update if URL exists)
            try:
                response = supabase.table("ivs_articles").upsert(data, on_conflict="url").execute()
                if response.data:
                    saved_count += 1
            except Exception as e:
                print(f"   Supabase upsert failed for {art['title'][:60]}...: {e}")
        
        print(f"✅ Saved/updated {saved_count} articles in Supabase.\n")
    
    # Console output (keep as-is)
    for i, art in enumerate(result["final_articles"], 1):
        print(f"{i}. [{art.get('category', 'Other')}] {art['title']}")
        print(f"   Scores → Rel: {art.get('score_relevance')} | Tech: {art.get('score_technical')} | Comp: {art.get('score_compelling')}")
        print(f"   Summary: {art.get('summary', '')[:180]}...")
        print(f"   URL: {art['url']}\n")
    
    return result["final_articles"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IVS News Pipeline")
    parser.add_argument("--run", action="store_true", help="Run the full pipeline")
    parser.add_argument("--queries", nargs="*", help="Override default search queries")
    parser.add_argument("--model", default="grok-4-1-fast-reasoning", help="Grok model to use")
    args = parser.parse_args()
    
    if args.run:
        run_pipeline(args.queries)
    else:
        print("Usage: python pipeline.py --run")