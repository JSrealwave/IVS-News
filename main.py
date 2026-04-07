from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="IVS News API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

class Article(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    url: str
    category: str
    score_relevance: int
    score_technical: int
    score_compelling: int

@app.get("/feed")
async def get_feed(limit: int = Query(20, le=50), category: Optional[str] = None):
    query = supabase.table("ivs_articles").select("*").order("run_at", desc=True).limit(limit)
    if category:
        query = query.eq("category", category)
    response = query.execute()
    return response.data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
