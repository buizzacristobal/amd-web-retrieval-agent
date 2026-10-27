"""
Structured Data Contracts for AMD Web Retrieval Agent.
Defines Pydantic models for Queries, Documents, Chunks, Citations, and Responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SubQuery(BaseModel):
    """Decomposed targeted search sub-query."""
    query: str
    intent: str
    priority: int = 1

class ScrapedDocument(BaseModel):
    """Raw scraped document with structured tables and metadata."""
    url: str
    title: str
    domain: str
    clean_text: str
    markdown_tables: List[str] = Field(default_factory=list)
    timestamp: str = ""
    word_count: int = 0
    scrape_status: str = "success"

class RankedChunk(BaseModel):
    """Passage or atomic table chunk scored via BM25 + Semantic similarity."""
    chunk_id: int
    url: str
    title: str
    text: str
    is_table: bool = False
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    combined_score: float = 0.0

class Citation(BaseModel):
    """Evidence citation mapped to a verified ranked chunk."""
    citation_id: int
    url: str
    title: str
    supporting_quote: str = ""
    confidence: float = 1.0

class AgentResponse(BaseModel):
    """Final grounded agent response with citations and verification telemetry."""
    original_query: str
    sub_queries: List[SubQuery] = Field(default_factory=list)
    answer_markdown: str
    citations: List[Citation] = Field(default_factory=list)
    ranked_chunks: List[RankedChunk] = Field(default_factory=list)
    scraped_docs: List[ScrapedDocument] = Field(default_factory=list)
    grounding_score: float = 0.0
    abstained: bool = False
    hops_executed: int = 1
    telemetry: Dict[str, Any] = Field(default_factory=dict)
