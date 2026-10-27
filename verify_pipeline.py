"""
Automated CLI Test & Benchmark Suite for AMD Web Retrieval Agent.
Verifies all pipeline stages: Schemas, Hardware Router, Query Planning,
Resilient Search, DOM Table Extraction, Hybrid BM25/Semantic Ranking,
Citation Verification, and Anti-Hallucination Abstention Guardrails.
"""

import sys
import time
from typing import List, Dict, Any

from agent.schemas import (
    SubQuery,
    ScrapedDocument,
    RankedChunk,
    Citation,
    AgentResponse
)
from agent.query_planner import QueryPlanner
from agent.searcher import ResilientSearcher
from agent.scraper import DOMScraper
from agent.ranker import HybridRanker
from agent.synthesizer import GroundedSynthesizer
from agent.orchestrator import RetrievalOrchestrator
import config

# Terminal ANSI color styling
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def log_test_start(name: str):
    print(f"\n{BOLD}{CYAN}>>> Running Test: {name}{RESET}")

def log_success(msg: str):
    print(f"  {GREEN}✔ [PASS]{RESET} {msg}")

def log_failure(msg: str):
    print(f"  {RED}✘ [FAIL]{RESET} {msg}")

def run_all_tests() -> bool:
    print(f"\n{BOLD}{'=' * 75}{RESET}")
    print(f"{BOLD}  AMD AI League (Match 4) — Automated Pipeline Verification Suite  {RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")

    total_tests = 8
    passed_tests = 0
    start_suite_time = time.time()

    # ------------------------------------------------------------------
    # Test 1: Schemas and Data Contract Validation
    # ------------------------------------------------------------------
    log_test_start("1. Schemas & Pydantic Data Contracts")
    try:
        sq = SubQuery(query="AMD MI300X specs", intent="retrieve specs", priority=1)
        doc = ScrapedDocument(
            url="https://amd.com/mi300x",
            title="MI300X Overview",
            domain="amd.com",
            clean_text="AMD Instinct MI300X accelerator specifications.",
            markdown_tables=["| Col1 | Col2 |\n|---|---|\n| A | B |"]
        )
        chunk = RankedChunk(
            chunk_id=1,
            url=doc.url,
            title=doc.title,
            text=doc.clean_text,
            is_table=False,
            bm25_score=0.95,
            semantic_score=0.88,
            combined_score=0.92
        )
        citation = Citation(
            citation_id=1,
            url=doc.url,
            title=doc.title,
            supporting_quote="MI300X specs",
            confidence=0.92
        )
        resp = AgentResponse(
            original_query="What is MI300X?",
            sub_queries=[sq],
            answer_markdown="Answer text [1](https://amd.com/mi300x)",
            citations=[citation],
            ranked_chunks=[chunk],
            scraped_docs=[doc],
            grounding_score=0.95,
            abstained=False,
            hops_executed=1,
            telemetry={"backend": "universal_fallback"}
        )

        dumped = resp.model_dump()
        assert dumped["original_query"] == "What is MI300X?"
        assert len(dumped["citations"]) == 1
        assert len(dumped["scraped_docs"][0]["markdown_tables"]) == 1
        log_success("All Pydantic v2 schemas serialize and validate accurately.")
        passed_tests += 1
    except Exception as e:
        log_failure(f"Schemas validation failed: {e}")

    # ------------------------------------------------------------------
    # Test 2: Hardware Routing & AMD Platform Telemetry
    # ------------------------------------------------------------------
    log_test_start("2. Hardware Routing & AMD Platform Telemetry")
    try:
        settings = config.get_settings()
        hw_target_default = config.get_hardware_target(settings.inference_backend)
        hw_target_cloud = config.get_hardware_target("amd_cloud_vllm")
        hw_target_rocm = config.get_hardware_target("local_rocm")
        vram_est = config.get_vram_utilization_est()
        rocm_info = config.detect_rocm_runtime()

        assert "MI300X" in hw_target_cloud
        assert "ROCm" in hw_target_rocm
        log_success(f"Detected hardware target: {hw_target_default}")
        log_success(f"AMD Cloud vLLM mapping: {hw_target_cloud}")
        log_success(f"Memory telemetry reading: {vram_est}")
        passed_tests += 1
    except Exception as e:
        log_failure(f"Hardware routing failed: {e}")

    # ------------------------------------------------------------------
    # Test 3: Query Planner Decomposition
    # ------------------------------------------------------------------
    log_test_start("3. Query Planner Decomposition & Heuristics")
    try:
        planner = QueryPlanner()
        sub_queries = planner.plan_queries("Compare AMD Instinct MI300X vs MI325X memory capacity, bandwidth, and FP8 performance specs.")
        assert len(sub_queries) >= 2, f"Expected at least 2 subqueries, got {len(sub_queries)}"
        queries_text = [q.query for q in sub_queries]
        assert any("MI300X" in q for q in queries_text), "Subqueries should target MI300X"
        assert any("MI325X" in q for q in queries_text), "Subqueries should target MI325X"
        for q in sub_queries:
            log_success(f"SubQuery [P{q.priority}]: {q.query} (Intent: {q.intent})")
        passed_tests += 1
    except Exception as e:
        log_failure(f"Query planner failed: {e}")

    # ------------------------------------------------------------------
    # Test 4: Resilient Searcher (Multi-Engine & Fallback)
    # ------------------------------------------------------------------
    log_test_start("4. Resilient Multi-Engine Searcher")
    try:
        searcher = ResilientSearcher()
        results = searcher.search("AMD Instinct MI300X specifications", max_results=3)
        assert len(results) > 0, "Searcher should return results"
        log_success(f"Retrieved {len(results)} search results via '{results[0].engine}' engine.")
        for r in results[:2]:
            log_success(f"Result: {r.title} | {r.url}")
        passed_tests += 1
    except Exception as e:
        log_failure(f"Resilient searcher failed: {e}")

    # ------------------------------------------------------------------
    # Test 5: DOM Scraper & HTML Table Parser to Markdown
    # ------------------------------------------------------------------
    log_test_start("5. DOM Scraper & HTML Table Parser")
    try:
        scraper = DOMScraper()
        test_url = "https://rocm.blogs.amd.com/artificial-intelligence/mi300x-vs-mi325x-benchmarks.html"
        doc = scraper.scrape_url(test_url)
        assert doc.word_count > 0, "Document clean text should be extracted"
        assert len(doc.markdown_tables) >= 1, "Should extract at least 1 Markdown table from DOM"
        table_md = doc.markdown_tables[0]
        assert "|" in table_md, "Table must be formatted in Markdown syntax"
        assert "AMD Instinct MI300X" in table_md
        assert "AMD Instinct MI325X" in table_md
        log_success(f"Scraped '{doc.title}' ({doc.word_count} words).")
        log_success(f"Extracted {len(doc.markdown_tables)} structured Markdown table(s).")
        print(f"\n{YELLOW}{table_md}{RESET}\n")
        passed_tests += 1
    except Exception as e:
        log_failure(f"DOM scraper and table parser failed: {e}")

    # ------------------------------------------------------------------
    # Test 6: Hybrid BM25 & Semantic Ranker with Atomic Tables
    # ------------------------------------------------------------------
    log_test_start("6. Hybrid BM25 + Semantic Ranker & Deduplication")
    try:
        ranker = HybridRanker()
        chunks = ranker.chunk_and_rank([doc], "MI300X vs MI325X memory bandwidth comparison", top_k=5)
        assert len(chunks) > 0, "Should generate ranked chunks"
        assert any(c.is_table for c in chunks), "Markdown tables must be preserved as atomic chunks"
        top_chunk = chunks[0]
        assert top_chunk.combined_score > 0.0
        log_success(f"Top chunk #{top_chunk.chunk_id} (is_table={top_chunk.is_table}) score: {top_chunk.combined_score}")
        log_success(f"BM25 Score: {top_chunk.bm25_score} | Semantic Score: {top_chunk.semantic_score}")
        passed_tests += 1
    except Exception as e:
        log_failure(f"Hybrid ranker failed: {e}")

    # ------------------------------------------------------------------
    # Test 7: Anti-Hallucination Abstention Guardrail
    # ------------------------------------------------------------------
    log_test_start("7. Anti-Hallucination Abstention Guardrail (Fictional & Ungrounded Entities)")
    try:
        orchestrator = RetrievalOrchestrator()

        # Case 7A: Explicit fictional hardware entity
        fictional_query = "What is the official release date and benchmark score of the fictional AMD Radeon RX 19990 XTX quantum GPU?"
        resp_fictional = orchestrator.run(fictional_query)
        assert resp_fictional.abstained is True, "Agent MUST abstain on fictional/unverified hardware queries"
        assert resp_fictional.grounding_score == 0.0
        assert "Abstention Reason" in resp_fictional.answer_markdown
        log_success("Agent successfully abstained on fictional GPU query.")

        # Case 7B: Comparative query with non-existent hardware target
        fictional_comp_query = "Compare AMD Instinct MI300X vs AMD Instinct MI800X"
        resp_fictional_comp = orchestrator.run(fictional_comp_query)
        assert resp_fictional_comp.abstained is True, "Agent MUST abstain when one comparison entity is unverified"
        assert resp_fictional_comp.grounding_score == 0.0
        log_success("Agent successfully rejected ungrounded comparative query (MI300X vs MI800X).")

        # Case 7C: Empty / whitespace query input
        resp_empty = orchestrator.run("   ")
        assert resp_empty.abstained is True, "Agent MUST abstain immediately on whitespace query"
        assert resp_empty.hops_executed == 0, "No search hops should execute for empty queries"
        log_success("Agent abstained cleanly on empty query with 0 hops.")

        passed_tests += 1
    except Exception as e:
        log_failure(f"Abstention guardrail failed: {e}")

    # ------------------------------------------------------------------
    # Test 8: End-to-End Multi-Hop Retrieval & Grounded Synthesis
    # ------------------------------------------------------------------
    log_test_start("8. End-to-End Multi-Hop Pipeline & Synthesis")
    try:
        spec_query = "Compare AMD Instinct MI300X vs MI325X memory capacity, bandwidth, and FP8 performance specs."
        resp_real = orchestrator.run(spec_query)
        assert resp_real.abstained is False, "Agent should NOT abstain on real AMD hardware query"
        assert len(resp_real.citations) > 0, "Response must include verified citations"
        assert len(resp_real.ranked_chunks) > 0
        assert "|" in resp_real.answer_markdown, "Answer must include structured comparison table"
        assert resp_real.telemetry["total_latency_ms"] > 0
        log_success(f"End-to-end completed in {resp_real.telemetry['total_latency_ms']} ms.")
        log_success(f"Grounding score: {resp_real.grounding_score * 100:.1f}% across {len(resp_real.citations)} verified citation(s).")
        log_success(f"Hardware HUD target: {resp_real.telemetry['hardware_target']}")

        # Verify secondary official comparison (MI300X vs NVIDIA H100)
        h100_query = "Compare AMD Instinct MI300X vs NVIDIA H100"
        resp_h100 = orchestrator.run(h100_query)
        assert resp_h100.abstained is False, "Should successfully synthesize real MI300X vs H100 comparison"
        assert "|" in resp_h100.answer_markdown, "Must preserve structured matrix"
        log_success(f"Verified secondary hardware benchmark: MI300X vs NVIDIA H100 ({len(resp_h100.citations)} citations).")

        passed_tests += 1
    except Exception as e:
        log_failure(f"End-to-end pipeline failed: {e}")

    # ------------------------------------------------------------------
    # Summary Table
    # ------------------------------------------------------------------
    suite_elapsed = round(time.time() - start_suite_time, 2)
    print(f"\n{BOLD}{'=' * 75}{RESET}")
    print(f"{BOLD}  VERIFICATION RESULTS: {passed_tests}/{total_tests} Tests Passed in {suite_elapsed}s  {RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")

    if passed_tests == total_tests:
        print(f"{GREEN}{BOLD}✔ ALL PIPELINE BENCHMARKS PASSED. READY FOR SUBMISSION.{RESET}\n")
        return True
    else:
        print(f"{RED}{BOLD}✘ {total_tests - passed_tests} TEST(S) FAILED.{RESET}\n")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
