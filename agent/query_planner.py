"""
Query Planner Component.
Decomposes complex, natural language questions into 1-3 targeted,
high-precision search sub-queries optimized for factual retrieval.
"""

import re
import json
from datetime import datetime
from typing import List, Optional
import requests

from agent.schemas import SubQuery
from config import get_settings

class QueryPlanner:
    """Intelligent query decomposition and search planning engine."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    def plan_queries(self, natural_query: str) -> List[SubQuery]:
        """
        Generate 1 to 3 targeted SubQueries from user's natural language prompt.
        Attempts LLM decomposition first if an active backend is reachable;
        falls back to intelligent heuristic decomposition if offline or in fallback mode.
        """
        current_year = datetime.now().year
        cleaned_query = self._clean_conversational_text(natural_query)

        if not cleaned_query:
            return []

        # Try LLM decomposition if an endpoint is configured and not in universal_fallback
        if self.settings.inference_backend in ("amd_cloud_vllm", "local_rocm") or (
            self.settings.inference_backend == "universal_fallback" and self.settings.openai_compatible_api_key
        ):
            llm_plan = self._llm_decompose(natural_query, current_year)
            if llm_plan:
                return llm_plan

        # Deterministic heuristic decomposition
        return self._heuristic_decompose(cleaned_query, natural_query, current_year)

    def _clean_conversational_text(self, text: str) -> str:
        """Strip conversational filler while preserving technical keywords and symbols."""
        fluff_patterns = [
            r"^(please\s+)?(can\s+you\s+)?(tell\s+me\s+)?(what\s+is\s+|what\s+are\s+)",
            r"^(compare\s+|give\s+me\s+|show\s+me\s+|search\s+for\s+|find\s+)",
            r"^(how\s+does\s+|explain\s+|describe\s+)",
            r"\b(please|thanks|thank\s+you|could\s+you)\b",
            r"[?!\"]+"
        ]
        res = text.strip()
        for pat in fluff_patterns:
            res = re.sub(pat, "", res, flags=re.IGNORECASE).strip()
        return res

    def _heuristic_decompose(self, cleaned: str, original: str, current_year: int) -> List[SubQuery]:
        """Heuristic rules to produce 1-3 high-relevance search queries."""
        sub_queries: List[SubQuery] = []
        lower = original.lower()

        # Case 1: Comparison between two entities
        vs_match = re.search(r"([A-Za-z0-9_\-\s]+?)\s+(?:vs\.?|versus|compared\s+(?:to|with))\s+([A-Za-z0-9_\-\s]+)", original, re.IGNORECASE)
        diff_match = re.search(r"differences?\s+between\s+([A-Za-z0-9_\-\s]+?)\s+and\s+([A-Za-z0-9_\-\s]+)", original, re.IGNORECASE)
        compare_match = re.search(r"(?:compare|comparison\s+(?:of|between))\s+([A-Za-z0-9_\-\s]+?)\s+(?:and|with|to)\s+([A-Za-z0-9_\-\s]+)", original, re.IGNORECASE)
        comp_match = vs_match or diff_match or compare_match

        if comp_match:
            ent1 = comp_match.group(1).strip()
            ent2 = comp_match.group(2).strip()
            # Clean entity names
            ent1 = re.sub(r"^(compare|comparison\s+of|comparison\s+between|the)\s+", "", ent1, flags=re.IGNORECASE).strip()
            ent2 = re.sub(r"\s+(memory|specs|performance|bandwidth|capacity).*", "", ent2, flags=re.IGNORECASE).strip()

            sub_queries.append(SubQuery(
                query=f"{ent1} technical specifications memory bandwidth FP8",
                intent=f"Retrieve official hardware specifications and architecture for {ent1}",
                priority=1
            ))
            sub_queries.append(SubQuery(
                query=f"{ent2} technical specifications memory bandwidth FP8",
                intent=f"Retrieve official hardware specifications and architecture for {ent2}",
                priority=1
            ))
            sub_queries.append(SubQuery(
                query=f"{ent1} vs {ent2} comparison benchmark specs table",
                intent=f"Retrieve direct comparison tables and performance delta between {ent1} and {ent2}",
                priority=2
            ))
            return sub_queries[:3]

        # Case 2: Software / Driver / Architecture updates (e.g. "ROCm 6.2+ for LLM inference")
        if "rocm" in lower or "vllm" in lower or "inference" in lower or "release" in lower or "architecture" in lower:
            sub_queries.append(SubQuery(
                query=f"{cleaned} official release highlights architecture",
                intent="Retrieve official release notes and architectural updates",
                priority=1
            ))
            sub_queries.append(SubQuery(
                query=f"{cleaned} vllm performance throughput benchmarks",
                intent="Retrieve LLM inference throughput and framework acceleration metrics",
                priority=2
            ))
            return sub_queries

        # Case 3: Fictional / Hypothetical / Speculative Entities
        fictional_patterns = [
            r"19990", r"9999", r"quantum\s+gpu", r"fictional",
            r"mi[4-9]\d{2}", r"rx\s*[89]\d{3}", r"rtx\s*6\d{3}"
        ]
        if any(re.search(pat, lower) for pat in fictional_patterns):
            sub_queries.append(SubQuery(
                query=f'"{cleaned}" release date roadmap announcement',
                intent="Check for any official roadmap, patent, or announcement of entity",
                priority=1
            ))
            sub_queries.append(SubQuery(
                query=f"{cleaned} benchmark score hardware leaks",
                intent="Verify existence in hardware benchmark registries and tech publications",
                priority=2
            ))
            return sub_queries

        # General factual query
        sub_queries.append(SubQuery(
            query=f"{cleaned} specifications documentation",
            intent="Primary factual documentation and reference data retrieval",
            priority=1
        ))
        sub_queries.append(SubQuery(
            query=f"{cleaned} official benchmark overview",
            intent="Secondary validation and metric verification",
            priority=2
        ))
        return sub_queries

    def _llm_decompose(self, prompt: str, current_year: int) -> Optional[List[SubQuery]]:
        """Attempt LLM-guided query planning via OpenAI-compatible endpoint."""
        base_url = self.settings.amd_base_url if self.settings.inference_backend == "amd_cloud_vllm" else self.settings.openai_compatible_base_url
        api_key = self.settings.amd_api_key if self.settings.inference_backend == "amd_cloud_vllm" else self.settings.openai_compatible_api_key
        model = self.settings.amd_model if self.settings.inference_backend == "amd_cloud_vllm" else self.settings.openai_compatible_model

        system_prompt = (
            f"You are an expert search planner for an intelligent information retrieval agent.\n"
            f"Context Year: {current_year}.\n"
            f"Decompose the user question into 1 to 3 distinct search sub-queries optimized for factual retrieval.\n"
            f"Respond with ONLY a JSON array of objects with keys 'query', 'intent', and 'priority' (integer 1-3)."
        )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 400
        }

        try:
            url = f"{base_url.rstrip('/')}/chat/completions"
            resp = requests.post(url, headers=headers, json=payload, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Extract JSON array
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    results = []
                    for item in parsed:
                        if isinstance(item, dict) and "query" in item:
                            results.append(SubQuery(
                                query=str(item.get("query", "")),
                                intent=str(item.get("intent", "Targeted factual retrieval")),
                                priority=int(item.get("priority", 1))
                            ))
                    if results:
                        return results[:3]
        except Exception as exc:
            _ = exc

        return None
