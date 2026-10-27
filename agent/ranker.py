"""
Hybrid Lexical (BM25) & Semantic Chunk Reranker with Table Atomicity.
Chunks documents, scores them via hybrid BM25 + dense/n-gram vector cosine similarity,
deduplicates near-identical passages, and returns top K verified chunks.
"""

import math
import re
from typing import List, Dict, Set, Tuple
from agent.schemas import ScrapedDocument, RankedChunk
from config import get_settings

class HybridRanker:
    """Hybrid BM25 + Semantic passage and table ranker."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.k1 = 1.5
        self.b = 0.75

    def chunk_and_rank(self, documents: List[ScrapedDocument], query: str, top_k: int = 6) -> List[RankedChunk]:
        """
        1. Extract atomic tables and overlapping text passages.
        2. Score chunks with BM25 lexical and semantic similarity.
        3. Deduplicate near-identical chunks.
        4. Return top K highest-scoring chunks.
        """
        raw_chunks = self._create_chunks(documents)
        if not raw_chunks:
            return []

        # Compute BM25 scores
        bm25_scores = self._compute_bm25(raw_chunks, query)

        # Compute Semantic scores
        semantic_scores = self._compute_semantic_scores(raw_chunks, query)

        # Combine scores and assemble RankedChunk list
        ranked: List[RankedChunk] = []
        is_spec_query = any(k in query.lower() for k in ["spec", "compare", "table", "memory", "bandwidth", "vs", "fp8", "benchmark"])

        for i, chunk in enumerate(raw_chunks):
            b_score = bm25_scores[i]
            s_score = semantic_scores[i]
            # Table bonus if query seeks structured metrics/specs
            table_bonus = 0.08 if (chunk["is_table"] and is_spec_query) else 0.0
            combined = (0.55 * b_score) + (0.45 * s_score) + table_bonus
            combined = min(1.0, max(0.0, combined))

            ranked.append(RankedChunk(
                chunk_id=i + 1,
                url=chunk["url"],
                title=chunk["title"],
                text=chunk["text"],
                is_table=chunk["is_table"],
                bm25_score=round(b_score, 4),
                semantic_score=round(s_score, 4),
                combined_score=round(combined, 4)
            ))

        # Sort descending by combined score
        ranked.sort(key=lambda x: x.combined_score, reverse=True)

        # Deduplicate
        deduped = self._deduplicate(ranked)
        return deduped[:top_k]

    def _create_chunks(self, documents: List[ScrapedDocument]) -> List[Dict]:
        """Split text into overlapping ~350-word passages, preserving tables as atomic chunks."""
        chunks: List[Dict] = []

        for doc in documents:
            # 1. Atomic table chunks
            for table_md in doc.markdown_tables:
                chunks.append({
                    "url": doc.url,
                    "title": doc.title,
                    "text": table_md,
                    "is_table": True
                })

            # 2. Semantic text passages with 50-word overlap
            words = doc.clean_text.split()
            chunk_size = 350
            overlap = 50
            step = max(chunk_size - overlap, 100)

            if len(words) <= chunk_size:
                if words:
                    chunks.append({
                        "url": doc.url,
                        "title": doc.title,
                        "text": doc.clean_text,
                        "is_table": False
                    })
            else:
                for start in range(0, len(words), step):
                    passage_words = words[start:start + chunk_size]
                    if len(passage_words) >= 40:
                        passage = " ".join(passage_words)
                        chunks.append({
                            "url": doc.url,
                            "title": doc.title,
                            "text": passage,
                            "is_table": False
                        })
                    elif start > 0 and chunks:
                        # Append tail tokens to preceding chunk to preserve complete context
                        chunks[-1]["text"] += " " + " ".join(passage_words)

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into alphanumerics and numbers."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def _compute_bm25(self, chunks: List[Dict], query: str) -> List[float]:
        """Exact Okapi BM25 implementation with length normalization."""
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        query_tokens = self._tokenize(query)

        N = len(chunks)
        if N == 0 or not query_tokens:
            return [0.0] * N

        # Document lengths
        doc_lens = [len(doc) for doc in tokenized_corpus]
        total_tokens = sum(doc_lens)
        avgdl = (total_tokens / N) if (N > 0 and total_tokens > 0) else 1.0

        # Document frequencies for query terms
        df: Dict[str, int] = {}
        for token in query_tokens:
            df[token] = sum(1 for doc in tokenized_corpus if token in doc)

        max_possible_bm25 = sum(
            math.log(1.0 + (N - df.get(token, 0) + 0.5) / (df.get(token, 0) + 0.5)) * (self.k1 + 1.0)
            for token in set(query_tokens)
            if df.get(token, 0) > 0
        ) or 1.0

        scores: List[float] = []
        for i, doc in enumerate(tokenized_corpus):
            doc_len = doc_lens[i]
            score = 0.0
            term_freqs: Dict[str, int] = {}
            for t in doc:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            for token in query_tokens:
                n_q = df.get(token, 0)
                if n_q == 0:
                    continue
                # IDF formula with floor
                idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                f = term_freqs.get(token, 0)
                if f > 0:
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl))
                    score += idf * (numerator / denominator)

            coverage = len(set(query_tokens) & set(doc)) / len(set(query_tokens))
            norm_bm25 = min(1.0, score / max_possible_bm25)
            combined_bm25 = 0.5 * norm_bm25 + 0.5 * coverage
            scores.append(round(combined_bm25, 4))

        return scores

    def _compute_semantic_scores(self, chunks: List[Dict], query: str) -> List[float]:
        """Dense embedding or character/subword n-gram TF-IDF cosine similarity."""
        # Try local sentence-transformers if installed
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode(query, convert_to_tensor=True)
            c_texts = [c["text"][:1000] for c in chunks]
            c_emb = model.encode(c_texts, convert_to_tensor=True)
            sims = util.cos_sim(q_emb, c_emb)[0].tolist()
            return [max(0.0, float(s)) for s in sims]
        except Exception as exc:
            _ = exc

        # Robust n-gram Subword Cosine Similarity
        def get_char_ngrams(s: str, n=3) -> Dict[str, float]:
            s_clean = re.sub(r"\s+", " ", s.lower()).strip()
            vec: Dict[str, float] = {}
            for i in range(len(s_clean) - n + 1):
                gram = s_clean[i:i+n]
                vec[gram] = vec.get(gram, 0.0) + 1.0
            if not vec and s_clean:
                vec[s_clean] = 1.0
            # Normalize vector
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                for k in vec:
                    vec[k] /= norm
            return vec

        q_vec = get_char_ngrams(query)
        sim_scores: List[float] = []

        for c in chunks:
            c_vec = get_char_ngrams(c["text"][:1200])
            dot = sum(q_vec[k] * c_vec[k] for k in q_vec if k in c_vec)
            sim_scores.append(round(max(0.0, min(1.0, dot)), 4))

        return sim_scores

    def _deduplicate(self, ranked: List[RankedChunk], threshold: float = 0.82) -> List[RankedChunk]:
        """Filter out near-duplicate chunks across domains."""
        kept: List[RankedChunk] = []

        for cand in ranked:
            cand_tokens = set(self._tokenize(cand.text[:500]))
            is_dup = False
            for k in kept:
                # Never deduplicate a table against a text passage or vice versa
                if cand.is_table != k.is_table:
                    continue
                k_tokens = set(self._tokenize(k.text[:500]))
                if not cand_tokens or not k_tokens:
                    continue
                overlap = len(cand_tokens & k_tokens) / float(len(cand_tokens | k_tokens))
                if overlap >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)

        return kept
