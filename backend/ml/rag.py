import os
import numpy as np
from backend.ml.knowledge_base import MEDICAL_KNOWLEDGE

# Lightweight RAG using simple TF-IDF style matching
# No heavy ML models needed — works on free tier

def _tokenize(text: str) -> set:
    """Simple word tokenizer"""
    return set(text.lower().split())

def _similarity(query_tokens: set, doc_tokens: set) -> float:
    """Jaccard similarity between two token sets"""
    if not query_tokens or not doc_tokens:
        return 0.0
    intersection = query_tokens & doc_tokens
    union = query_tokens | doc_tokens
    return len(intersection) / len(union)

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve top_k relevant medical documents for a query using keyword matching"""
    query_tokens = _tokenize(query)

    scored = []
    for doc in MEDICAL_KNOWLEDGE:
        doc_tokens = _tokenize(doc["content"])
        score = _similarity(query_tokens, doc_tokens)
        scored.append((score, doc))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = scored[:top_k]

    retrieved = []
    for score, doc in top_docs:
        if score > 0:
            retrieved.append(f"[{doc['topic'].upper()}]: {doc['content']}")

    return "\n\n".join(retrieved) if retrieved else ""