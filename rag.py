"""
RAG System for Cameron's AI Assistant
Uses TF-IDF for retrieval + Claude API for generation
"""

import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def chunk_text(text, source_name, chunk_size=200, overlap=40):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source_name
        })
        i += chunk_size - overlap
    return chunks


def load_documents(doc_dir="documents"):
    """Load all .txt files from the documents directory."""
    chunks = []
    if not os.path.exists(doc_dir):
        return chunks
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(doc_dir, filename)
            source_name = filename.replace(".txt", "").replace("_", " ").title()
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                file_chunks = chunk_text(content, source_name)
                chunks.extend(file_chunks)
    return chunks


class RAGSystem:
    def __init__(self, doc_dir="documents"):
        self.doc_dir = doc_dir
        self.chunks = []
        self.vectorizer = None
        self.chunk_vectors = None
        self._build_index()

    def _build_index(self):
        """Load documents and build TF-IDF index."""
        self.chunks = load_documents(self.doc_dir)
        if not self.chunks:
            return
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
        self.chunk_vectors = self.vectorizer.fit_transform(texts)

    def refresh(self):
        """Reload documents — call if files change."""
        self._build_index()

    def retrieve(self, query, top_k=5):
        """Return top_k most relevant chunks for the query."""
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # relevance threshold
                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(scores[idx])
                })
        return results

    def build_context(self, query, top_k=5):
        """Build a context string from retrieved chunks."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return "No specific documents found for this query."
        context_parts = []
        for r in results:
            context_parts.append(f"[Source: {r['source']}]\n{r['text']}")
        return "\n\n---\n\n".join(context_parts)
