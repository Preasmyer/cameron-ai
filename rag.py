"""
RAG System for Cameron's AI Assistant
Uses sentence-transformers for semantic embedding retrieval + Claude API for generation
"""

import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def chunk_text(text, source_name, chunk_size=200, overlap=40):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text_str,
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
        self.embeddings = None
        self.model = None
        self._load_model()
        self._build_index()

    def _load_model(self):
        """Load the sentence transformer model once."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _build_index(self):
        """Load documents and build semantic embedding index."""
        self.chunks = load_documents(self.doc_dir)
        if not self.chunks:
            return
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False)

    def refresh(self):
        """Reload documents — call if files change."""
        self._build_index()

    def retrieve(self, query, top_k=5):
        """Return top_k most semantically relevant chunks for the query."""
        if not self.chunks or self.embeddings is None:
            return []
        query_embedding = self.model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:
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
