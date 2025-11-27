"""RAG retrieval system using TF-IDF."""
import re
from pathlib import Path
from typing import List, Dict
from collections import Counter
import math


class Chunk:
    """Represents a document chunk."""
    def __init__(self, chunk_id: str, content: str, source: str, score: float = 0.0):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.score = score
    
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, source={self.source}, score={self.score:.3f})"


class TFIDFRetriever:
    """TF-IDF based retriever for document chunks."""
    
    def __init__(self, docs_dir: str):
        """Initialize retriever with documents directory."""
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Chunk] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.chunk_vectors: List[Dict[str, float]] = []
        self._load_documents()
        self._build_index()
    
    def _load_documents(self):
        """Load and chunk documents."""
        chunk_idx = 0
        
        for doc_file in self.docs_dir.glob("*.md"):
            content = doc_file.read_text()
            source = doc_file.stem
            
            # Split into paragraphs (chunks)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for para_idx, para in enumerate(paragraphs):
                chunk_id = f"{source}::chunk{para_idx}"
                self.chunks.append(Chunk(chunk_id, para, source))
                chunk_idx += 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _build_index(self):
        """Build TF-IDF index."""
        # Build vocabulary
        doc_freq: Dict[str, int] = Counter()
        all_docs_tokens: List[List[str]] = []
        
        for chunk in self.chunks:
            tokens = self._tokenize(chunk.content)
            all_docs_tokens.append(tokens)
            doc_freq.update(set(tokens))  # Count documents containing each term
        
        # Build vocabulary
        self.vocab = {term: idx for idx, term in enumerate(sorted(set(doc_freq.keys())))}
        
        # Calculate IDF
        num_docs = len(self.chunks)
        self.idf = {
            term: math.log(num_docs / (doc_freq[term] + 1))
            for term in self.vocab.keys()
        }
        
        # Build TF vectors for each chunk
        for tokens in all_docs_tokens:
            term_freq = Counter(tokens)
            max_freq = max(term_freq.values()) if term_freq else 1
            
            vector = {}
            for term, freq in term_freq.items():
                if term in self.vocab:
                    tf = freq / max_freq
                    vector[term] = tf * self.idf[term]
            
            self.chunk_vectors.append(vector)
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        """
        Retrieve top-k chunks for a query.
        
        Args:
            query: Query string
            top_k: Number of chunks to return
        
        Returns:
            List of Chunk objects sorted by relevance score
        """
        query_tokens = self._tokenize(query)
        query_term_freq = Counter(query_tokens)
        max_freq = max(query_term_freq.values()) if query_term_freq else 1
        
        # Build query vector
        query_vector = {}
        for term, freq in query_term_freq.items():
            if term in self.vocab:
                tf = freq / max_freq
                query_vector[term] = tf * self.idf.get(term, 0)
        
        # Calculate similarity scores
        scored_chunks = []
        for idx, chunk_vector in enumerate(self.chunk_vectors):
            score = self._cosine_similarity(query_vector, chunk_vector)
            chunk = self.chunks[idx]
            chunk.score = score
            scored_chunks.append(chunk)
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]




