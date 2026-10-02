"""
Embedding service for semantic search in RAG memory.

Supports multiple backends:
- Ollama (local, recommended)
- HuggingFace API (cloud)
- sentence-transformers (local Python)
- Mock (for testing, returns zero vectors)
"""
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Abstract base for embedding services."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dim = 384  # Default dimension
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return np.array([self.embed(t) for t in texts])


class MockEmbedding(EmbeddingService):
    """Mock embedding for testing (returns zero vectors)."""
    
    def __init__(self, dim: int = 384):
        super().__init__("mock")
        self._dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        # Return a deterministic "embedding" based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self._dim).astype(np.float32) * 0.1


class OllamaEmbedding(EmbeddingService):
    """
    Ollama-based embedding service.
    
    Requires Ollama running locally with a supported model:
        ollama pull nomic-embed-text  # or mxbai-embed-large, all-minilm
    
    Usage:
        embedder = OllamaEmbedding("nomic-embed-text")
        embedding = embedder.embed("hello world")
    """
    
    def __init__(self, model_name: str = "nomic-embed-text", 
                 base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        self._dim = self._detect_dimension()
    
    def _detect_dimension(self) -> int:
        """Detect embedding dimension from model."""
        try:
            import requests
            # Try to get model info
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # nomic-embed-text: 768, all-minilm: 384, mxbai-embed-large: 1024
                model_dims = {
                    "nomic-embed-text": 768,
                    "nomic-embed-text:v1.5": 768,
                    "mxbai-embed-large": 1024,
                    "all-minilm": 384,
                    "snowflake-arctic-embed": 1024,
                }
                return model_dims.get(self.model_name, 768)
        except Exception as e:
            logger.debug("Could not detect Ollama model dimension: %s", e)
        return 768  # Default for nomic-embed-text
    
    def embed(self, text: str) -> np.ndarray:
        try:
            import requests
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return np.array(data["embedding"], dtype=np.float32)
        except Exception as e:
            logger.warning("Ollama embedding failed: %s", e)
            # Fallback to mock
            return MockEmbedding(self._dim).embed(text)


class HuggingFaceEmbedding(EmbeddingService):
    """
    HuggingFace Inference API embedding service.
    
    Requires HF_API_KEY environment variable.
    
    Usage:
        embedder = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
        embedding = embedder.embed("hello world")
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("HF_API_KEY", "")
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self._dim = self._get_model_dim(model_name)
    
    def _get_model_dim(self, model_name: str) -> int:
        """Get known dimension for common models."""
        dims = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        return dims.get(model_name, 384)
    
    def embed(self, text: str) -> np.ndarray:
        try:
            import requests
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # HF returns list of embeddings for each input
            if isinstance(data, list) and len(data) > 0:
                return np.array(data[0], dtype=np.float32)
            return MockEmbedding(self._dim).embed(text)
        except Exception as e:
            logger.warning("HuggingFace embedding failed: %s", e)
            return MockEmbedding(self._dim).embed(text)


class SentenceTransformersEmbedding(EmbeddingService):
    """
    Local sentence-transformers embedding service.
    
    Requires: pip install sentence-transformers
    
    Usage:
        embedder = SentenceTransformersEmbedding("all-MiniLM-L6-v2")
        embedding = embedder.embed("hello world")
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self._model = None
        self._dim = self._get_model_dim(model_name)
    
    def _get_model_dim(self, model_name: str) -> int:
        """Get known dimension for common models."""
        dims = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
        }
        return dims.get(model_name, 384)
    
    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed")
                self._model = "mock"
    
    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        
        if self._model == "mock" or self._model is None:
            return MockEmbedding(self._dim).embed(text)
        
        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning("sentence-transformers failed: %s", e)
            return MockEmbedding(self._dim).embed(text)


# =============================================================================
# Factory function
# =============================================================================

def create_embedder(backend: str = "auto", model_name: Optional[str] = None) -> EmbeddingService:
    """
    Create an embedding service.
    
    Args:
        backend: "auto", "ollama", "huggingface", "sentence-transformers", "mock"
        model_name: Optional model name override
    
    Returns:
        EmbeddingService instance
    """
    if backend == "auto":
        # Try in order of preference
        try:
            # Try Ollama first (fastest if running)
            import requests
            requests.get("http://localhost:11434/api/tags", timeout=2)
            logger.info("Using Ollama for embeddings")
            return OllamaEmbedding(model_name or "nomic-embed-text")
        except:
            pass
        
        try:
            # Try sentence-transformers
            from sentence_transformers import SentenceTransformer
            logger.info("Using sentence-transformers for embeddings")
            return SentenceTransformersEmbedding(model_name or "all-MiniLM-L6-v2")
        except ImportError:
            pass
        
        # Fall back to mock
        logger.info("Using mock embeddings (no backend available)")
        return MockEmbedding(384)
    
    elif backend == "ollama":
        return OllamaEmbedding(model_name or "nomic-embed-text")
    
    elif backend == "huggingface":
        return HuggingFaceEmbedding(model_name or "BAAI/bge-small-en-v1.5")
    
    elif backend == "sentence-transformers":
        return SentenceTransformersEmbedding(model_name or "all-MiniLM-L6-v2")
    
    elif backend == "mock":
        return MockEmbedding(384)
    
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


# =============================================================================
# Similarity functions
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def top_k_similar(query_embedding: np.ndarray, 
                  doc_embeddings: np.ndarray,
                  k: int = 5) -> List[int]:
    """
    Find indices of top-k most similar documents.
    
    Args:
        query_embedding: Query vector (dim,)
        doc_embeddings: Document matrix (n_docs, dim)
        k: Number of results
    
    Returns:
        List of indices sorted by similarity (descending)
    """
    # Normalize
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity via matrix multiplication
    similarities = np.dot(doc_norms, query_norm)
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:k]
    return top_indices.tolist()
