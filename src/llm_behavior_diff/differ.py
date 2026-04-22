"""Semantic diff engine using embeddings."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from llm_behavior_diff.models import SemanticScore


class EmbeddingDiffer:
    """Compute semantic similarity using embeddings."""
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize embedding differ.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding-based diff. "
                "Install with: pip install sentence-transformers"
            )
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy load the embedding model.
        
        Returns:
            Loaded sentence transformer model
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Array of embeddings
        """
        model = self._load_model()
        # Normalize embeddings for cosine similarity
        return model.encode(texts, normalize_embeddings=True)
    
    def compute_similarity(
        self,
        text_a: str,
        text_b: str
    ) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not text_a.strip() or not text_b.strip():
            return 0.0
        
        embeddings = self.encode([text_a, text_b])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def compute_similarity_batch(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Compute similarities for multiple pairs efficiently.
        
        Args:
            pairs: List of (text_a, text_b) tuples
            
        Returns:
            List of similarity scores
        """
        if not pairs:
            return []
        
        # Collect all unique texts
        all_texts = []
        text_to_idx = {}
        for text_a, text_b in pairs:
            if text_a not in text_to_idx:
                text_to_idx[text_a] = len(all_texts)
                all_texts.append(text_a)
            if text_b not in text_to_idx:
                text_to_idx[text_b] = len(all_texts)
                all_texts.append(text_b)
        
        # Encode all texts
        embeddings = self.encode(all_texts)
        
        # Compute similarities
        similarities = []
        for text_a, text_b in pairs:
            idx_a = text_to_idx[text_a]
            idx_b = text_to_idx[text_b]
            sim = cosine_similarity([embeddings[idx_a]], [embeddings[idx_b]])[0][0]
            similarities.append(float(sim))
        
        return similarities
    
    def compute_semantic_score(
        self,
        text_a: str,
        text_b: str,
        llm_judge_score: Optional[float] = None,
        judge_reasoning: Optional[str] = None,
        embedding_weight: float = 1.0,
        judge_weight: float = 0.0
    ) -> SemanticScore:
        """Compute comprehensive semantic score.
        
        Args:
            text_a: First text
            text_b: Second text
            llm_judge_score: Optional LLM judge score
            judge_reasoning: Optional reasoning from judge
            embedding_weight: Weight for embedding similarity
            judge_weight: Weight for judge score
            
        Returns:
            SemanticScore with combined metrics
        """
        embedding_sim = self.compute_similarity(text_a, text_b)
        
        # Calculate combined score
        total_weight = embedding_weight
        if llm_judge_score is not None:
            total_weight += judge_weight
        
        if total_weight == 0:
            combined = embedding_sim
        else:
            combined = (embedding_weight * embedding_sim) / total_weight
            if llm_judge_score is not None:
                combined += (judge_weight * llm_judge_score) / total_weight
        
        return SemanticScore(
            embedding_similarity=embedding_sim,
            llm_judge_score=llm_judge_score,
            combined_score=min(1.0, max(0.0, combined)),
            judge_reasoning=judge_reasoning
        )


class SimpleDiffer:
    """Simple differ using basic text similarity (no ML required)."""
    
    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between token sets.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Jaccard similarity score
        """
        if not text_a.strip() or not text_b.strip():
            return 0.0
        
        # Tokenize by words (simple approach)
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        
        if not tokens_a or not tokens_b:
            return 0.0
        
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        
        return len(intersection) / len(union)
    
    def compute_semantic_score(
        self,
        text_a: str,
        text_b: str,
        llm_judge_score: Optional[float] = None,
        judge_reasoning: Optional[str] = None
    ) -> SemanticScore:
        """Compute semantic score using simple similarity.
        
        Args:
            text_a: First text
            text_b: Second text
            llm_judge_score: Optional LLM judge score
            judge_reasoning: Optional reasoning
            
        Returns:
            SemanticScore
        """
        simple_sim = self.compute_similarity(text_a, text_b)
        
        # Combine with judge score if available
        if llm_judge_score is not None:
            combined = (simple_sim + llm_judge_score) / 2
        else:
            combined = simple_sim
        
        return SemanticScore(
            embedding_similarity=simple_sim,
            llm_judge_score=llm_judge_score,
            combined_score=min(1.0, max(0.0, combined)),
            judge_reasoning=judge_reasoning
        )


def create_differ(
    use_embeddings: bool = True,
    model_name: Optional[str] = None
) -> EmbeddingDiffer | SimpleDiffer:
    """Factory function to create appropriate differ.
    
    Args:
        use_embeddings: Whether to use embedding-based differ
        model_name: Model name for embedding differ
        
    Returns:
        Differ instance
    """
    if use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE:
        return EmbeddingDiffer(model_name)
    return SimpleDiffer()
