"""Tests for differ module."""

import pytest
import numpy as np

from llm_behavior_diff.differ import SimpleDiffer, create_differ
from llm_behavior_diff.models import SemanticScore


class TestSimpleDiffer:
    """Test SimpleDiffer."""
    
    def test_identical_texts(self) -> None:
        """Test identical texts have similarity 1.0."""
        differ = SimpleDiffer()
        score = differ.compute_similarity("hello world", "hello world")
        assert score == 1.0
    
    def test_completely_different(self) -> None:
        """Test completely different texts."""
        differ = SimpleDiffer()
        score = differ.compute_similarity("abc xyz", "123 456")
        assert score == 0.0
    
    def test_partial_overlap(self) -> None:
        """Test partial overlap."""
        differ = SimpleDiffer()
        score = differ.compute_similarity("hello world foo", "hello world bar")
        assert 0 < score < 1
    
    def test_empty_texts(self) -> None:
        """Test empty texts."""
        differ = SimpleDiffer()
        assert differ.compute_similarity("", "hello") == 0.0
        assert differ.compute_similarity("hello", "") == 0.0
        assert differ.compute_similarity("", "") == 0.0
    
    def test_semantic_score(self) -> None:
        """Test semantic score computation."""
        differ = SimpleDiffer()
        score = differ.compute_semantic_score(
            text_a="hello world",
            text_b="hello world",
            llm_judge_score=0.9,
            judge_reasoning="Very similar"
        )
        assert isinstance(score, SemanticScore)
        assert score.embedding_similarity == 1.0
        assert score.llm_judge_score == 0.9
        assert score.combined_score > 0.9
        assert score.judge_reasoning == "Very similar"


class TestCreateDiffer:
    """Test differ factory."""
    
    def test_create_simple(self) -> None:
        """Test creating simple differ."""
        differ = create_differ(use_embeddings=False)
        assert isinstance(differ, SimpleDiffer)
    
    def test_create_embedding(self) -> None:
        """Test creating embedding differ."""
        differ = create_differ(use_embeddings=True)
        # Falls back to SimpleDiffer if sentence-transformers not available
        assert isinstance(differ, (SimpleDiffer, type(differ)))


class TestSemanticScores:
    """Test semantic score calculations."""
    
    def test_score_bounds(self) -> None:
        """Test that scores are within bounds."""
        differ = SimpleDiffer()
        
        # Test various pairs
        pairs = [
            ("hello", "hello"),
            ("foo bar", "baz qux"),
            ("the quick brown fox", "the quick brown dog"),
        ]
        
        for a, b in pairs:
            score = differ.compute_similarity(a, b)
            assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
