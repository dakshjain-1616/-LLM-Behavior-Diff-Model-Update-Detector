"""Tests for Pydantic data models."""

import pytest
from datetime import datetime
from pathlib import Path

from llm_behavior_diff.models import (
    Prompt,
    PromptSuite,
    ModelConfig,
    LLMResponse,
    SemanticScore,
    DiffResult,
    ComparisonRun,
    ReportConfig,
    Report,
    Settings,
    ProviderType,
    PromptCategory,
)


class TestPrompt:
    """Test Prompt model."""
    
    def test_valid_prompt(self) -> None:
        """Test creating a valid prompt."""
        prompt = Prompt(
            id="test-1",
            text="What is 2+2?",
            category=PromptCategory.REASONING,
            tags=["math", "simple"],
            expected_behavior="Should answer 4"
        )
        assert prompt.id == "test-1"
        assert prompt.text == "What is 2+2?"
        assert prompt.category == PromptCategory.REASONING
        assert "math" in prompt.tags
        
    def test_empty_id_raises(self) -> None:
        """Test that empty ID raises validation error."""
        with pytest.raises(ValueError):
            Prompt(id="", text="Hello", category=PromptCategory.CONVERSATIONAL)
    
    def test_empty_text_raises(self) -> None:
        """Test that empty text raises validation error."""
        with pytest.raises(ValueError):
            Prompt(id="test", text="", category=PromptCategory.CONVERSATIONAL)


class TestPromptSuite:
    """Test PromptSuite model."""
    
    def test_valid_suite(self) -> None:
        """Test creating a valid prompt suite."""
        prompts = [
            Prompt(id=f"p{i}", text=f"Prompt {i}", category=PromptCategory.FACTUAL)
            for i in range(3)
        ]
        suite = PromptSuite(
            name="Test Suite",
            version="1.0.0",
            description="A test suite",
            prompts=prompts
        )
        assert suite.name == "Test Suite"
        assert len(suite.prompts) == 3
        assert suite.created_at is not None
    
    def test_duplicate_ids_raises(self) -> None:
        """Test that duplicate IDs raise validation error."""
        prompts = [
            Prompt(id="same-id", text="Prompt 1", category=PromptCategory.FACTUAL),
            Prompt(id="same-id", text="Prompt 2", category=PromptCategory.FACTUAL),
        ]
        with pytest.raises(ValueError, match="Duplicate prompt IDs"):
            PromptSuite(name="Test", prompts=prompts)
    
    def test_empty_prompts_raises(self) -> None:
        """Test that empty prompts list raises validation error."""
        with pytest.raises(ValueError):
            PromptSuite(name="Test", prompts=[])


class TestModelConfig:
    """Test ModelConfig model."""
    
    def test_valid_config(self) -> None:
        """Test creating a valid model config."""
        config = ModelConfig(
            name="llama2",
            provider=ProviderType.OLLAMA,
            temperature=0.7,
            max_tokens=100,
            system_prompt="You are a helpful assistant"
        )
        assert config.name == "llama2"
        assert config.provider == ProviderType.OLLAMA
        assert config.temperature == 0.7
    
    def test_temperature_bounds(self) -> None:
        """Test temperature bounds validation."""
        with pytest.raises(ValueError):
            ModelConfig(name="test", provider=ProviderType.OLLAMA, temperature=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(name="test", provider=ProviderType.OLLAMA, temperature=2.1)


class TestLLMResponse:
    """Test LLMResponse model."""
    
    def test_successful_response(self) -> None:
        """Test a successful response."""
        config = ModelConfig(name="test", provider=ProviderType.OLLAMA)
        response = LLMResponse(
            prompt_id="p1",
            model_config_used=config,
            text="The answer is 4",
            tokens_used=10,
            latency_ms=150.5
        )
        assert response.success is True
        assert response.error is None
        assert response.text == "The answer is 4"
    
    def test_failed_response(self) -> None:
        """Test a failed response."""
        config = ModelConfig(name="test", provider=ProviderType.OLLAMA)
        response = LLMResponse(
            prompt_id="p1",
            model_config_used=config,
            text="",
            error="Connection timeout"
        )
        assert response.success is False
        assert response.error == "Connection timeout"


class TestSemanticScore:
    """Test SemanticScore model."""
    
    def test_valid_score(self) -> None:
        """Test creating a valid semantic score."""
        score = SemanticScore(
            embedding_similarity=0.85,
            llm_judge_score=0.90,
            combined_score=0.87,
            judge_reasoning="Responses are very similar"
        )
        assert score.embedding_similarity == 0.85
        assert score.combined_score == 0.87
    
    def test_score_bounds(self) -> None:
        """Test score bounds validation."""
        with pytest.raises(ValueError):
            SemanticScore(embedding_similarity=1.1, combined_score=0.5)
        with pytest.raises(ValueError):
            SemanticScore(embedding_similarity=0.5, combined_score=-0.1)


class TestDiffResult:
    """Test DiffResult model."""
    
    def test_valid_result(self) -> None:
        """Test creating a valid diff result."""
        config_a = ModelConfig(name="model-a", provider=ProviderType.OLLAMA)
        config_b = ModelConfig(name="model-b", provider=ProviderType.OLLAMA)
        
        response_a = LLMResponse(prompt_id="p1", model_config_used=config_a, text="Answer A")
        response_b = LLMResponse(prompt_id="p1", model_config_used=config_b, text="Answer B")
        
        score = SemanticScore(embedding_similarity=0.8, combined_score=0.82)
        
        result = DiffResult(
            prompt_id="p1",
            prompt_text="What is 2+2?",
            response_a=response_a,
            response_b=response_b,
            semantic_score=score,
            behavioral_change_detected=True,
            change_severity="minor"
        )
        assert result.behavioral_change_detected is True
        assert result.change_severity == "minor"
    
    def test_invalid_severity(self) -> None:
        """Test that invalid severity raises error."""
        config = ModelConfig(name="test", provider=ProviderType.OLLAMA)
        response = LLMResponse(prompt_id="p1", model_config_used=config, text="Test")
        score = SemanticScore(embedding_similarity=0.5, combined_score=0.5)
        
        with pytest.raises(ValueError):
            DiffResult(
                prompt_id="p1",
                prompt_text="Test",
                response_a=response,
                response_b=response,
                semantic_score=score,
                behavioral_change_detected=False,
                change_severity="invalid"
            )


class TestComparisonRun:
    """Test ComparisonRun model."""
    
    def test_run_stats(self) -> None:
        """Test run statistics calculation."""
        config_a = ModelConfig(name="model-a", provider=ProviderType.OLLAMA)
        config_b = ModelConfig(name="model-b", provider=ProviderType.OLLAMA)
        
        prompts = [Prompt(id=f"p{i}", text=f"Prompt {i}", category=PromptCategory.FACTUAL) for i in range(4)]
        suite = PromptSuite(name="Test", prompts=prompts)
        
        run = ComparisonRun(
            id="run-1",
            model_a=config_a,
            model_b=config_b,
            prompt_suite=suite,
            results=[]
        )
        
        assert run.behavioral_change_rate == 0.0
        assert run.average_similarity == 0.0


class TestReportConfig:
    """Test ReportConfig model."""
    
    def test_default_config(self) -> None:
        """Test default report configuration."""
        config = ReportConfig()
        assert config.title == "LLM Behavior Diff Report"
        assert config.theme == "light"
        assert config.include_raw_responses is False


class TestSettings:
    """Test Settings model."""
    
    def test_default_settings(self) -> None:
        """Test default settings."""
        settings = Settings()
        assert settings.ollama_host == "http://localhost:11434"
        assert settings.similarity_threshold == 0.85
        assert isinstance(settings.output_dir, Path)
    
    def test_custom_output_dir(self) -> None:
        """Test custom output directory."""
        settings = Settings(output_dir="/custom/path")
        assert settings.output_dir == Path("/custom/path")
    
    def test_threshold_bounds(self) -> None:
        """Test threshold bounds."""
        with pytest.raises(ValueError):
            Settings(similarity_threshold=1.1)


class TestProviderType:
    """Test ProviderType enum."""
    
    def test_enum_values(self) -> None:
        """Test enum values."""
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.OPENROUTER.value == "openrouter"


class TestPromptCategory:
    """Test PromptCategory enum."""
    
    def test_enum_values(self) -> None:
        """Test enum values."""
        assert PromptCategory.REASONING.value == "reasoning"
        assert PromptCategory.CODING.value == "coding"
        assert PromptCategory.SAFETY.value == "safety"
