import pytest
from datetime import datetime
from src.llm_behavior_diff.models import (
    ComparisonRun, ModelConfig, LLMResponse, DiffResult, 
    SemanticScore, Settings, ProviderType
)
from src.llm_behavior_diff.report import ReportGenerator
import os

@pytest.fixture
def mock_run():
    model_a = ModelConfig(name="gpt-3.5-turbo", provider=ProviderType.OPENROUTER)
    model_b = ModelConfig(name="gpt-4", provider=ProviderType.OPENROUTER)
    
    results = [
        DiffResult(
            prompt_id="p1",
            prompt_text="Hello",
            response_a=LLMResponse(text="Hi there", model="gpt-3.5-turbo"),
            response_b=LLMResponse(text="Hello! How can I help?", model="gpt-4"),
            semantic_score=SemanticScore(embedding_score=0.8, judge_score=0.7, combined_score=0.75, judge_reasoning="Slightly different phrasing"),
            behavioral_change_detected=False,
            change_severity="none"
        ),
        DiffResult(
            prompt_id="p2",
            prompt_text="Write a poem",
            response_a=LLMResponse(text="Roses are red", model="gpt-3.5-turbo"),
            response_b=LLMResponse(text="Violets are blue", model="gpt-4"),
            semantic_score=SemanticScore(embedding_score=0.2, judge_score=0.1, combined_score=0.15, judge_reasoning="Completely different content"),
            behavioral_change_detected=True,
            change_severity="major"
        )
    ]
    
    return ComparisonRun(
        run_id="test-run",
        model_a=model_a,
        model_b=model_b,
        results=results,
        started_at=datetime.now(),
        completed_at=datetime.now()
    )

@pytest.fixture
def settings():
    return Settings(title="Test Report")

def test_report_stats(mock_run):
    generator = ReportGenerator(templates_dir="templates")
    stats = generator.generate_stats(mock_run)
    
    assert stats["total_prompts"] == 2
    assert stats["behavioral_changes"] == 1
    assert stats["change_rate"] == 0.5
    assert stats["avg_similarity"] == 0.45

def test_report_generation(mock_run, settings, tmp_path):
    # Use the actual templates directory from the project root
    project_root = "/app/llm_behavior_diff_2355"
    generator = ReportGenerator(templates_dir=os.path.join(project_root, "templates"))
    
    output_file = tmp_path / "report.html"
    report = generator.save_report(mock_run, settings, str(output_file))
    
    assert os.path.exists(str(output_file))
    assert report.summary_stats["total_prompts"] == 2
    
    with open(output_file, "r") as f:
        content = f.read()
        assert "Test Report" in content
        assert "gpt-3.5-turbo" in content
        assert "gpt-4" in content
        assert "Roses are red" in content
        assert "Violets are blue" in content
