import pytest
from datetime import datetime
from src.llm_behavior_diff.models import (
    ComparisonRun, ModelConfig, LLMResponse, DiffResult, 
    SemanticScore, Settings, ProviderType, PromptSuite, Prompt, PromptCategory, ReportConfig
)
from src.llm_behavior_diff.report import ReportGenerator
import os

@pytest.fixture
def mock_run():
    model_a = ModelConfig(name="gpt-3.5-turbo", provider=ProviderType.OPENROUTER)
    model_b = ModelConfig(name="gpt-4", provider=ProviderType.OPENROUTER)
    
    suite = PromptSuite(
        name="Test Suite",
        prompts=[
            Prompt(id="p1", text="Hello", category=PromptCategory.CONVERSATIONAL),
            Prompt(id="p2", text="Write a poem", category=PromptCategory.CREATIVITY)
        ]
    )
    
    results = [
        DiffResult(
            prompt_id="p1",
            prompt_text="Hello",
            response_a=LLMResponse(
                prompt_id="p1", 
                text="Hi there", 
                model_config_used=model_a
            ),
            response_b=LLMResponse(
                prompt_id="p1", 
                text="Hello! How can I help?", 
                model_config_used=model_b
            ),
            semantic_score=SemanticScore(
                embedding_similarity=0.8, 
                llm_judge_score=0.7, 
                combined_score=0.75, 
                judge_reasoning="Slightly different phrasing"
            ),
            behavioral_change_detected=False,
            change_severity="none"
        ),
        DiffResult(
            prompt_id="p2",
            prompt_text="Write a poem",
            response_a=LLMResponse(
                prompt_id="p2", 
                text="Roses are red", 
                model_config_used=model_a
            ),
            response_b=LLMResponse(
                prompt_id="p2", 
                text="Violets are blue", 
                model_config_used=model_b
            ),
            semantic_score=SemanticScore(
                embedding_similarity=0.2, 
                llm_judge_score=0.1, 
                combined_score=0.15, 
                judge_reasoning="Completely different content"
            ),
            behavioral_change_detected=True,
            change_severity="major"
        )
    ]
    
    return ComparisonRun(
        id="test-run",
        model_a=model_a,
        model_b=model_b,
        prompt_suite=suite,
        results=results,
        started_at=datetime.utcnow()
    )

@pytest.fixture
def settings():
    # We use a standard Settings object. 
    # We will NOT try to set a 'title' attribute on it since it's not in the model.
    return Settings()

def test_report_stats(mock_run):
    generator = ReportGenerator(templates_dir="templates")
    stats = generator.generate_stats(mock_run)
    
    assert stats["total_prompts"] == 2
    assert stats["behavioral_changes"] == 1
    assert stats["change_rate"] == 0.5
    assert stats["avg_similarity"] == 0.45

def test_report_generation(mock_run, settings, tmp_path):
    project_root = "/app/llm_behavior_diff_2355"
    generator = ReportGenerator(templates_dir=os.path.join(project_root, "templates"))
    
    output_file = tmp_path / "report.html"
    report_obj = generator.save_report(mock_run, settings, str(output_file))
    
    assert os.path.exists(str(output_file))
    summary = report_obj.get_summary_stats()
    assert summary["total_prompts"] == 2
    
    with open(output_file, "r") as f:
        content = f.read()
        # The default title should be present
        assert "LLM Behavior Diff Report" in content
        assert "gpt-3.5-turbo" in content
        assert "gpt-4" in content
        assert "Roses are red" in content
        assert "Violets are blue" in content
