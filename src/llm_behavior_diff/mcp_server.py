"""MCP Server for LLM Behavior Diff — Model Update Detector."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .models import (
    ComparisonRun,
    DiffResult,
    ModelConfig,
    PromptSuite,
    ProviderType,
    Report,
    ReportConfig,
    SemanticScore,
    Settings,
)
from .runner import LLMRunner
from .differ import create_differ
from .judge import CombinedScorer, LLMJudge
from .report import ReportGenerator

# Initialize FastMCP server
mcp = FastMCP("llm-behavior-diff")


class CompareModelsRequest(BaseModel):
    """Request to compare two models."""
    model_a: str = Field(..., description="Model A identifier (e.g., 'qwen3:8b')")
    provider_a: str = Field(default="ollama", description="Provider for Model A: 'ollama' or 'openrouter'")
    model_b: str = Field(..., description="Model B identifier (e.g., 'gemma4:e4b')")
    provider_b: str = Field(default="ollama", description="Provider for Model B: 'ollama' or 'openrouter'")
    prompts_path: str = Field(default="prompts/default.yaml", description="Path to prompt suite YAML file")
    threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity threshold for change detection")
    use_judge: bool = Field(default=True, description="Use LLM judge for scoring")


class CompareModelsResponse(BaseModel):
    """Response from model comparison."""
    success: bool = Field(..., description="Whether comparison succeeded")
    run_id: Optional[str] = Field(None, description="Unique run identifier")
    total_prompts: int = Field(default=0, description="Number of prompts tested")
    changes_detected: int = Field(default=0, description="Number of behavioral changes detected")
    change_rate: float = Field(default=0.0, description="Percentage of prompts with changes")
    avg_similarity: float = Field(default=0.0, description="Average semantic similarity score")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed comparison results")
    error: Optional[str] = Field(None, description="Error message if failed")


class AnalyzeDriftRequest(BaseModel):
    """Request to analyze drift between two model responses."""
    prompt_text: str = Field(..., description="The prompt text")
    response_a: str = Field(..., description="Response from Model A")
    response_b: str = Field(..., description="Response from Model B")
    use_embeddings: bool = Field(default=True, description="Use embedding-based similarity")
    use_judge: bool = Field(default=False, description="Use LLM judge (requires OpenRouter API key)")


class AnalyzeDriftResponse(BaseModel):
    """Response from drift analysis."""
    success: bool = Field(..., description="Whether analysis succeeded")
    embedding_similarity: Optional[float] = Field(None, description="Embedding-based similarity (0-1)")
    judge_score: Optional[float] = Field(None, description="LLM judge score (0-1)")
    combined_score: float = Field(default=0.0, description="Combined similarity score (0-1)")
    behavioral_change_detected: bool = Field(default=False, description="Whether significant change detected")
    severity: str = Field(default="none", description="Change severity: none, minor, moderate, major")
    judge_reasoning: Optional[str] = Field(None, description="Reasoning from LLM judge")
    error: Optional[str] = Field(None, description="Error message if failed")


class GenerateReportRequest(BaseModel):
    """Request to generate a report from comparison results."""
    results_json: str = Field(..., description="JSON string of comparison results")
    output_path: str = Field(..., description="Path to save HTML report")
    title: str = Field(default="LLM Behavior Diff Report", description="Report title")
    include_raw_responses: bool = Field(default=False, description="Include full response text in report")


class GenerateReportResponse(BaseModel):
    """Response from report generation."""
    success: bool = Field(..., description="Whether report generation succeeded")
    output_path: Optional[str] = Field(None, description="Path where report was saved")
    html_preview: Optional[str] = Field(None, description="Preview of HTML content (first 1000 chars)")
    error: Optional[str] = Field(None, description="Error message if failed")


def _load_settings() -> Settings:
    """Load application settings from environment."""
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


def _load_prompt_suite(path: str) -> PromptSuite:
    """Load a prompt suite from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return PromptSuite(**data)


@mcp.tool()
async def compare_models(request: CompareModelsRequest) -> CompareModelsResponse:
    """Compare two LLM models using a prompt suite.
    
    Runs the same set of prompts through both models and detects behavioral changes
    using semantic similarity and optional LLM-as-judge scoring.
    
    Args:
        request: Comparison request with model configurations and options
        
    Returns:
        Comparison results with change statistics and per-prompt details
    """
    settings = _load_settings()
    
    try:
        # Load prompt suite
        suite = _load_prompt_suite(request.prompts_path)
        
        # Configure models
        config_a = ModelConfig(
            name=request.model_a,
            provider=ProviderType(request.provider_a.lower())
        )
        config_b = ModelConfig(
            name=request.model_b,
            provider=ProviderType(request.provider_b.lower())
        )
        
        # Initialize components
        runner = LLMRunner(settings)
        differ = create_differ(settings)
        
        scorer = CombinedScorer(differ=differ)
        if request.use_judge and settings.openrouter_api_key:
            judge = LLMJudge(runner, settings)
            scorer.judge = judge
        
        # Run comparison
        results: List[DiffResult] = []
        
        for prompt in suite.prompts:
            # Run models
            resp_a, resp_b = await runner.compare(prompt, config_a, config_b)
            
            # Score
            score = await scorer.score(prompt, resp_a, resp_b)
            
            # Detect change
            is_change = score.combined_score < request.threshold
            severity = "none"
            if is_change:
                if score.combined_score < 0.4:
                    severity = "major"
                elif score.combined_score < 0.7:
                    severity = "moderate"
                else:
                    severity = "minor"
            
            results.append(DiffResult(
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response_a=resp_a,
                response_b=resp_b,
                semantic_score=score,
                behavioral_change_detected=is_change,
                change_severity=severity
            ))
        
        # Calculate statistics
        changed_count = sum(1 for r in results if r.behavioral_change_detected)
        change_rate = changed_count / len(results) if results else 0.0
        avg_sim = sum(r.semantic_score.combined_score for r in results) / len(results) if results else 0.0
        
        # Build response
        results_data = []
        for r in results:
            results_data.append({
                "prompt_id": r.prompt_id,
                "prompt_text": r.prompt_text[:200] + "..." if len(r.prompt_text) > 200 else r.prompt_text,
                "similarity_score": round(r.semantic_score.combined_score, 4),
                "behavioral_change": r.behavioral_change_detected,
                "severity": r.change_severity,
                "response_a_preview": r.response_a.text[:150] + "..." if len(r.response_a.text) > 150 else r.response_a.text,
                "response_b_preview": r.response_b.text[:150] + "..." if len(r.response_b.text) > 150 else r.response_b.text,
            })
        
        await runner.close()
        
        return CompareModelsResponse(
            success=True,
            run_id=f"run-{int(asyncio.get_event_loop().time())}",
            total_prompts=len(results),
            changes_detected=changed_count,
            change_rate=round(change_rate * 100, 2),
            avg_similarity=round(avg_sim, 4),
            results=results_data
        )
        
    except Exception as e:
        return CompareModelsResponse(
            success=False,
            error=str(e)
        )


@mcp.tool()
async def analyze_drift(request: AnalyzeDriftRequest) -> AnalyzeDriftResponse:
    """Analyze semantic drift between two model responses.
    
    Compares two responses to the same prompt and calculates similarity scores
    using embeddings and optionally an LLM judge.
    
    Args:
        request: Drift analysis request with responses to compare
        
    Returns:
        Similarity scores and change detection results
    """
    settings = _load_settings()
    
    try:
        # Create differ
        if request.use_embeddings:
            from .differ import EmbeddingDiffer
            differ = EmbeddingDiffer()
        else:
            from .differ import SimpleDiffer
            differ = SimpleDiffer()
        
        # Compute embedding similarity
        embedding_sim = differ.compute_similarity(
            request.response_a,
            request.response_b
        )
        
        # Optional LLM judge
        judge_score: Optional[float] = None
        judge_reasoning: Optional[str] = None
        
        if request.use_judge and settings.openrouter_api_key:
            runner = LLMRunner(settings)
            judge = LLMJudge(runner, settings)
            
            try:
                judge_score, judge_reasoning = await judge.judge_similarity(
                    request.prompt_text,
                    request.response_a,
                    request.response_b
                )
            except Exception:
                pass  # Fall back to embedding only
            finally:
                await runner.close()
        
        # Calculate combined score
        if judge_score is not None:
            combined = (0.6 * embedding_sim + 0.4 * judge_score)
        else:
            combined = embedding_sim
        
        # Determine severity
        threshold = 0.85
        is_change = combined < threshold
        severity = "none"
        if is_change:
            if combined < 0.4:
                severity = "major"
            elif combined < 0.7:
                severity = "moderate"
            else:
                severity = "minor"
        
        return AnalyzeDriftResponse(
            success=True,
            embedding_similarity=round(embedding_sim, 4),
            judge_score=round(judge_score, 4) if judge_score else None,
            combined_score=round(combined, 4),
            behavioral_change_detected=is_change,
            severity=severity,
            judge_reasoning=judge_reasoning
        )
        
    except Exception as e:
        return AnalyzeDriftResponse(
            success=False,
            error=str(e)
        )


@mcp.tool()
async def generate_report(request: GenerateReportRequest) -> GenerateReportResponse:
    """Generate an HTML report from comparison results.
    
    Creates a detailed HTML report with visualizations of behavioral changes
    between model versions.
    
    Args:
        request: Report generation request with results data
        
    Returns:
        Path to generated report and HTML preview
    """
    try:
        # Parse results JSON
        results_data = json.loads(request.results_json)
        
        # Generate HTML directly using a simplified approach
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{request.title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background: #e8f5e9; padding: 20px; border-radius: 4px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .severity-major {{ color: #d32f2f; font-weight: bold; }}
        .severity-moderate {{ color: #f57c00; font-weight: bold; }}
        .severity-minor {{ color: #fbc02d; }}
        .severity-none {{ color: #388e3c; }}
        .timestamp {{ color: #999; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{request.title}</h1>
        <p class="timestamp">Generated: {asyncio.get_event_loop().time()}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Comparisons: {len(results_data)}</p>
        </div>
        
        <h2>Results</h2>
        <table>
            <tr>
                <th>Prompt ID</th>
                <th>Similarity Score</th>
                <th>Change Detected</th>
                <th>Severity</th>
            </tr>
"""
        
        for result in results_data:
            severity_class = f"severity-{result.get('severity', 'none')}"
            html_content += f"""
            <tr>
                <td>{result.get('prompt_id', 'N/A')}</td>
                <td>{result.get('similarity_score', 'N/A')}</td>
                <td>{'Yes' if result.get('behavioral_change') else 'No'}</td>
                <td class="{severity_class}">{result.get('severity', 'none').title()}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # Ensure output directory exists
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write report
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return GenerateReportResponse(
            success=True,
            output_path=str(output_path.absolute()),
            html_preview=html_content[:1000]
        )
        
    except Exception as e:
        return GenerateReportResponse(
            success=False,
            error=str(e)
        )


def main():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
