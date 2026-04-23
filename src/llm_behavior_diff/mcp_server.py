"""MCP Server for LLM Behavior Diff — Model Update Detector."""

from __future__ import annotations

import asyncio
import json
import os
import time
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
    Settings,
)
from .runner import LLMRunner
from .differ import create_differ, EmbeddingDiffer, SimpleDiffer
from .judge import LLMJudge
from .report import ReportGenerator

mcp = FastMCP("llm-behavior-diff")


class CompareModelsRequest(BaseModel):
    model_a: str = Field(..., description="Model A identifier")
    provider_a: str = Field(default="ollama", description="ollama | openrouter | stub")
    model_b: str = Field(..., description="Model B identifier")
    provider_b: str = Field(default="ollama", description="ollama | openrouter | stub")
    prompts_path: str = Field(default="prompts/default.yaml", description="Path to prompt suite YAML")
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    use_judge: bool = Field(default=False)
    use_embeddings: bool = Field(default=True)


class CompareModelsResponse(BaseModel):
    success: bool
    run_id: Optional[str] = None
    total_prompts: int = 0
    changes_detected: int = 0
    change_rate: float = 0.0
    avg_similarity: float = 0.0
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class AnalyzeDriftRequest(BaseModel):
    prompt_text: str
    response_a: str
    response_b: str
    use_embeddings: bool = Field(default=True)
    use_judge: bool = Field(default=False)


class AnalyzeDriftResponse(BaseModel):
    success: bool
    embedding_similarity: Optional[float] = None
    judge_score: Optional[float] = None
    combined_score: float = 0.0
    behavioral_change_detected: bool = False
    severity: str = "none"
    judge_reasoning: Optional[str] = None
    error: Optional[str] = None


class GenerateReportRequest(BaseModel):
    results_json: str = Field(..., description="JSON array of per-prompt results")
    output_path: str = Field(..., description="Where to save the HTML report")
    title: str = Field(default="LLM Behavior Diff Report")


class GenerateReportResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    html_preview: Optional[str] = None
    error: Optional[str] = None


def _settings() -> Settings:
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    )


def _load_suite(path: str) -> PromptSuite:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return PromptSuite(**data)


def _severity(score: float, threshold: float) -> tuple[bool, str]:
    is_change = score < threshold
    if not is_change:
        return False, "none"
    if score < 0.4:
        return True, "major"
    if score < 0.7:
        return True, "moderate"
    return True, "minor"


@mcp.tool()
async def compare_models(request: CompareModelsRequest) -> CompareModelsResponse:
    """Run a prompt suite through two models and report behavioral drift."""
    s = _settings()
    try:
        suite = _load_suite(request.prompts_path)
        config_a = ModelConfig(name=request.model_a, provider=ProviderType(request.provider_a.lower()))
        config_b = ModelConfig(name=request.model_b, provider=ProviderType(request.provider_b.lower()))

        runner = LLMRunner(
            ollama_host=s.ollama_host,
            openrouter_api_key=s.openrouter_api_key,
        )
        differ = create_differ(use_embeddings=request.use_embeddings)
        judge: Optional[LLMJudge] = None
        if request.use_judge and s.openrouter_api_key:
            judge = LLMJudge(judge_model=s.judge_model, openrouter_api_key=s.openrouter_api_key)

        results: List[DiffResult] = []
        for prompt in suite.prompts:
            resp_a = await runner.run_prompt(config_a, prompt.id, prompt.text)
            resp_b = await runner.run_prompt(config_b, prompt.id, prompt.text)

            judge_score = None
            judge_reason = None
            if judge is not None:
                try:
                    judge_score, judge_reason = await judge.judge_similarity(
                        prompt.text, resp_a.text, resp_b.text
                    )
                except Exception as e:
                    judge_reason = f"judge error: {e}"

            score = differ.compute_semantic_score(
                text_a=resp_a.text,
                text_b=resp_b.text,
                llm_judge_score=judge_score,
                judge_reasoning=judge_reason,
            )
            is_change, severity = _severity(score.combined_score, request.threshold)
            results.append(DiffResult(
                prompt_id=prompt.id,
                prompt_text=prompt.text,
                response_a=resp_a,
                response_b=resp_b,
                semantic_score=score,
                behavioral_change_detected=is_change,
                change_severity=severity,
            ))

        await runner.close()
        if judge:
            await judge.close()

        changed = sum(1 for r in results if r.behavioral_change_detected)
        avg = (sum(r.semantic_score.combined_score for r in results) / len(results)) if results else 0.0

        results_data = []
        for r in results:
            results_data.append({
                "prompt_id": r.prompt_id,
                "prompt_text": r.prompt_text,
                "similarity_score": round(r.semantic_score.combined_score, 4),
                "behavioral_change": r.behavioral_change_detected,
                "severity": r.change_severity,
                "response_a": r.response_a.text,
                "response_b": r.response_b.text,
            })

        return CompareModelsResponse(
            success=True,
            run_id=f"run-{int(time.time())}",
            total_prompts=len(results),
            changes_detected=changed,
            change_rate=round((changed / len(results)) if results else 0.0, 4),
            avg_similarity=round(avg, 4),
            results=results_data,
        )
    except Exception as e:
        return CompareModelsResponse(success=False, error=str(e))


@mcp.tool()
async def analyze_drift(request: AnalyzeDriftRequest) -> AnalyzeDriftResponse:
    """Compute semantic similarity between two candidate responses to one prompt."""
    s = _settings()
    try:
        differ = EmbeddingDiffer() if request.use_embeddings else SimpleDiffer()
        embedding_sim = differ.compute_similarity(request.response_a, request.response_b)

        judge_score: Optional[float] = None
        judge_reasoning: Optional[str] = None
        if request.use_judge and s.openrouter_api_key:
            judge = LLMJudge(judge_model=s.judge_model, openrouter_api_key=s.openrouter_api_key)
            try:
                judge_score, judge_reasoning = await judge.judge_similarity(
                    request.prompt_text, request.response_a, request.response_b
                )
            finally:
                await judge.close()

        combined = (0.6 * embedding_sim + 0.4 * judge_score) if judge_score is not None else embedding_sim
        is_change, severity = _severity(combined, 0.85)

        return AnalyzeDriftResponse(
            success=True,
            embedding_similarity=round(embedding_sim, 4),
            judge_score=round(judge_score, 4) if judge_score is not None else None,
            combined_score=round(combined, 4),
            behavioral_change_detected=is_change,
            severity=severity,
            judge_reasoning=judge_reasoning,
        )
    except Exception as e:
        return AnalyzeDriftResponse(success=False, error=str(e))


@mcp.tool()
async def generate_report(request: GenerateReportRequest) -> GenerateReportResponse:
    """Render an HTML summary from a JSON array of per-prompt results."""
    try:
        results_data = json.loads(request.results_json)
        # Lightweight HTML rendering independent of a ComparisonRun.
        rows = []
        for r in results_data:
            sev = r.get("severity", "none")
            rows.append(
                f"<tr><td>{r.get('prompt_id','')}</td>"
                f"<td>{r.get('similarity_score','')}</td>"
                f"<td>{'Yes' if r.get('behavioral_change') else 'No'}</td>"
                f"<td class='sev-{sev}'>{sev}</td></tr>"
            )
        html = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{request.title}</title>"
            "<style>body{font-family:sans-serif;margin:40px}table{border-collapse:collapse;width:100%}"
            "th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}"
            ".sev-major{color:#b91c1c;font-weight:600}.sev-moderate{color:#c2410c}.sev-minor{color:#a16207}.sev-none{color:#15803d}</style>"
            f"</head><body><h1>{request.title}</h1>"
            f"<p>Total: {len(results_data)}</p>"
            "<table><tr><th>Prompt ID</th><th>Similarity</th><th>Change</th><th>Severity</th></tr>"
            + "".join(rows) +
            "</table></body></html>"
        )

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return GenerateReportResponse(
            success=True,
            output_path=str(out.absolute()),
            html_preview=html[:1000],
        )
    except Exception as e:
        return GenerateReportResponse(success=False, error=str(e))


def main() -> None:
    """Run the MCP server on stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
