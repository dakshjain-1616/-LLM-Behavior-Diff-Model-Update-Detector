"""Pydantic data models for LLM Behavior Diff."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class PromptCategory(str, Enum):
    """Categories for test prompts."""
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVITY = "creativity"
    SAFETY = "safety"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"


class Prompt(BaseModel):
    """A single test prompt."""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Unique identifier for the prompt")
    text: str = Field(..., min_length=1, description="The prompt text")
    category: PromptCategory = Field(..., description="Category of the prompt")
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    expected_behavior: Optional[str] = Field(None, description="Expected behavior description")
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate prompt ID format."""
        if not v or not v.strip():
            raise ValueError("Prompt ID cannot be empty")
        return v.strip()


class PromptSuite(BaseModel):
    """A collection of test prompts."""
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., description="Name of the prompt suite")
    version: str = Field(default="1.0.0", description="Version of the suite")
    description: Optional[str] = Field(None, description="Description of the suite")
    prompts: List[Prompt] = Field(..., min_length=1, description="List of prompts")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @field_validator("prompts")
    @classmethod
    def validate_unique_ids(cls, v: List[Prompt]) -> List[Prompt]:
        """Ensure all prompt IDs are unique."""
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate prompt IDs found")
        return v


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    model_config = ConfigDict(frozen=True)
    
    name: str = Field(..., description="Model name/identifier")
    provider: ProviderType = Field(..., description="Provider type")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    system_prompt: Optional[str] = Field(None, description="System prompt")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific params")


class LLMResponse(BaseModel):
    """Response from an LLM."""
    model_config = ConfigDict(frozen=True)
    
    prompt_id: str = Field(..., description="ID of the prompt that generated this response")
    model_config_used: ModelConfig = Field(..., description="Model configuration used")
    text: str = Field(..., description="The response text")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    latency_ms: Optional[float] = Field(None, description="Response latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = Field(None, description="Raw API response")
    error: Optional[str] = Field(None, description="Error message if request failed")
    
    @property
    def success(self) -> bool:
        """Check if the response was successful."""
        return self.error is None


class SemanticScore(BaseModel):
    """Semantic similarity score between two responses."""
    model_config = ConfigDict(frozen=True)
    
    embedding_similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity of embeddings")
    llm_judge_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM judge score")
    combined_score: float = Field(..., ge=0.0, le=1.0, description="Combined semantic score")
    judge_reasoning: Optional[str] = Field(None, description="Reasoning from LLM judge")
    
    @field_validator("combined_score")
    @classmethod
    def validate_combined(cls, v: float, info: Any) -> float:
        """Ensure combined score is valid."""
        if v < 0.0 or v > 1.0:
            raise ValueError("Combined score must be between 0 and 1")
        return v


class DiffResult(BaseModel):
    """Result of comparing two LLM responses."""
    model_config = ConfigDict(frozen=True)
    
    prompt_id: str = Field(..., description="ID of the prompt")
    prompt_text: str = Field(..., description="The prompt text")
    response_a: LLMResponse = Field(..., description="Response from model A")
    response_b: LLMResponse = Field(..., description="Response from model B")
    semantic_score: SemanticScore = Field(..., description="Semantic similarity score")
    behavioral_change_detected: bool = Field(..., description="Whether behavior changed significantly")
    change_severity: str = Field(..., description="Severity: none, minor, moderate, major")
    
    @field_validator("change_severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity level."""
        valid = {"none", "minor", "moderate", "major"}
        if v.lower() not in valid:
            raise ValueError(f"Severity must be one of {valid}")
        return v.lower()


class ComparisonRun(BaseModel):
    """A complete comparison run between two models."""
    model_config = ConfigDict(frozen=True)
    
    id: str = Field(..., description="Unique run identifier")
    model_a: ModelConfig = Field(..., description="Configuration for model A")
    model_b: ModelConfig = Field(..., description="Configuration for model B")
    prompt_suite: PromptSuite = Field(..., description="Prompt suite used")
    results: List[DiffResult] = Field(default_factory=list, description="Comparison results")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def behavioral_change_rate(self) -> float:
        """Calculate percentage of prompts with behavioral changes."""
        if not self.results:
            return 0.0
        changed = sum(1 for r in self.results if r.behavioral_change_detected)
        return changed / len(self.results)
    
    @property
    def average_similarity(self) -> float:
        """Calculate average semantic similarity."""
        if not self.results:
            return 0.0
        return sum(r.semantic_score.combined_score for r in self.results) / len(self.results)


class ReportConfig(BaseModel):
    """Configuration for report generation."""
    model_config = ConfigDict(frozen=True)
    
    title: str = Field(default="LLM Behavior Diff Report", description="Report title")
    include_raw_responses: bool = Field(default=False, description="Include full response text")
    include_embeddings: bool = Field(default=False, description="Include embedding visualizations")
    theme: str = Field(default="light", description="Report theme: light or dark")
    max_samples_per_category: int = Field(default=10, ge=1, description="Max samples to show per category")


class Report(BaseModel):
    """Generated report."""
    model_config = ConfigDict(frozen=True)
    
    run: ComparisonRun = Field(..., description="The comparison run")
    config: ReportConfig = Field(..., description="Report configuration")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    html_content: Optional[str] = Field(None, description="Generated HTML content")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the report."""
        return {
            "total_prompts": len(self.run.results),
            "behavioral_changes": sum(1 for r in self.run.results if r.behavioral_change_detected),
            "change_rate": self.run.behavioral_change_rate,
            "avg_similarity": self.run.average_similarity,
            "model_a": self.run.model_a.name,
            "model_b": self.run.model_b.name,
            "duration_seconds": self.run.duration_seconds,
        }


class Settings(BaseModel):
    """Application settings loaded from environment."""
    model_config = ConfigDict(frozen=False)
    
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama host URL")
    ollama_timeout: int = Field(default=120, ge=1, description="Ollama request timeout in seconds")
    openrouter_timeout: int = Field(default=60, ge=1, description="OpenRouter request timeout")
    default_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Default embedding model"
    )
    judge_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",
        description="Model used for LLM judging"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting behavioral changes"
    )
    output_dir: Path = Field(default=Path("./output"), description="Default output directory")
    
    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Union[str, Path]) -> Path:
        """Ensure output directory is a Path object."""
        return Path(v) if isinstance(v, str) else v


class MCPToolRequest(BaseModel):
    """Request to an MCP tool."""
    model_config = ConfigDict(frozen=True)
    
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    request_id: str = Field(..., description="Unique request ID")


class MCPToolResponse(BaseModel):
    """Response from an MCP tool."""
    model_config = ConfigDict(frozen=True)
    
    request_id: str = Field(..., description="Request ID matching the request")
    success: bool = Field(..., description="Whether the tool call succeeded")
    result: Optional[Any] = Field(None, description="Tool result")
    error: Optional[str] = Field(None, description="Error message if failed")
