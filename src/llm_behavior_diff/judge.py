"""LLM judge for semantic comparison scoring."""

from __future__ import annotations

import json
from typing import Optional

from llm_behavior_diff.models import ModelConfig, ProviderType
from llm_behavior_diff.runner import LLMRunner


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator comparing two AI responses to the same prompt.

Your task is to evaluate how similar the responses are in meaning, intent, and quality.

Original Prompt:
{prompt_text}

Response A:
{response_a}

Response B:
{response_b}

Rate the semantic similarity between these responses on a scale of 0.0 to 1.0, where:
- 1.0 = Nearly identical in meaning and quality
- 0.8 = Very similar, minor differences only
- 0.6 = Similar main points, some differences
- 0.4 = Partially overlapping content
- 0.2 = Mostly different with some overlap
- 0.0 = Completely different or contradictory

Provide your rating and a brief explanation of your reasoning.

Respond in JSON format:
{{
    "similarity_score": float,
    "reasoning": "brief explanation"
}}
"""


class LLMJudge:
    """Judge semantic similarity using an LLM."""
    
    def __init__(
        self,
        judge_model: str = "openai/gpt-4o-mini",
        openrouter_api_key: Optional[str] = None,
        temperature: float = 0.3
    ) -> None:
        """Initialize LLM judge.
        
        Args:
            judge_model: Model to use for judging
            openrouter_api_key: OpenRouter API key
            temperature: Temperature for judge model
        """
        self.judge_model = judge_model
        self.openrouter_api_key = openrouter_api_key
        self.temperature = temperature
        self._runner: Optional[LLMRunner] = None
    
    def _get_runner(self) -> LLMRunner:
        """Get or create LLM runner.
        
        Returns:
            LLMRunner instance
        """
        if self._runner is None:
            self._runner = LLMRunner(
                openrouter_api_key=self.openrouter_api_key
            )
        return self._runner
    
    async def judge_similarity(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str
    ) -> tuple[float, str]:
        """Judge semantic similarity between two responses.
        
        Args:
            prompt_text: Original prompt text
            response_a: First response
            response_b: Second response
            
        Returns:
            Tuple of (similarity_score, reasoning)
        """
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt_text=prompt_text,
            response_a=response_a[:2000],  # Limit length
            response_b=response_b[:2000]
        )
        
        # Determine provider from model name
        if "/" in self.judge_model and not self.judge_model.startswith("ollama/"):
            provider = ProviderType.OPENROUTER
            model_name = self.judge_model
        else:
            provider = ProviderType.OLLAMA
            model_name = self.judge_model.replace("ollama/", "")
        
        model_config = ModelConfig(
            name=model_name,
            provider=provider,
            temperature=self.temperature,
            max_tokens=500
        )
        
        runner = self._get_runner()
        
        try:
            result = await runner.run_prompt(
                model_config=model_config,
                prompt_id="judge",
                prompt_text=judge_prompt
            )
            
            if not result.success:
                return 0.5, f"Judge model failed: {result.error}"
            
            # Parse JSON response
            score, reasoning = self._parse_judge_response(result.text)
            return score, reasoning
            
        except Exception as e:
            return 0.5, f"Error during judging: {str(e)}"
    
    def _parse_judge_response(self, response: str) -> tuple[float, str]:
        """Parse judge model response.
        
        Args:
            response: Raw response text
            
        Returns:
            Tuple of (similarity_score, reasoning)
        """
        try:
            # Try to find JSON in response
            response = response.strip()
            
            # Handle markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response
            
            data = json.loads(json_str)
            
            score = float(data.get("similarity_score", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")
            
            # Clamp score to valid range
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: try to extract score from text
            return self._extract_score_fallback(response), response[:200]
    
    def _extract_score_fallback(self, text: str) -> float:
        """Extract similarity score from text when JSON parsing fails.
        
        Args:
            text: Response text
            
        Returns:
            Extracted score or 0.5
        """
        text_lower = text.lower()
        
        # Look for explicit score mentions
        import re
        patterns = [
            r'similarity[\s:]*([0-9.]+)',
            r'score[\s:]*([0-9.]+)',
            r'rating[\s:]*([0-9.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    score = float(match.group(1))
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue
        
        return 0.5
    
    async def close(self) -> None:
        """Close the judge runner."""
        if self._runner:
            await self._runner.close()
            self._runner = None


class CombinedScorer:
    """Combine embedding and LLM judge scores."""
    
    def __init__(
        self,
        embedding_weight: float = 0.6,
        judge_weight: float = 0.4,
        openrouter_api_key: Optional[str] = None,
        judge_model: str = "openai/gpt-4o-mini"
    ) -> None:
        """Initialize combined scorer.
        
        Args:
            embedding_weight: Weight for embedding similarity
            judge_weight: Weight for LLM judge score
            openrouter_api_key: OpenRouter API key for judge
            judge_model: Model to use for judging
        """
        self.embedding_weight = embedding_weight
        self.judge_weight = judge_weight
        self.judge = LLMJudge(
            judge_model=judge_model,
            openrouter_api_key=openrouter_api_key
        ) if openrouter_api_key else None
    
    async def score(
        self,
        prompt_text: str,
        response_a: str,
        response_b: str,
        embedding_similarity: float
    ) -> tuple[float, Optional[float], Optional[str]]:
        """Compute combined score.
        
        Args:
            prompt_text: Original prompt
            response_a: First response
            response_b: Second response
            embedding_similarity: Pre-computed embedding similarity
            
        Returns:
            Tuple of (combined_score, judge_score, reasoning)
        """
        judge_score: Optional[float] = None
        reasoning: Optional[str] = None
        
        if self.judge:
            try:
                judge_score, reasoning = await self.judge.judge_similarity(
                    prompt_text, response_a, response_b
                )
            except Exception:
                pass  # Fall back to embedding only
        
        # Calculate combined score
        if judge_score is not None:
            combined = (
                self.embedding_weight * embedding_similarity +
                self.judge_weight * judge_score
            ) / (self.embedding_weight + self.judge_weight)
        else:
            combined = embedding_similarity
        
        return combined, judge_score, reasoning
    
    async def close(self) -> None:
        """Close resources."""
        if self.judge:
            await self.judge.close()
