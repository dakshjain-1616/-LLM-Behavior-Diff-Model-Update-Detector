"""LLM runner for Ollama and OpenRouter clients."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx

from llm_behavior_diff.models import LLMResponse, ModelConfig, ProviderType


class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120) -> None:
        """Initialize Ollama client.
        
        Args:
            host: Ollama host URL
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a response from Ollama.
        
        Args:
            model: Model name
            prompt: Prompt text
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
        """
        url = f"{self.host}/api/generate"
        
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system:
            payload["system"] = system
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Add any additional options
        for key, value in kwargs.items():
            payload["options"][key] = value
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> List[str]:
        """List available models.
        
        Returns:
            List of model names
        """
        url = f"{self.host}/api/tags"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    

    async def health_check(self) -> bool:
        """Check if Ollama is reachable.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.host}/api/tags"
            response = await self.client.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class OpenRouterClient:
    """Client for OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str, timeout: int = 60) -> None:
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/example/llm-behavior-diff",
                "X-Title": "LLM Behavior Diff"
            }
        )
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a response from OpenRouter.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            prompt: Prompt text
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            API response dictionary
        """
        url = f"{self.BASE_URL}/chat/completions"
        
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        
        # Add any additional parameters
        payload.update(kwargs)
        
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models.
        
        Returns:
            List of model information dictionaries
        """
        url = "https://openrouter.ai/api/v1/models"
        response = await self.client.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    
    async def health_check(self) -> bool:
        """Check if OpenRouter API is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            url = "https://openrouter.ai/api/v1/models"
            response = await self.client.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class LLMRunner:
    """Runner for executing LLM requests."""
    
    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        openrouter_api_key: Optional[str] = None,
        ollama_timeout: int = 120,
        openrouter_timeout: int = 60
    ) -> None:
        """Initialize LLM runner.
        
        Args:
            ollama_host: Ollama host URL
            openrouter_api_key: OpenRouter API key
            ollama_timeout: Ollama request timeout
            openrouter_timeout: OpenRouter request timeout
        """
        self.ollama_client = OllamaClient(ollama_host, ollama_timeout)
        self.openrouter_client = (
            OpenRouterClient(openrouter_api_key, openrouter_timeout)
            if openrouter_api_key
            else None
        )
    
    async def run_prompt(
        self,
        model_config: ModelConfig,
        prompt_id: str,
        prompt_text: str
    ) -> LLMResponse:
        """Run a single prompt against a model.
        
        Args:
            model_config: Model configuration
            prompt_id: Prompt identifier
            prompt_text: Prompt text
            
        Returns:
            LLM response object
        """
        start_time = time.time()
        
        try:
            if model_config.provider == ProviderType.OLLAMA:
                result = await self._run_ollama(model_config, prompt_text)
            elif model_config.provider == ProviderType.OPENROUTER:
                if not self.openrouter_client:
                    raise ValueError("OpenRouter API key not provided")
                result = await self._run_openrouter(model_config, prompt_text)
            elif model_config.provider == ProviderType.STUB:
                result = self._run_stub(model_config, prompt_text)
            else:
                raise ValueError(f"Unknown provider: {model_config.provider}")
            
            latency_ms = (time.time() - start_time) * 1000
            
            return LLMResponse(
                prompt_id=prompt_id,
                model_config_used=model_config,
                text=result["text"],
                tokens_used=result.get("tokens_used"),
                latency_ms=latency_ms,
                raw_response=result.get("raw")
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResponse(
                prompt_id=prompt_id,
                model_config_used=model_config,
                text="",
                latency_ms=latency_ms,
                error=str(e)
            )
    
    async def _run_ollama(
        self,
        model_config: ModelConfig,
        prompt_text: str
    ) -> Dict[str, Any]:
        """Run prompt against Ollama.
        
        Args:
            model_config: Model configuration
            prompt_text: Prompt text
            
        Returns:
            Result dictionary with text and metadata
        """
        response = await self.ollama_client.generate(
            model=model_config.name,
            prompt=prompt_text,
            system=model_config.system_prompt,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            **model_config.extra_params
        )
        
        return {
            "text": response.get("response", ""),
            "tokens_used": response.get("eval_count"),
            "raw": response
        }
    
    async def _run_openrouter(
        self,
        model_config: ModelConfig,
        prompt_text: str
    ) -> Dict[str, Any]:
        """Run prompt against OpenRouter.
        
        Args:
            model_config: Model configuration
            prompt_text: Prompt text
            
        Returns:
            Result dictionary with text and metadata
        """
        response = await self.openrouter_client.generate(
            model=model_config.name,
            prompt=prompt_text,
            system=model_config.system_prompt,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            top_p=model_config.top_p,
            **model_config.extra_params
        )
        
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No response from OpenRouter")
        
        message = choices[0].get("message", {})
        usage = response.get("usage", {})
        
        return {
            "text": message.get("content", ""),
            "tokens_used": usage.get("total_tokens"),
            "raw": response
        }
    
    def _run_stub(self, model_config: ModelConfig, prompt_text: str) -> Dict[str, Any]:
        """Return a deterministic fake response for offline/demo use.

        The seed is (model_name, temperature) so different models produce
        different text on the same prompt, letting the diff pipeline be
        exercised end to end with no external services.
        """
        import hashlib
        seed = f"{model_config.name}|{model_config.temperature}|{prompt_text}"
        h = hashlib.sha256(seed.encode()).hexdigest()
        # Pick a short templated reply that varies by model name.
        templates = [
            f"[{model_config.name}] {prompt_text[:60]} -> answer token {h[:6]}.",
            f"[{model_config.name}] reply: {h[:8]}. prompt was: {prompt_text[:40]}",
            f"Model {model_config.name} says: {h[:10]}",
        ]
        idx = int(h[:2], 16) % len(templates)
        text = templates[idx]
        return {"text": text, "tokens_used": len(text.split()), "raw": {"stub": True, "hash": h}}

    async def health_check(self, provider: ProviderType) -> bool:
        """Check health of a provider.
        
        Args:
            provider: Provider to check
            
        Returns:
            True if healthy, False otherwise
        """
        if provider == ProviderType.OLLAMA:
            return await self.ollama_client.health_check()
        elif provider == ProviderType.OPENROUTER:
            if not self.openrouter_client:
                return False
            return await self.openrouter_client.health_check()
        return False
    
    async def close(self) -> None:
        """Close all clients."""
        await self.ollama_client.close()
        if self.openrouter_client:
            await self.openrouter_client.close()
    
    async def __aenter__(self) -> LLMRunner:
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Convenience function for synchronous usage
def run_prompt_sync(
    model_config: ModelConfig,
    prompt_id: str,
    prompt_text: str,
    ollama_host: str = "http://localhost:11434",
    openrouter_api_key: Optional[str] = None
) -> LLMResponse:
    """Run a prompt synchronously.
    
    Args:
        model_config: Model configuration
        prompt_id: Prompt identifier
        prompt_text: Prompt text
        ollama_host: Ollama host URL
        openrouter_api_key: OpenRouter API key
        
    Returns:
        LLM response object
    """
    runner = LLMRunner(
        ollama_host=ollama_host,
        openrouter_api_key=openrouter_api_key
    )
    return asyncio.run(runner.run_prompt(model_config, prompt_id, prompt_text))
