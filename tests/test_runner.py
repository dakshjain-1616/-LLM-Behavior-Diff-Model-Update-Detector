"""Tests for LLM runner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_behavior_diff.models import ModelConfig, ProviderType, LLMResponse
from llm_behavior_diff.runner import OllamaClient, OpenRouterClient, LLMRunner


class TestOllamaClient:
    """Test Ollama client."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        client = OllamaClient()
        
        with patch.object(client.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        """Test failed health check."""
        client = OllamaClient()
        
        with patch.object(client.client, 'get', side_effect=Exception("Connection failed")):
            result = await client.health_check()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Test successful generation."""
        client = OllamaClient()
        
        with patch.object(client.client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "Hello!", "eval_count": 10}
            mock_post.return_value = mock_response
            
            result = await client.generate("llama2", "Hi")
            assert result["response"] == "Hello!"
            assert result["eval_count"] == 10


class TestOpenRouterClient:
    """Test OpenRouter client."""
    
    @pytest.mark.asyncio
    async def test_health_check_no_key(self) -> None:
        """Test health check without API key."""
        client = OpenRouterClient(api_key="test-key")
        
        with patch.object(client.client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = await client.health_check()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Test successful generation."""
        client = OpenRouterClient(api_key="test-key")
        
        with patch.object(client.client, 'post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Hello!"}}],
                "usage": {"total_tokens": 15}
            }
            mock_post.return_value = mock_response
            
            result = await client.generate("gpt-4", "Hi")
            assert result["choices"][0]["message"]["content"] == "Hello!"


class TestLLMRunner:
    """Test LLM Runner."""
    
    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test runner initialization."""
        runner = LLMRunner()
        assert runner.ollama_client is not None
        assert runner.openrouter_client is None
    
    @pytest.mark.asyncio
    async def test_init_with_openrouter(self) -> None:
        """Test runner with OpenRouter key."""
        runner = LLMRunner(openrouter_api_key="test-key")
        assert runner.ollama_client is not None
        assert runner.openrouter_client is not None
    
    @pytest.mark.asyncio
    async def test_run_prompt_ollama(self) -> None:
        """Test running prompt with Ollama."""
        runner = LLMRunner()
        config = ModelConfig(name="llama2", provider=ProviderType.OLLAMA)
        
        with patch.object(runner.ollama_client, 'generate') as mock_gen:
            mock_gen.return_value = {
                "response": "Test response",
                "eval_count": 10
            }
            
            response = await runner.run_prompt(config, "test-1", "Hello")
            assert isinstance(response, LLMResponse)
            assert response.text == "Test response"
            assert response.success is True
    
    @pytest.mark.asyncio
    async def test_run_prompt_openrouter_no_key(self) -> None:
        """Test running prompt with OpenRouter but no key."""
        runner = LLMRunner()
        config = ModelConfig(name="gpt-4", provider=ProviderType.OPENROUTER)
        
        response = await runner.run_prompt(config, "test-1", "Hello")
        assert isinstance(response, LLMResponse)
        assert response.success is False
        assert "OpenRouter API key not provided" in response.error
    
    @pytest.mark.asyncio
    async def test_run_prompt_error(self) -> None:
        """Test handling of errors."""
        runner = LLMRunner()
        config = ModelConfig(name="llama2", provider=ProviderType.OLLAMA)
        
        with patch.object(runner.ollama_client, 'generate', side_effect=Exception("API Error")):
            response = await runner.run_prompt(config, "test-1", "Hello")
            assert isinstance(response, LLMResponse)
            assert response.success is False
            assert "API Error" in response.error
    
    @pytest.mark.asyncio
    async def test_health_check_ollama(self) -> None:
        """Test health check for Ollama."""
        runner = LLMRunner()
        
        with patch.object(runner.ollama_client, 'health_check') as mock_health:
            mock_health.return_value = True
            result = await runner.health_check(ProviderType.OLLAMA)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_openrouter_no_client(self) -> None:
        """Test health check for OpenRouter without client."""
        runner = LLMRunner()
        result = await runner.health_check(ProviderType.OPENROUTER)
        assert result is False
