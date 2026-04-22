import pytest
from typer.testing import CliRunner
from src.llm_behavior_diff.cli import app
from unittest.mock import MagicMock, patch
import os

runner = CliRunner()

def test_cli_help():
    # Test top-level help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM Behavior Diff" in result.stdout

def test_cli_run_help():
    # Test 'run' command help
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run a comparison between two models" in result.stdout

@patch("src.llm_behavior_diff.cli._run_async")
def test_cli_run_invocation(mock_run_async):
    # Mock the async run to avoid actual LLM calls
    mock_run_async.return_value = None
    
    # We need to provide required arguments: --model-a and --model-b
    result = runner.invoke(app, [
        "run",
        "--model-a", "test-a",
        "--model-b", "test-b",
        "--prompts", "prompts/default.yaml",
        "--output", "output/test_report.html"
    ])
    
    if result.exit_code != 0:
        print(f"EXIT CODE: {result.exit_code}")
        print(f"STDOUT: {result.stdout}")
        
    assert result.exit_code == 0
    mock_run_async.assert_called_once()
