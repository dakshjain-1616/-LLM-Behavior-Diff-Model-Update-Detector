# LLM Behavior Diff — Model Update Detector

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)]()

A production-grade tool for detecting semantic behavioral changes between Large Language Model versions.

## 🎯 Overview

LLM Behavior Diff helps you understand how model updates affect behavior by:
- Running identical prompts through two model versions
- Computing semantic similarity using embeddings
- Using LLM-as-judge for nuanced behavioral analysis
- Generating detailed HTML reports with change visualizations
- Exposing functionality via MCP server for AI agent integration

## ✨ Features

- **🔍 Semantic Diff Engine**: Compare LLM responses using sentence embeddings and cosine similarity
- **🤖 Multi-Provider Support**: Works with Ollama (local) and OpenRouter (cloud) APIs
- **📊 Rich CLI**: Interactive command-line interface with progress bars and formatted tables
- **🌐 MCP Server**: Model Context Protocol server for integration with Claude Code and other AI assistants
- **📈 HTML Reporting**: Generate beautiful, interactive HTML reports with behavioral change analytics
- **🧪 Prompt Suites**: Define test suites in YAML for systematic evaluation
- **⚡ Async Architecture**: Fully asynchronous for efficient parallel processing

## 📦 Installation

### From Source

```bash
git clone https://github.com/example/llm-behavior-diff.git
cd llm-behavior-diff
pip install -e ".[dev]"
```

### Requirements

- Python 3.11+
- For local models: [Ollama](https://ollama.ai/) running locally
- For cloud models: [OpenRouter](https://openrouter.ai/) API key

## 🚀 Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run a Comparison

```bash
# Compare two Ollama models
llm-diff run \
  --model-a qwen3:8b \
  --model-b gemma4:e4b \
  --prompts prompts/default.yaml \
  --output output/report.html

# Compare OpenRouter models
llm-diff run \
  --model-a openrouter/qwen/qwen3-8b \
  --provider-a openrouter \
  --model-b openrouter/google/gemma-4-26b-a4b-it \
  --provider-b openrouter \
  --prompts prompts/default.yaml
```

### 3. View the Report

Open `output/report.html` in your browser to see detailed behavioral change analysis.

## 📖 Usage

### CLI Commands

```bash
# Show help
llm-diff --help

# Run comparison with custom threshold
llm-diff run \
  --model-a llama3 \
  --model-b llama3.1 \
  --threshold 0.80 \
  --use-judge \
  --output report.html

# Run without LLM judge (faster, embedding-only)
llm-diff run \
  --model-a qwen3:8b \
  --model-b gemma4:e4b \
  --no-use-judge
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | None |
| `OLLAMA_HOST` | Ollama server URL | http://localhost:11434 |
| `DEFAULT_SIMILARITY_THRESHOLD` | Change detection threshold | 0.85 |

### MCP Server

The MCP server exposes three tools for AI agent integration:

#### `compare_models`

Compare two models using a prompt suite:

```json
{
  "model_a": "qwen3:8b",
  "provider_a": "ollama",
  "model_b": "gemma4:e4b",
  "provider_b": "ollama",
  "prompts_path": "prompts/default.yaml",
  "threshold": 0.85,
  "use_judge": true
}
```

#### `analyze_drift`

Analyze semantic drift between two responses:

```json
{
  "prompt_text": "What is 2+2?",
  "response_a": "The answer is 4.",
  "response_b": "2 plus 2 equals 4.",
  "use_embeddings": true,
  "use_judge": false
}
```

#### `generate_report`

Generate an HTML report from comparison results:

```json
{
  "results_json": "[{...}]",
  "output_path": "output/report.html",
  "title": "My Comparison Report",
  "include_raw_responses": false
}
```

### Running the MCP Server

```bash
# Using the MCP CLI
mcp run src/llm_behavior_diff/mcp_server.py

# Or directly
python -m llm_behavior_diff.mcp_server
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / MCP Server                      │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM Runner                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │   Ollama     │  │    OpenRouter    │  │   LLM Judge    │  │
│  │   Client     │  │     Client       │  │   (Optional)   │  │
│  └──────────────┘  └──────────────────┘  └────────────────┘  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Semantic Diff Engine                      │
│  ┌──────────────────┐  ┌──────────────────────────────────┐  │
│  │ Embedding Differ │  │      Combined Scorer             │  │
│  │ (sentence-       │  │  (embeddings + LLM judge)        │  │
│  │  transformers)   │  │                                  │  │
│  └──────────────────┘  └──────────────────────────────────┘  │
└──────────────────────┬────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Report Generator                           │
│              (Jinja2 → HTML/Markdown)                        │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Prompt Suites

The default prompt suite (`prompts/default.yaml`) includes 50+ prompts across categories:

- **Reasoning**: Logic puzzles, math problems, optimization
- **Coding**: Code generation, debugging, algorithm design
- **Creativity**: Story writing, brainstorming, style adaptation
- **Safety**: Refusal patterns, harmful content handling
- **Instruction Following**: Multi-step tasks, format compliance
- **Factual**: Knowledge recall, fact checking
- **Conversational**: Dialogue, tone consistency

### Custom Prompt Suites

Create your own YAML file:

```yaml
name: "My Custom Suite"
version: "1.0.0"
description: "Custom prompts for my use case"
prompts:
  - id: "custom-001"
    text: "Generate a Python function to..."
    category: "coding"
    tags: ["python", "functions"]
    expected_behavior: "Should return a valid function"
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_behavior_diff

# Run specific test file
pytest tests/test_differ.py -v
```

## 📝 Example Output

### CLI Output

```
╭──────────────────────────────────────────╮
│     LLM Behavior Diff                      │
│     Detecting behavioral shifts...        │
╰──────────────────────────────────────────╯
Comparing models... ━━━━━━━━━━━━━━━━━━━ 100% 56/56

Comparison Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric           ┃ Value    ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Total Prompts    │ 56       │
│ Changes Detected │ 12       │
│ Change Rate      │ 21.4%    │
│ Avg Similarity   │ 87.3%    │
└──────────────────┴──────────┘

✔ Report saved to: output/report.html
```

### HTML Report

The HTML report includes:
- Summary statistics (change rate, average similarity)
- Per-prompt comparison with side-by-side diffs
- Severity classification (minor/moderate/major)
- Category breakdown charts
- Exportable JSON results

## 🔧 Configuration

### Model Identifiers (April 2026)

**Ollama Models:**
- `gemma4:e4b` - Google Gemma 4B
- `gemma4:e2b` - Google Gemma E2B
- `qwen3:8b` - Alibaba Qwen3 8B
- `qwen3:4b` - Alibaba Qwen3 4B

**OpenRouter Models:**
- `google/gemma-4-26b-a4b-it`
- `qwen/qwen3-8b`
- `mistralai/mistral-small-2603`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [Ollama](https://ollama.ai/) for local LLM hosting
- [OpenRouter](https://openrouter.ai/) for unified API access
- [MCP](https://modelcontextprotocol.io/) for AI agent integration
