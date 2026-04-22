# LLM Behavior Diff — Model Update Detector

## Goal
Build a production-grade Python tool and MCP server that detects semantic behavioral changes between two LLM versions (Ollama or OpenRouter) using embedding similarity and LLM-as-judge scoring.

## Research Summary
- **Model IDs (April 2026)**: 
    - Ollama: `gemma4:e4b`, `gemma4:e2b`, `qwen3:8b`, `qwen3:4b`.
    - OpenRouter: `google/gemma-4-26b-a4b-it`, `qwen/qwen3-8b`, `mistralai/mistral-small-2603`.
- **Tech Stack**: Python 3.11+, `ollama`, `openai` (for OpenRouter), `sentence-transformers`, `mcp`, `click`, `rich`, `jinja2`, `pydantic`, `pytest`.
- **MCP Server**: Will use the `mcp` Python SDK to expose `run_diff` as a tool.

## Approach
1. **Data Models**: Define Pydantic schemas for Prompts, ModelResponses, and DiffReports.
2. **Runner**: Abstract client to handle both Ollama (local) and OpenRouter (OpenAI-compatible) with retry logic.
3. **Differ**: 
    - Semantic similarity using `sentence-transformers` (all-MiniLM-L6-v2).
    - LLM-as-judge using a "Judge" model (defaulting to the stronger of the two or a specified third model) to categorize changes (verbosity, tone, refusal, etc.).
4. **CLI**: Rich-powered interface with progress bars and formatted tables.
5. **MCP**: FastMCP-based server for agentic integration.
6. **Report**: Jinja2 templates for a clean, interactive HTML report and a concise Markdown summary.

## Subtasks
1. **Project Scaffold**: Create `pyproject.toml` with all dependencies and the directory structure. (verify: `pip install .` succeeds)
2. **Core Data Models**: Implement `src/llm_behavior_diff/models.py` with Pydantic. (verify: model validation tests)
3. **Prompt Suite**: Create `src/llm_behavior_diff/prompts/default.yaml` with 50+ diverse prompts. (verify: file exists and is valid YAML)
4. **Model Runner**: Implement `runner.py` to support Ollama and OpenRouter. (verify: successful mock response from both)
5. **Semantic Differ & Judge**: Implement `differ.py` and `judge.py`. (verify: similarity scores and judge labels for sample pairs)
6. **Report Generator**: Create Jinja2 templates and `report.py`. (verify: HTML file generated with sample data)
7. **CLI Implementation**: Build `cli.py` using `click` and `rich`. (verify: `llm-diff --help` works)
8. **MCP Server**: Implement `mcp_server.py`. (verify: `mcp dev` or inspector can see the tool)
9. **Testing**: Write 15+ unit tests in `tests/`. (verify: `pytest` passes)
10. **Final Integration & Documentation**: Create README.md and .env.example. (verify: end-to-end run with mock/local models)

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/app/llm_behavior_diff_2355/pyproject.toml` | Dependency and build config |
| `/app/llm_behavior_diff_2355/src/llm_behavior_diff/` | Source code directory |
| `/app/llm_behavior_diff_2355/tests/` | Test suite |
| `/app/llm_behavior_diff_2355/README.md` | Documentation |

## Evaluation Criteria
- Successful semantic diffing of two models.
- HTML report generated with behavioral categories.
- MCP server correctly exposes the diff tool.
- CLI provides a polished UX with progress bars.
- Test coverage for core logic.
