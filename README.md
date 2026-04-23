# llm-behavior-diff

Compare two LLM versions on the same prompts and flag meaningful behavioral
drift: embedding-based semantic similarity, optional LLM-as-judge scoring, and
an HTML report you can drop into a CI artifact or diff review. Ships a CLI, a
Python API, and an MCP server so Claude Code or any MCP-compatible agent can
run a diff before a model swap.

## Install

```
pip install -e .
```

Python 3.11+. Embedding similarity uses `sentence-transformers/all-MiniLM-L6-v2`
(downloaded on first use). The LLM-judge path requires `OPENROUTER_API_KEY`;
without it, scoring falls back to embeddings-only.

## Quickstart

A `stub` provider returns deterministic hashed responses, so the whole
pipeline runs offline without Ollama or an API key:

```
llm-diff run \
  --model-a stub-a --provider-a stub \
  --model-b stub-b --provider-b stub \
  --prompts prompts/default.yaml \
  --output output/report.html \
  --no-use-embeddings
```

Observed output (real run, stub + Jaccard, threshold 0.5):

```
╭───────────────────────────────────────────────────╮
│ LLM Behavior Diff                                 │
│ Detecting behavioral shifts between model updates │
╰───────────────────────────────────────────────────╯
  Processing: safety-001 ━━━━━━━━━━━━━━━━━━━━ 100%

Comparison Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric           ┃ Value ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Prompts    │ 5     │
│ Changes Detected │ 3     │
│ Change Rate      │ 60.0% │
│ Avg Similarity   │ 40.0% │
└──────────────────┴───────┘
Report saved to: output/stub_jaccard.html
```

Same stub run with `--use-embeddings` (cosine on `all-MiniLM-L6-v2`)
gives Avg Similarity 77.9% (stub strings share the prompt prefix so
embeddings rate them more similar than Jaccard does).

### Real OpenRouter run

```
export OPENROUTER_API_KEY=sk-or-...
llm-diff run \
  --model-a meta-llama/llama-3.2-3b-instruct --provider-a openrouter \
  --model-b google/gemini-2.0-flash-lite-001 --provider-b openrouter \
  --prompts prompts/default.yaml \
  --output output/or_emb.html \
  --use-embeddings --threshold 0.85
```

Observed (real API, embeddings only):

```
Comparison Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Total Prompts    │ 5     │
┃ Changes Detected │ 0     │
┃ Change Rate      │ 0.0%  │
┃ Avg Similarity   │ 91.4% │
└──────────────────┴───────┘
```

With `--use-judge` added (Gemini 2.0 Flash Lite as judge, the default),
Avg Similarity 91.8% and sample judge reasoning:

> "Both responses correctly answer 'yes' and provide essentially the
> same explanation... Response A is slightly more verbose, but the core
> meaning is identical."

With `--no-use-embeddings --threshold 0.7` (Jaccard) the same two
OpenRouter models diverge sharply: Avg Similarity 25.0%, 5/5 changes —
the Llama and Gemini answers share few exact tokens even when semantically
identical, which is exactly why the embeddings path is on by default.

Swap `--provider-a ollama --provider-b ollama` and real model names
(`qwen3:8b`, `gemma4:e4b`, etc.) to diff actual Ollama models.

## CLI reference

### `llm-diff --help`

```
 Usage: llm-diff [OPTIONS] COMMAND [ARGS]...

 LLM Behavior Diff — Model Update Detector

 --version                Show version information
 --help                   Show this message and exit.

 Commands
   run  Run a comparison between two models.
```

### `llm-diff --version`

```
LLM Behavior Diff version 0.1.0
```

### `llm-diff run --help`

Key options:

| Option                       | Default                | Description                                              |
|------------------------------|------------------------|----------------------------------------------------------|
| `--model-a`                  | required               | Model A identifier                                       |
| `--provider-a`               | `ollama`               | `ollama`, `openrouter`, or `stub`                        |
| `--model-b`                  | required               | Model B identifier                                       |
| `--provider-b`               | `ollama`               | `ollama`, `openrouter`, or `stub`                        |
| `--prompts`                  | `prompts/default.yaml` | Prompt suite YAML                                        |
| `--output`                   | `output/report.html`   | HTML report path                                         |
| `--threshold`                | `0.85`                 | Below this combined score, a prompt counts as a change   |
| `--use-judge / --no-use-judge` | off                  | LLM-as-judge scoring (needs `OPENROUTER_API_KEY`)        |
| `--use-embeddings / --no-use-embeddings` | on         | Sentence-transformer embeddings vs. Jaccard fallback     |

Severity buckets (applied only when a change is detected):
`combined >= 0.7` -> minor, `>= 0.4` -> moderate, `< 0.4` -> major.

## Prompt suite format

`prompts/default.yaml` ships 5 prompts spanning reasoning, coding, factual,
instruction-following, and safety. Schema:

```yaml
name: "My suite"
version: "1.0.0"
prompts:
  - id: "code-001"
    text: "Write a Python function reverse_string(s)..."
    category: "coding"          # reasoning|coding|creativity|safety|instruction_following|factual|conversational
    tags: ["python"]
    expected_behavior: "Short correct function"
```

IDs must be unique. `category` must be one of the enum values above.

## Python API

```python
import asyncio
from llm_behavior_diff.runner import LLMRunner, run_prompt_sync
from llm_behavior_diff.differ import EmbeddingDiffer, SimpleDiffer, create_differ
from llm_behavior_diff.models import ModelConfig, ProviderType

# Synchronous one-shot call
resp = run_prompt_sync(
    ModelConfig(name="stub-m", provider=ProviderType.STUB),
    prompt_id="p1",
    prompt_text="hello world",
)
print(resp.text, resp.success)
# -> Model stub-m says: 921fac0c4c True

# Similarity
d = SimpleDiffer()
print(d.compute_similarity("the cat sat", "the cat ran"))   # 0.5

# Or with embeddings (downloads model on first call)
e = EmbeddingDiffer()
print(e.compute_similarity("The answer is 4.", "Two plus two equals four."))
# -> ~0.59
```

`create_differ(use_embeddings=False)` returns a `SimpleDiffer` (Jaccard);
`True` returns an `EmbeddingDiffer` if `sentence-transformers` is importable,
otherwise falls back to `SimpleDiffer`.

Report generation from a `ComparisonRun`:

```python
from llm_behavior_diff.report import ReportGenerator
from llm_behavior_diff.models import Settings
ReportGenerator().save_report(run, Settings(), "out.html")
```

`ReportGenerator` looks for a `report.html` Jinja template in the CWD, the
package directory, and a legacy path, then falls back to a built-in template
so reports always render.

## MCP server

Start the server (stdio transport):

```
llm-diff-mcp
# or: python -m llm_behavior_diff.mcp_server
```

It exposes three tools:

- `compare_models` — run a full prompt suite through two models and return
  per-prompt similarity + severity + response text.
- `analyze_drift` — score drift between two candidate responses for one prompt.
- `generate_report` — render a simple HTML summary from a JSON list of results.

Smoke test (all three, offline, via Python):

```python
import asyncio, json
from llm_behavior_diff.mcp_server import (
    compare_models, CompareModelsRequest,
    analyze_drift, AnalyzeDriftRequest,
    generate_report, GenerateReportRequest,
)

async def main():
    a = await compare_models(CompareModelsRequest(
        model_a="stub-a", provider_a="stub",
        model_b="stub-b", provider_b="stub",
        prompts_path="prompts/default.yaml",
        threshold=0.5, use_embeddings=False,
    ))
    print(a.total_prompts, a.changes_detected, a.avg_similarity)
    # real: 5 4 0.3446

    b = await analyze_drift(AnalyzeDriftRequest(
        prompt_text="math",
        response_a="The answer is 4.",
        response_b="2+2 equals 4.",
        use_embeddings=True,
    ))
    print(b.embedding_similarity, b.severity)
    # real: 0.5572 moderate

    c = await generate_report(GenerateReportRequest(
        results_json=json.dumps([
            {"prompt_id":"p1","similarity_score":0.9,"behavioral_change":False,"severity":"none"},
            {"prompt_id":"p2","similarity_score":0.3,"behavioral_change":True,"severity":"major"},
        ]),
        output_path="output/mcp_report.html",
        title="MCP Smoke",
    ))
    print(c.success, c.output_path)

asyncio.run(main())
```

### Verified over stdio JSON-RPC

```
llm-diff-mcp   # speaks MCP 2024-11-05 on stdio
# tools/list -> compare_models, analyze_drift, generate_report
# tools/call analyze_drift {"prompt_text":"...","response_a":"Paris",
#   "response_b":"The capital is Paris.","use_embeddings":true}
# -> {"embedding_similarity":0.7761,"severity":"minor", ...}
```

### Claude Code config

```json
{
  "mcpServers": {
    "llm-behavior-diff": {
      "command": "llm-diff-mcp"
    }
  }
}
```

## How it works

1. Load a prompt suite (YAML -> `PromptSuite` pydantic model).
2. For each prompt, run it through Model A and Model B via `LLMRunner`
   (Ollama `/api/generate`, OpenRouter chat completions, or the deterministic
   `stub` provider).
3. Score each response pair with `EmbeddingDiffer` (cosine on
   `all-MiniLM-L6-v2` embeddings) or `SimpleDiffer` (Jaccard over words).
   Optionally combine with an LLM-as-judge score (OpenRouter, default
   `google/gemini-2.0-flash-lite-001`).
4. Classify each prompt as none/minor/moderate/major change against
   `--threshold`.
5. Render an HTML report and print a rich summary table.

## Testing

```
pytest
# 46 passed
```

## Limitations

- LLM-as-judge requires an OpenRouter API key; with none, judging is skipped
  and the combined score equals the embedding/Jaccard similarity.
- First embedding run downloads `all-MiniLM-L6-v2` from Hugging Face and is
  slow. Subsequent runs use the local cache.
- The Ollama client talks to `http://localhost:11434` by default
  (`OLLAMA_HOST` env var overrides); it does not spawn Ollama for you.
- The `stub` provider produces deterministic fake text keyed on model name,
  temperature, and prompt. It is useful for CI and demos, not for real
  behavioral conclusions.
- `llm-diff --version` exits before running a subcommand; `llm-diff` with no
  args prints help (no default action).

## License

MIT. See `LICENSE`.
