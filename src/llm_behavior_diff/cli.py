"""CLI for LLM Behavior Diff — Model Update Detector."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

from .models import (
    ModelConfig, ProviderType, PromptSuite, Settings, ComparisonRun, 
    ReportConfig, DiffResult
)
from .runner import LLMRunner
from .differ import create_differ
from .judge import CombinedScorer, LLMJudge
from .report import ReportGenerator

# Main app - use invoke_without_command to show help when no subcommand given
app = typer.Typer(
    help="LLM Behavior Diff — Model Update Detector",
    rich_markup_mode="rich",
    invoke_without_command=True
)

console = Console()


def load_prompt_suite(path: Path) -> PromptSuite:
    """Load a prompt suite from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return PromptSuite(**data)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version information"),
):
    """LLM Behavior Diff — Model Update Detector.
    
    A tool for detecting semantic behavioral changes between two LLM versions.
    """
    if version:
        console.print("[bold]LLM Behavior Diff[/bold] version 0.1.0")
        raise typer.Exit()
    # If no subcommand was invoked, show help
    if ctx.invoked_subcommand is None:
        console.print("[bold blue]LLM Behavior Diff[/bold blue]")
        console.print("\nUsage: llm-diff [COMMAND] [OPTIONS]")
        console.print("\nCommands:")
        console.print("  run    Run a comparison between two models")
        console.print("\nRun 'llm-diff --help' for more information.")


@app.command()
def run(
    model_a: str = typer.Option(..., "--model-a", help="Model A name (e.g., llama3)"),
    provider_a: ProviderType = typer.Option(ProviderType.OLLAMA, "--provider-a", help="Provider for Model A"),
    model_b: str = typer.Option(..., "--model-b", help="Model B name (e.g., llama3.1)"),
    provider_b: ProviderType = typer.Option(ProviderType.OLLAMA, "--provider-b", help="Provider for Model B"),
    prompts: Path = typer.Option(Path("prompts/default.yaml"), "--prompts", help="Path to prompt suite YAML"),
    output: Path = typer.Option(Path("output/report.html"), "--output", help="Path to save HTML report"),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold for change detection"),
    use_judge: bool = typer.Option(True, "--use-judge/--no-use-judge", help="Use LLM judge for scoring"),
):
    """Run a comparison between two models."""
    asyncio.run(_run_async(
        model_a, provider_a, model_b, provider_b, prompts, output, threshold, use_judge
    ))


async def _run_async(
    m_a: str, p_a: ProviderType, m_b: str, p_b: ProviderType, 
    prompts_path: Path, output_path: Path, threshold: float, use_judge: bool
):
    """Async execution of the comparison."""
    console.print(Panel.fit(
        "[bold blue]LLM Behavior Diff[/bold blue]\n"
        "[dim]Detecting behavioral shifts between model updates[/dim]"
    ))

    # 1. Setup
    settings = Settings(
        similarity_threshold=threshold,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    try:
        suite = load_prompt_suite(prompts_path)
    except Exception as e:
        console.print(f"[bold red]Error loading prompt suite:[/bold red] {e}")
        raise typer.Exit(1)

    config_a = ModelConfig(name=m_a, provider=p_a)
    config_b = ModelConfig(name=m_b, provider=p_b)
    
    runner = LLMRunner(settings)
    differ = create_differ(settings)
    
    scorer = CombinedScorer(differ=differ)
    if use_judge:
        judge = LLMJudge(runner, settings)
        scorer.judge = judge

    # 2. Execution
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Comparing models...", total=len(suite.prompts))
        
        for prompt in suite.prompts:
            progress.update(task, description=f"[cyan]Processing: {prompt.id}")
            
            # Run models
            resp_a, resp_b = await runner.compare(prompt, config_a, config_b)
            
            # Score
            score = await scorer.score(prompt, resp_a, resp_b)
            
            # Detect change
            is_change = score.combined_score < settings.similarity_threshold
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
            
            progress.advance(task)

    # 3. Reporting
    run_data = ComparisonRun(
        id=f"run-{int(os.times().elapsed)}",
        model_a=config_a,
        model_b=config_b,
        prompt_suite=suite,
        results=results
    )
    
    report_gen = ReportGenerator()
    report = report_gen.save_report(run_data, settings, str(output_path))
    
    # 4. Summary Table
    console.print("\n[bold]Comparison Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    stats = report.get_summary_stats()
    table.add_row("Total Prompts", str(stats["total_prompts"]))
    table.add_row("Changes Detected", f"[bold red]{stats['behavioral_changes']}[/bold red]")
    table.add_row("Change Rate", f"{stats['change_rate']*100:.1f}%")
    table.add_row("Avg Similarity", f"{stats['avg_similarity']*100:.1f}%")
    
    console.print(table)
    console.print(f"\n[bold green]✔ Report saved to:[/bold green] {output_path}")


if __name__ == "__main__":
    app()
