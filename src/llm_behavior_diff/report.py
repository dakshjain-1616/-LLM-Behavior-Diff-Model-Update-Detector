import os
from datetime import datetime
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .models import ComparisonRun, Report, Settings

class ReportGenerator:
    def __init__(self, templates_dir: str = "templates"):
        # Handle both absolute and relative paths
        if not os.path.isabs(templates_dir):
            # Assume relative to project root if not absolute
            templates_dir = os.path.join("/app/llm_behavior_diff_2355", templates_dir)
            
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_stats(self, run: ComparisonRun) -> Dict[str, Any]:
        total = len(run.results)
        changes = sum(1 for r in run.results if r.behavioral_change_detected)
        avg_sim = sum(r.semantic_score.combined_score for r in run.results) / total if total > 0 else 0
        
        return {
            "total_prompts": total,
            "behavioral_changes": changes,
            "change_rate": changes / total if total > 0 else 0,
            "avg_similarity": avg_sim
        }

    def generate_html(self, run: ComparisonRun, settings: Settings) -> str:
        template = self.env.get_template("report.html")
        stats = self.generate_stats(run)
        
        return template.render(
            run=run,
            config=settings,
            stats=stats,
            generated_at=datetime.now()
        )

    def save_report(self, run: ComparisonRun, settings: Settings, output_path: str) -> Report:
        html_content = self.generate_html(run, settings)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
            
        stats = self.generate_stats(run)
        return Report(
            run_id=run.run_id,
            generated_at=datetime.now(),
            output_path=output_path,
            summary_stats=stats
        )
