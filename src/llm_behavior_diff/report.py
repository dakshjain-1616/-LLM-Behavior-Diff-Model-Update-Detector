import os
from datetime import datetime
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .models import ComparisonRun, Report, Settings, ReportConfig

class ReportGenerator:
    def __init__(self, templates_dir: str = "templates"):
        if not os.path.isabs(templates_dir):
            templates_dir = os.path.join("/app/llm_behavior_diff_2355", templates_dir)

        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_stats(self, run: ComparisonRun) -> Dict[str, Any]:
        return {
            "total_prompts": len(run.results),
            "behavioral_changes": sum(1 for r in run.results if r.behavioral_change_detected),
            "change_rate": run.behavioral_change_rate,
            "avg_similarity": run.average_similarity
        }

    def generate_html(self, run: ComparisonRun, settings: Settings) -> str:
        template = self.env.get_template("report.html")
        stats = self.generate_stats(run)

        title = getattr(settings, 'title', "LLM Behavior Diff Report")
        report_config = ReportConfig(title=title)

        return template.render(
            run=run,
            config=report_config,
            stats=stats,
            generated_at=datetime.utcnow()
        )

    def save_report(self, run: ComparisonRun, settings: Settings, output_path: str) -> Report:
        html_content = self.generate_html(run, settings)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)

        title = getattr(settings, 'title', "LLM Behavior Diff Report")
        report_config = ReportConfig(title=title)

        return Report(
            run=run,
            config=report_config,
            generated_at=datetime.utcnow(),
            html_content=html_content
        )
