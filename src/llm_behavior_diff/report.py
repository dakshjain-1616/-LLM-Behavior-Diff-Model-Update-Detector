"""Report generator using Jinja2 templates."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape, DictLoader

from .models import ComparisonRun, Report, Settings, ReportConfig


# Inline fallback template so the report works even if the repo has no templates/ directory.
_FALLBACK_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{{ config.title }}</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:40px;background:#f7f7f8;color:#222}
.container{max-width:1100px;margin:0 auto;background:#fff;padding:28px 32px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
h1{margin-top:0;border-bottom:2px solid #3a7bd5;padding-bottom:8px}
.meta{color:#666;font-size:13px;margin-bottom:20px}
.summary{background:#eef4ff;padding:14px 18px;border-radius:6px;margin:16px 0}
.summary span{display:inline-block;margin-right:24px}
table{width:100%;border-collapse:collapse;margin-top:16px}
th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
th{background:#f3f4f6;font-weight:600}
.sev-major{color:#b91c1c;font-weight:600}
.sev-moderate{color:#c2410c;font-weight:600}
.sev-minor{color:#a16207}
.sev-none{color:#15803d}
pre{white-space:pre-wrap;background:#f9fafb;padding:8px;border-radius:4px;font-size:12px}
</style>
</head>
<body>
<div class="container">
<h1>{{ config.title }}</h1>
<div class="meta">Generated {{ generated_at }} · Model A: <b>{{ run.model_a.name }}</b> ({{ run.model_a.provider.value }}) vs Model B: <b>{{ run.model_b.name }}</b> ({{ run.model_b.provider.value }})</div>
<div class="summary">
<span><b>Prompts:</b> {{ stats.total_prompts }}</span>
<span><b>Changes:</b> {{ stats.behavioral_changes }}</span>
<span><b>Change rate:</b> {{ "%.1f%%" % (stats.change_rate * 100) }}</span>
<span><b>Avg similarity:</b> {{ "%.3f" % stats.avg_similarity }}</span>
</div>
<table>
<tr><th>Prompt</th><th>Response A</th><th>Response B</th><th>Similarity</th><th>Severity</th></tr>
{% for r in run.results %}
<tr>
<td><b>{{ r.prompt_id }}</b><br><small>{{ r.prompt_text }}</small></td>
<td><pre>{{ r.response_a.text }}</pre></td>
<td><pre>{{ r.response_b.text }}</pre></td>
<td>{{ "%.3f" % r.semantic_score.combined_score }}</td>
<td class="sev-{{ r.change_severity }}">{{ r.change_severity }}</td>
</tr>
{% endfor %}
</table>
</div>
</body>
</html>
"""


class ReportGenerator:
    """Render HTML reports from a ComparisonRun."""

    def __init__(self, templates_dir: str = "templates") -> None:
        """Initialize the generator.

        Looks for a ``report.html`` template in the given directory, then falls
        back to a built-in template so reports always render.
        """
        self._external_env = None
        # Try the provided path and a few sensible fallbacks.
        candidates = []
        if os.path.isabs(templates_dir):
            candidates.append(templates_dir)
        else:
            candidates.append(templates_dir)
            candidates.append(os.path.join(os.getcwd(), templates_dir))
            here = Path(__file__).resolve()
            candidates.append(str(here.parent / templates_dir))
            candidates.append(str(here.parent.parent.parent / templates_dir))
            # Legacy path baked into older builds.
            candidates.append("/app/llm_behavior_diff_2355/templates")

        for c in candidates:
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "report.html")):
                self._external_env = Environment(
                    loader=FileSystemLoader(c),
                    autoescape=select_autoescape(["html", "xml"]),
                )
                break

        self._fallback_env = Environment(
            loader=DictLoader({"report.html": _FALLBACK_TEMPLATE}),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def _env(self) -> Environment:
        return self._external_env or self._fallback_env

    def generate_stats(self, run: ComparisonRun) -> Dict[str, Any]:
        """Return summary statistics for a run."""
        return {
            "total_prompts": len(run.results),
            "behavioral_changes": sum(1 for r in run.results if r.behavioral_change_detected),
            "change_rate": run.behavioral_change_rate,
            "avg_similarity": run.average_similarity,
        }

    def generate_html(self, run: ComparisonRun, settings: Settings) -> str:
        """Render HTML for a run."""
        template = self._env().get_template("report.html")
        stats = self.generate_stats(run)
        title = getattr(settings, "title", "LLM Behavior Diff Report")
        report_config = ReportConfig(title=title)
        return template.render(
            run=run,
            config=report_config,
            stats=stats,
            generated_at=datetime.utcnow(),
        )

    def save_report(self, run: ComparisonRun, settings: Settings, output_path: str) -> Report:
        """Render HTML and save it to ``output_path``; return a Report object."""
        html_content = self.generate_html(run, settings)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html_content)
        title = getattr(settings, "title", "LLM Behavior Diff Report")
        report_config = ReportConfig(title=title)
        return Report(
            run=run,
            config=report_config,
            generated_at=datetime.utcnow(),
            html_content=html_content,
        )
