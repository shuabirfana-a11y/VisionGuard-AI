from html import escape
import json

from app.schemas import AnalysisResponse


def report_json(result: AnalysisResponse) -> bytes:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")


def report_html(result: AnalysisResponse) -> str:
    evidence = "".join(
        f"<li><b>{escape(item.evidence_id)}</b>：{escape(item.label)}，置信度 {item.confidence:.2%}，"
        f"位置 ({item.bbox.x1}, {item.bbox.y1})-({item.bbox.x2}, {item.bbox.y2})</li>"
        for item in result.vision.detections
    )
    classification = result.vision.fire_classification
    if classification and classification.available:
        evidence += (
            f"<li><b>{escape(classification.evidence_id)}</b>：整图可见火焰判断"
            f"{'阳性' if classification.prediction else '阴性'}，概率 {classification.probability:.2%}，"
            "不提供检测框</li>"
        )
    evidence = evidence or "<li>未获得明确视觉风险证据。</li>"
    citations = "".join(
        f"<li>[{escape(item.citation_id)}] "
        + (
            f'<a href="{escape(item.source_url, quote=True)}">{escape(item.source_title)}</a>'
            if item.source_url
            else escape(item.source_title)
        )
        + f" / {escape(item.source_section)} / {escape(item.source_version)}"
        + f"<br><small>检索方式：{escape(item.retrieval_method)}；相关度：{item.retrieval_score:.1%}</small>"
        + f"<br><small>适用边界：{escape(item.applicability)}</small></li>"
        for item in result.knowledge
    )
    actions = "".join(f"<li>{escape(item)}</li>" for item in result.report.actions)
    uncertainties = "".join(f"<li>{escape(item)}</li>" for item in result.reasoning.uncertainties)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(result.report.title)}</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;color:#172334;max-width:850px;margin:40px auto;line-height:1.7}}
h1{{color:#17365d;border-bottom:3px solid #2e74b5;padding-bottom:12px}}h2{{color:#2e74b5;margin-top:28px}}
.meta{{background:#eef4fa;padding:14px;border-radius:8px}}.level{{font-size:24px;font-weight:700;color:#b64b20}}
.boundary{{border-left:4px solid #d79b29;background:#fff8e8;padding:12px}}footer{{margin-top:35px;color:#64748b;font-size:12px}}
@media print{{body{{margin:15mm}}button{{display:none}}}}
</style></head><body>
<button onclick="window.print()">打印 / 另存为PDF</button>
<h1>{escape(result.report.title)}</h1>
<div class="meta"><b>任务：</b>{escape(result.task)}<br><b>报告编号：</b>{escape(result.request_id)}<br>
<b>视觉模型：</b>{escape(result.vision.inference.model_name)} / {escape(result.vision.inference.model_version)}<br>
<b>推理器：</b>{escape(result.reasoning.provider)} / {escape(result.reasoning.model)}</div>
<h2>风险结论</h2><div class="level">{escape(result.risk.overall_level.upper())}</div><p>{escape(result.risk.summary)}</p>
<h2>视觉证据</h2><ul>{evidence}</ul>
<h2>可信推理</h2><p>{escape(result.reasoning.explanation)}</p><div class="boundary">{escape(result.reasoning.safety_boundary)}</div>
<h2>知识依据</h2><ul>{citations}</ul>
<h2>不确定性</h2><ul>{uncertainties}</ul>
<h2>建议动作</h2><ul>{actions}</ul>
<footer>{escape(result.report.disclaimer)}</footer></body></html>"""
