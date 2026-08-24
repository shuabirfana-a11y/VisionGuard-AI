from html import escape
import json

from app.schemas import AnalysisResponse


def report_json(result: AnalysisResponse) -> bytes:
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2).encode("utf-8")


def report_html(result: AnalysisResponse) -> str:
    risk_labels = {
        "critical": "极高",
        "high": "高",
        "medium": "中",
        "low": "低",
        "unknown": "待核查",
    }
    risk_label = risk_labels.get(result.risk.overall_level, result.risk.overall_level)
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
    uncertainties = uncertainties or "<li>未记录额外不确定性；仍须遵守人工复核边界。</li>"
    trace = "".join(
        "<li>"
        f'<span class="step">{index}</span><div><b>{escape(item.tool)}</b>'
        f"<p>{escape(item.summary)}</p>"
        f"<small>步骤耗时：{item.duration_ms:.2f} ms"
        + (
            "；引用：" + "、".join(escape(reference) for reference in item.references)
            if item.references
            else ""
        )
        + "</small></div></li>"
        for index, item in enumerate(result.agent_trace, start=1)
    )
    visual_fallback = "是" if result.vision.inference.fallback_used else "否"
    reasoning_fallback = "是" if result.reasoning.fallback_used else "否"
    visual_evidence_count = len(result.vision.detections) + int(
        bool(classification and classification.available)
    )
    iou = (
        f"{result.vision.inference.iou_threshold:.2f}"
        if result.vision.inference.iou_threshold is not None
        else "不适用"
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{escape(result.report.title)}</title>
<style>
*{{box-sizing:border-box}}body{{font-family:'Microsoft YaHei',sans-serif;color:#172334;max-width:900px;margin:36px auto;padding:0 24px;line-height:1.7}}
h1{{color:#17365d;border-bottom:3px solid #2e74b5;padding-bottom:12px;margin-bottom:8px}}h2{{color:#2e74b5;margin-top:28px;border-bottom:1px solid #d9e5ef;padding-bottom:5px}}
.subtitle{{color:#64748b;margin-top:0}}.meta{{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px;background:#eef4fa;padding:16px;border-radius:8px}}
.meta .wide{{grid-column:1/-1}}.level{{display:inline-block;font-size:24px;font-weight:700;color:#9b3518;border-left:5px solid #d96032;padding:5px 12px;background:#fff3ed}}
.audit{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}}.audit div{{border:1px solid #cbdcea;border-radius:7px;padding:9px}}.audit b,.audit small{{display:block}}.audit small{{color:#64748b}}
.boundary{{border-left:4px solid #d79b29;background:#fff8e8;padding:12px}}.trace{{padding:0;list-style:none}}.trace li{{display:grid;grid-template-columns:28px 1fr;gap:10px;margin:12px 0}}.trace p{{margin:0}}.trace small{{color:#64748b}}.step{{display:grid;width:25px;height:25px;place-items:center;border-radius:50%;background:#2e74b5;color:white;font-weight:700}}
.disclaimer{{margin-top:30px;border:1px solid #e8c56d;background:#fff9e9;padding:12px;color:#6d5316}}footer{{margin-top:18px;color:#64748b;font-size:12px}}
button{{border:0;border-radius:6px;padding:9px 14px;background:#2e74b5;color:white;cursor:pointer}}
@media(max-width:650px){{.meta,.audit{{grid-template-columns:1fr}}.meta .wide{{grid-column:auto}}}}
@media print{{body{{margin:10mm auto;padding:0}}button{{display:none}}a{{color:inherit;text-decoration:none}}h2{{break-after:avoid}}li,.meta,.boundary{{break-inside:avoid}}}}
</style></head><body>
<button onclick="window.print()">打印 / 另存为PDF</button>
<h1>{escape(result.report.title)}</h1>
<p class="subtitle">工业安全视觉风险辅助分析报告 · 结论须由现场安全人员复核</p>
<div class="meta"><div class="wide"><b>分析任务：</b>{escape(result.task)}</div><div><b>报告编号：</b>{escape(result.request_id)}</div>
<div><b>视觉模型：</b>{escape(result.vision.inference.model_name)} / {escape(result.vision.inference.model_version)}</div>
<div><b>模型阈值：</b>置信度 {result.vision.inference.confidence_threshold:.2f} / IoU {iou}</div><div><b>视觉设备：</b>{escape(result.vision.inference.device)} / {result.vision.inference.inference_ms:.2f} ms</div>
<div><b>推理器：</b>{escape(result.reasoning.provider)} / {escape(result.reasoning.model)}</div><div><b>回退状态：</b>视觉 {visual_fallback} / 推理 {reasoning_fallback}</div></div>
<h2>风险结论</h2><div class="level">{escape(risk_label)}</div><p>{escape(result.risk.summary)}</p>
<div class="audit"><div><small>可引用视觉证据</small><b>{visual_evidence_count} 条</b></div><div><small>知识依据</small><b>{len(result.knowledge)} 条</b></div><div><small>Agent工具步骤</small><b>{len(result.agent_trace)} 步</b></div><div><small>人工复核</small><b>必须</b></div></div>
<h2>视觉证据</h2><ul>{evidence}</ul>
<h2>可信推理</h2><p>{escape(result.reasoning.explanation)}</p><div class="boundary">{escape(result.reasoning.safety_boundary)}</div>
<h2>知识依据</h2><ul>{citations}</ul>
<h2>不确定性</h2><ul>{uncertainties}</ul>
<h2>建议动作</h2><ul>{actions}</ul>
<h2>Agent审计轨迹</h2><ol class="trace">{trace}</ol>
<div class="disclaimer"><b>使用边界：</b>{escape(result.report.disclaimer)}</div>
<footer>VisionGuard AI · 专业视觉取证 + Agent任务编排 + 安全知识增强</footer></body></html>"""
