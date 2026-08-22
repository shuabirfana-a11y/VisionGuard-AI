const form = document.querySelector("#analysis-form");
const imageInput = document.querySelector("#image");
const evidenceStage = document.querySelector("#evidence-stage");
const evidenceCanvas = document.querySelector("#evidence-canvas");
const statusEl = document.querySelector("#status");
const resultEl = document.querySelector("#result");
const emptyEl = document.querySelector("#empty");
let sourceImage = null;
let latestVision = null;
let latestRequestId = null;
let sourceUrl = null;

loadDemoCases();
updateMetrics();
loadRuntimeProfile();

async function loadRuntimeProfile() {
  const notice = document.querySelector("#runtime-notice");
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) throw new Error("运行状态不可用");
    const health = await response.json();
    const capabilities = health.capabilities || {};
    const vision = health.vision_backend === "yolo"
      ? "专业YOLO视觉定位"
      : "Demo视觉回退";
    const reasoning = capabilities.llm_reasoning === "configured"
      ? `大模型可信推理（${capabilities.llm_model || "已配置"}）`
      : "确定性可信推理";
    const classifier = capabilities.fire_classifier_runtime?.state === "ready"
      ? "三路火情确认已就绪"
      : "火情确认分支未就绪";
    notice.textContent = `比赛运行配置 ${health.version}：${vision} · ${classifier} · ${reasoning} · 可解释混合RAG。所有结论仍须现场人员复核。`;
    notice.classList.toggle("runtime-ready", health.vision_backend === "yolo");
  } catch (error) {
    notice.textContent = `无法读取运行配置：${error.message}`;
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = URL.createObjectURL(file);
  latestVision = null;
  sourceImage = new Image();
  sourceImage.onload = () => {
    evidenceStage.hidden = false;
    drawEvidence();
  };
  sourceImage.src = sourceUrl;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  statusEl.textContent = "Agent 分析中";
  statusEl.className = "status running";
  const payload = new FormData(form);
  try {
    const response = await fetch("/api/v1/analyze", { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "分析失败");
    renderResult(data);
    updateMetrics();
    statusEl.textContent = "分析完成";
    statusEl.className = "status done";
  } catch (error) {
    emptyEl.hidden = false;
    resultEl.hidden = true;
    emptyEl.textContent = error.message;
    statusEl.textContent = "任务失败";
    statusEl.className = "status error";
  } finally {
    button.disabled = false;
  }
});

function renderResult(data) {
  latestRequestId = data.request_id;
  emptyEl.hidden = true;
  resultEl.hidden = false;
  const riskLabels = {critical: "极高", high: "高", medium: "中", low: "低", unknown: "待核查"};
  document.querySelector("#risk-level").textContent = `${riskLabels[data.risk.overall_level] || data.risk.overall_level} · ${data.risk.overall_level}`;
  document.querySelector("#risk-summary").textContent = data.risk.summary;
  latestVision = data.vision;
  drawEvidence();
  const inference = data.vision.inference;
  const fallback = inference.fallback_used
    ? `<span class="fallback-badge">Demo回退：${escapeHtml(inference.fallback_reason || "未说明")}</span>`
    : `<span class="professional-badge">${inference.backend === "yolo" ? "专业视觉后端" : "演示视觉后端"}</span>`;
  document.querySelector("#model-meta").innerHTML = `
    <div>${fallback}</div>
    <dl>
      <div><dt>模型</dt><dd>${escapeHtml(inference.model_name)} · ${escapeHtml(inference.model_version)}</dd></div>
      <div><dt>阈值</dt><dd>置信度 ${Number(inference.confidence_threshold).toFixed(2)}${inference.iou_threshold == null ? "" : ` · IoU ${Number(inference.iou_threshold).toFixed(2)}`}</dd></div>
      <div><dt>推理耗时</dt><dd>${Number(inference.inference_ms).toFixed(2)} ms</dd></div>
      <div><dt>设备</dt><dd>${escapeHtml(inference.device)}</dd></div>
    </dl>`;
  const detections = document.querySelector("#detections");
  const detectorCards = data.vision.detections.map(item => `
      <div class="evidence">
        <b>${escapeHtml(item.label)}</b>
        <div>置信度 ${(item.confidence * 100).toFixed(1)}% · 区域占比 ${(item.area_ratio * 100).toFixed(2)}%</div>
        <div>定位 (${item.bbox.x1}, ${item.bbox.y1}) → (${item.bbox.x2}, ${item.bbox.y2})</div>
        <small>证据编号 ${escapeHtml(item.evidence_id)} · 来源 ${escapeHtml(item.source)}</small>
      </div>`).join("");
  const classifier = data.vision.fire_classification;
  const classifierCard = classifier
    ? classifier.available
      ? `<div class="evidence classifier-evidence">
          <b>高精度整图火情确认</b>
          <div>${classifier.prediction ? "判断存在可见火焰" : "未确认可见火焰"} · 概率 ${(classifier.probability * 100).toFixed(1)}%</div>
          <div>阈值 ${(classifier.threshold * 100).toFixed(1)}% · 不提供检测框</div>
          <small>证据编号 ${escapeHtml(classifier.evidence_id)} · ${escapeHtml(classifier.model_version)} · ${Number(classifier.inference_ms).toFixed(1)} ms</small>
        </div>`
      : `<div class="evidence classifier-unavailable"><b>高精度火情确认分支不可用</b><div>${escapeHtml(classifier.error || "未说明原因")}</div></div>`
    : "";
  const fusionCard = data.vision.fusion
    ? `<div class="evidence fusion-evidence"><b>双模型证据融合 · ${escapeHtml(data.vision.fusion.status)}</b><div>${escapeHtml(data.vision.fusion.summary)}</div></div>`
    : "";
  detections.innerHTML = detectorCards || classifierCard || fusionCard
    ? `${fusionCard}${detectorCards}${classifierCard}`
    : '<div class="evidence">未获得明确风险目标证据，仍需人工复核。</div>';
  const reasoning = data.reasoning;
  const reasoningBadge = reasoning.fallback_used
    ? `<span class="fallback-badge">推理回退：${escapeHtml(reasoning.fallback_reason || "未说明")}</span>`
    : reasoning.used_llm
      ? '<span class="professional-badge">大模型证据约束推理</span>'
      : '<span class="neutral-badge">确定性证据推理</span>';
  document.querySelector("#reasoning").innerHTML = `
    <div>${reasoningBadge}</div>
    <p>${escapeHtml(reasoning.explanation)}</p>
    <div class="reference-row"><b>视觉引用</b> ${renderReferences(reasoning.evidence_ids)}</div>
    <div class="reference-row"><b>知识引用</b> ${renderReferences(reasoning.knowledge_ids)}</div>
    <div class="reasoning-boundary">${escapeHtml(reasoning.safety_boundary)}</div>
    ${reasoning.uncertainties.length ? `<ul class="uncertainties">${reasoning.uncertainties.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}`;
  document.querySelector("#knowledge").innerHTML = data.knowledge.length
    ? data.knowledge.map(item => {
      const sourceUrl = safeHttpUrl(item.source_url);
      const sourceTitle = sourceUrl
        ? `<a class="source-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.source_title)}</a>`
        : escapeHtml(item.source_title);
      const authority = { law: "法律", department_rule: "部门规章", internal_method: "方法边界" }[item.authority_level] || "知识依据";
      return `
      <div class="evidence knowledge-card">
        <b>[${escapeHtml(item.citation_id)}] ${escapeHtml(item.title)}</b>
        <div>${sourceTitle} / ${escapeHtml(item.source_section)}</div>
        <small>${escapeHtml(authority)} · 版本 ${escapeHtml(item.source_version)} · 检索相关度 ${(item.retrieval_score * 100).toFixed(1)}%</small>
        <small>检索方式：${escapeHtml(item.retrieval_method || "知识检索")} ${item.matched_terms?.length ? `· 命中词 ${escapeHtml(item.matched_terms.join("、"))}` : ""}</small>
        <div class="knowledge-boundary">适用边界：${escapeHtml(item.applicability || "须结合现场适用条件复核")}</div>
      </div>`;
    }).join("")
    : '<div class="evidence">未检索到知识依据。</div>';
  document.querySelector("#actions").innerHTML = data.report.actions.map(item => `<li>${escapeHtml(item)}</li>`).join("");
  document.querySelector("#html-report").href = `/api/v1/analyses/${encodeURIComponent(data.request_id)}/report.html`;
  document.querySelector("#json-report").href = `/api/v1/analyses/${encodeURIComponent(data.request_id)}/report.json`;
  document.querySelector("#json-report").download = `visionguard-${data.request_id}.json`;
  document.querySelector("#trace").innerHTML = data.agent_trace.map(item => `<li><b>${escapeHtml(item.tool)}</b>：${escapeHtml(item.summary)}</li>`).join("");
  document.querySelector("#json").textContent = JSON.stringify(data, null, 2);
}

document.querySelector("#qa-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!latestRequestId) return;
  const question = document.querySelector("#qa-question").value.trim();
  const answer = document.querySelector("#qa-answer");
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  answer.hidden = false;
  answer.textContent = "Agent 正在基于本次证据回答……";
  try {
    const response = await fetch(`/api/v1/analyses/${encodeURIComponent(latestRequestId)}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "追问失败");
    answer.innerHTML = `
      <p>${escapeHtml(data.answer)}</p>
      <div class="reference-row"><b>视觉引用</b> ${renderReferences(data.reasoning.evidence_ids)}</div>
      <div class="reference-row"><b>知识引用</b> ${renderReferences(data.reasoning.knowledge_ids)}</div>
      <small>${escapeHtml(data.agent_trace.map(item => item.summary).join("；"))}</small>`;
  } catch (error) {
    answer.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function loadDemoCases() {
  const container = document.querySelector("#demo-cases");
  try {
    const response = await fetch("/api/v1/demo-cases");
    const cases = await response.json();
    container.innerHTML = cases.map(item => `
      <button type="button" class="demo-case" data-case-id="${escapeHtml(item.case_id)}" title="${escapeHtml(item.description)}">
        ${escapeHtml(item.name)}
      </button>`).join("");
    container.querySelectorAll(".demo-case").forEach(button => {
      button.addEventListener("click", async () => {
        try {
          await runDemoCase(button.dataset.caseId, button.textContent.trim());
        } catch (error) {
          statusEl.textContent = "案例加载失败";
          statusEl.className = "status error";
        }
      });
    });
  } catch (error) {
    container.textContent = "演示案例加载失败";
  }
}

async function runDemoCase(caseId, name) {
  const response = await fetch(`/api/v1/demo-cases/${encodeURIComponent(caseId)}/image`);
  if (!response.ok) throw new Error("演示案例加载失败");
  const blob = await response.blob();
  const file = new File([blob], `${caseId}.png`, {type: "image/png"});
  const transfer = new DataTransfer();
  transfer.items.add(file);
  imageInput.files = transfer.files;
  imageInput.dispatchEvent(new Event("change"));
  document.querySelector("#task").value = `运行比赛合成案例“${name}”，输出视觉证据、可信推理和人工复核建议。`;
  form.requestSubmit();
}

async function updateMetrics() {
  const container = document.querySelector("#metrics");
  try {
    const response = await fetch("/api/v1/metrics");
    const metrics = await response.json();
    container.innerHTML = `
      <div><span>分析次数</span><b>${metrics.total_analyses}</b></div>
      <div><span>平均总耗时</span><b>${Number(metrics.average_total_ms).toFixed(1)} ms</b></div>
      <div><span>平均视觉耗时</span><b>${Number(metrics.average_inference_ms).toFixed(1)} ms</b></div>
      <div><span>视觉回退</span><b>${metrics.vision_fallback_count}</b></div>`;
  } catch (error) {
    container.textContent = "指标暂不可用";
  }
}

function renderReferences(values) {
  return values.length
    ? values.map(value => `<code>${escapeHtml(value)}</code>`).join(" ")
    : '<span class="muted">无</span>';
}

function drawEvidence() {
  if (!sourceImage || !sourceImage.complete) return;
  const availableWidth = Math.max(260, evidenceStage.clientWidth || 480);
  const scale = Math.min(availableWidth / sourceImage.naturalWidth, 420 / sourceImage.naturalHeight, 1.5);
  const width = Math.round(sourceImage.naturalWidth * scale);
  const height = Math.round(sourceImage.naturalHeight * scale);
  evidenceCanvas.width = width;
  evidenceCanvas.height = height;
  const context = evidenceCanvas.getContext("2d");
  context.drawImage(sourceImage, 0, 0, width, height);
  if (!latestVision) return;
  const scaleX = width / latestVision.image_width;
  const scaleY = height / latestVision.image_height;
  latestVision.detections.forEach(item => {
    const color = item.category === "fire" ? "#ff7a35" : "#8dd4ff";
    const x = item.bbox.x1 * scaleX;
    const y = item.bbox.y1 * scaleY;
    const boxWidth = (item.bbox.x2 - item.bbox.x1) * scaleX;
    const boxHeight = (item.bbox.y2 - item.bbox.y1) * scaleY;
    context.strokeStyle = color;
    context.lineWidth = Math.max(2, width / 240);
    context.strokeRect(x, y, boxWidth, boxHeight);
    const text = `${item.label} ${(item.confidence * 100).toFixed(1)}%`;
    context.font = `bold ${Math.max(12, width / 36)}px sans-serif`;
    const textWidth = context.measureText(text).width + 12;
    const labelHeight = Math.max(22, width / 22);
    const labelY = Math.max(0, y - labelHeight);
    context.fillStyle = color;
    context.fillRect(x, labelY, Math.min(textWidth, width - x), labelHeight);
    context.fillStyle = "#07111f";
    context.fillText(text, x + 6, labelY + labelHeight * 0.72);
  });
}

window.addEventListener("resize", drawEvidence);

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
}

function safeHttpUrl(value) {
  if (!value) return null;
  try {
    const url = new URL(String(value));
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch (_) {
    return null;
  }
}
