const form = document.querySelector("#analysis-form");
const imageInput = document.querySelector("#image");
const evidenceStage = document.querySelector("#evidence-stage");
const evidenceCanvas = document.querySelector("#evidence-canvas");
const evidenceDownload = document.querySelector("#evidence-download");
const statusEl = document.querySelector("#status");
const resultEl = document.querySelector("#result");
const emptyEl = document.querySelector("#empty");
let sourceImage = null;
let latestVision = null;
let latestRequestId = null;
let sourceUrl = null;
let sourceVersion = 0;
let analysisVersion = 0;
let caseVersion = 0;
let imageReady = Promise.resolve(false);
let analysisController = null;
let questionController = null;
let caseController = null;
let waitingTimer = null;
const analyzeButton = document.querySelector("#analyze-button");
const cancelButton = document.querySelector("#cancel-analysis");
const imageFeedback = document.querySelector("#image-feedback");
const REQUEST_TIMEOUT_MS = 120000;

loadDemoCases();
updateMetrics();
loadRuntimeProfile();

document.querySelector("#readiness-button").addEventListener("click", runReadinessCheck);

async function runReadinessCheck() {
  const button = document.querySelector("#readiness-button");
  const overall = document.querySelector("#readiness-overall");
  const results = document.querySelector("#readiness-results");
  button.disabled = true;
  button.textContent = "正在检查模型与工具…";
  overall.textContent = "自检进行中";
  try {
    const response = await fetch("/api/v1/readiness", {cache: "no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "环境自检失败");
    const labels = {ready: "全部就绪", degraded: "可运行但存在降级", not_ready: "存在阻断项"};
    overall.textContent = labels[data.status] || data.status;
    overall.className = `readiness-overall ${data.status}`;
    results.hidden = false;
    results.innerHTML = data.checks.map(item => `
      <div class="readiness-item ${escapeHtml(item.status)}">
        <span>${item.status === "ready" ? "✓" : item.status === "error" ? "×" : "!"}</span>
        <div><b>${escapeHtml(item.label)}</b><small>${escapeHtml(item.detail)}</small></div>
      </div>`).join("");
  } catch (error) {
    overall.textContent = error.message;
    overall.className = "readiness-overall not_ready";
  } finally {
    button.disabled = false;
    button.textContent = "重新运行环境自检";
  }
}

async function loadRuntimeProfile() {
  const notice = document.querySelector("#runtime-notice");
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) throw new Error("运行状态不可用");
    const health = await response.json();
    const capabilities = health.capabilities || {};
    const vision = health.vision_backend === "yolo"
      ? "已配置专业视觉检测，实际运行情况以本次结果为准"
      : "当前为演示视觉模式，结果仅用于体验操作流程，不能作为专业检测结论";
    const reasoning = capabilities.llm_reasoning === "configured"
      ? "已配置大模型解释"
      : "当前使用规则解释";
    notice.textContent = vision + "。" + reasoning + "。结论须由现场人员复核。";
    notice.classList.toggle("runtime-ready", health.vision_backend === "yolo");
  } catch (error) {
    notice.textContent = `无法读取运行配置：${error.message}`;
  }
}

function setStatus(text, state = "idle") {
  statusEl.textContent = text;
  statusEl.className = "status " + state;
}

function stopCaseLoading() {
  caseVersion += 1;
  caseController?.abort();
  caseController = null;
}

function resetAnalysis(message = "图片已准备好，可以开始分析。") {
  analysisVersion += 1;
  analysisController?.abort();
  questionController?.abort();
  analysisController = null;
  questionController = null;
  clearInterval(waitingTimer);
  waitingTimer = null;
  latestVision = null;
  latestRequestId = null;
  resultEl.hidden = true;
  emptyEl.hidden = false;
  emptyEl.textContent = message;
  evidenceDownload.disabled = true;
  document.querySelector("#html-report").removeAttribute("href");
  document.querySelector("#json-report").removeAttribute("href");
  document.querySelector("#json-report").removeAttribute("download");
  document.querySelector("#qa-answer").hidden = true;
  document.querySelector("#qa-answer").textContent = "";
  document.querySelector("#qa-question").value = "";
  document.querySelector("#qa-form button").disabled = false;
  document.querySelector(".result-panel").setAttribute("aria-busy", "false");
  cancelButton.hidden = true;
  analyzeButton.disabled = !sourceImage;
  analyzeButton.textContent = "开始风险分析";
  setStatus(sourceImage ? "待分析" : "等待图片");
  drawEvidence();
}

function loadSelectedImage(file) {
  const version = ++sourceVersion;
  sourceImage = null;
  resetAnalysis("正在准备图片……");
  evidenceStage.hidden = true;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = null;
  if (!file) {
    imageFeedback.textContent = "请先选择一张图片。";
    emptyEl.textContent = "选择图片后开始分析。";
    return Promise.resolve(false);
  }
  const message = !["image/jpeg", "image/png", "image/webp"].includes(file.type)
    ? "仅支持 JPEG、PNG 或 WebP 图片，请重新选择。"
    : file.size > 10 * 1024 * 1024 ? "图片超过 10 MB，请压缩后重新选择。"
    : file.size === 0 ? "图片为空，请重新选择。" : null;
  if (message) {
    imageFeedback.textContent = message;
    emptyEl.textContent = message;
    setStatus("图片不可用", "error");
    return Promise.resolve(false);
  }
  imageFeedback.textContent = "正在读取图片……";
  const image = new Image();
  sourceUrl = URL.createObjectURL(file);
  return new Promise(resolve => {
    image.onload = () => {
      if (version !== sourceVersion) { resolve(false); return; }
      sourceImage = image;
      evidenceStage.hidden = false;
      imageFeedback.textContent = file.name + " · " + image.naturalWidth + " × " + image.naturalHeight + " 像素";
      emptyEl.textContent = "图片已准备好，可以开始分析。";
      analyzeButton.disabled = false;
      setStatus("待分析");
      drawEvidence();
      resolve(true);
    };
    image.onerror = () => {
      if (version === sourceVersion) {
        imageFeedback.textContent = "无法读取这张图片，请换一张完整的图片重试。";
        emptyEl.textContent = imageFeedback.textContent;
        setStatus("图片不可用", "error");
      }
      resolve(false);
    };
    image.src = sourceUrl;
  });
}

imageInput.addEventListener("change", () => {
  stopCaseLoading();
  imageReady = loadSelectedImage(imageInput.files[0]);
});

document.querySelector("#task").addEventListener("input", () => {
  if (latestRequestId || analysisController) {
    resetAnalysis("核查要求已更改，请重新分析当前图片。");
  }
});

cancelButton.addEventListener("click", () => {
  resetAnalysis("已停止等待本次结果，可修改任务后重新分析。服务器可能仍在完成原任务。");
});

async function responseData(response, defaultMessage) {
  let data;
  try { data = await response.json(); }
  catch (_) { throw new Error("服务返回异常，请稍后重试。"); }
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : defaultMessage);
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const version = sourceVersion;
  if (!await imageReady || version !== sourceVersion || !sourceImage || analysisController) return;
  resetAnalysis();
  const run = analysisVersion;
  const controller = new AbortController();
  analysisController = controller;
  const started = Date.now();
  analyzeButton.disabled = true;
  analyzeButton.textContent = "正在分析……";
  cancelButton.hidden = false;
  document.querySelector(".result-panel").setAttribute("aria-busy", "true");
  setStatus("正在分析", "running");
  emptyEl.textContent = "正在分析当前图片，完成后会显示结果。";
  waitingTimer = setInterval(() => {
    if (run === analysisVersion) emptyEl.textContent = "正在分析，已等待 " + Math.floor((Date.now() - started) / 1000) + " 秒。可以停止等待，或继续等候结果。";
  }, 1000);
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const payload = new FormData(form);
  try {
    const response = await fetch("/api/v1/analyze", { method: "POST", body: payload, signal: controller.signal });
    const data = await responseData(response, "分析未完成，请重试。");
    if (run !== analysisVersion || version !== sourceVersion) return;
    renderResult(data);
    updateMetrics();
    setStatus("分析完成", "done");
  } catch (error) {
    if (run !== analysisVersion || version !== sourceVersion) return;
    resetAnalysis(error.name === "AbortError" ? "等待超时，请检查服务状态后重试。" : error.message);
    setStatus("未完成，可重试", "error");
  } finally {
    clearTimeout(timeout);
    if (run === analysisVersion) {
      clearInterval(waitingTimer);
      analysisController = null;
      analyzeButton.disabled = !sourceImage;
      analyzeButton.textContent = "开始风险分析";
      cancelButton.hidden = true;
      document.querySelector(".result-panel").setAttribute("aria-busy", "false");
    }
  }
});

function renderResult(data) {
  latestRequestId = data.request_id;
  emptyEl.hidden = true;
  resultEl.hidden = false;
  document.querySelector("#result-context").textContent = "本次任务：" + data.task;
  const modeNotice = document.querySelector("#result-notice");
  const notices = [];
  if (data.vision.inference.backend !== "yolo" || data.vision.inference.fallback_used) notices.push("本次使用演示视觉模式，检测结果不能作为专业模型效果证明。");
  if (data.reasoning.fallback_used) notices.push("解释服务未完成，本次改用规则解释。");
  if (data.vision.fire_classification && !data.vision.fire_classification.available) notices.push("火情复核模型不可用，当前缺少该分支的确认。");
  modeNotice.textContent = notices.join(" ");
  modeNotice.hidden = notices.length === 0;
  const riskLabels = {critical: "极高", high: "高", medium: "中", low: "低", unknown: "待核查"};
  const riskCard = document.querySelector(".risk-card");
  riskCard.dataset.level = data.risk.overall_level;
  document.querySelector("#risk-level").textContent = riskLabels[data.risk.overall_level] || data.risk.overall_level;
  document.querySelector("#risk-summary").textContent = data.risk.summary;
  const visualEvidenceCount = data.vision.detections.length
    + (data.vision.fire_classification?.available ? 1 : 0);
  const fallbackCount = Number(Boolean(data.vision.inference.fallback_used))
    + Number(Boolean(data.reasoning.fallback_used));
  const toolStepCount = data.agent_trace.filter(item => item.tool !== "agent.plan").length;
  document.querySelector("#audit-summary").innerHTML = `
    <div><span>视觉证据</span><b>${visualEvidenceCount}</b><small>条可引用证据</small></div>
    <div><span>知识依据</span><b>${data.knowledge.length}</b><small>条来源记录</small></div>
    <div><span>Agent 工具</span><b>${toolStepCount}</b><small>个专业工具</small></div>
    <div><span>回退次数</span><b>${fallbackCount}</b><small>${fallbackCount ? "已明确标注" : "未发生回退"}</small></div>`;
  const plan = data.agent_plan;
  document.querySelector("#agent-plan").innerHTML = `
    <div class="plan-heading"><span>Agent 任务规划</span><b>${escapeHtml(plan.intent)}</b></div>
    <div class="plan-route">${plan.tool_sequence.map((tool, index) => `<code>${index + 1}. ${escapeHtml(tool)}</code>`).join("<i>→</i>")}</div>
    <p>${escapeHtml(plan.rationale)}</p>
    <small>目标输出：${escapeHtml(plan.requested_outputs.join("、"))}</small>
    <small>安全约束：${escapeHtml(plan.safety_constraints.join("；"))}</small>`;
  latestVision = data.vision;
  drawEvidence();
  evidenceDownload.disabled = !sourceImage || !sourceImage.complete;
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
          <b>整图火情复核</b>
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
  document.querySelector("#trace").innerHTML = data.agent_trace.map((item, index) => `
    <li>
      <span class="trace-index">${index + 1}</span>
      <div><b>${escapeHtml(item.tool)}</b><p>${escapeHtml(item.summary)}</p>
      <small>${Number(item.duration_ms || 0).toFixed(2)} ms${item.references?.length ? ` · 引用 ${renderReferences(item.references)}` : ""}</small></div>
    </li>`).join("");
  document.querySelector("#json").textContent = JSON.stringify(data, null, 2);
}

evidenceDownload.addEventListener("click", () => {
  if (!sourceImage || !latestVision || evidenceDownload.disabled) return;
  const link = document.createElement("a");
  link.download = `visionguard-evidence-${latestRequestId || "analysis"}.png`;
  const exportCanvas = document.createElement("canvas");
  paintEvidence(exportCanvas, sourceImage.naturalWidth, sourceImage.naturalHeight);
  link.href = exportCanvas.toDataURL("image/png");
  link.click();
});

document.querySelector("#qa-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!latestRequestId || questionController) return;
  const run = analysisVersion;
  const requestId = latestRequestId;
  const controller = new AbortController();
  questionController = controller;
  const question = document.querySelector("#qa-question").value.trim();
  const answer = document.querySelector("#qa-answer");
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  answer.hidden = false;
  answer.textContent = "正在根据这次结果核对你的问题……";
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`/api/v1/analyses/${encodeURIComponent(requestId)}/ask`, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await responseData(response, "暂时无法回答，请重试。");
    if (run !== analysisVersion || requestId !== latestRequestId) return;
    answer.innerHTML = `
      <p>${escapeHtml(data.answer)}</p>
      <div class="reference-row"><b>视觉引用</b> ${renderReferences(data.reasoning.evidence_ids)}</div>
      <div class="reference-row"><b>知识引用</b> ${renderReferences(data.reasoning.knowledge_ids)}</div>
      <small>${escapeHtml(data.agent_trace.map(item => item.summary).join("；"))}</small>`;
  } catch (error) {
    if (run === analysisVersion && requestId === latestRequestId) answer.textContent = error.name === "AbortError" ? "回答超时，请稍后重试。" : error.message;
  } finally {
    clearTimeout(timeout);
    if (run === analysisVersion && requestId === latestRequestId) {
      questionController = null;
      button.disabled = false;
    }
  }
});

async function loadDemoCases() {
  const container = document.querySelector("#demo-cases");
  try {
    const response = await fetch("/api/v1/demo-cases");
    const cases = await responseData(response, "示例暂不可用，请上传自己的图片。");
    container.innerHTML = cases.map(item => `
      <button type="button" class="demo-case" data-case-id="${escapeHtml(item.case_id)}" title="${escapeHtml(item.description)}">
        <span>${escapeHtml(item.name)}</span>
        <small>${escapeHtml(item.description)}</small>
        <small class="case-source">${escapeHtml(item.source_note)} · ${escapeHtml(item.license)}</small>
      </button>`).join("");
    container.querySelectorAll(".demo-case").forEach(button => {
      button.addEventListener("click", async () => {
        try {
          await runDemoCase(button.dataset.caseId, button.querySelector("span").textContent.trim());
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
  stopCaseLoading();
  const version = caseVersion;
  const controller = new AbortController();
  caseController = controller;
  imageInput.value = "";
  imageReady = loadSelectedImage(null);
  imageFeedback.textContent = "正在加载示例图片……";
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`/api/v1/demo-cases/${encodeURIComponent(caseId)}/image`, {signal: controller.signal});
    if (!response.ok) throw new Error("示例图片暂不可用，请重试或上传自己的图片。");
    const blob = await response.blob();
    if (version !== caseVersion) return;
    const extension = blob.type === "image/jpeg" ? "jpg" : "png";
    const file = new File([blob], `${caseId}.${extension}`, {type: blob.type});
    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
    document.querySelector("#task").value = `核查示例“${name}”中的火焰与烟雾线索，说明依据及需要现场确认的事项。`;
    imageReady = loadSelectedImage(file);
    const loaded = await imageReady;
    if (version !== caseVersion || !loaded) return;
    form.requestSubmit();
  } catch (error) {
    if (version !== caseVersion) return;
    const message = error.name === "AbortError" ? "示例加载超时，请重试。" : error.message;
    imageFeedback.textContent = message;
    emptyEl.textContent = message;
    setStatus("示例加载失败", "error");
  } finally {
    clearTimeout(timeout);
    if (version === caseVersion) caseController = null;
  }
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
  paintEvidence(evidenceCanvas, width, height);
}

function paintEvidence(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
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
