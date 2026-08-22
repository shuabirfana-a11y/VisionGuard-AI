$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$appPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$yoloModel = Join-Path $repoRoot "models\third_party\yolov8-fire-smoke-v1.0.0\baseline_best.pt"
$classifierRoot = Join-Path $repoRoot "models\fire_highscore_submission"
$classifierCpuPython = Join-Path $repoRoot ".venv-firebench\Scripts\python.exe"
$classifierDmlPython = Join-Path $repoRoot ".venv-firebench-dml\Scripts\python.exe"
$classifierPython = if (Test-Path -LiteralPath $classifierDmlPython) {
    $classifierDmlPython
} else {
    $classifierCpuPython
}
$llamaServer = Join-Path $repoRoot "runtime\llama-b10581-vulkan\llama-server.exe"
$llmModel = Join-Path $repoRoot "models\llm\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$llmProcess = $null
$llmStartedHere = $false

if (-not (Test-Path -LiteralPath $appPython)) {
    throw "Application environment is missing. Create .venv and install project dependencies first."
}
if (-not (Test-Path -LiteralPath $yoloModel)) {
    & (Join-Path $PSScriptRoot "fetch_yolo_model.ps1")
}

$env:VISION_BACKEND = "yolo"
$env:YOLO_MODEL_PATH = $yoloModel
$env:YOLO_MODEL_VERSION = "fire-smoke-yolov8n-upstream-v1.0.0"
$env:YOLO_EXPECTED_SHA256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"
$env:YOLO_CONFIDENCE_THRESHOLD = "0.10"
$env:YOLO_IOU_THRESHOLD = "0.30"
$env:YOLO_DEVICE = "auto"
$env:VISION_FALLBACK_ENABLED = "true"

if ((Test-Path -LiteralPath $classifierRoot) -and (Test-Path -LiteralPath $classifierPython)) {
    & $classifierPython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    $classifierHasCuda = $LASTEXITCODE -eq 0
    & $classifierPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch_directml') else 1)"
    $classifierHasDirectML = $LASTEXITCODE -eq 0
    $env:FIRE_CLASSIFIER_ENABLED = "true"
    $env:FIRE_CLASSIFIER_ROOT = $classifierRoot
    $env:FIRE_CLASSIFIER_PYTHON = $classifierPython
    $env:FIRE_CLASSIFIER_MODE = "persistent"
    if ($classifierHasCuda) {
        # A single resident ensemble avoids duplicating the two large backbones in
        # VRAM. GPU execution already supplies the competition latency profile.
        $env:FIRE_CLASSIFIER_DEVICE = "cuda"
        $env:FIRE_CLASSIFIER_WORKER_COUNT = "1"
        Write-Host "FireBench competition profile: CUDA, 1 resident worker"
    } elseif ($classifierHasDirectML) {
        $env:FIRE_CLASSIFIER_DEVICE = "dml"
        $env:FIRE_CLASSIFIER_WORKER_COUNT = "1"
        Write-Host "FireBench competition profile: DirectML GPU, 1 resident worker"
    } else {
        $env:FIRE_CLASSIFIER_DEVICE = "cpu"
        $env:FIRE_CLASSIFIER_WORKER_COUNT = "4"
        Write-Warning "CUDA PyTorch is unavailable; using the slower CPU profile."
    }
    $env:FIRE_CLASSIFIER_EAGER_START = "true"
} else {
    $env:FIRE_CLASSIFIER_ENABLED = "false"
    Write-Warning "The optional teammate classifier is unavailable; starting with YOLO localization only."
}

if ((Test-Path -LiteralPath $llamaServer) -and (Test-Path -LiteralPath $llmModel)) {
    $llmReady = $false
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -TimeoutSec 2
        $llmReady = $true
    } catch {
        $logRoot = Join-Path $repoRoot "runtime\logs"
        New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
        $llmProcess = Start-Process -FilePath $llamaServer -ArgumentList @(
            "-m", $llmModel,
            "--alias", "visionguard-qwen2.5-1.5b",
            "--host", "127.0.0.1",
            "--port", "8081",
            "--ctx-size", "16384",
            "--gpu-layers", "all",
            "--no-webui"
        ) -PassThru -WindowStyle Hidden -WorkingDirectory (Split-Path -Parent $llamaServer) `
          -RedirectStandardOutput (Join-Path $logRoot "local-llm.stdout.log") `
          -RedirectStandardError (Join-Path $logRoot "local-llm.stderr.log")
        $llmStartedHere = $true
        for ($attempt = 0; $attempt -lt 120; $attempt++) {
            Start-Sleep -Milliseconds 500
            if ($llmProcess.HasExited) { break }
            try {
                $null = Invoke-RestMethod -Uri "http://127.0.0.1:8081/health" -TimeoutSec 2
                $llmReady = $true
                break
            } catch { }
        }
    }
    if ($llmReady) {
        $env:REASONING_BACKEND = "llm"
        $env:LLM_BASE_URL = "http://127.0.0.1:8081/v1"
        $env:LLM_API_KEY = "local-visionguard"
        $env:LLM_MODEL = "visionguard-qwen2.5-1.5b"
        $env:LLM_TIMEOUT_SECONDS = "30"
        $env:REASONING_FALLBACK_ENABLED = "true"
        Write-Host "Reasoning profile: local Qwen2.5-1.5B via llama.cpp Vulkan"
    } else {
        $env:REASONING_BACKEND = "deterministic"
        Write-Warning "Local LLM failed to become ready; deterministic reasoning remains active."
    }
} else {
    Write-Warning "Local LLM is not installed; run scripts\setup_local_llm.ps1 to enable it."
}

Set-Location -LiteralPath $repoRoot
try {
    & $appPython -m uvicorn app.main:app --host 127.0.0.1 --port 8000
} finally {
    if ($llmStartedHere -and $null -ne $llmProcess -and -not $llmProcess.HasExited) {
        Stop-Process -Id $llmProcess.Id -Force
    }
}
