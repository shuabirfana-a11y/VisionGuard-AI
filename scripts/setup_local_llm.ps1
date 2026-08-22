$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot "runtime\llama-b10581-vulkan"
$runtimeCache = Join-Path $repoRoot "runtime\cache"
$modelRoot = Join-Path $repoRoot "models\llm"
$archive = Join-Path $runtimeCache "llama-b10581-bin-win-vulkan-x64.zip"
$server = Join-Path $runtimeRoot "llama-server.exe"
$model = Join-Path $modelRoot "qwen2.5-1.5b-instruct-q4_k_m.gguf"
$runtimeUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b10581/llama-b10581-bin-win-vulkan-x64.zip"
$runtimeSha256 = "03d22a6267330005a56de5841a39ee6efaf5524337b9c08030122ef17e189a80"
$modelUrl = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
$modelModelScopeUrl = "https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/master/qwen2.5-1.5b-instruct-q4_k_m.gguf"
$modelMirrorUrl = "https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf?download=true"
$modelSha256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"

function Assert-FileHash([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        return $false
    }
    return $true
}

New-Item -ItemType Directory -Force -Path $runtimeCache, $modelRoot | Out-Null
if (-not (Assert-FileHash $archive $runtimeSha256)) {
    Write-Host "Downloading pinned llama.cpp b10581 Vulkan runtime..."
    & curl.exe --location --fail --retry 3 --continue-at - --output $archive $runtimeUrl
    if ($LASTEXITCODE -ne 0) { throw "llama.cpp runtime download failed." }
    if (-not (Assert-FileHash $archive $runtimeSha256)) {
        throw "SHA-256 mismatch for downloaded llama.cpp runtime."
    }
}
if (-not (Test-Path -LiteralPath $server)) {
    New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot -Force
}
if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe is missing after extraction."
}
if (-not (Assert-FileHash $model $modelSha256)) {
    Write-Host "Downloading pinned Qwen2.5-1.5B-Instruct Q4_K_M model (about 1.04 GiB)..."
    & curl.exe --location --fail --retry 3 --continue-at - --output $model $modelUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Hugging Face is unavailable; resuming from Qwen's official ModelScope repository."
        & curl.exe --location --fail --retry 3 --continue-at - --output $model $modelModelScopeUrl
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Official hosts are unavailable; using a mirror before official SHA-256 verification."
        & curl.exe --location --fail --retry 3 --continue-at - --output $model $modelMirrorUrl
    }
    if ($LASTEXITCODE -ne 0) { throw "Qwen model download failed." }
    if (-not (Assert-FileHash $model $modelSha256)) {
        throw "SHA-256 mismatch for downloaded Qwen model."
    }
}

Write-Host "Local LLM runtime is ready."
Write-Host "Runtime: $server"
Write-Host "Model:   $model"
