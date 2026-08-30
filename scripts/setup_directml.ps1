$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$appPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$runtimePython = if ($env:VISIONGUARD_BOOTSTRAP_PYTHON) {
    $env:VISIONGUARD_BOOTSTRAP_PYTHON
} elseif (Test-Path -LiteralPath $appPython) {
    $appPython
} else {
    "python"
}
$environmentRoot = Join-Path $repoRoot ".venv-firebench-dml"
$environmentPython = Join-Path $environmentRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $runtimePython -m venv $environmentRoot
}

& $environmentPython -m pip install --upgrade pip
& $environmentPython -m pip install `
    torch-directml==0.2.5.dev240914 `
    timm==1.0.21 `
    joblib==1.5.2 `
    numpy==1.26.4 `
    scikit-learn==1.8.0 `
    safetensors==0.7.0 `
    pillow==11.3.0

& $environmentPython -c "import torch, torch_directml; d=torch_directml.device(); x=torch.tensor([1.0]).to(d); print({'torch': torch.__version__, 'device': str(d), 'probe': float(x.cpu()[0])})"
