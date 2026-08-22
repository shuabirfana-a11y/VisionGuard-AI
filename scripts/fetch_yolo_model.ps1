param(
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Destination) {
    $Destination = Join-Path $repoRoot "models\third_party\yolov8-fire-smoke-v1.0.0\baseline_best.pt"
}

$downloadUrl = "https://github.com/z11hzy/yolov8-fire-smoke-detection/releases/download/v1.0.0/baseline_best.pt"
$expectedSha256 = "b91633799ceb052c814b4f8b77a37efc9a40f002d528df97d74463585fa4f28f"
$destinationPath = [IO.Path]::GetFullPath($Destination)
$destinationDir = Split-Path -Parent $destinationPath

New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null

if (Test-Path -LiteralPath $destinationPath) {
    $existingHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existingHash -eq $expectedSha256) {
        Write-Output "Model already present and verified: $destinationPath"
        exit 0
    }
    throw "Existing model hash does not match the published release; refusing to overwrite: $destinationPath"
}

$temporaryPath = "$destinationPath.download"
try {
    Invoke-WebRequest -Headers @{"User-Agent" = "VisionGuard-AI-model-download"} -Uri $downloadUrl -OutFile $temporaryPath
    $actualHash = (Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expectedSha256) {
        throw "Downloaded model hash mismatch. Expected $expectedSha256, got $actualHash"
    }
    Move-Item -LiteralPath $temporaryPath -Destination $destinationPath
    Write-Output "Downloaded and verified: $destinationPath"
} finally {
    if (Test-Path -LiteralPath $temporaryPath) {
        Remove-Item -LiteralPath $temporaryPath -Force
    }
}
