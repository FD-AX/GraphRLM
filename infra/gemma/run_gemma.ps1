param(
    [string]$ModelPath = "/models/huggingface/google/gemma-3-4b-it",
    [string]$ServedModelName = "gemma-3-4b-it",
    [string]$ContainerName = "jmlc-gemma-host",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$cachePath = Join-Path $repoRoot "hf-cache"

if (-not (Test-Path $cachePath)) {
    throw "Local hf-cache was not found: $cachePath"
}

$existing = docker ps -a --filter "name=$ContainerName" --format "{{.Names}}"

if ($existing -contains $ContainerName) {
    docker start $ContainerName | Out-Null
    Write-Host "Started existing container: $ContainerName"
    exit 0
}

docker run -d --name $ContainerName --gpus all `
    -p "${Port}:8000" `
    -e HF_TOKEN=$env:HF_TOKEN `
    -v "${cachePath}:/models/huggingface:ro" `
    graphmemory-gemma:dev `
    $ModelPath `
    --host 0.0.0.0 `
    --port 8000 `
    --served-model-name $ServedModelName

Write-Host "Created and started container: $ContainerName"
