param(
    [string]$FactLensRepo = "C:\tmp\factlens_official_repo",
    [string]$OutputDir = "artifacts\metrics_run",
    [int]$DiscoveryCases = 10,
    [int]$DiscoveryRepeat = 1,
    [switch]$IncludeModelControllers,
    [string]$ModelName = "gpt-5-mini"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )
    Write-Host ""
    Write-Host "== $Name =="
    & $Body
}

Invoke-Step "FactLens structured completeness matrix" {
    python -m scripts.run_factlens_external_matrix `
        --factlens-repo $FactLensRepo `
        --output-dir (Join-Path $OutputDir "factlens_external_matrix")
}

$discoveryArgs = @(
    "-m", "scripts.run_factlens_rlm_discovery",
    "--factlens-repo", $FactLensRepo,
    "--output-dir", (Join-Path $OutputDir "factlens_rlm_discovery"),
    "--cases", "$DiscoveryCases",
    "--repeat", "$DiscoveryRepeat"
)

if ($IncludeModelControllers) {
    if (-not $env:OPENAI_API_KEY) {
        throw "OPENAI_API_KEY is required for -IncludeModelControllers"
    }
    $discoveryArgs += @("--include-model-controllers", "--model-name", $ModelName)
}

Invoke-Step "FactLens RLM discovery" {
    python @discoveryArgs
}

Invoke-Step "Metric invariant checks" {
    python -m scripts.check_metrics --artifact-root $OutputDir
}
