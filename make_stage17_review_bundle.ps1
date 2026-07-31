$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "scripts"))) {
    throw "Place this script in the root of multimodal-hte-breast-cancer before running it."
}

$BundleDir = Join-Path $Root "_stage17_review_bundle"
$ZipPath = Join-Path $Root "stage17_review_bundle.zip"

if (Test-Path $BundleDir) {
    Remove-Item $BundleDir -Recurse -Force
}
New-Item -ItemType Directory -Path $BundleDir -Force | Out-Null

function Copy-RelativeFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [switch]$Required
    )

    $Source = Join-Path $Root $RelativePath
    if (-not (Test-Path $Source -PathType Leaf)) {
        if ($Required) {
            throw "Required file not found: $RelativePath"
        }
        Write-Warning "Optional file not found: $RelativePath"
        return
    }

    $Destination = Join-Path $BundleDir $RelativePath
    $DestinationDir = Split-Path $Destination -Parent
    New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    Copy-Item $Source $Destination -Force
}

function Copy-MatchingFiles {
    param(
        [Parameter(Mandatory = $true)][string]$RelativeDirectory,
        [Parameter(Mandatory = $true)][string[]]$Patterns,
        [int64]$MaximumBytes = 52428800
    )

    $Directory = Join-Path $Root $RelativeDirectory
    if (-not (Test-Path $Directory -PathType Container)) {
        Write-Warning "Optional directory not found: $RelativeDirectory"
        return
    }

    foreach ($Pattern in $Patterns) {
        Get-ChildItem -Path $Directory -Filter $Pattern -File -ErrorAction SilentlyContinue |
            Sort-Object FullName -Unique |
            ForEach-Object {
                if ($_.Length -gt $MaximumBytes) {
                    Write-Warning "Skipped large file ($([math]::Round($_.Length / 1MB, 1)) MB): $($_.FullName)"
                    return
                }
                $Relative = $_.FullName.Substring($Root.Length).TrimStart('\')
                Copy-RelativeFile -RelativePath $Relative
            }
    }
}

function Write-CsvSample {
    param(
        [Parameter(Mandatory = $true)][string]$RelativeSource,
        [Parameter(Mandatory = $true)][string]$RelativeDestination,
        [int]$Rows = 5
    )

    $Source = Join-Path $Root $RelativeSource
    if (-not (Test-Path $Source -PathType Leaf)) {
        Write-Warning "Sample source not found: $RelativeSource"
        return
    }

    $Destination = Join-Path $BundleDir $RelativeDestination
    $DestinationDir = Split-Path $Destination -Parent
    New-Item -ItemType Directory -Path $DestinationDir -Force | Out-Null
    Get-Content -Path $Source -TotalCount ($Rows + 1) | Set-Content -Path $Destination -Encoding UTF8
}

Write-Host "Collecting Stage 16 source code and outputs..." -ForegroundColor Cyan

$RequiredFiles = @(
    "stage16_config.json",
    "scripts\_stage12_utils.py",
    "scripts\_stage16_utils.py",
    "scripts\41_replicate_estimators.py",
    "scripts\61_stage16_preflight.py",
    "scripts\62_decompose_exact_landmark_aipw.py",
    "scripts\63_outcome_model_robustness.py",
    "scripts\64_fold_and_influence_stability.py",
    "scripts\65_generate_stage16_decision.py"
)

foreach ($RelativePath in $RequiredFiles) {
    Copy-RelativeFile -RelativePath $RelativePath -Required
}

$OptionalFiles = @(
    "run_stage16_outcome_augmentation_audit.ps1",
    "README_STAGE16.md",
    "results\tables\57_common_target_estimator_bridge.csv",
    "results\tables\57_bridge_diagnostics.csv",
    "results\tables\59_stage15_decision.csv",
    "results\tables\59_stage15_decision.md"
)

foreach ($RelativePath in $OptionalFiles) {
    Copy-RelativeFile -RelativePath $RelativePath
}

Copy-MatchingFiles -RelativeDirectory "results\tables" -Patterns @(
    "61_*",
    "62_*",
    "63_*",
    "64_*",
    "65_*"
)

Copy-MatchingFiles -RelativeDirectory "data\derived\stage16" -Patterns @(
    "*.csv",
    "*.json",
    "*.txt",
    "*.md",
    "*.npy",
    "*.npz",
    "*.pkl",
    "*.joblib"
) -MaximumBytes 31457280

$LogDirectory = Join-Path $Root "results\logs"
if (Test-Path $LogDirectory -PathType Container) {
    $LatestLog = Get-ChildItem $LogDirectory -Filter "stage16_outcome_augmentation_audit_*.log" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $LatestLog) {
        $RelativeLog = $LatestLog.FullName.Substring($Root.Length).TrimStart('\')
        Copy-RelativeFile -RelativePath $RelativeLog
    } else {
        Write-Warning "No Stage 16 transcript was found."
    }
}

Write-CsvSample `
    -RelativeSource "data\derived\stage14_trace\53_candidate_06.csv" `
    -RelativeDestination "samples\53_candidate_06_first5_rows.csv" `
    -Rows 5

Write-CsvSample `
    -RelativeSource "data\derived\stage15\60_ccw_long_with_patient_id.csv" `
    -RelativeDestination "samples\60_ccw_long_with_patient_id_first5_rows.csv" `
    -Rows 5

$InventoryRows = Get-ChildItem $BundleDir -Recurse -File | ForEach-Object {
    [pscustomobject]@{
        relative_path = $_.FullName.Substring($BundleDir.Length).TrimStart('\')
        bytes = $_.Length
        sha256 = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$InventoryPath = Join-Path $BundleDir "bundle_inventory.csv"
$InventoryRows | Sort-Object relative_path | Export-Csv -Path $InventoryPath -NoTypeInformation -Encoding UTF8

$ReadmePath = Join-Path $BundleDir "README_STAGE17_REVIEW_BUNDLE.txt"
@"
Purpose
-------
This bundle contains the exact Stage 16 implementation and small supporting outputs needed to design Stage 17 without guessing function names, folds, nuisance-model interfaces, or output schemas.

Scientific target for Stage 17
------------------------------
1. Patient-level influence audit.
2. Original fold-2 forensic audit.
3. Prespecified repeated cross-fitting with bounded arm-specific ridge.
4. Repeated-score aggregation pilot.
5. No publication bootstrap and no new learner search yet.

The bundle intentionally excludes large raw or patient-level datasets when they exceed the configured size limit.
"@ | Set-Content -Path $ReadmePath -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item $ZipPath -Force
}
Compress-Archive -Path (Join-Path $BundleDir "*") -DestinationPath $ZipPath -Force

Write-Host "" 
Write-Host "Stage 17 review bundle created:" -ForegroundColor Green
Write-Host $ZipPath
Write-Host "Files included: $((Get-ChildItem $BundleDir -Recurse -File).Count)"
Write-Host "Upload stage17_review_bundle.zip for the exact Stage 17 implementation."
