param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$OUT_ROOT = Join-Path $ROOT "results\review_bundles"
New-Item -ItemType Directory -Force -Path $OUT_ROOT | Out-Null

$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$BUNDLE_DIR = Join-Path $OUT_ROOT "stage25_v10_review_bundle_$STAMP"
$ZIP_PATH = "$BUNDLE_DIR.zip"

New-Item -ItemType Directory -Force -Path $BUNDLE_DIR | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "code") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "configs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "protocol") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "stage24") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "samples") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $BUNDLE_DIR "metadata") | Out-Null

function Copy-OptionalFile {
    param(
        [string]$RelativeSource,
        [string]$RelativeDestination
    )

    $source = Join-Path $ROOT $RelativeSource
    $destination = Join-Path $BUNDLE_DIR $RelativeDestination

    if (Test-Path -LiteralPath $source -PathType Leaf) {
        $destinationParent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
        return $true
    }

    return $false
}

function Write-CsvSample {
    param(
        [string]$RelativeSource,
        [string]$RelativeDestination,
        [int]$Rows = 5
    )

    $source = Join-Path $ROOT $RelativeSource
    $destination = Join-Path $BUNDLE_DIR $RelativeDestination

    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        return $false
    }

    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null

    $data = Import-Csv -LiteralPath $source
    $data | Select-Object -First $Rows |
        Export-Csv -LiteralPath $destination -NoTypeInformation -Encoding UTF8

    $headerPath = [System.IO.Path]::ChangeExtension($destination, ".columns.txt")
    $header = (Get-Content -LiteralPath $source -First 1)
    $header | Set-Content -LiteralPath $headerPath -Encoding UTF8

    return $true
}

Write-Host ("#" * 124)
Write-Host "STAGE 25 REVIEW BUNDLE - CANDIDATE V10 DESIGN"
Write-Host ("#" * 124)
Write-Host "This script does not fit a model and does not modify Candidate V9."
Write-Host "It collects the exact V9 estimator implementation and Stage 24 diagnostics."

$files = @(
    @{ Source = "stage20_config.json"; Destination = "configs\stage20_config.json" },
    @{ Source = "stage21_config.json"; Destination = "configs\stage21_config.json" },
    @{ Source = "paper_a_stage24_config.json"; Destination = "configs\paper_a_stage24_config.json" },

    @{ Source = "scripts\_common.py"; Destination = "code\_common.py" },
    @{ Source = "scripts\_stage12_utils.py"; Destination = "code\_stage12_utils.py" },
    @{ Source = "scripts\_stage16_utils.py"; Destination = "code\_stage16_utils.py" },
    @{ Source = "scripts\_stage17_utils.py"; Destination = "code\_stage17_utils.py" },
    @{ Source = "scripts\_stage18_utils.py"; Destination = "code\_stage18_utils.py" },
    @{ Source = "scripts\_stage19_utils.py"; Destination = "code\_stage19_utils.py" },
    @{ Source = "scripts\_stage20_utils.py"; Destination = "code\_stage20_utils.py" },
    @{ Source = "scripts\_stage21_utils.py"; Destination = "code\_stage21_utils.py" },

    @{ Source = "scripts\79_final_20_partition_point_estimate.py"; Destination = "code\79_final_20_partition_point_estimate.py" },
    @{ Source = "scripts\80_create_candidate_v9_protocol_lock.py"; Destination = "code\80_create_candidate_v9_protocol_lock.py" },
    @{ Source = "scripts\81_verify_candidate_v9_protocol_lock.py"; Destination = "code\81_verify_candidate_v9_protocol_lock.py" },
    @{ Source = "scripts\82_full_publication_bootstrap.py"; Destination = "code\82_full_publication_bootstrap.py" },
    @{ Source = "scripts\83_summarize_publication_bootstrap.py"; Destination = "code\83_summarize_publication_bootstrap.py" },
    @{ Source = "scripts\84_generate_publication_decision.py"; Destination = "code\84_generate_publication_decision.py" },

    @{ Source = "scripts\_paper_a_stage24_utils.py"; Destination = "code\_paper_a_stage24_utils.py" },
    @{ Source = "scripts\90_stage24_verify_v9_and_discover_cohort.py"; Destination = "code\90_stage24_verify_v9_and_discover_cohort.py" },
    @{ Source = "scripts\91_stage24_build_treatment_sequence_registry.py"; Destination = "code\91_stage24_build_treatment_sequence_registry.py" },
    @{ Source = "scripts\92_stage24_sequence_imbalance_and_feasibility.py"; Destination = "code\92_stage24_sequence_imbalance_and_feasibility.py" },
    @{ Source = "scripts\93_stage24_sequence_audit_decision.py"; Destination = "code\93_stage24_sequence_audit_decision.py" },

    @{ Source = "run_stage20_candidate_v9_protocol_lock.ps1"; Destination = "code\run_stage20_candidate_v9_protocol_lock.ps1" },
    @{ Source = "run_stage21_publication_bootstrap.ps1"; Destination = "code\run_stage21_publication_bootstrap.ps1" },
    @{ Source = "run_stage24_chemo_sequencing_audit.ps1"; Destination = "code\run_stage24_chemo_sequencing_audit.ps1" },

    @{ Source = "data\derived\manifests\80_candidate_v9_protocol_lock_manifest.json"; Destination = "protocol\80_candidate_v9_protocol_lock_manifest.json" },
    @{ Source = "paper_A_treatment_effects\analysis_plan_FINAL.md"; Destination = "protocol\analysis_plan_FINAL.md" },
    @{ Source = "paper_A_treatment_effects\primary_estimand_FINAL.json"; Destination = "protocol\primary_estimand_FINAL.json" },
    @{ Source = "paper_A_treatment_effects\model_registry_FINAL.json"; Destination = "protocol\model_registry_FINAL.json" },
    @{ Source = "paper_A_treatment_effects\bootstrap_registry_FINAL.json"; Destination = "protocol\bootstrap_registry_FINAL.json" },

    @{ Source = "results\tables\79_candidate_v9_final_point_estimate.csv"; Destination = "protocol\79_candidate_v9_final_point_estimate.csv" },
    @{ Source = "results\tables\79_candidate_v9_partition_estimates.csv"; Destination = "protocol\79_candidate_v9_partition_estimates.csv" },
    @{ Source = "results\tables\83_publication_bootstrap_summary.csv"; Destination = "protocol\83_publication_bootstrap_summary.csv" },
    @{ Source = "results\tables\84_publication_decision.csv"; Destination = "protocol\84_publication_decision.csv" },

    @{ Source = "results\tables\stage24_chemo_sequencing\s24_90_selected_v9_cohort.json"; Destination = "stage24\s24_90_selected_v9_cohort.json" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_91_source_registry.csv"; Destination = "stage24\s24_91_source_registry.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_91_sequence_registry_checks.csv"; Destination = "stage24\s24_91_sequence_registry_checks.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_92_binary_sequence_imbalance.csv"; Destination = "stage24\s24_92_binary_sequence_imbalance.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_92_chemo_landmark_categories.csv"; Destination = "stage24\s24_92_chemo_landmark_categories.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_92_chemo_hormone_sequence_categories.csv"; Destination = "stage24\s24_92_chemo_hormone_sequence_categories.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_92_sensitivity_population_feasibility.csv"; Destination = "stage24\s24_92_sensitivity_population_feasibility.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_93_decision_checks.csv"; Destination = "stage24\s24_93_decision_checks.csv" },
    @{ Source = "results\tables\stage24_chemo_sequencing\s24_93_chemo_sequencing_decision.json"; Destination = "stage24\s24_93_chemo_sequencing_decision.json" }
)

$inventory = foreach ($item in $files) {
    $copied = Copy-OptionalFile `
        -RelativeSource $item.Source `
        -RelativeDestination $item.Destination

    [pscustomobject]@{
        source = $item.Source
        destination = $item.Destination
        copied = $copied
    }
}

$sampleRequests = @(
    @{
        Source = "data\derived\landmark_cohorts\outer_hormone_hrpos_her2neg_landmark180.csv"
        Destination = "samples\v9_full_cohort_first5.csv"
        Rows = 5
    },
    @{
        Source = "data\derived\landmark_compact\outer_hormone_hrpos_her2neg_landmark180.csv"
        Destination = "samples\v9_compact_cohort_first5.csv"
        Rows = 5
    },
    @{
        Source = "data\derived\landmark_weights\outer_hormone_hrpos_her2neg_landmark180.csv"
        Destination = "samples\v9_weights_first5.csv"
        Rows = 5
    },
    @{
        Source = "results\tables\stage24_chemo_sequencing\s24_91_patient_sequence_registry_LOCAL_ONLY.csv"
        Destination = "samples\stage24_sequence_registry_first10.csv"
        Rows = 10
    }
)

$sampleInventory = foreach ($item in $sampleRequests) {
    $written = Write-CsvSample `
        -RelativeSource $item.Source `
        -RelativeDestination $item.Destination `
        -Rows $item.Rows

    [pscustomobject]@{
        source = $item.Source
        destination = $item.Destination
        rows_requested = $item.Rows
        written = $written
    }
}

$inventory |
    Export-Csv `
        -LiteralPath (Join-Path $BUNDLE_DIR "metadata\file_copy_inventory.csv") `
        -NoTypeInformation `
        -Encoding UTF8

$sampleInventory |
    Export-Csv `
        -LiteralPath (Join-Path $BUNDLE_DIR "metadata\sample_inventory.csv") `
        -NoTypeInformation `
        -Encoding UTF8

$hashInventory = Get-ChildItem -Path $BUNDLE_DIR -Recurse -File |
    ForEach-Object {
        [pscustomobject]@{
            relative_path = $_.FullName.Substring($BUNDLE_DIR.Length).TrimStart('\')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }

$hashInventory |
    Sort-Object relative_path |
    Export-Csv `
        -LiteralPath (Join-Path $BUNDLE_DIR "metadata\bundle_hash_inventory.csv") `
        -NoTypeInformation `
        -Encoding UTF8

$readme = @"
Purpose
-------
This bundle contains the exact locked Candidate V9 implementation and the Stage 24
chemotherapy-sequencing diagnostics needed to design Candidate V10 without
guessing estimator interfaces or data schemas.

Scientific conclusion from Stage 24
-----------------------------------
The full V9 cohort has severe chemotherapy-sequencing imbalance. The most useful
candidate primary sensitivity population is patients with no chemotherapy initiated
by day 180, subject to explicit exclusion of unascertainable chemotherapy timing.

What the next package should do
-------------------------------
1. Define a clean Candidate V10 population without future-treatment exclusions.
2. Preserve Candidate V9 as an immutable historical analysis.
3. Lock the V10 estimand and cohort before looking at a new effect estimate.
4. Reuse the exact V9 ATO-AIPW RMST estimator rather than rewriting it from memory.
5. Refit all nuisance models and run a new patient bootstrap only after V10 lock.
6. Keep chemotherapy-started-by-day180 patients descriptive because overlap and
   event counts are inadequate.

This bundle does not fit any model and does not modify Candidate V9.
"@

$readme |
    Set-Content `
        -LiteralPath (Join-Path $BUNDLE_DIR "README_STAGE25_V10_REVIEW.txt") `
        -Encoding UTF8

if (Test-Path -LiteralPath $ZIP_PATH) {
    Remove-Item -LiteralPath $ZIP_PATH -Force
}

Compress-Archive `
    -Path (Join-Path $BUNDLE_DIR "*") `
    -DestinationPath $ZIP_PATH `
    -Force

Write-Host ""
Write-Host "Stage 25 review bundle created:"
Write-Host $ZIP_PATH
Write-Host "Files in bundle: $((Get-ChildItem $BUNDLE_DIR -Recurse -File).Count)"
Write-Host ""
Write-Host "Upload the ZIP file for exact Candidate V10 implementation."
