param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

$LOG_DIR = Join-Path $ROOT "results\logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG = Join-Path $LOG_DIR "stage23_manuscript_integration_preflight_$STAMP.log"

$EXCLUDED_DIR_NAMES = @(
    ".git", ".venv", "venv", "__pycache__", "data", "results",
    "publication_assets_candidate_v9", "node_modules", "dist", "build"
)

$STALE_TOKENS = @(
    "28.773173", "27.731760", "29.890478", "39.508207",
    "39.548570", "41.758043", "10.972223", "79.850375"
)

$LOCKED_TOKENS = @(
    "22.951284", "-4.173791", "91.009574", "0.956667"
)

$STRONG_PATTERNS = @(
    "statistically\s+significant",
    "\bsignificantly\b",
    "causal\s+effect",
    "improved?\s+survival",
    "survival\s+benefit",
    "treatment\s+benefit",
    "\bproves?\b",
    "\bconfirmed?\b",
    "\bdefinitive\b",
    "positive\s+(causal\s+)?effect"
)

function Test-ExcludedPath {
    param([string]$FullName)
    $parts = $FullName -split '[\\/]'
    foreach ($part in $parts) {
        if ($EXCLUDED_DIR_NAMES -contains $part) { return $true }
    }
    return $false
}

function Get-TextRaw {
    param([string]$Path)
    return [System.IO.File]::ReadAllText($Path)
}

function Get-RelativePathSafe {
    param([string]$Path)
    try {
        return (Resolve-Path -LiteralPath $Path).Path.Substring($ROOT.Length).TrimStart('\')
    }
    catch {
        return $Path
    }
}

function Resolve-IncludePath {
    param(
        [string]$ParentFile,
        [string]$Token
    )

    if ([string]::IsNullOrWhiteSpace($Token)) { return $null }
    if ($Token.Contains('\')) { return $null }

    $parentDir = Split-Path -Parent $ParentFile
    $candidates = New-Object System.Collections.Generic.List[string]

    if ([System.IO.Path]::GetExtension($Token)) {
        $candidates.Add((Join-Path $parentDir $Token))
        $candidates.Add((Join-Path $ROOT $Token))
    }
    else {
        $candidates.Add((Join-Path $parentDir ($Token + ".tex")))
        $candidates.Add((Join-Path $parentDir $Token))
        $candidates.Add((Join-Path $ROOT ($Token + ".tex")))
        $candidates.Add((Join-Path $ROOT $Token))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Get-IncludeTokens {
    param([string]$Path)
    $text = Get-TextRaw $Path
    $pattern = '\\(input|include|subfile)\s*\{([^}]+)\}'
    $tokens = @()
    foreach ($m in [regex]::Matches($text, $pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)) {
        $tokens += $m.Groups[2].Value.Trim()
    }
    return $tokens
}

function Walk-ManuscriptDependencies {
    param([string]$MainFile)

    $visited = @{}
    $ordered = New-Object System.Collections.Generic.List[string]
    $edges = New-Object System.Collections.Generic.List[object]

    function Visit-One {
        param([string]$File)
        $resolved = (Resolve-Path -LiteralPath $File).Path
        if ($visited.ContainsKey($resolved)) { return }
        $visited[$resolved] = $true
        $ordered.Add($resolved)

        foreach ($token in (Get-IncludeTokens $resolved)) {
            $child = Resolve-IncludePath -ParentFile $resolved -Token $token
            $edges.Add([pscustomobject]@{
                Parent = Get-RelativePathSafe $resolved
                IncludeToken = $token
                Resolved = $(if ($child) { Get-RelativePathSafe $child } else { "" })
                Exists = [bool]$child
            })
            if ($child) { Visit-One $child }
        }
    }

    Visit-One $MainFile
    return [pscustomobject]@{ Files = $ordered; Edges = $edges }
}

function Show-SectionContext {
    param(
        [string]$Path,
        [int]$LineNumber,
        [int]$Before = 2,
        [int]$After = 8
    )
    $lines = Get-Content -LiteralPath $Path
    $start = [Math]::Max(1, $LineNumber - $Before)
    $end = [Math]::Min($lines.Count, $LineNumber + $After)
    for ($i = $start; $i -le $end; $i++) {
        Write-Host (("{0,5}: " -f $i) + $lines[$i - 1])
    }
}

Start-Transcript -Path $LOG -Force
try {
    Write-Host ("#" * 124)
    Write-Host "STAGE 23 - MANUSCRIPT DISCOVERY AND CANDIDATE V9 INTEGRATION PREFLIGHT"
    Write-Host ("#" * 124)
    Write-Host "Project root: $ROOT"
    Write-Host "Transcript: $LOG"
    Write-Host "No estimator, bootstrap, protocol-lock file, or manuscript source is modified."
    Write-Host "Everything needed for the next exact patch is printed into this single log."

    $texFiles = Get-ChildItem -Path $ROOT -Recurse -File -Filter *.tex |
        Where-Object { -not (Test-ExcludedPath $_.FullName) }

    if (-not $texFiles -or $texFiles.Count -eq 0) {
        throw "No manuscript .tex files were found outside excluded directories."
    }

    $candidateRows = foreach ($file in $texFiles) {
        $text = Get-TextRaw $file.FullName
        $score = 0
        if ($text -match '\\documentclass') { $score += 100 }
        if ($text -match '\\begin\s*\{document\}') { $score += 80 }
        if ($text -match '\\title') { $score += 20 }
        if ($text -match '\\maketitle') { $score += 10 }
        if ($file.FullName -match 'paper_A_treatment_effects') { $score += 25 }
        if ($file.Name -match '(?i)(main|manuscript|paper|article)') { $score += 15 }
        $includeCount = ([regex]::Matches($text, '\\(input|include|subfile)\s*\{')).Count
        $sectionCount = ([regex]::Matches($text, '\\(section|subsection|subsubsection)\*?\s*\{')).Count
        $score += [Math]::Min($includeCount, 30) * 2
        $score += [Math]::Min($sectionCount, 30)

        [pscustomobject]@{
            Score = $score
            Path = Get-RelativePathSafe $file.FullName
            DocumentClass = [bool]($text -match '\\documentclass')
            BeginDocument = [bool]($text -match '\\begin\s*\{document\}')
            Includes = $includeCount
            Sections = $sectionCount
            Bytes = $file.Length
            FullName = $file.FullName
        }
    }

    $candidateRows = $candidateRows | Sort-Object -Property @{Expression='Score';Descending=$true}, Path
    $main = $candidateRows[0]

    Write-Host ""
    Write-Host "RANKED MANUSCRIPT CANDIDATES"
    $candidateRows | Select-Object -First 25 Score,Path,DocumentClass,BeginDocument,Includes,Sections,Bytes | Format-Table -AutoSize | Out-String -Width 240 | Write-Host

    Write-Host "SELECTED MAIN MANUSCRIPT"
    Write-Host "  Path: $($main.Path)"
    Write-Host "  Score: $($main.Score)"

    $walk = Walk-ManuscriptDependencies $main.FullName
    $includedFiles = @($walk.Files)

    Write-Host ""
    Write-Host "SELECTED SOURCE TREE"
    $order = 0
    foreach ($file in $includedFiles) {
        $order++
        Write-Host (("  {0,2}. " -f $order) + (Get-RelativePathSafe $file))
    }

    Write-Host ""
    Write-Host "INCLUDE RESOLUTION"
    if ($walk.Edges.Count -eq 0) {
        Write-Host "  <no input/include/subfile commands found>"
    }
    else {
        $walk.Edges | Format-Table -AutoSize | Out-String -Width 260 | Write-Host
    }

    Write-Host ""
    Write-Host "SECTION LOCATIONS AND CONTEXT"
    $sectionPattern = '\\(section|subsection|subsubsection)\*?\s*\{([^}]*)\}'
    foreach ($file in $includedFiles) {
        $lines = Get-Content -LiteralPath $file
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $m = [regex]::Match($lines[$i], $sectionPattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
            if ($m.Success) {
                $title = $m.Groups[2].Value
                Write-Host ("- " + (Get-RelativePathSafe $file) + ":" + ($i + 1) + " [" + $title + "]")
                if ($title -match '(?i)(method|result|discussion|conclusion|abstract|survival|treatment|causal)') {
                    Show-SectionContext -Path $file -LineNumber ($i + 1) -Before 1 -After 10
                }
            }
        }
    }

    Write-Host ""
    Write-Host "LOCKED RESULT COVERAGE"
    foreach ($token in $LOCKED_TOKENS) {
        $count = 0
        foreach ($file in $includedFiles) {
            $text = Get-TextRaw $file
            $count += ([regex]::Matches($text, [regex]::Escape($token))).Count
        }
        Write-Host ("  {0}: {1}" -f $token, $count)
    }

    Write-Host ""
    Write-Host "STALE NUMERIC TOKENS"
    $staleFound = 0
    foreach ($file in $includedFiles) {
        $lines = Get-Content -LiteralPath $file
        for ($i = 0; $i -lt $lines.Count; $i++) {
            foreach ($token in $STALE_TOKENS) {
                if ($lines[$i].Contains($token)) {
                    $staleFound++
                    Write-Host ((Get-RelativePathSafe $file) + ":" + ($i + 1) + " token=" + $token)
                    Write-Host ("    " + $lines[$i].Trim())
                }
            }
        }
    }
    if ($staleFound -eq 0) { Write-Host "  <none found>" }

    Write-Host ""
    Write-Host "POTENTIALLY OVERSTRONG CLAIMS"
    $claimFound = 0
    foreach ($file in $includedFiles) {
        $lines = Get-Content -LiteralPath $file
        for ($i = 0; $i -lt $lines.Count; $i++) {
            foreach ($pattern in $STRONG_PATTERNS) {
                if ($lines[$i] -match $pattern) {
                    $claimFound++
                    Write-Host ((Get-RelativePathSafe $file) + ":" + ($i + 1) + " pattern=" + $pattern)
                    Write-Host ("    " + $lines[$i].Trim())
                }
            }
        }
    }
    if ($claimFound -eq 0) { Write-Host "  <none found by automatic patterns>" }

    $snippetDir = Join-Path $ROOT "paper_A_treatment_effects\publication_assets_candidate_v9\manuscript_snippets"
    $snippetFiles = @(
        "88_abstract_result_candidate_v9.txt",
        "88_methods_candidate_v9.tex",
        "88_results_candidate_v9.tex",
        "88_discussion_candidate_v9.tex",
        "88_conclusion_candidate_v9.tex"
    )

    Write-Host ""
    Write-Host "STAGE 22 MANUSCRIPT SNIPPETS - FULL CONTENT"
    foreach ($name in $snippetFiles) {
        $path = Join-Path $snippetDir $name
        Write-Host ("-" * 124)
        Write-Host $name
        Write-Host ("-" * 124)
        if (Test-Path -LiteralPath $path) {
            Get-Content -LiteralPath $path | Write-Host
        }
        else {
            Write-Host "MISSING: $path"
        }
    }

    Write-Host ""
    Write-Host "INTEGRATION RULES"
    Write-Host "  1. Locked point estimate: 22.951284 days."
    Write-Host "  2. Primary percentile 95% CI: -4.173791 to 91.009574 days."
    Write-Host "  3. The primary interval includes zero."
    Write-Host "  4. The 95.7% positive-bootstrap fraction is descriptive, not a p-value."
    Write-Host "  5. Describe an observational overlap-population RMST contrast, not proven efficacy."
    Write-Host "  6. Keep CCW results labelled as non-comparable design sensitivities."
    Write-Host "  7. Do not modify locked Candidate V9 code, configs, manifests, or final registries."

    Write-Host ""
    Write-Host ("#" * 124)
    Write-Host "STAGE 23 COMPLETED"
    Write-Host ("#" * 124)
    Write-Host "Selected main manuscript: $($main.Path)"
    Write-Host "Included source files: $($includedFiles.Count)"
    Write-Host "Stale numeric hits: $staleFound"
    Write-Host "Potential overstrong claim hits: $claimFound"
    Write-Host "No manuscript source was modified."
    Write-Host "Keep this single log file: $LOG"
}
finally {
    Stop-Transcript
}
