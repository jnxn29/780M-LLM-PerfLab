[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$ReleaseDate = "",
    [string]$OutRoot = "",
    [string]$ReleaseProfile = "resume_clean_no_ort",
    [bool]$SanitizePaths = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Split-Path -Parent $PSScriptRoot
}
$sourceRootResolved = [System.IO.Path]::GetFullPath($SourceRoot)

if ([string]::IsNullOrWhiteSpace($ReleaseDate)) {
    $ReleaseDate = (Get-Date).ToString("yyyyMMdd")
}
if ([string]::IsNullOrWhiteSpace($OutRoot)) {
    $OutRoot = Join-Path $sourceRootResolved ("release/v0.1.2-clean-no-ort-{0}/780m-llm-perflab" -f $ReleaseDate)
}
$targetRoot = [System.IO.Path]::GetFullPath($OutRoot)
$profilePath = Join-Path $sourceRootResolved ("release/profiles/{0}.json" -f $ReleaseProfile)
if (-not (Test-Path $profilePath)) {
    throw "release profile not found: $profilePath"
}

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath
    )
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    $args = @(
        $SourcePath,
        $DestinationPath,
        "/E",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
        "/XD",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache"
    )
    & robocopy @args | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed: $SourcePath -> $DestinationPath (exit=$LASTEXITCODE)"
    }
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    try {
        return [System.IO.Path]::GetRelativePath($BasePath, $TargetPath)
    }
    catch {
        $base = (Resolve-Path -LiteralPath $BasePath).Path.TrimEnd("\")
        $target = (Resolve-Path -LiteralPath $TargetPath).Path
        $baseUri = [System.Uri]::new(($base + "\"))
        $targetUri = [System.Uri]::new($target)
        $relative = $baseUri.MakeRelativeUri($targetUri).ToString()
        return [System.Uri]::UnescapeDataString($relative).Replace("/", "\")
    }
}

function Get-StringArray {
    param([object]$Value)
    $items = @()
    foreach ($item in @($Value)) {
        $text = ([string]$item).Trim()
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $items += $text
        }
    }
    return ,$items
}

function Copy-RelativeFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$SourceRootPath,
        [Parameter(Mandatory = $true)][string]$TargetRootPath
    )
    $src = Join-Path $SourceRootPath $RelativePath
    if (-not (Test-Path $src)) {
        return
    }
    $dst = Join-Path $TargetRootPath $RelativePath
    $dstParent = Split-Path -Parent $dst
    if (-not [string]::IsNullOrWhiteSpace($dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }
    Copy-Item -Path $src -Destination $dst -Force
}

function Convert-PathToForwardSlash {
    param([Parameter(Mandatory = $true)][string]$PathLike)
    return ($PathLike -replace "\\", "/")
}

function Get-PathPrivacyValue {
    param(
        [Parameter(Mandatory = $true)][string]$PathLike,
        [Parameter(Mandatory = $true)][string]$SourceRootPath,
        [Parameter(Mandatory = $true)][string]$ReleaseRootPath
    )
    if ([string]::IsNullOrWhiteSpace($PathLike)) {
        return $PathLike
    }
    if ($PathLike -match '^[a-zA-Z][a-zA-Z0-9+\.-]*://') {
        return $PathLike
    }
    $raw = $PathLike
    $candidate = $raw
    try {
        $candidate = [System.IO.Path]::GetFullPath($raw)
    }
    catch {
        return $raw
    }
    try {
        $sourceFull = [System.IO.Path]::GetFullPath($SourceRootPath)
        $releaseFull = [System.IO.Path]::GetFullPath($ReleaseRootPath)
        if ($candidate.StartsWith($sourceFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rel = Get-RelativePathCompat -BasePath $sourceFull -TargetPath $candidate
            return (Convert-PathToForwardSlash -PathLike $rel)
        }
        if ($candidate.StartsWith($releaseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
            $rel = Get-RelativePathCompat -BasePath $releaseFull -TargetPath $candidate
            return (Convert-PathToForwardSlash -PathLike $rel)
        }
    }
    catch {
        # fallback below
    }
    return [System.IO.Path]::GetFileName($candidate)
}

function Convert-ObjectPathFields {
    param(
        [Parameter(Mandatory = $true)]$InputObject,
        [Parameter(Mandatory = $true)][string]$SourceRootPath,
        [Parameter(Mandatory = $true)][string]$ReleaseRootPath
    )

    if ($null -eq $InputObject) {
        return $null
    }
    if ($InputObject -is [string]) {
        return (Get-PathPrivacyValue -PathLike $InputObject -SourceRootPath $SourceRootPath -ReleaseRootPath $ReleaseRootPath)
    }
    if ($InputObject -is [System.Collections.IDictionary]) {
        $result = [ordered]@{}
        foreach ($key in $InputObject.Keys) {
            $result[$key] = Convert-ObjectPathFields -InputObject $InputObject[$key] -SourceRootPath $SourceRootPath -ReleaseRootPath $ReleaseRootPath
        }
        return $result
    }
    if ($InputObject -is [System.Collections.IEnumerable] -and -not ($InputObject -is [string])) {
        $items = @()
        foreach ($item in $InputObject) {
            $items += @(Convert-ObjectPathFields -InputObject $item -SourceRootPath $SourceRootPath -ReleaseRootPath $ReleaseRootPath)
        }
        return ,$items
    }
    if ($InputObject -is [psobject]) {
        $result = [ordered]@{}
        foreach ($prop in $InputObject.PSObject.Properties) {
            $result[$prop.Name] = Convert-ObjectPathFields -InputObject $prop.Value -SourceRootPath $SourceRootPath -ReleaseRootPath $ReleaseRootPath
        }
        return $result
    }
    return $InputObject
}

function Get-ReleaseTextFiles {
    param([Parameter(Mandatory = $true)][string]$RootPath)
    $exts = @(
        ".md", ".txt", ".json", ".csv", ".yaml", ".yml",
        ".ps1", ".py", ".toml", ".ini", ".cfg", ".log"
    )
    return @(Get-ChildItem -Path $RootPath -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
            $ext = [System.IO.Path]::GetExtension($_.Name).ToLowerInvariant()
            $ext -in $exts
        })
}

function Invoke-ReleasePathSanitize {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [Parameter(Mandatory = $true)][string]$SourceRootPath,
        [Parameter(Mandatory = $true)][string]$ReleaseRootPath
    )
    $sourceNorm = Convert-PathToForwardSlash -PathLike ([System.IO.Path]::GetFullPath($SourceRootPath))
    $releaseNorm = Convert-PathToForwardSlash -PathLike ([System.IO.Path]::GetFullPath($ReleaseRootPath))
    $patternUser = '(?i)\b[A-Z]:[\\/](Users|Documents and Settings)[\\/][^\\/:\s]+(?:[\\/][^ \r\n\t"''`<>|]+)*'
    $patternAbs = '(?i)\b[A-Z]:[\\/][^ \r\n\t"''`<>|]+(?:[\\/][^ \r\n\t"''`<>|]+)*'
    foreach ($file in (Get-ReleaseTextFiles -RootPath $RootPath)) {
        $text = Get-Content -Path $file.FullName -Raw -Encoding UTF8
        $original = $text
        $text = $text.Replace((Convert-PathToForwardSlash -PathLike $sourceNorm), "<repo-root>")
        $text = $text.Replace(([System.IO.Path]::GetFullPath($SourceRootPath)), "<repo-root>")
        $text = $text.Replace((Convert-PathToForwardSlash -PathLike $releaseNorm), "<release-root>")
        $text = $text.Replace(([System.IO.Path]::GetFullPath($ReleaseRootPath)), "<release-root>")
        $text = [System.Text.RegularExpressions.Regex]::Replace($text, $patternUser, "<redacted-path>")
        $text = [System.Text.RegularExpressions.Regex]::Replace($text, $patternAbs, "<redacted-path>")
        if ($text -ne $original) {
            Set-Content -Path $file.FullName -Value $text -Encoding UTF8
        }
    }
}

if (Test-Path $targetRoot) {
    Remove-Item -Path $targetRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $targetRoot -Force | Out-Null

$profile = Get-Content -Path $profilePath -Raw -Encoding UTF8 | ConvertFrom-Json
$rootFiles = Get-StringArray -Value $profile.root_files
$copyDirs = Get-StringArray -Value $profile.copy_dirs
$copyFiles = Get-StringArray -Value $profile.copy_files
$perfFiles = Get-StringArray -Value $profile.perf_files
$perfDirs = Get-StringArray -Value $profile.perf_dirs
$excludedMarkers = Get-StringArray -Value $profile.excluded_markers

foreach ($file in $rootFiles) {
    Copy-RelativeFile -RelativePath $file -SourceRootPath $sourceRootResolved -TargetRootPath $targetRoot
}
foreach ($dir in $copyDirs) {
    $srcDir = Join-Path $sourceRootResolved $dir
    if (Test-Path $srcDir) {
        Copy-Tree -SourcePath $srcDir -DestinationPath (Join-Path $targetRoot $dir)
    }
}
foreach ($path in $copyFiles) {
    Copy-RelativeFile -RelativePath $path -SourceRootPath $sourceRootResolved -TargetRootPath $targetRoot
}
# Always include the active release profile so the bundle can re-run prepare/freeze.
Copy-RelativeFile -RelativePath ("release/profiles/{0}.json" -f $ReleaseProfile) -SourceRootPath $sourceRootResolved -TargetRootPath $targetRoot

$perfRoot = Join-Path $sourceRootResolved "reports/perf_timeline"
$perfTarget = Join-Path $targetRoot "reports/perf_timeline"
New-Item -ItemType Directory -Path $perfTarget -Force | Out-Null
foreach ($name in $perfFiles) {
    $src = Join-Path $perfRoot $name
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $perfTarget $name) -Force
    }
}
foreach ($dirName in $perfDirs) {
    $srcDir = Join-Path $perfRoot $dirName
    if (Test-Path $srcDir) {
        Copy-Tree -SourcePath $srcDir -DestinationPath (Join-Path $perfTarget $dirName)
    }
}

if ($SanitizePaths) {
    Invoke-ReleasePathSanitize -RootPath $targetRoot -SourceRootPath $sourceRootResolved -ReleaseRootPath $targetRoot
}

$manifestPath = Join-Path $targetRoot "MANIFEST.json"
$hashPath = Join-Path $targetRoot "SHA256SUMS.txt"
$runLockPath = Join-Path $targetRoot "RUN_IDS.lock.json"
$qualityPath = Join-Path $targetRoot "QUALITY_GATES.json"

$resumeSummaryPath = Join-Path $sourceRootResolved "docs/resume_release_v0_1/resume_kpi_summary.json"
$repeatStabilityPath = Join-Path $sourceRootResolved "docs/resume_release_v0_1/repeat_stability.json"
$readinessPath = Join-Path $sourceRootResolved "reports/perf_timeline/release_readiness.json"
$runLock = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    source_root = "."
    release_root = "."
    snapshot_inputs = @{}
}
if (Test-Path $resumeSummaryPath) {
    try {
        $summary = Get-Content -Path $resumeSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($summary.snapshot_inputs) {
            $runLock.snapshot_inputs = $summary.snapshot_inputs
        }
    }
    catch {
    }
}
$runLockSanitized = Convert-ObjectPathFields -InputObject $runLock -SourceRootPath $sourceRootResolved -ReleaseRootPath $targetRoot
$runLockSanitized | ConvertTo-Json -Depth 20 | Set-Content -Path $runLockPath -Encoding UTF8

$quality = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    release_readiness_path = "reports/perf_timeline/release_readiness.json"
    release_readiness_exists = (Test-Path $readinessPath)
    repeat_stability_path = "docs/resume_release_v0_1/repeat_stability.json"
    repeat_stability_exists = (Test-Path $repeatStabilityPath)
}
if (Test-Path $readinessPath) {
    try {
        $ready = Get-Content -Path $readinessPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $quality["release_ready"] = [bool]$ready.ready
    }
    catch {
        $quality["release_ready"] = $false
    }
}
$qualitySanitized = Convert-ObjectPathFields -InputObject $quality -SourceRootPath $sourceRootResolved -ReleaseRootPath $targetRoot
$qualitySanitized | ConvertTo-Json -Depth 20 | Set-Content -Path $qualityPath -Encoding UTF8

$manifest = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    source_root = "."
    release_root = "."
    release_profile = $ReleaseProfile
    release_profile_path = ("release/profiles/{0}.json" -f $ReleaseProfile)
    release_date = $ReleaseDate
    included_root_files = $rootFiles
    included_copy_dirs = $copyDirs
    included_copy_files = $copyFiles
    included_perf_files = $perfFiles
    included_perf_dirs = $perfDirs
    generated_files = @(
        "MANIFEST.json",
        "SHA256SUMS.txt",
        "RUN_IDS.lock.json",
        "QUALITY_GATES.json"
    )
    excluded = $excludedMarkers
    sanitize_paths = [bool]$SanitizePaths
}
$manifestSanitized = Convert-ObjectPathFields -InputObject $manifest -SourceRootPath $sourceRootResolved -ReleaseRootPath $targetRoot
$manifestSanitized | ConvertTo-Json -Depth 20 | Set-Content -Path $manifestPath -Encoding UTF8

# Hashes must include generated metadata files.
$allFiles = Get-ChildItem -Path $targetRoot -Recurse -File -ErrorAction SilentlyContinue
$hashLines = New-Object System.Collections.Generic.List[string]
foreach ($f in $allFiles) {
    $rel = Get-RelativePathCompat -BasePath $targetRoot -TargetPath $f.FullName
    $h = (Get-FileHash -Algorithm SHA256 -Path $f.FullName).Hash.ToLower()
    $hashLines.Add("$h  $($rel -replace '\\', '/')")
}
$hashLines | Set-Content -Path $hashPath -Encoding UTF8

$fileCount = (Get-ChildItem -Path $targetRoot -Recurse -File -ErrorAction SilentlyContinue).Count
$sizeBytes = (Get-ChildItem -Path $targetRoot -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
if (-not $sizeBytes) { $sizeBytes = 0 }

Write-Host "release_root: $targetRoot"
Write-Host ("file_count: {0}" -f $fileCount)
Write-Host ("size_mb: {0:N2}" -f ($sizeBytes / 1MB))
Write-Host "manifest: $manifestPath"
Write-Host "hashes: $hashPath"
