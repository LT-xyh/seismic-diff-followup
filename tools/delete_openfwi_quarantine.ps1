param(
    [string]$OpenFWIRoot = "src\data\openfwi",
    [string]$QuarantineName = "",
    [switch]$All,
    [switch]$Apply
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath([string]$Path) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Assert-UnderRoot([string]$Path, [string]$RootPath) {
    $full = Resolve-FullPath $Path
    $rootFull = Resolve-FullPath $RootPath
    $prefix = $rootFull.TrimEnd('\') + '\'
    if ($full -ne $rootFull -and -not $full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside OpenFWI root: $full"
    }
    return $full
}

function Count-NpyRecursive([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Dir -Recurse -Filter *.npy -File).Count
}

$root = Assert-UnderRoot $OpenFWIRoot $OpenFWIRoot
if (-not (Test-Path -LiteralPath $root)) {
    throw "OpenFWI root does not exist: $root"
}

$candidates = @()
if ($QuarantineName.Trim().Length -gt 0) {
    $target = Assert-UnderRoot (Join-Path $root $QuarantineName) $root
    if (-not (Test-Path -LiteralPath $target)) {
        throw "Quarantine directory does not exist: $target"
    }
    $item = Get-Item -LiteralPath $target
    if (-not $item.PSIsContainer -or -not $item.Name.StartsWith("_quarantine_storage_")) {
        throw "Target is not an OpenFWI quarantine directory: $target"
    }
    $candidates = @($item)
} else {
    $allQuarantine = @(Get-ChildItem -LiteralPath $root -Directory -Filter "_quarantine_storage_*" | Sort-Object Name -Descending)
    if ($allQuarantine.Count -eq 0) {
        Write-Host "No OpenFWI quarantine directories found under: $root"
        exit 0
    }
    if ($All) {
        $candidates = $allQuarantine
    } else {
        $candidates = @($allQuarantine[0])
    }
}

Write-Host "OpenFWI root: $root"
Write-Host "Apply: $Apply"
if (-not $Apply) {
    Write-Host "This is a dry run. Re-run with -Apply to delete."
}
Write-Host ""
Write-Host "Directories selected for deletion:"

foreach ($dir in $candidates) {
    $full = Assert-UnderRoot $dir.FullName $root
    $npyCount = Count-NpyRecursive $full
    Write-Host ("- {0}  npy_files={1}" -f $full, $npyCount)
}

Write-Host ""
foreach ($dir in $candidates) {
    $full = Assert-UnderRoot $dir.FullName $root
    if ($Apply) {
        Write-Host "[APPLY] Delete $full"
        Remove-Item -LiteralPath $full -Recurse -Force
    } else {
        Write-Host "[DRY-RUN] Would delete $full"
    }
}

Write-Host ""
if ($Apply) {
    Write-Host "Quarantine deletion completed."
} else {
    Write-Host "Dry run completed. No files were changed."
}
