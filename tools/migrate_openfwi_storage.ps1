param(
    [string]$OpenFWIRoot = "src\data\openfwi",
    [string[]]$Datasets = @("FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB", "CurveFaultA"),
    [ValidateSet("Quarantine", "Delete")]
    [string]$Mode = "Quarantine",
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

function Count-Npy([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Dir -Filter *.npy -File).Count
}

function Step-Message([string]$Message) {
    if ($Apply) {
        Write-Host "[APPLY] $Message"
    } else {
        Write-Host "[DRY-RUN] $Message"
    }
}

$root = Assert-UnderRoot $OpenFWIRoot $OpenFWIRoot
if (-not (Test-Path -LiteralPath $root)) {
    throw "OpenFWI root does not exist: $root"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$quarantineRoot = Join-Path $root "_quarantine_storage_$timestamp"

Write-Host "OpenFWI root: $root"
Write-Host "Mode: $Mode"
Write-Host "Apply: $Apply"
if (-not $Apply) {
    Write-Host "This is a dry run. Re-run with -Apply to change files."
}

$plans = @()

foreach ($set in $Datasets) {
    $datasetDir = Assert-UnderRoot (Join-Path $root $set) $root
    if (-not (Test-Path -LiteralPath $datasetDir)) {
        throw "Missing dataset directory: $datasetDir"
    }

    $depthDir = Assert-UnderRoot (Join-Path $datasetDir "depth_vel") $root
    $oldRmsDir = Assert-UnderRoot (Join-Path $datasetDir "rms_vel") $root
    $rawRmsDir = Assert-UnderRoot (Join-Path $datasetDir "rms_vel_raw") $root
    $wellDir = Assert-UnderRoot (Join-Path $datasetDir "well_log") $root
    $timeDir = Assert-UnderRoot (Join-Path $datasetDir "time_vel") $root

    $depthCount = Count-Npy $depthDir
    $oldRmsExists = Test-Path -LiteralPath $oldRmsDir
    $rawRmsExists = Test-Path -LiteralPath $rawRmsDir
    $oldRmsCount = Count-Npy $oldRmsDir
    $rawRmsCount = Count-Npy $rawRmsDir
    $wellExists = Test-Path -LiteralPath $wellDir
    $timeExists = Test-Path -LiteralPath $timeDir

    if ($depthCount -le 0) {
        throw "$set has no depth_vel npy files."
    }
    if (-not $rawRmsExists -and -not $oldRmsExists) {
        throw "$set has neither rms_vel_raw nor rms_vel."
    }
    if ($rawRmsExists -and $oldRmsExists -and $oldRmsCount -ne $rawRmsCount) {
        throw "$set rms_vel count $oldRmsCount != rms_vel_raw count $rawRmsCount."
    }
    if ($rawRmsExists -and $rawRmsCount -le 0) {
        throw "$set has empty rms_vel_raw."
    }

    $plans += [PSCustomObject]@{
        Dataset = $set
        DatasetDir = $datasetDir
        OldRmsDir = $oldRmsDir
        RawRmsDir = $rawRmsDir
        WellDir = $wellDir
        TimeDir = $timeDir
        DepthCount = $depthCount
        OldRmsExists = $oldRmsExists
        RawRmsExists = $rawRmsExists
        OldRmsCount = $oldRmsCount
        RawRmsCount = $rawRmsCount
        WellExists = $wellExists
        TimeExists = $timeExists
    }
}

Write-Host ""
Write-Host "Validation summary:"
$plans | Select-Object Dataset, DepthCount, OldRmsExists, OldRmsCount, RawRmsExists, RawRmsCount, WellExists, TimeExists | Format-Table -AutoSize

if ($Mode -eq "Quarantine") {
    Step-Message "Quarantine root will be: $quarantineRoot"
    if ($Apply) {
        New-Item -ItemType Directory -Force -Path $quarantineRoot | Out-Null
    }
}

foreach ($plan in $plans) {
    Step-Message "Processing $($plan.Dataset)"

    if ($Mode -eq "Quarantine") {
        $quarantineSet = Join-Path $quarantineRoot $plan.Dataset
        if ($Apply) {
            New-Item -ItemType Directory -Force -Path $quarantineSet | Out-Null
        }

        if ($plan.RawRmsExists -and $plan.OldRmsExists) {
            $oldRmsDest = Join-Path $quarantineSet "rms_vel_noisy"
            Step-Message "Move old noisy rms_vel -> $oldRmsDest"
            if ($Apply) {
                Move-Item -LiteralPath $plan.OldRmsDir -Destination $oldRmsDest
            }
        }
        if ($plan.RawRmsExists) {
            Step-Message "Rename rms_vel_raw -> rms_vel"
            if ($Apply) {
                Rename-Item -LiteralPath $plan.RawRmsDir -NewName "rms_vel"
            }
        }
        if ($plan.WellExists) {
            $wellDest = Join-Path $quarantineSet "well_log"
            Step-Message "Move stored well_log -> $wellDest"
            if ($Apply) {
                Move-Item -LiteralPath $plan.WellDir -Destination $wellDest
            }
        }
        if ($plan.TimeExists) {
            $timeDest = Join-Path $quarantineSet "time_vel"
            Step-Message "Move time_vel -> $timeDest"
            if ($Apply) {
                Move-Item -LiteralPath $plan.TimeDir -Destination $timeDest
            }
        }
    } else {
        if ($plan.RawRmsExists -and $plan.OldRmsExists) {
            Step-Message "Delete old noisy rms_vel"
            if ($Apply) {
                Remove-Item -LiteralPath $plan.OldRmsDir -Recurse -Force
            }
        }
        if ($plan.RawRmsExists) {
            Step-Message "Rename rms_vel_raw -> rms_vel"
            if ($Apply) {
                Rename-Item -LiteralPath $plan.RawRmsDir -NewName "rms_vel"
            }
        }
        if ($plan.WellExists) {
            Step-Message "Delete stored well_log"
            if ($Apply) {
                Remove-Item -LiteralPath $plan.WellDir -Recurse -Force
            }
        }
        if ($plan.TimeExists) {
            Step-Message "Delete time_vel"
            if ($Apply) {
                Remove-Item -LiteralPath $plan.TimeDir -Recurse -Force
            }
        }
    }
}

Write-Host ""
if ($Apply) {
    Write-Host "Migration completed."
} else {
    Write-Host "Dry run completed. No files were changed."
}
