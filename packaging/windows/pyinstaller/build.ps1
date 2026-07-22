<#
.SYNOPSIS
    Build and smoke-test the frozen Windows custback engine (WIN-5.1).

.DESCRIPTION
    Creates an isolated build venv, installs custback with the Windows runtime
    profile, freezes it with PyInstaller (onedir), and runs a clean-environment
    smoke test of the produced artifact.  The smoke test scrubs PATH / PYTHONPATH
    and unsets CUDA_PATH so the run proves the artifact is self-contained
    (WIN-5.1 acceptance: "runs with PATH/PYTHONPATH/Python/Node/CUDA toolkit
    assumptions removed").

    Run from the repository root on windows-latest:
        pwsh packaging/windows/pyinstaller/build.ps1

.PARAMETER Extras
    The pip extras to freeze.  Defaults to the CPU profile (rvm,mediapipe,windows).
    Use "gpu,mediapipe,windows" on a CUDA runner.
#>
[CmdletBinding()]
param(
    [string]$Extras = "rvm,mediapipe,windows",
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$specDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $specDir "..\..\..")).Path
$buildRoot = Join-Path $specDir "build"
$distRoot = Join-Path $specDir "dist"
$venv = Join-Path $buildRoot "venv"

Write-Host "==> repo root: $repoRoot"
Remove-Item -Recurse -Force $buildRoot, $distRoot -ErrorAction SilentlyContinue

python -m venv $venv
$py = Join-Path $venv "Scripts\python.exe"

& $py -m pip install --disable-pip-version-check --upgrade "pip>=23,<27"
# The engine plus the Windows security backend; PyInstaller is bounded so a
# resolver change cannot silently alter the freezing tool under us.
& $py -m pip install --disable-pip-version-check "$repoRoot[$Extras]" "pyinstaller>=6.6,<7"

Push-Location $specDir
try {
    & $py -m PyInstaller --clean --noconfirm `
        --distpath $distRoot --workpath (Join-Path $buildRoot "pyi") `
        custback.spec
}
finally {
    Pop-Location
}

$exe = Join-Path $distRoot "custback\custback.exe"
if (-not (Test-Path $exe)) {
    throw "PyInstaller did not produce $exe"
}
Write-Host "==> built $exe"

if ($SkipSmoke) { return }

# Clean-environment smoke: no Python, no repo on PYTHONPATH, no CUDA toolkit.
Write-Host "==> clean-environment smoke test"
$cleanEnv = @{
    "PATH"       = "$env:SystemRoot\System32;$env:SystemRoot"
    "PYTHONPATH" = ""
    "PYTHONHOME" = ""
    "CUDA_PATH"  = ""
}
$prior = @{}
foreach ($k in $cleanEnv.Keys) {
    $prior[$k] = [Environment]::GetEnvironmentVariable($k)
    [Environment]::SetEnvironmentVariable($k, $cleanEnv[$k])
}
try {
    & $exe --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "frozen --help failed ($LASTEXITCODE)" }

    # A short synthetic run with no camera, no vcam, and no API proves the
    # segmentation/compositor path imports and runs end to end while frozen.
    $proc = Start-Process -FilePath $exe `
        -ArgumentList "--synthetic", "--no-vcam", "--no-api", "--no-file-log" `
        -PassThru -NoNewWindow
    Start-Sleep -Seconds 8
    if ($proc.HasExited -and $proc.ExitCode -ne 0) {
        throw "frozen synthetic run exited early with $($proc.ExitCode)"
    }
    if (-not $proc.HasExited) { $proc.CloseMainWindow() | Out-Null; Start-Sleep 2 }
    if (-not $proc.HasExited) { $proc.Kill() }
    Write-Host "==> smoke test passed"
}
finally {
    foreach ($k in $prior.Keys) {
        [Environment]::SetEnvironmentVariable($k, $prior[$k])
    }
}
