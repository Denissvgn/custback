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
    Use "gpu,mediapipe,windows" on a CUDA runner, "directml,mediapipe,windows"
    for the WIN-6.2 DirectML profile.

.PARAMETER Arch
    Target architecture: x64 (default) or arm64 (WIN-6.3).  PyInstaller cannot
    cross-freeze, so the script verifies the running Python matches and trims
    the default extras to the ARM64 dependency profile (WINDOWS_ARM64.md).

.PARAMETER AvatarProfile
    Avatar driver stack for custback-avatar.exe (WIN-6.4): "vision" (default,
    MediaPipe) or "audio2face" (gRPC client).  The two are mutually exclusive
    per payload because their protobuf requirements conflict; the spec excludes
    the other stack accordingly.
#>
[CmdletBinding()]
param(
    [string]$Extras = "rvm,mediapipe,windows",
    [ValidateSet("x64", "arm64")][string]$Arch = "x64",
    [ValidateSet("vision", "audio2face")][string]$AvatarProfile = "vision",
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Arch -eq "arm64" -and -not $PSBoundParameters.ContainsKey("Extras")) {
    # ARM64 dependency profile (WIN-6.3 / WINDOWS_ARM64.md): mediapipe ships
    # no win_arm64 wheel, so the default trims to RVM CPU + the security
    # backend. The core PEP 508 marker and PyInstaller spec also omit the
    # wheel-less pyvirtualcam package. DirectML may be added explicitly once
    # WIN-6.2 evidence exists.
    $Extras = "rvm,windows"
}
if ($AvatarProfile -eq "audio2face" -and -not $PSBoundParameters.ContainsKey("Extras")) {
    # The audio2face flavor swaps the conflicting driver stack (WIN-6.4).
    $Extras = "rvm,audio2face,windows"
}

$extraNames = $Extras.Split(",") | ForEach-Object { $_.Trim().ToLowerInvariant() }

# WIN-6.4 driver-stack exclusivity: nvidia protocol wheels need protobuf>=5.29
# while mediapipe needs protobuf<5 (see pyproject audio2face extra); a venv
# holding both cannot resolve, so refuse before pip discovers it slowly.
if ($AvatarProfile -eq "audio2face") {
    if ($extraNames -contains "mediapipe") {
        throw "the audio2face avatar profile cannot include the 'mediapipe' extra (protobuf conflict); use -Extras rvm,audio2face,windows"
    }
    if ($Arch -eq "arm64") {
        throw "the audio2face avatar profile is x64-only (NVIDIA protocol wheels ship no ARM64 build; WINDOWS_ARM64.md)"
    }
}
if ($AvatarProfile -eq "vision" -and ($extraNames -contains "audio2face")) {
    throw "the vision avatar profile does not freeze the 'audio2face' extra; pass -AvatarProfile audio2face"
}

# WIN-6.2 wheel-conflict guard: onnxruntime-gpu (CUDA) and
# onnxruntime-directml both install the same `onnxruntime` package; freezing
# an environment that mixes them produces an artifact whose provider set is
# resolver-order luck. The installer profile must pick exactly one.
if (($extraNames -contains "gpu") -and ($extraNames -contains "directml")) {
    throw "extras 'gpu' (CUDA) and 'directml' are mutually exclusive; choose one acceleration profile"
}

# WIN-6.3 architecture guards: no CUDA runtime exists for Windows-on-ARM, and
# mediapipe publishes no win_arm64 wheel — fail at freeze time, not on the
# clean VM.
if ($Arch -eq "arm64") {
    if ($extraNames -contains "gpu") {
        throw "the 'gpu' (CUDA) extra is not available on Windows ARM64; use 'directml' or CPU (WINDOWS_ARM64.md)"
    }
    if ($extraNames -contains "mediapipe") {
        throw "the 'mediapipe' extra has no Windows ARM64 wheel (WINDOWS_ARM64.md)"
    }
}

$specDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $specDir "..\..\..")).Path
$buildRoot = Join-Path $specDir "build"
$distRoot = Join-Path $specDir "dist"
$venv = Join-Path $buildRoot "venv"

Write-Host "==> repo root: $repoRoot"
Remove-Item -Recurse -Force $buildRoot, $distRoot -ErrorAction SilentlyContinue

python -m venv $venv
$py = Join-Path $venv "Scripts\python.exe"

# PyInstaller freezes for the running interpreter's architecture only; a
# mismatched host would produce an artifact that silently targets the wrong
# arch (WIN-6.3). AMD64 = x64 in platform.machine() terms.
$machine = (& $py -c "import platform; print(platform.machine().lower())").Trim()
$expected = if ($Arch -eq "arm64") { "arm64" } else { "amd64" }
if ($machine -ne $expected) {
    throw "python reports architecture '$machine' but -Arch $Arch requires '$expected'"
}

& $py -m pip install --disable-pip-version-check --upgrade "pip>=23,<27"
# The engine plus the Windows security backend; PyInstaller is bounded so a
# resolver change cannot silently alter the freezing tool under us.
& $py -m pip install --disable-pip-version-check "$repoRoot[$Extras]" "pyinstaller>=6.6,<7"

Push-Location $specDir
try {
    # The spec reads the avatar driver profile from the environment (WIN-6.4).
    $env:CUSTBACK_AVATAR_PROFILE = $AvatarProfile
    & $py -m PyInstaller --clean --noconfirm `
        --distpath $distRoot --workpath (Join-Path $buildRoot "pyi") `
        custback.spec
}
finally {
    Remove-Item Env:CUSTBACK_AVATAR_PROFILE -ErrorAction SilentlyContinue
    Pop-Location
}

$exe = Join-Path $distRoot "custback\custback.exe"
if (-not (Test-Path $exe)) {
    throw "PyInstaller did not produce $exe"
}
$avatarExe = Join-Path $distRoot "custback\custback-avatar.exe"
if (-not (Test-Path $avatarExe)) {
    throw "PyInstaller did not produce $avatarExe (WIN-6.4 second executable)"
}
Write-Host "==> built $exe (+ custback-avatar.exe, $AvatarProfile profile)"

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

    # WIN-6.4: the frozen avatar service initializes and tears down a
    # hardware-free idle renderer (its bundled avatar.yaml + rig resources)
    # in the same scrubbed environment.
    & $avatarExe --smoke | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "frozen avatar smoke failed ($LASTEXITCODE)" }
    Write-Host "==> smoke tests passed (engine + avatar)"
}
finally {
    foreach ($k in $prior.Keys) {
        [Environment]::SetEnvironmentVariable($k, $prior[$k])
    }
}
