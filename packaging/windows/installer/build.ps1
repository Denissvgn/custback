<#
.SYNOPSIS
    Build and sign the per-user Custback installer bundle (WIN-5.6 / WIN-5.7).

.DESCRIPTION
    Stages the frozen engine (../pyinstaller/dist/custback) and the published
    shell (../shell/dist/shell), harvests them into WiX component groups,
    builds the per-user MSI (Package.wxs), then wraps it with the runtime
    prerequisites into the signed Burn bundle Custback-Setup-<version>.exe
    (Bundle.wxs).

    Every executable is signed and timestamped when a code-signing certificate
    thumbprint is supplied, satisfying the SmartScreen requirement (WIN-5.6).
    The Burn bundle is signed via the detach/reattach dance so both the engine
    and the wrapper carry a valid signature.

    Phase 6 options:
      -VCamDll        stages the native Media Foundation virtual camera DLL
                      (WIN-6.1, packaging/windows/vcam/build.ps1 output) next
                      to the shell and enables its per-user COM registration.
      -IncludeAvatar  keeps custback-avatar.exe in the engine payload
                      (WIN-6.4); by default the avatar service is not shipped,
                      and the shell supervises it only when present (WIN-5.5).
      -Arch           x64 (default) or arm64 (WIN-6.3): selects the MSI/bundle
                      platform and the matching VC++ redistributable.

.NOTES
    Requires the WiX 5 CLI (`dotnet tool install --global wix`) with the Util
    and Bal extensions, plus signtool from the Windows SDK. Runs on
    windows-latest in the release workflow (WIN-5.8).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$EngineDir,
    [Parameter(Mandatory = $true)][string]$ShellDir,
    [ValidateSet("x64", "arm64")][string]$Arch = "x64",
    [string]$VCamDll,
    [switch]$IncludeAvatar,
    [string]$OutputDir = "dist",
    [string]$CertThumbprint,
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$here = $PSScriptRoot
$out = Join-Path $here $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Invoke-Sign([string]$file) {
    if (-not $CertThumbprint) {
        Write-Warning "no -CertThumbprint: leaving $([IO.Path]::GetFileName($file)) unsigned"
        return
    }
    & signtool sign /sha1 $CertThumbprint /fd SHA256 `
        /tr $TimestampUrl /td SHA256 /d "Custback" $file
    if ($LASTEXITCODE -ne 0) { throw "signtool failed for $file" }
}

# 1. Stage the payload trees so Phase 6 options can shape them without
#    touching the build outputs: the vcam DLL joins the shell (WIN-6.1) and
#    the avatar service ships only on request (WIN-6.4 / WIN-5.5 gating).
$stage = Join-Path $out "stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
$stagedEngine = Join-Path $stage "engine"
$stagedShell = Join-Path $stage "shell"
Copy-Item -Recurse $EngineDir $stagedEngine
Copy-Item -Recurse $ShellDir $stagedShell

$defines = @()
if ($VCamDll) {
    if (-not (Test-Path $VCamDll)) { throw "native vcam DLL not found: $VCamDll" }
    Copy-Item $VCamDll (Join-Path $stagedShell "CustbackVCam.dll")
    $defines += @("-d", "IncludeNativeVCam=1")
}
if (-not $IncludeAvatar) {
    Remove-Item (Join-Path $stagedEngine "custback-avatar.exe") -ErrorAction SilentlyContinue
}

# 2. Sign the payload executables BEFORE harvesting so their hashes in the MSI
#    match the signed files a clean VM will run.
Get-ChildItem -Path $stagedEngine, $stagedShell -Recurse -Include *.exe, *.dll |
    ForEach-Object { Invoke-Sign $_.FullName }

# 3. Harvest payload trees into component groups referenced by Package.wxs.
wix extension add --global WixToolset.Heat
wix extension add --global WixToolset.Util.wixext
wix extension add --global WixToolset.Bal.wixext

& heat dir $stagedEngine -cg EngineComponents -dr EngineFolder -srd -scom -sreg -gg -g1 `
    -var var.EngineDir -out (Join-Path $out "EngineComponents.wxs")
& heat dir $stagedShell -cg ShellComponents -dr INSTALLFOLDER -srd -scom -sreg -gg -g1 `
    -var var.ShellDir -out (Join-Path $out "ShellComponents.wxs")

# 4. Build the per-user MSI for the target architecture.
$msi = Join-Path $out "Custback.msi"
wix build `
    (Join-Path $here "Package.wxs") `
    (Join-Path $out "EngineComponents.wxs") `
    (Join-Path $out "ShellComponents.wxs") `
    -ext WixToolset.Util.wixext `
    -arch $Arch `
    -d Version=$Version -d EngineDir=$stagedEngine -d ShellDir=$stagedShell `
    @defines `
    -o $msi
Invoke-Sign $msi

# 5. Build the bundle EXE and sign via detach/reattach so the Burn engine and
#    the wrapper are both signed. The bundle chains the arch-matched VC++
#    redistributable (redist\VC_redist.$Arch.exe).
$setup = Join-Path $out "Custback-Setup-$Version.exe"
wix build (Join-Path $here "Bundle.wxs") `
    -ext WixToolset.Bal.wixext -ext WixToolset.Util.wixext `
    -arch $Arch `
    -d Version=$Version -d Arch=$Arch -o $setup

if ($CertThumbprint) {
    $engine = Join-Path $out "burnengine.exe"
    wix burn detach $setup -engine $engine
    Invoke-Sign $engine
    wix burn reattach $setup -engine $engine -o $setup
    Invoke-Sign $setup
    Remove-Item $engine -Force
}

Write-Host "==> built $setup"
