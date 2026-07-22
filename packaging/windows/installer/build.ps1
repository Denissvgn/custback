<#
.SYNOPSIS
    Build and sign the per-user Custback installer bundle (WIN-5.6 / WIN-5.7).

.DESCRIPTION
    Harvests the frozen engine (../pyinstaller/dist/custback) and the published
    shell (../shell/dist/shell) into WiX component groups, builds the per-user
    MSI (Package.wxs), then wraps it with the runtime prerequisites into the
    signed Burn bundle Custback-Setup-<version>.exe (Bundle.wxs).

    Every executable is signed and timestamped when a code-signing certificate
    thumbprint is supplied, satisfying the SmartScreen requirement (WIN-5.6).
    The Burn bundle is signed via the detach/reattach dance so both the engine
    and the wrapper carry a valid signature.

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

# 1. Sign the payload executables BEFORE harvesting so their hashes in the MSI
#    match the signed files a clean VM will run.
Get-ChildItem -Path $EngineDir, $ShellDir -Recurse -Include *.exe, *.dll |
    ForEach-Object { Invoke-Sign $_.FullName }

# 2. Harvest payload trees into component groups referenced by Package.wxs.
wix extension add --global WixToolset.Heat
wix extension add --global WixToolset.Util.wixext
wix extension add --global WixToolset.Bal.wixext

& heat dir $EngineDir -cg EngineComponents -dr EngineFolder -srd -scom -sreg -gg -g1 `
    -var var.EngineDir -out (Join-Path $out "EngineComponents.wxs")
& heat dir $ShellDir -cg ShellComponents -dr INSTALLFOLDER -srd -scom -sreg -gg -g1 `
    -var var.ShellDir -out (Join-Path $out "ShellComponents.wxs")

# 3. Build the per-user MSI.
$msi = Join-Path $out "Custback.msi"
wix build `
    (Join-Path $here "Package.wxs") `
    (Join-Path $out "EngineComponents.wxs") `
    (Join-Path $out "ShellComponents.wxs") `
    -ext WixToolset.Util.wixext `
    -d Version=$Version -d EngineDir=$EngineDir -d ShellDir=$ShellDir `
    -o $msi
Invoke-Sign $msi

# 4. Build the bundle EXE and sign via detach/reattach so the Burn engine and
#    the wrapper are both signed.
$setup = Join-Path $out "Custback-Setup-$Version.exe"
wix build (Join-Path $here "Bundle.wxs") `
    -ext WixToolset.Bal.wixext -ext WixToolset.Util.wixext `
    -d Version=$Version -o $setup

if ($CertThumbprint) {
    $engine = Join-Path $out "burnengine.exe"
    wix burn detach $setup -engine $engine
    Invoke-Sign $engine
    wix burn reattach $setup -engine $engine -o $setup
    Invoke-Sign $setup
    Remove-Item $engine -Force
}

Write-Host "==> built $setup"
