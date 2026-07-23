<#
.SYNOPSIS
    Build the native Media Foundation virtual camera DLL (WIN-6.1).

.DESCRIPTION
    Compiles CustbackVCam.vcxproj with MSBuild for the requested architecture
    and verifies the produced DLL exports the in-proc COM surface.  Runs on
    windows-latest (VS 2022 build tools + Windows 11 SDK 22000+).

    The DLL is consumed by the installer (Package.wxs registers its CLSID
    per-user and removes it on uninstall) and by the shell, which creates the
    camera with MFCreateVirtualCamera at run time (session lifetime — nothing
    outlives the shell process except the COM registration the MSI owns).

.PARAMETER Arch
    Target architecture: x64 (default) or arm64 (WIN-6.3).

.PARAMETER GateDiagnostics
    Enable the MIT-C1 OutputDebugString trace for frame-ring open failures.
    This is intended only for WIN-6.1 gate payloads.
#>
[CmdletBinding()]
param(
    [ValidateSet("x64", "arm64")][string]$Arch = "x64",
    [string]$Configuration = "Release",
    [switch]$GateDiagnostics
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$here = $PSScriptRoot
$platform = if ($Arch -eq "arm64") { "ARM64" } else { "x64" }
$gateDiagnosticsValue = if ($GateDiagnostics) { "true" } else { "false" }

$msbuild = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -requires Microsoft.Component.MSBuild `
    -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) { throw "MSBuild not found (install VS 2022 build tools)" }

& $msbuild (Join-Path $here "CustbackVCam.vcxproj") `
    /nologo /m `
    /p:Configuration=$Configuration `
    /p:Platform=$platform `
    /p:CustbackVcamGateDiagnostics=$gateDiagnosticsValue
if ($LASTEXITCODE -ne 0) { throw "msbuild failed ($LASTEXITCODE)" }

$dll = Join-Path $here "dist\$platform\$Configuration\CustbackVCam.dll"
if (-not (Test-Path $dll)) { throw "build did not produce $dll" }

# The COM surface the installer/shell relies on must actually be exported.
$exports = & dumpbin /nologo /exports $dll
foreach ($required in "DllGetClassObject", "DllCanUnloadNow") {
    if ($exports -notmatch $required) {
        throw "$dll does not export $required"
    }
}

Write-Host "==> built $dll"
