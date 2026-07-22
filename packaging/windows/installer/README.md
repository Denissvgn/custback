# Signed per-user installer (WiX) — WIN-5.6 / WIN-5.7

Produces `Custback-Setup-<version>.exe`: a signed, per-user Burn bundle that
chains the runtime prerequisites and installs the WebView2/tray shell plus the
frozen engine, with no elevation and no machine-wide state (D5).

## Files

| File | Purpose |
| --- | --- |
| `Package.wxs` | Per-user MSI: shell + harvested engine payload, Start-Menu shortcut, upgrade/rollback, uninstall, and the WIN-5.7 retention rules. |
| `Bundle.wxs` | Burn bundle: detects/installs VC++ and WebView2 Evergreen, detects (never installs) OBS Virtual Camera, then runs the MSI. Output matches `^Custback-Setup-[0-9]+\.[0-9]+\.[0-9]+.*\.exe$`. |
| `build.ps1` | Harvest → build MSI → build bundle → sign+timestamp everything. |

## Prerequisites handled (WIN-5.6)

* **WebView2 Evergreen runtime** — detected via `EdgeUpdate` registry; the
  redistributable bootstrapper is installed only when absent.
* **VC++ runtime** — the pinned `VC_redist.x64.exe`, installed only when absent.
* **OBS Virtual Camera** — *detected only* and reported to the app (WIN-3.7);
  never redistributed (CC-5, D3).
* **ORT/CUDA/cuDNN** — ship inside the frozen engine payload (WIN-5.1); the
  exact pinned components come from the freeze profile, not a separate chain.

Every `.exe`/`.dll` in the payload, the MSI, the Burn engine, and the wrapper
are Authenticode-signed and RFC-3161 timestamped for SmartScreen.

## Data-retention policy (WIN-5.7)

* An **ordinary uninstall removes only application binaries.** User data —
  backgrounds, rigs, config, tokens, logs under `%APPDATA%\Custback` and
  `%LOCALAPPDATA%\Custback` — is created at runtime by the engine and is not
  tracked by any installer component, so it is never deleted by default.
* **Opt-in erase:** `msiexec /x … REMOVEUSERDATA=1` (surfaced as a checkbox by
  the shell's uninstall entry) removes the data trees via `util:RemoveFolderEx`.
* **Native virtual-camera registration:** when the native MF camera (WIN-6.1)
  is later registered, uninstall removes its per-user COM/MF registration
  (`UninstallVirtualCamera` component); a safe no-op when it was never
  installed.

## Build

```powershell
pwsh packaging/windows/installer/build.ps1 `
    -Version 0.4.0 `
    -EngineDir packaging/windows/pyinstaller/dist/custback `
    -ShellDir  packaging/windows/shell/dist/shell `
    -CertThumbprint <thumbprint>
```

## Status

`IMPL*` — WiX source and signing flow complete and reviewable; the MSI/bundle
build, clean-VM install/upgrade/uninstall, and signature verification run on the
`windows-latest` release job (WIN-5.8), which is where WIN-5.6/5.7 flip to
`DONE`. Bundling of GPL model/vcam components stays blocked until WIN-0.4 legal
sign-off (CC-5).
