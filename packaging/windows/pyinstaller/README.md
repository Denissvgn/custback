# Build the Windows engine

This directory builds the engine and avatar service as a shared PyInstaller
`onedir` payload. These are developer builds; signed Windows installers are not
part of the currently published source support.

From the repository root on a matching Windows host:

```powershell
pwsh packaging/windows/pyinstaller/build.ps1
pwsh packaging/windows/pyinstaller/build.ps1 -Extras "gpu,mediapipe,windows"
pwsh packaging/windows/pyinstaller/build.ps1 -Extras "directml,mediapipe,windows"
pwsh packaging/windows/pyinstaller/build.ps1 -Arch arm64
pwsh packaging/windows/pyinstaller/build.ps1 -AvatarProfile audio2face
```

Choose one acceleration profile. ARM64 requires native ARM64 Python and cannot
use the x64 MediaPipe, CUDA, or Audio2Face wheel profiles. The build script
rejects incompatible selections. Model weights are acquired separately and
are not bundled.

The output contains `custback.exe`, `custback-avatar.exe`, and their shared DLLs
and package data. Preserve the whole directory. The spec collects bundled
PyAV libraries, optional runtime providers, YAML templates, and pywin32.
`-AvatarProfile` selects one supported avatar driver stack.

See the [desktop shell](../shell/README.md) and
[installer](../installer/README.md) build instructions. External components
retain their upstream licenses; a successful build is not distribution approval.
