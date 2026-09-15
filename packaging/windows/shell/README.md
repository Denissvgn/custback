# Build the Windows desktop shell

The shell supervises the engine, hosts its authenticated local UI in WebView2,
and provides a system tray entry. It can supervise a separately installed
avatar executable. Native camera support is experimental.

Requires the .NET 8 SDK. From the repository root:

```powershell
dotnet publish packaging/windows/shell/Custback.Shell.csproj -c Release -r win-x64 --self-contained true -o dist/shell
```

The WebView2 runtime must be available on the target machine. Keep the frozen
engine payload in the shell's `engine` directory.

## Local configuration

The shell reads `%APPDATA%\Custback\config.yaml` and, when the avatar service is
installed, `%APPDATA%\Custback\avatar.yaml`. Missing files use built-in defaults.
The shell owns process supervision, loopback endpoints, and credential paths;
remote deployment settings belong to source-run services instead.

Credentials are read from owner-only files. The shell obtains an HttpOnly
browser session without placing bearer tokens in URLs, command arguments, or
browser storage. Closing the tray application shuts down the supervised
processes and removes its session-lifetime native camera.

These are developer builds. See [source installation](../../../README.md) for
the supported entry path and [engine builds](../pyinstaller/README.md) for the
required runtime payload.
