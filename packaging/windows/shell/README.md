# Desktop shell (C# WebView2 + tray) — WIN-5.2 / 5.3 / 5.4 / 5.5

A thin, single-instance supervisor around the frozen `custback` engine
(`../pyinstaller`). It hosts the existing local web UI (`custback.api.webui`) in
WebView2 and lives in the system tray. The signed per-user installer
(`../installer`, WIN-5.6) ships the produced `Custback.exe` next to the frozen
engine in `engine\`.

## Responsibilities → source

| Task | Concern | Source |
| --- | --- | --- |
| WIN-5.2 | Single-instance mutex; reserve loopback port; supervise engine; wait for explicit readiness; host UI; show startup/doctor failures first; do **not** weaken the engine's Host/Origin/cookie/bearer boundary | `Program.cs`, `Engine.cs`, `TrayApplicationContext.cs` |
| WIN-5.3 | HttpOnly session without the bearer in a URL/cmdline/web storage/log; private lifecycle/shutdown channel | `SessionBootstrap.cs`, `Engine.RequestShutdownAsync` |
| WIN-5.4 | Keep processing when hidden to tray; graceful shutdown drains camera/output; crash/suspend handling; launch-at-login | `TrayApplicationContext.cs`, `AutoStart` |
| WIN-5.5 | Supervise the avatar second process only when installed | `Engine` (`SuperviseAvatar`) |

## Security model (WIN-5.3)

The bearer token is the root credential and never leaves the trusted boundary:

1. The engine writes the token to a private, owner-only-DACL file (WIN-2.3);
   the shell passes only the *path* on the command line, never the value.
2. The shell reads the bearer, POSTs it in the **body** of `/auth/session` over
   loopback, and receives an opaque `custback_session` cookie.
3. Only that HttpOnly, `SameSite=Strict` session cookie is injected into
   WebView2. The page loads already-authenticated and never sees the bearer.
4. Shutdown uses `POST /lifecycle/shutdown` with the **bearer** (not the
   session), so loaded web content — which holds only the HttpOnly cookie —
   cannot stop the engine. The engine enforces this bearer-only rule server-side
   (`tests/test_api.py::test_lifecycle_shutdown_rejects_browser_session`).

The shell adds no origins, no host objects, and no second bearer; the engine's
loopback boundary remains the sole authority even though both processes are
local. Off-origin navigations open in the system browser, not the WebView.

## Build

```powershell
dotnet publish packaging/windows/shell/Custback.Shell.csproj -c Release -r win-x64 --self-contained true -o dist/shell
```

Requires the .NET 8 SDK and restores the Evergreen `Microsoft.Web.WebView2`
loader. The WebView2 **runtime** is an installer prerequisite (WIN-5.6), not
bundled here.

## Status

`IMPL*` — reviewable source complete; compilation and the WebView2/tray
integration run on the `windows-latest` release job (WIN-5.8), which is where
WIN-5.2/5.3/5.4/5.5 flip to `DONE`. The engine-side contract this shell depends
on (the `/auth/session` bootstrap and the bearer-only `/lifecycle/shutdown`
channel) is verified today on Linux in `tests/test_api.py`.
