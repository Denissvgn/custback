using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text;

namespace Custback.Shell;

/// <summary>
/// Resolved paths and feature flags for the supervised engine. Paths follow the
/// Known-Folder routing the engine itself uses (WIN-1.5): per-user roaming
/// <c>%APPDATA%\Custback</c> for tokens/config, local <c>%LOCALAPPDATA%</c> for
/// cache/logs. The engine payload ships in a sibling <c>engine\</c> directory
/// next to the shell executable (installer layout, WIN-5.6).
/// </summary>
internal sealed record EngineOptions(
    string EngineExe,
    string? AvatarExe,
    string TokenFile,
    string RendererTokenFile,
    string LogFile)
{
    internal bool SuperviseAvatar => AvatarExe is not null && File.Exists(AvatarExe);

    internal static EngineOptions Discover()
    {
        string shellDir = AppContext.BaseDirectory;
        string engineDir = Path.Combine(shellDir, "engine");
        string appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        string configDir = Path.Combine(appData, "Custback");
        string logDir = Path.Combine(localAppData, "Custback", "logs");
        Directory.CreateDirectory(configDir);
        Directory.CreateDirectory(logDir);

        string avatarExe = Path.Combine(engineDir, "custback-avatar.exe");
        return new EngineOptions(
            EngineExe: Path.Combine(engineDir, "custback.exe"),
            // Only supervised when the avatar feature was actually installed
            // (D7 ships core first; avatar Windows parity is WIN-6.4).
            AvatarExe: File.Exists(avatarExe) ? avatarExe : null,
            TokenFile: Path.Combine(configDir, "api-token"),
            RendererTokenFile: Path.Combine(configDir, "renderer-token"),
            LogFile: Path.Combine(logDir, "shell.log"));
    }
}

/// <summary>
/// Launches and supervises the frozen engine (and the optional avatar service),
/// waits for explicit readiness, and owns the private lifecycle channel
/// (WIN-5.2/5.4/5.5). The bearer token never appears on a command line, in a
/// URL, or in the shell log — the engine writes it to a private token file that
/// only this process (same user) reads.
/// </summary>
internal sealed class Engine : IDisposable
{
    private readonly EngineOptions _options;
    private readonly HttpClient _http = new(new SocketsHttpHandler
    {
        // Loopback only; no proxy, no redirects, tight timeouts.
        UseProxy = false,
        AllowAutoRedirect = false,
        ConnectTimeout = TimeSpan.FromSeconds(2),
    })
    { Timeout = TimeSpan.FromSeconds(5) };

    private Process? _engineProcess;
    private Process? _avatarProcess;
    private string? _bearer;
    private volatile bool _shuttingDown;

    internal Engine(EngineOptions options) => _options = options;

    internal int Port { get; private set; }

    internal string BaseUrl => $"http://127.0.0.1:{Port}";

    /// <summary>Raised when the engine exits without a shutdown request.</summary>
    internal event EventHandler? Crashed;

    internal async Task StartAsync(CancellationToken cancellationToken)
    {
        Port = ReserveLoopbackPort();
        _engineProcess = StartProcess(_options.EngineExe, new[]
        {
            "--api-host", "127.0.0.1",
            "--api-port", Port.ToString(),
            // Paths, not secrets: the engine mints these files with owner-only
            // DACLs (WIN-2.3) and writes the token values into them.
            "--api-token-file", _options.TokenFile,
            "--renderer-token-file", _options.RendererTokenFile,
        });

        if (_options.SuperviseAvatar)
        {
            // The avatar service is a second process with its own storage and
            // drivers; supervise it only when installed (WIN-5.5).
            _avatarProcess = StartProcess(_options.AvatarExe!, new[] { "serve" });
        }

        await WaitForReadyAsync(cancellationToken).ConfigureAwait(false);
    }

    private static int ReserveLoopbackPort()
    {
        // Ask the OS for a free ephemeral loopback port, then release it and
        // hand the number to the engine. A benign TOCTOU remains; engine
        // startup fails loudly (EXIT_API) if the port was taken meanwhile, and
        // the shell surfaces that rather than silently continuing.
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        try
        {
            return ((IPEndPoint)listener.LocalEndpoint).Port;
        }
        finally
        {
            listener.Stop();
        }
    }

    private Process StartProcess(string exe, string[] arguments)
    {
        var info = new ProcessStartInfo(exe)
        {
            // CREATE_NO_WINDOW: no console flashes, but stdout/stderr are still
            // captured for supervision/doctor output (the engine is console=True).
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = Path.GetDirectoryName(exe)!,
        };
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }

        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, e) => AppendLog(e.Data);
        process.ErrorDataReceived += (_, e) => AppendLog(e.Data);
        process.Exited += OnProcessExited;
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return process;
    }

    private void OnProcessExited(object? sender, EventArgs e)
    {
        if (_shuttingDown || !ReferenceEquals(sender, _engineProcess))
        {
            return;
        }

        Crashed?.Invoke(this, EventArgs.Empty);
    }

    private async Task WaitForReadyAsync(CancellationToken cancellationToken)
    {
        // Explicit readiness (WIN-5.2): the engine only begins serving the API
        // after the pipeline has started, so a 200 from an authenticated
        // /status is a truthful "ready" — not merely "port open".
        _bearer = await ReadTokenWhenAvailableAsync(_options.TokenFile, cancellationToken)
            .ConfigureAwait(false);

        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(30);
        while (DateTime.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (_engineProcess is { HasExited: true })
            {
                throw new InvalidOperationException(
                    $"engine exited during startup (code {_engineProcess.ExitCode})");
            }

            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, $"{BaseUrl}/status");
                request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {_bearer}");
                using var response = await _http.SendAsync(request, cancellationToken)
                    .ConfigureAwait(false);
                if (response.StatusCode == HttpStatusCode.OK)
                {
                    return;
                }
            }
            catch (HttpRequestException)
            {
                // Not listening yet; keep polling until the deadline.
            }

            await Task.Delay(200, cancellationToken).ConfigureAwait(false);
        }

        throw new TimeoutException("engine did not become ready within 30s");
    }

    private static async Task<string> ReadTokenWhenAvailableAsync(
        string path, CancellationToken cancellationToken)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(15);
        while (DateTime.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (File.Exists(path))
            {
                var value = (await File.ReadAllTextAsync(path, cancellationToken)
                    .ConfigureAwait(false)).Trim();
                if (value.Length > 0)
                {
                    return value;
                }
            }

            await Task.Delay(100, cancellationToken).ConfigureAwait(false);
        }

        throw new TimeoutException("engine did not provision the API token file");
    }

    /// <summary>The management bearer, available after <see cref="StartAsync"/>.</summary>
    internal string Bearer => _bearer ?? throw new InvalidOperationException("engine not started");

    /// <summary>
    /// Ask the engine to stop gracefully over the private lifecycle channel
    /// (WIN-5.3), then wait for the pipeline to drain (WIN-5.4).
    /// </summary>
    internal async Task RequestShutdownAsync(TimeSpan timeout)
    {
        _shuttingDown = true;
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, $"{BaseUrl}/lifecycle/shutdown");
            request.Headers.TryAddWithoutValidation("Authorization", $"Bearer {Bearer}");
            request.Content = new StringContent(string.Empty);
            using var response = await _http.SendAsync(request).ConfigureAwait(false);
        }
        catch (HttpRequestException)
        {
            // Engine may already be gone; fall through to the wait/kill below.
        }

        await StopProcessAsync(_avatarProcess, timeout).ConfigureAwait(false);
        await StopProcessAsync(_engineProcess, timeout).ConfigureAwait(false);
    }

    private static async Task StopProcessAsync(Process? process, TimeSpan timeout)
    {
        if (process is null)
        {
            return;
        }

        try
        {
            using var cts = new CancellationTokenSource(timeout);
            await process.WaitForExitAsync(cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Graceful stop did not complete in time; terminate the tree so the
            // camera/output device is released rather than held indefinitely.
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
        }
    }

    private void AppendLog(string? line)
    {
        if (string.IsNullOrEmpty(line))
        {
            return;
        }

        try
        {
            File.AppendAllText(_options.LogFile, line + Environment.NewLine, Encoding.UTF8);
        }
        catch (IOException)
        {
            // Losing a diagnostic line must never take down supervision.
        }
    }

    public void Dispose()
    {
        _http.Dispose();
        _engineProcess?.Dispose();
        _avatarProcess?.Dispose();
    }
}
