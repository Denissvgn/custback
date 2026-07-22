using Microsoft.Win32;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace Custback.Shell;

/// <summary>
/// Tray-resident lifecycle for the desktop product (WIN-5.4). Hosts the local
/// web UI in WebView2, keeps the engine processing while minimised to tray, and
/// stops it gracefully only on explicit Quit. Startup/doctor failures are shown
/// before any UI is presented (WIN-5.2).
/// </summary>
internal sealed class TrayApplicationContext : ApplicationContext
{
    private readonly Engine _engine;
    private readonly CancellationTokenSource _cts = new();
    private readonly NotifyIcon _tray;
    private readonly SynchronizationContext _ui;
    private Form? _window;
    private WebView2? _webView;
    private EventWaitHandle? _activate;
    private bool _quitting;

    internal int ExitCode { get; private set; }

    internal TrayApplicationContext(EngineOptions options)
    {
        _engine = new Engine(options);
        _engine.Crashed += OnEngineCrashed;
        _ui = SynchronizationContext.Current ?? new WindowsFormsSynchronizationContext();

        _tray = new NotifyIcon
        {
            Text = "Custback",
            Visible = true,
            Icon = SystemIcons.Application,
            ContextMenuStrip = BuildMenu(),
        };
        _tray.DoubleClick += (_, _) => ShowWindow();

        StartActivationListener();
        _ = BootAsync();
    }

    private ContextMenuStrip BuildMenu()
    {
        var menu = new ContextMenuStrip();
        menu.Items.Add("Open Custback", null, (_, _) => ShowWindow());
        var launchAtLogin = new ToolStripMenuItem("Launch at login")
        {
            CheckOnClick = true,
            Checked = AutoStart.IsEnabled,
        };
        launchAtLogin.CheckedChanged += (_, _) => AutoStart.Set(launchAtLogin.Checked);
        menu.Items.Add(launchAtLogin);
        menu.Items.Add(new ToolStripSeparator());
        menu.Items.Add("Quit", null, async (_, _) => await QuitAsync().ConfigureAwait(true));
        return menu;
    }

    private async Task BootAsync()
    {
        try
        {
            await _engine.StartAsync(_cts.Token).ConfigureAwait(true);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Surface the failure before the UI: a clean VM with, say, a missing
            // WebView2 runtime or an occupied port must not silently hang.
            ShowFatal("Custback could not start its engine.", ex);
            return;
        }

        await ShowWindowAsync().ConfigureAwait(true);
    }

    private void ShowWindow() => _ = ShowWindowAsync();

    private async Task ShowWindowAsync()
    {
        if (_window is { IsDisposed: false })
        {
            _window.Show();
            _window.WindowState = FormWindowState.Normal;
            _window.Activate();
            return;
        }

        _window = new Form
        {
            Text = "Custback",
            Width = 1100,
            Height = 720,
            StartPosition = FormStartPosition.CenterScreen,
        };
        // Closing the window hides it; the engine keeps running so the virtual
        // camera stays live in the meeting app (WIN-5.4). Only Quit stops it.
        _window.FormClosing += (_, e) =>
        {
            if (!_quitting && e.CloseReason == CloseReason.UserClosing)
            {
                e.Cancel = true;
                _window!.Hide();
            }
        };

        _webView = new WebView2 { Dock = DockStyle.Fill };
        _window.Controls.Add(_webView);
        _window.Show();

        string userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Custback", "webview");
        var environment = await CoreWebView2Environment.CreateAsync(userDataFolder: userData)
            .ConfigureAwait(true);
        await _webView.EnsureCoreWebView2Async(environment).ConfigureAwait(true);
        HardenWebView(_webView.CoreWebView2);

        await SessionBootstrap.InjectSessionAsync(_webView.CoreWebView2, _engine).ConfigureAwait(true);
        _webView.CoreWebView2.Navigate(_engine.BaseUrl + "/");
    }

    private void HardenWebView(CoreWebView2 core)
    {
        var settings = core.Settings;
        settings.AreDevToolsEnabled = false;
        settings.AreDefaultContextMenusEnabled = false;
        settings.IsStatusBarEnabled = false;
        settings.AreHostObjectsAllowed = false;
        // The shell adds no bearer, no host objects, and no extra origins: the
        // engine's Host/Origin/cookie/bearer boundary is the only authority and
        // is not weakened just because both processes are local (WIN-5.2). Any
        // navigation away from the loopback origin opens in the system browser
        // instead of inside the trusted WebView.
        core.NewWindowRequested += (_, e) =>
        {
            e.Handled = true;
            OpenExternally(e.Uri);
        };
        core.NavigationStarting += (_, e) =>
        {
            if (!e.Uri.StartsWith(_engine.BaseUrl, StringComparison.OrdinalIgnoreCase))
            {
                e.Cancel = true;
                OpenExternally(e.Uri);
            }
        };
    }

    private static void OpenExternally(string uri)
    {
        if (Uri.TryCreate(uri, UriKind.Absolute, out var parsed) &&
            (parsed.Scheme == Uri.UriSchemeHttp || parsed.Scheme == Uri.UriSchemeHttps))
        {
            System.Diagnostics.Process.Start(
                new System.Diagnostics.ProcessStartInfo(uri) { UseShellExecute = true });
        }
    }

    private void OnEngineCrashed(object? sender, EventArgs e)
    {
        if (_quitting)
        {
            return;
        }

        // Post back to the UI thread; a crashed engine is user-visible and the
        // shell should not keep pretending to run (WIN-5.4).
        _ui.Post(_ =>
        {
            _tray.ShowBalloonTip(5000, "Custback",
                "The Custback engine stopped unexpectedly.", ToolTipIcon.Error);
            ExitCode = 1;
        }, null);
    }

    private void StartActivationListener()
    {
        // A second launch signals this event (see Program); restore the window.
        _activate = new EventWaitHandle(false, EventResetMode.AutoReset, Program.ActivateEventName);
        var thread = new Thread(() =>
        {
            while (!_cts.IsCancellationRequested)
            {
                if (_activate.WaitOne(500))
                {
                    _ui.Post(_ => ShowWindow(), null);
                }
            }
        })
        { IsBackground = true, Name = "activation-listener" };
        thread.Start();
    }

    private async Task QuitAsync()
    {
        if (_quitting)
        {
            return;
        }

        _quitting = true;
        _tray.Visible = false;
        try
        {
            // Graceful stop drains camera/output before the process exits.
            await _engine.RequestShutdownAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(true);
        }
        finally
        {
            _cts.Cancel();
            ExitThread();
        }
    }

    private void ShowFatal(string summary, Exception ex)
    {
        _tray.Visible = false;
        MessageBox.Show(
            $"{summary}\n\n{ex.Message}\n\nSee %LOCALAPPDATA%\\Custback\\logs for details.",
            "Custback", MessageBoxButtons.OK, MessageBoxIcon.Error);
        ExitCode = 1;
        _cts.Cancel();
        ExitThread();
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _cts.Cancel();
            _activate?.Dispose();
            _webView?.Dispose();
            _window?.Dispose();
            _tray.Dispose();
            _engine.Dispose();
        }

        base.Dispose(disposing);
    }
}

/// <summary>Per-user launch-at-login via the HKCU Run key (WIN-5.4).</summary>
internal static class AutoStart
{
    private const string RunKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    private const string ValueName = "Custback";

    internal static bool IsEnabled
    {
        get
        {
            using var key = Registry.CurrentUser.OpenSubKey(RunKey);
            return key?.GetValue(ValueName) is not null;
        }
    }

    internal static void Set(bool enabled)
    {
        using var key = Registry.CurrentUser.CreateSubKey(RunKey);
        if (key is null)
        {
            return;
        }

        if (enabled)
        {
            key.SetValue(ValueName, $"\"{Application.ExecutablePath}\"");
        }
        else
        {
            key.DeleteValue(ValueName, throwOnMissingValue: false);
        }
    }
}
