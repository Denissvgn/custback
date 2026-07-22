using System.Windows.Forms;

namespace Custback.Shell;

/// <summary>
/// Entry point and single-instance guard for the desktop shell (WIN-5.2).
/// </summary>
internal static class Program
{
    // Per-user single instance. The installer is per-user (D5), so a session
    // ("Local\") name is the correct scope: a second launch must surface the
    // running window rather than start a rival engine on a second loopback
    // port and fight over the camera/output device.
    internal const string InstanceMutexName = @"Local\Custback.Shell.SingleInstance";
    internal const string ActivateEventName = @"Local\Custback.Shell.Activate";

    [STAThread]
    private static int Main()
    {
        using var singleInstance = new Mutex(initiallyOwned: true, InstanceMutexName, out bool createdNew);
        if (!createdNew)
        {
            // Wake the existing instance so it can restore its window, then exit.
            if (EventWaitHandle.TryOpenExisting(ActivateEventName, out var activate))
            {
                using (activate)
                {
                    activate.Set();
                }
            }

            return 0;
        }

        ApplicationConfiguration.Initialize();
        Application.SetHighDpiMode(HighDpiMode.PerMonitorV2);

        using var context = new TrayApplicationContext(EngineOptions.Discover());
        Application.Run(context);
        return context.ExitCode;
    }
}
