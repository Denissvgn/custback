using System.Runtime.InteropServices;

namespace Custback.Shell;

/// <summary>
/// Owns the native Media Foundation virtual camera for the life of the shell
/// (WIN-6.1). The camera is registered with <b>session</b> lifetime and
/// <b>current-user</b> access: it appears in consumer apps while the shell
/// runs and vanishes with the process, so no persistent device state can leak
/// past uninstall (WIN-5.7 — the MSI removes only the COM registration).
///
/// The session is best-effort and feature-gated: it activates only when the
/// media-source DLL is installed next to the shell, and any failure degrades
/// to the OBS/pyvirtualcam path rather than blocking startup — the native
/// camera is opt-in until its clean-machine gate passes (feature matrix).
/// </summary>
internal sealed class VirtualCameraSession : IDisposable
{
    // Must match Activator.h, vcam_native.VCAM_CLSID, and Package.wxs.
    internal const string ActivatorClsid = "{7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}";
    internal const string FriendlyName = "Custback Camera";
    internal const string MediaSourceDll = "CustbackVCam.dll";

    private IMFVirtualCamera? _camera;
    private bool _mediaFoundationStarted;

    /// <summary>The DLL ships only when the native-camera feature is installed.</summary>
    internal static bool IsInstalled =>
        File.Exists(Path.Combine(AppContext.BaseDirectory, MediaSourceDll));

    /// <summary>
    /// Creates and starts the virtual camera. Returns false (never throws)
    /// when the feature is absent or the OS refuses — the caller logs and the
    /// engine keeps using its configured output backend.
    /// </summary>
    internal bool TryStart(Action<string> log)
    {
        if (_camera is not null)
        {
            return true;
        }

        if (!IsInstalled)
        {
            return false;
        }

        try
        {
            int hr = NativeMethods.MFStartup(NativeMethods.MF_VERSION, 0);
            Marshal.ThrowExceptionForHR(hr);
            _mediaFoundationStarted = true;

            hr = NativeMethods.MFCreateVirtualCamera(
                NativeMethods.MFVirtualCameraType_SoftwareCameraSource,
                NativeMethods.MFVirtualCameraLifetime_Session,
                NativeMethods.MFVirtualCameraAccess_CurrentUser,
                FriendlyName,
                ActivatorClsid,
                categories: IntPtr.Zero,
                categoryCount: 0,
                out IMFVirtualCamera camera);
            Marshal.ThrowExceptionForHR(hr);
            // Own the RCW before any gated assertion or Start call so the
            // catch path can deterministically release a partially created
            // camera before shutting Media Foundation down.
            _camera = camera;

#if DEBUG || CUSTBACK_GATE_BUILD
            // MIT-C4: exercise a late inherited IMFAttributes slot before
            // Start so an incorrect IMFVirtualCamera projection fails at the
            // gate boundary instead of corrupting a later virtual-camera call.
            hr = camera.GetCount(out uint attributeCount);
            log(
                "native virtual camera COM projection self-check: " +
                $"IMFAttributes::GetCount HRESULT=0x{hr:X8}, count={attributeCount}");
            Marshal.ThrowExceptionForHR(hr);
#endif

            // Start(null): no per-app callback; Frame Server owns activation.
            Marshal.ThrowExceptionForHR(camera.Start(IntPtr.Zero));
            log($"native virtual camera started: {FriendlyName}");
            return true;
        }
        catch (Exception ex)
        {
            log($"native virtual camera unavailable ({ex.Message}); " +
                "continuing with the configured output backend");
            Stop(log);
            return false;
        }
    }

    /// <summary>Stops and removes the camera; safe to call repeatedly.</summary>
    internal void Stop(Action<string>? log = null)
    {
        if (_camera is not null)
        {
            // Stop the stream, tear down Frame Server state, then remove the
            // session registration so the device disappears immediately
            // rather than at session end.
            try { _camera.Stop(); } catch { /* already stopped */ }
            try { _camera.Shutdown(); } catch { /* already down */ }
            try { _camera.Remove(); } catch { /* session cleanup races OS */ }
            Marshal.FinalReleaseComObject(_camera);
            _camera = null;
            log?.Invoke("native virtual camera removed");
        }

        if (_mediaFoundationStarted)
        {
            _ = NativeMethods.MFShutdown();
            _mediaFoundationStarted = false;
        }
    }

    public void Dispose() => Stop();

    private static class NativeMethods
    {
        // MF_SDK_VERSION 0x0002, MF_API_VERSION 0x0070.
        internal const int MF_VERSION = 0x00020070;

        internal const int MFVirtualCameraType_SoftwareCameraSource = 0;
        internal const int MFVirtualCameraLifetime_Session = 0;
        internal const int MFVirtualCameraAccess_CurrentUser = 0;

        [DllImport("mfplat.dll", ExactSpelling = true)]
        internal static extern int MFStartup(int version, int flags);

        [DllImport("mfplat.dll", ExactSpelling = true)]
        internal static extern int MFShutdown();

        [DllImport("mfsensorgroup.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
        internal static extern int MFCreateVirtualCamera(
            int type,
            int lifetime,
            int access,
            string friendlyName,
            string sourceId,
            IntPtr categories,
            uint categoryCount,
            out IMFVirtualCamera virtualCamera);
    }
}

/// <summary>
/// Minimal COM projection of IMFVirtualCamera (mfvirtualcamera.h). The vtable
/// must list every inherited IMFAttributes method, in declaration order,
/// before the IMFVirtualCamera methods — the placeholders below exist solely
/// to keep the slots aligned and are never called by the shell.
/// </summary>
[ComImport]
[Guid("1C08A864-EF6C-4C75-AF59-5F2D68DA9563")]
[InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
internal interface IMFVirtualCamera
{
    // IMFAttributes (30 slots, order fixed by mfobjects.h).
    void GetItem(in Guid key, IntPtr value);
    void GetItemType(in Guid key, out int type);
    void CompareItem(in Guid key, IntPtr value, out bool result);
    void Compare(IntPtr theirs, int matchType, out bool result);
    void GetUINT32(in Guid key, out uint value);
    void GetUINT64(in Guid key, out ulong value);
    void GetDouble(in Guid key, out double value);
    void GetGUID(in Guid key, out Guid value);
    void GetStringLength(in Guid key, out uint length);
    void GetString(in Guid key, IntPtr value, uint size, out uint length);
    void GetAllocatedString(in Guid key, out IntPtr value, out uint length);
    void GetBlobSize(in Guid key, out uint size);
    void GetBlob(in Guid key, IntPtr buffer, uint size, out uint copied);
    void GetAllocatedBlob(in Guid key, out IntPtr buffer, out uint size);
    void GetUnknown(in Guid key, in Guid riid, out IntPtr obj);
    void SetItem(in Guid key, IntPtr value);
    void DeleteItem(in Guid key);
    void DeleteAllItems();
    void SetUINT32(in Guid key, uint value);
    void SetUINT64(in Guid key, ulong value);
    void SetDouble(in Guid key, double value);
    void SetGUID(in Guid key, in Guid value);
    void SetString(in Guid key, [MarshalAs(UnmanagedType.LPWStr)] string value);
    void SetBlob(in Guid key, IntPtr buffer, uint size);
    void SetUnknown(in Guid key, IntPtr obj);
    void LockStore();
    void UnlockStore();
    [PreserveSig] int GetCount(out uint count);
    void GetItemByIndex(uint index, out Guid key, IntPtr value);
    void CopyAllItems(IntPtr destination);

    // IMFVirtualCamera (mfvirtualcamera.h order).
    void AddDeviceSourceInfo([MarshalAs(UnmanagedType.LPWStr)] string deviceSourceInfo);
    void AddProperty(IntPtr property, uint propertyLength, IntPtr data, uint dataLength);
    void AddRegistryEntry(
        [MarshalAs(UnmanagedType.LPWStr)] string entryName,
        [MarshalAs(UnmanagedType.LPWStr)] string subkeyPath,
        int entryType,
        IntPtr data,
        uint dataLength);

    [PreserveSig] int Start(IntPtr callback);
    [PreserveSig] int Stop();
    [PreserveSig] int Remove();
    [PreserveSig] int Shutdown();

    void GetMediaSource(out IntPtr mediaSource);
    void SendCameraProperty(
        in Guid propertySet,
        uint propertyId,
        uint propertyFlags,
        IntPtr inputPayload,
        uint inputPayloadLength,
        IntPtr outputPayload,
        uint outputPayloadLength,
        out uint bytesReturned);
}
