// Custom media source of the Custback native virtual camera (WIN-6.1).
//
// Instantiated by the Windows 11 Frame Server (inside its service process)
// through the Activator COM class after the desktop shell registers the
// camera with MFCreateVirtualCamera (session lifetime, current user).  It
// exposes exactly one always-selected RGB32 video stream whose frames come
// from the engine's shared ring (FrameRing.h).

#pragma once

#include <ks.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <winrt/base.h>

#include <mutex>

#include "MediaStream.h"

namespace custback::vcam {

// Media types offered on the single stream, in preference order.  720p first:
// it matches the engine's default camera geometry, so the common negotiation
// needs no centering pass.
struct StreamFormat {
    uint32_t width;
    uint32_t height;
    uint32_t fps;
};
inline constexpr StreamFormat kStreamFormats[] = {
    {1280, 720, 30},
    {1920, 1080, 30},
};

struct MediaSource : winrt::implements<MediaSource, IMFMediaSourceEx,
                                       IMFMediaSource, IMFMediaEventGenerator,
                                       IMFGetService, IKsControl> {
    HRESULT RuntimeClassInitialize();

    // IMFMediaEventGenerator
    IFACEMETHODIMP GetEvent(DWORD flags, IMFMediaEvent** event) noexcept override;
    IFACEMETHODIMP BeginGetEvent(IMFAsyncCallback* callback,
                                 IUnknown* state) noexcept override;
    IFACEMETHODIMP EndGetEvent(IMFAsyncResult* result,
                               IMFMediaEvent** event) noexcept override;
    IFACEMETHODIMP QueueEvent(MediaEventType type, REFGUID extendedType,
                              HRESULT status,
                              const PROPVARIANT* value) noexcept override;

    // IMFMediaSource
    IFACEMETHODIMP GetCharacteristics(DWORD* characteristics) noexcept override;
    IFACEMETHODIMP CreatePresentationDescriptor(
        IMFPresentationDescriptor** descriptor) noexcept override;
    IFACEMETHODIMP Start(IMFPresentationDescriptor* descriptor,
                         const GUID* timeFormat,
                         const PROPVARIANT* startPosition) noexcept override;
    IFACEMETHODIMP Stop() noexcept override;
    IFACEMETHODIMP Pause() noexcept override;
    IFACEMETHODIMP Shutdown() noexcept override;

    // IMFMediaSourceEx
    IFACEMETHODIMP GetSourceAttributes(
        IMFAttributes** attributes) noexcept override;
    IFACEMETHODIMP GetStreamAttributes(
        DWORD streamId, IMFAttributes** attributes) noexcept override;
    IFACEMETHODIMP SetD3DManager(IUnknown* manager) noexcept override;

    // IMFGetService
    IFACEMETHODIMP GetService(REFGUID service, REFIID riid,
                              void** object) noexcept override;

    // IKsControl — probed by Frame Server; no property sets are implemented.
    IFACEMETHODIMP KsProperty(PKSPROPERTY property, ULONG propertyLength,
                              void* data, ULONG dataLength,
                              ULONG* bytesReturned) noexcept override;
    IFACEMETHODIMP KsMethod(PKSMETHOD method, ULONG methodLength, void* data,
                            ULONG dataLength,
                            ULONG* bytesReturned) noexcept override;
    IFACEMETHODIMP KsEvent(PKSEVENT event, ULONG eventLength, void* data,
                           ULONG dataLength,
                           ULONG* bytesReturned) noexcept override;

 private:
    HRESULT CheckShutdown() const {
        return m_shutdown ? MF_E_SHUTDOWN : S_OK;
    }
    HRESULT CreateStreamDescriptor(IMFStreamDescriptor** descriptor);

    std::mutex m_lock;
    bool m_shutdown = false;
    bool m_streamAnnounced = false;

    winrt::com_ptr<IMFMediaEventQueue> m_eventQueue;
    winrt::com_ptr<IMFAttributes> m_sourceAttributes;
    winrt::com_ptr<IMFStreamDescriptor> m_streamDescriptor;
    winrt::com_ptr<IMFPresentationDescriptor> m_presentationDescriptor;
    winrt::com_ptr<MediaStream> m_stream;
};

}  // namespace custback::vcam
