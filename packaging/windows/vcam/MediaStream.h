// Video stream of the Custback native virtual camera (WIN-6.1).
//
// One always-selected RGB32 stream.  Frame Server drives the cadence through
// RequestSample; each request is answered synchronously with the newest
// complete engine frame from the shared ring, or with the placeholder when
// the engine is not publishing.  There is no worker thread and no queue —
// latest-frame-wins is the whole delivery model.

#pragma once

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <winrt/base.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "FrameRing.h"

namespace custback::vcam {

struct MediaSource;

struct MediaStream : winrt::implements<MediaStream, IMFMediaStream2,
                                       IMFMediaStream, IMFMediaEventGenerator> {
    static winrt::com_ptr<MediaStream> Create(MediaSource* parent,
                                              IMFStreamDescriptor* descriptor);

    // IMFMediaEventGenerator
    IFACEMETHODIMP GetEvent(DWORD flags, IMFMediaEvent** event) noexcept override;
    IFACEMETHODIMP BeginGetEvent(IMFAsyncCallback* callback,
                                 IUnknown* state) noexcept override;
    IFACEMETHODIMP EndGetEvent(IMFAsyncResult* result,
                               IMFMediaEvent** event) noexcept override;
    IFACEMETHODIMP QueueEvent(MediaEventType type, REFGUID extendedType,
                              HRESULT status,
                              const PROPVARIANT* value) noexcept override;

    // IMFMediaStream
    IFACEMETHODIMP GetMediaSource(IMFMediaSource** source) noexcept override;
    IFACEMETHODIMP GetStreamDescriptor(
        IMFStreamDescriptor** descriptor) noexcept override;
    IFACEMETHODIMP RequestSample(IUnknown* token) noexcept override;

    // IMFMediaStream2
    IFACEMETHODIMP SetStreamState(MF_STREAM_STATE state) noexcept override;
    IFACEMETHODIMP GetStreamState(MF_STREAM_STATE* state) noexcept override;

    // Called by the owning MediaSource, under its lock.
    HRESULT Start();
    HRESULT Stop();
    HRESULT Shutdown();

 private:
    HRESULT CheckShutdown() const {
        return m_shutdown ? MF_E_SHUTDOWN : S_OK;
    }

    // Reads the negotiated geometry from the descriptor's current media type.
    HRESULT RefreshNegotiatedType();

    // Builds the outgoing RGB32 payload: the engine frame centered into the
    // negotiated geometry, or the placeholder when the ring is unavailable.
    void ComposeFrame(std::vector<uint8_t>& out);

    winrt::com_ptr<IMFMediaEventQueue> m_eventQueue;
    winrt::com_ptr<IMFStreamDescriptor> m_descriptor;
    MediaSource* m_parent = nullptr;  // weak: parent outlives its stream

    std::mutex m_lock;
    bool m_shutdown = false;
    MF_STREAM_STATE m_state = MF_STREAM_STATE_STOPPED;

    FrameRingReader m_ring;
    std::vector<uint8_t> m_ringFrame;
    uint32_t m_width = 1280;
    uint32_t m_height = 720;
    uint32_t m_fpsNumerator = 30;
    uint32_t m_fpsDenominator = 1;
    LONGLONG m_startTime = 0;
    UINT64 m_sampleIndex = 0;
};

}  // namespace custback::vcam
