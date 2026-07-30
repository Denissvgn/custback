#include "MediaStream.h"

#include <mferror.h>

#include <algorithm>
#include <cstring>

#include "MediaSource.h"

namespace custback::vcam {

winrt::com_ptr<MediaStream> MediaStream::Create(MediaSource* parent,
                                                IMFStreamDescriptor* descriptor) {
    auto stream = winrt::make_self<MediaStream>();
    stream->m_parent = parent;
    stream->m_descriptor.copy_from(descriptor);
    winrt::check_hresult(
        MFCreateEventQueue(stream->m_eventQueue.put()));
    winrt::check_hresult(stream->RefreshNegotiatedType());
    return stream;
}

// ---- IMFMediaEventGenerator -----------------------------------------------

IFACEMETHODIMP MediaStream::GetEvent(DWORD flags, IMFMediaEvent** event) noexcept {
    winrt::com_ptr<IMFMediaEventQueue> queue;
    {
        std::lock_guard guard(m_lock);
        if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
            return hr;
        }
        queue = m_eventQueue;
    }
    return queue->GetEvent(flags, event);
}

IFACEMETHODIMP MediaStream::BeginGetEvent(IMFAsyncCallback* callback,
                                          IUnknown* state) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->BeginGetEvent(callback, state);
}

IFACEMETHODIMP MediaStream::EndGetEvent(IMFAsyncResult* result,
                                        IMFMediaEvent** event) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->EndGetEvent(result, event);
}

IFACEMETHODIMP MediaStream::QueueEvent(MediaEventType type, REFGUID extendedType,
                                       HRESULT status,
                                       const PROPVARIANT* value) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->QueueEventParamVar(type, extendedType, status, value);
}

// ---- IMFMediaStream --------------------------------------------------------

IFACEMETHODIMP MediaStream::GetMediaSource(IMFMediaSource** source) noexcept {
    if (source == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_parent->QueryInterface(IID_PPV_ARGS(source));
}

IFACEMETHODIMP MediaStream::GetStreamDescriptor(
    IMFStreamDescriptor** descriptor) noexcept {
    if (descriptor == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    m_descriptor.copy_to(descriptor);
    return S_OK;
}

IFACEMETHODIMP MediaStream::RequestSample(IUnknown* token) noexcept try {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    if (m_state != MF_STREAM_STATE_RUNNING) {
        return MF_E_INVALIDREQUEST;
    }

    // Geometry can change between Start calls (the consumer may renegotiate);
    // read it back cheaply on every sample so the payload always matches the
    // current media type.
    winrt::check_hresult(RefreshNegotiatedType());

    std::vector<uint8_t> payload;
    ComposeFrame(payload);

    winrt::com_ptr<IMFMediaBuffer> buffer;
    winrt::check_hresult(MFCreateMemoryBuffer(
        static_cast<DWORD>(payload.size()), buffer.put()));
    BYTE* destination = nullptr;
    DWORD maxLength = 0;
    winrt::check_hresult(buffer->Lock(&destination, &maxLength, nullptr));
    std::memcpy(destination, payload.data(), payload.size());
    winrt::check_hresult(buffer->Unlock());
    winrt::check_hresult(
        buffer->SetCurrentLength(static_cast<DWORD>(payload.size())));

    winrt::com_ptr<IMFSample> sample;
    winrt::check_hresult(MFCreateSample(sample.put()));
    winrt::check_hresult(sample->AddBuffer(buffer.get()));

    const LONGLONG now = MFGetSystemTime();
    if (m_startTime == 0) {
        m_startTime = now;
    }
    winrt::check_hresult(sample->SetSampleTime(now - m_startTime));
    const LONGLONG duration =
        (10'000'000ll * m_fpsDenominator) / std::max(1u, m_fpsNumerator);
    winrt::check_hresult(sample->SetSampleDuration(duration));
    ++m_sampleIndex;

    if (token != nullptr) {
        winrt::check_hresult(
            sample->SetUnknown(MFSampleExtension_Token, token));
    }

    return m_eventQueue->QueueEventParamUnk(
        MEMediaSample, GUID_NULL, S_OK, sample.get());
} catch (...) {
    return winrt::to_hresult();
}

// ---- IMFMediaStream2 -------------------------------------------------------

IFACEMETHODIMP MediaStream::SetStreamState(MF_STREAM_STATE state) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    switch (state) {
        case MF_STREAM_STATE_RUNNING:
            if (!m_ring.IsOpen()) {
                m_ring.Open();  // soft: placeholder until the engine appears
            }
            m_state = state;
            return S_OK;
        case MF_STREAM_STATE_STOPPED:
        case MF_STREAM_STATE_PAUSED:
            m_state = state;
            return S_OK;
        default:
            return MF_E_INVALID_STATE_TRANSITION;
    }
}

IFACEMETHODIMP MediaStream::GetStreamState(MF_STREAM_STATE* state) noexcept {
    if (state == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    *state = m_state;
    return S_OK;
}

// ---- source-driven lifecycle ----------------------------------------------

HRESULT MediaStream::Start() {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    if (!m_ring.IsOpen()) {
        m_ring.Open();  // soft-fail: the placeholder covers an idle engine
    }
    RefreshNegotiatedType();
    m_state = MF_STREAM_STATE_RUNNING;
    m_startTime = 0;
    return m_eventQueue->QueueEventParamVar(
        MEStreamStarted, GUID_NULL, S_OK, nullptr);
}

HRESULT MediaStream::Stop() {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    m_state = MF_STREAM_STATE_STOPPED;
    return m_eventQueue->QueueEventParamVar(
        MEStreamStopped, GUID_NULL, S_OK, nullptr);
}

HRESULT MediaStream::Shutdown() {
    std::lock_guard guard(m_lock);
    if (m_shutdown) {
        return S_OK;
    }
    m_shutdown = true;
    m_state = MF_STREAM_STATE_STOPPED;
    if (m_eventQueue) {
        m_eventQueue->Shutdown();
    }
    m_ring.Close();
    m_descriptor = nullptr;
    return S_OK;
}

// ---- frame composition -----------------------------------------------------

HRESULT MediaStream::RefreshNegotiatedType() {
    winrt::com_ptr<IMFMediaTypeHandler> handler;
    HRESULT hr = m_descriptor->GetMediaTypeHandler(handler.put());
    if (FAILED(hr)) {
        return hr;
    }
    winrt::com_ptr<IMFMediaType> current;
    hr = handler->GetCurrentMediaType(current.put());
    if (FAILED(hr)) {
        return hr;
    }
    UINT32 width = 0;
    UINT32 height = 0;
    if (SUCCEEDED(MFGetAttributeSize(current.get(), MF_MT_FRAME_SIZE, &width,
                                     &height)) &&
        width != 0 && height != 0) {
        m_width = width;
        m_height = height;
    }
    UINT32 numerator = 0;
    UINT32 denominator = 0;
    if (SUCCEEDED(MFGetAttributeRatio(current.get(), MF_MT_FRAME_RATE,
                                      &numerator, &denominator)) &&
        numerator != 0 && denominator != 0) {
        m_fpsNumerator = numerator;
        m_fpsDenominator = denominator;
    }
    return S_OK;
}

namespace {

// Idle placeholder: a dark slate so consumers show a deliberate "camera on,
// engine idle" image instead of black or a frozen frame.
void FillPlaceholder(std::vector<uint8_t>& out, uint32_t width,
                     uint32_t height) {
    out.assign(size_t{width} * height * kBytesPerPixel, 0);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t* row = out.data() + size_t{y} * width * kBytesPerPixel;
        const uint8_t shade =
            static_cast<uint8_t>(24 + (16 * y) / std::max(1u, height));
        for (uint32_t x = 0; x < width; ++x) {
            row[x * kBytesPerPixel + 0] = shade;       // B
            row[x * kBytesPerPixel + 1] = shade;       // G
            row[x * kBytesPerPixel + 2] = shade;       // R
            row[x * kBytesPerPixel + 3] = 0xFF;        // X
        }
    }
}

}  // namespace

void MediaStream::ComposeFrame(std::vector<uint8_t>& out) {
    uint32_t ringWidth = 0;
    uint32_t ringHeight = 0;
    FrameReadStatus readStatus = FrameReadStatus::Unavailable;
    if (m_ring.IsOpen() || m_ring.Open()) {
        readStatus = m_ring.CopyLatest(m_ringFrame, ringWidth, ringHeight);
    }
    if (readStatus != FrameReadStatus::Complete) {
        const size_t expected =
            size_t{m_width} * m_height * kBytesPerPixel;
        if (readStatus == FrameReadStatus::Transient &&
            m_ringFrame.size() == expected) {
            out = m_ringFrame;
            return;
        }
        m_ringFrame.clear();
        FillPlaceholder(out, m_width, m_height);
        return;
    }
    if (ringWidth == m_width && ringHeight == m_height) {
        out = m_ringFrame;
        return;
    }
    // The Python writer and consumer selected different advertised exact
    // modes.  Overlap-copying would silently crop or letterbox the scene, and
    // this constrained Phase-1 path deliberately implements no native scaler.
    // Publish only the input-independent placeholder until the consumer
    // negotiates the active ring geometry.
    m_ringFrame.clear();
    FillPlaceholder(out, m_width, m_height);
}

}  // namespace custback::vcam
