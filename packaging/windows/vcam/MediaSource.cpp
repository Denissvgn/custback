#include "MediaSource.h"

#include <ksmedia.h>
#include <mferror.h>

namespace custback::vcam {

namespace {

winrt::com_ptr<IMFMediaType> MakeVideoType(const StreamFormat& format) {
    winrt::com_ptr<IMFMediaType> type;
    winrt::check_hresult(MFCreateMediaType(type.put()));
    winrt::check_hresult(type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video));
    winrt::check_hresult(type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32));
    winrt::check_hresult(MFSetAttributeSize(type.get(), MF_MT_FRAME_SIZE,
                                            format.width, format.height));
    winrt::check_hresult(MFSetAttributeRatio(type.get(), MF_MT_FRAME_RATE,
                                             format.fps, 1));
    winrt::check_hresult(MFSetAttributeRatio(type.get(),
                                             MF_MT_PIXEL_ASPECT_RATIO, 1, 1));
    winrt::check_hresult(type->SetUINT32(MF_MT_INTERLACE_MODE,
                                         MFVideoInterlace_Progressive));
    // The Python pipeline publishes canonical full-range display-referred
    // sRGB BGR, and the ring writer only appends an opaque X byte.  Therefore
    // RGB32 consumers receive the same proven BT.709/sRGB/0..255 contract.
    winrt::check_hresult(type->SetUINT32(
        MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709));
    winrt::check_hresult(type->SetUINT32(
        MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_sRGB));
    winrt::check_hresult(type->SetUINT32(
        MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_0_255));
    winrt::check_hresult(type->SetUINT32(MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE));
    winrt::check_hresult(type->SetUINT32(MF_MT_DEFAULT_STRIDE,
                                         format.width * kBytesPerPixel));
    winrt::check_hresult(type->SetUINT32(
        MF_MT_SAMPLE_SIZE, format.width * format.height * kBytesPerPixel));
    return type;
}

}  // namespace

HRESULT MediaSource::RuntimeClassInitialize() try {
    winrt::check_hresult(MFCreateEventQueue(m_eventQueue.put()));
    winrt::check_hresult(MFCreateAttributes(m_sourceAttributes.put(), 1));

    winrt::check_hresult(CreateStreamDescriptor(m_streamDescriptor.put()));
    IMFStreamDescriptor* descriptors[] = {m_streamDescriptor.get()};
    winrt::check_hresult(MFCreatePresentationDescriptor(
        1, descriptors, m_presentationDescriptor.put()));
    winrt::check_hresult(m_presentationDescriptor->SelectStream(0));

    m_stream = MediaStream::Create(this, m_streamDescriptor.get());
    return S_OK;
} catch (...) {
    return winrt::to_hresult();
}

HRESULT MediaSource::CreateStreamDescriptor(IMFStreamDescriptor** descriptor) try {
    size_t selectedFormat = 0;
    FrameRingReader ring;
    uint32_t ringWidth = 0;
    uint32_t ringHeight = 0;
    if (ring.Open()) {
        if (!ring.ReadGeometry(ringWidth, ringHeight)) {
            winrt::check_hresult(MF_E_INVALIDMEDIATYPE);
        }
        bool matched = false;
        for (size_t i = 0; i < ARRAYSIZE(kStreamFormats); ++i) {
            if (kStreamFormats[i].width == ringWidth &&
                kStreamFormats[i].height == ringHeight) {
                selectedFormat = i;
                matched = true;
                break;
            }
        }
        if (!matched) {
            winrt::check_hresult(MF_E_INVALIDMEDIATYPE);
        }
    }

    winrt::com_ptr<IMFMediaType> type =
        MakeVideoType(kStreamFormats[selectedFormat]);
    IMFMediaType* raw[] = {type.get()};

    winrt::com_ptr<IMFStreamDescriptor> sd;
    winrt::check_hresult(MFCreateStreamDescriptor(
        0, ARRAYSIZE(raw), raw, sd.put()));

    winrt::com_ptr<IMFMediaTypeHandler> handler;
    winrt::check_hresult(sd->GetMediaTypeHandler(handler.put()));
    winrt::check_hresult(handler->SetCurrentMediaType(raw[0]));

    // Frame Server stream identity: a color capture pin, shareable, id 0.
    winrt::check_hresult(sd->SetGUID(MF_DEVICESTREAM_STREAM_CATEGORY,
                                     PINNAME_VIDEO_CAPTURE));
    winrt::check_hresult(sd->SetUINT32(MF_DEVICESTREAM_STREAM_ID, 0));
    winrt::check_hresult(sd->SetUINT32(MF_DEVICESTREAM_FRAMESERVER_SHARED, 1));
    winrt::check_hresult(sd->SetUINT32(
        MF_DEVICESTREAM_ATTRIBUTE_FRAMESOURCE_TYPES, MFFrameSourceTypes_Color));

    *descriptor = sd.detach();
    return S_OK;
} catch (...) {
    return winrt::to_hresult();
}

// ---- IMFMediaEventGenerator -----------------------------------------------

IFACEMETHODIMP MediaSource::GetEvent(DWORD flags, IMFMediaEvent** event) noexcept {
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

IFACEMETHODIMP MediaSource::BeginGetEvent(IMFAsyncCallback* callback,
                                          IUnknown* state) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->BeginGetEvent(callback, state);
}

IFACEMETHODIMP MediaSource::EndGetEvent(IMFAsyncResult* result,
                                        IMFMediaEvent** event) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->EndGetEvent(result, event);
}

IFACEMETHODIMP MediaSource::QueueEvent(MediaEventType type, REFGUID extendedType,
                                       HRESULT status,
                                       const PROPVARIANT* value) noexcept {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_eventQueue->QueueEventParamVar(type, extendedType, status, value);
}

// ---- IMFMediaSource --------------------------------------------------------

IFACEMETHODIMP MediaSource::GetCharacteristics(DWORD* characteristics) noexcept {
    if (characteristics == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    // Live capture device: no pause, no seek.
    *characteristics = MFMEDIASOURCE_IS_LIVE;
    return S_OK;
}

IFACEMETHODIMP MediaSource::CreatePresentationDescriptor(
    IMFPresentationDescriptor** descriptor) noexcept {
    if (descriptor == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    return m_presentationDescriptor->Clone(descriptor);
}

IFACEMETHODIMP MediaSource::Start(IMFPresentationDescriptor* descriptor,
                                  const GUID* timeFormat,
                                  const PROPVARIANT* startPosition) noexcept try {
    if (descriptor == nullptr) {
        return E_INVALIDARG;
    }
    if (timeFormat != nullptr && *timeFormat != GUID_NULL) {
        return MF_E_UNSUPPORTED_TIME_FORMAT;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }

    // Announce the stream: MENewStream on first start, MEUpdatedStream after,
    // then let the stream publish MEStreamStarted before MESourceStarted.
    winrt::com_ptr<IUnknown> streamUnknown;
    winrt::check_hresult(m_stream->QueryInterface(
        IID_PPV_ARGS(streamUnknown.put())));
    winrt::check_hresult(m_eventQueue->QueueEventParamUnk(
        m_streamAnnounced ? MEUpdatedStream : MENewStream, GUID_NULL, S_OK,
        streamUnknown.get()));
    m_streamAnnounced = true;

    winrt::check_hresult(m_stream->Start());

    PROPVARIANT startVariant;
    PropVariantInit(&startVariant);
    if (startPosition != nullptr) {
        startVariant = *startPosition;  // shallow: queued by value below
    }
    const HRESULT queued = m_eventQueue->QueueEventParamVar(
        MESourceStarted, GUID_NULL, S_OK,
        startPosition != nullptr ? &startVariant : nullptr);
    return queued;
} catch (...) {
    return winrt::to_hresult();
}

IFACEMETHODIMP MediaSource::Stop() noexcept try {
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    winrt::check_hresult(m_stream->Stop());
    return m_eventQueue->QueueEventParamVar(
        MESourceStopped, GUID_NULL, S_OK, nullptr);
} catch (...) {
    return winrt::to_hresult();
}

IFACEMETHODIMP MediaSource::Pause() noexcept {
    // A live camera source does not pause.
    return MF_E_INVALID_STATE_TRANSITION;
}

IFACEMETHODIMP MediaSource::Shutdown() noexcept {
    std::lock_guard guard(m_lock);
    if (m_shutdown) {
        return S_OK;
    }
    m_shutdown = true;
    if (m_stream) {
        m_stream->Shutdown();
    }
    if (m_eventQueue) {
        m_eventQueue->Shutdown();
    }
    m_stream = nullptr;
    m_presentationDescriptor = nullptr;
    m_streamDescriptor = nullptr;
    m_sourceAttributes = nullptr;
    return S_OK;
}

// ---- IMFMediaSourceEx ------------------------------------------------------

IFACEMETHODIMP MediaSource::GetSourceAttributes(
    IMFAttributes** attributes) noexcept {
    if (attributes == nullptr) {
        return E_POINTER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    m_sourceAttributes.copy_to(attributes);
    return S_OK;
}

IFACEMETHODIMP MediaSource::GetStreamAttributes(
    DWORD streamId, IMFAttributes** attributes) noexcept {
    if (attributes == nullptr) {
        return E_POINTER;
    }
    if (streamId != 0) {
        return MF_E_INVALIDSTREAMNUMBER;
    }
    std::lock_guard guard(m_lock);
    if (HRESULT hr = CheckShutdown(); FAILED(hr)) {
        return hr;
    }
    winrt::com_ptr<IMFStreamDescriptor> descriptor = m_streamDescriptor;
    return descriptor->QueryInterface(IID_PPV_ARGS(attributes));
}

IFACEMETHODIMP MediaSource::SetD3DManager(IUnknown* /*manager*/) noexcept {
    // Samples are system-memory RGB32; a device manager is accepted and
    // ignored, per the software-camera-source contract.
    return S_OK;
}

// ---- IMFGetService ---------------------------------------------------------

IFACEMETHODIMP MediaSource::GetService(REFGUID /*service*/, REFIID /*riid*/,
                                       void** object) noexcept {
    if (object != nullptr) {
        *object = nullptr;
    }
    return MF_E_UNSUPPORTED_SERVICE;
}

// ---- IKsControl ------------------------------------------------------------

IFACEMETHODIMP MediaSource::KsProperty(PKSPROPERTY /*property*/,
                                       ULONG /*propertyLength*/, void* /*data*/,
                                       ULONG /*dataLength*/,
                                       ULONG* bytesReturned) noexcept {
    if (bytesReturned != nullptr) {
        *bytesReturned = 0;
    }
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

IFACEMETHODIMP MediaSource::KsMethod(PKSMETHOD /*method*/,
                                     ULONG /*methodLength*/, void* /*data*/,
                                     ULONG /*dataLength*/,
                                     ULONG* bytesReturned) noexcept {
    if (bytesReturned != nullptr) {
        *bytesReturned = 0;
    }
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

IFACEMETHODIMP MediaSource::KsEvent(PKSEVENT /*event*/, ULONG /*eventLength*/,
                                    void* /*data*/, ULONG /*dataLength*/,
                                    ULONG* bytesReturned) noexcept {
    if (bytesReturned != nullptr) {
        *bytesReturned = 0;
    }
    return HRESULT_FROM_WIN32(ERROR_SET_NOT_FOUND);
}

}  // namespace custback::vcam
