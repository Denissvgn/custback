// Shared frame-ring protocol between the custback engine and the native
// Media Foundation virtual camera (WIN-6.1).
//
// This header is the C++ mirror of src/custback/vcam_native.py, which is the
// single source of truth for the layout.  tests/test_windows_vcam.py parses
// the constants below and fails when the two sides drift.
//
// Contract (seqlock, latest-frame-wins):
//   * One named page-file section: 64-byte header + one BGRX frame.
//   * The writer (engine) makes `seq` odd, writes payload + header, makes
//     `seq` even.  A reader observing an odd `seq`, or a `seq` change across
//     its copy, discards the torn frame and retries a bounded number of times.
//   * No cross-process event: the camera produces samples on its own clock
//     and holds the last good frame (or a placeholder while FLAG_ACTIVE is
//     clear / the section is absent) when the producer stalls.

#pragma once

#include <windows.h>

#if defined(CUSTBACK_VCAM_GATE_DIAGNOSTICS)
#include <strsafe.h>
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace custback::vcam {

// Mirrors vcam_native.SECTION_NAME.
inline constexpr wchar_t kSectionName[] = L"Local\\CustbackVCamFrame0";

// Mirrors vcam_native.MAGIC ("CBVC") and PROTOCOL_VERSION.
inline constexpr uint32_t kMagic = 0x43564243u;  // "CBVC" little-endian
inline constexpr uint32_t kProtocolVersion = 1u;

// Mirrors vcam_native.FOURCC ("BGRX"): MFVideoFormat_RGB32 memory order.
inline constexpr uint32_t kFourcc = 0x58524742u;  // "BGRX" little-endian
inline constexpr uint32_t kBytesPerPixel = 4u;

// Mirrors vcam_native.HEADER_SIZE / FLAG_ACTIVE.
inline constexpr uint32_t kHeaderSize = 64u;
inline constexpr uint32_t kFlagActive = 0x1u;

#pragma pack(push, 1)
struct FrameRingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t fourcc;
    uint32_t seq;
    uint32_t flags;
    uint64_t timestamp_100ns;
    uint64_t frame_counter;
    uint8_t reserved[16];
};
#pragma pack(pop)

static_assert(sizeof(FrameRingHeader) == kHeaderSize,
              "header layout must match vcam_native.HEADER_FORMAT");

// Read-only view over the engine's shared ring.  All failures are soft: the
// camera must keep serving frames (placeholder) when the engine is not
// running, so Open()/CopyLatest() report false rather than throwing.
class FrameRingReader {
 public:
    FrameRingReader() = default;
    FrameRingReader(const FrameRingReader&) = delete;
    FrameRingReader& operator=(const FrameRingReader&) = delete;

    ~FrameRingReader() { Close(); }

    bool Open(const wchar_t* sectionName = kSectionName) {
        Close();
        m_section = ::OpenFileMappingW(FILE_MAP_READ, FALSE, sectionName);
        if (m_section == nullptr) {
#if defined(CUSTBACK_VCAM_GATE_DIAGNOSTICS)
            const DWORD error = ::GetLastError();
            TraceOpenFailure(sectionName, error);
#endif
            return false;
        }
#if defined(CUSTBACK_VCAM_GATE_DIAGNOSTICS)
        m_reportedOpenFailure = false;
#endif
        m_view = static_cast<const uint8_t*>(
            ::MapViewOfFile(m_section, FILE_MAP_READ, 0, 0, 0));
        if (m_view == nullptr) {
            Close();
            return false;
        }
        MEMORY_BASIC_INFORMATION info{};
        if (::VirtualQuery(m_view, &info, sizeof(info)) == 0) {
            Close();
            return false;
        }
        m_viewSize = info.RegionSize;
        return m_viewSize >= kHeaderSize;
    }

    bool IsOpen() const { return m_view != nullptr; }

    void Close() {
        if (m_view != nullptr) {
            ::UnmapViewOfFile(m_view);
            m_view = nullptr;
        }
        if (m_section != nullptr) {
            ::CloseHandle(m_section);
            m_section = nullptr;
        }
        m_viewSize = 0;
    }

    // Copies the newest complete frame into `frame` (resized to fit) and
    // reports its geometry.  Mirrors vcam_native.read_latest_frame().
    bool CopyLatest(std::vector<uint8_t>& frame, uint32_t& width,
                    uint32_t& height, int maxAttempts = 4) {
        if (!IsOpen()) {
            return false;
        }
        for (int attempt = 0; attempt < maxAttempts; ++attempt) {
            FrameRingHeader header{};
            std::memcpy(&header, m_view, sizeof(header));
            const auto& sharedSeq = *reinterpret_cast<const uint32_t*>(
                m_view + offsetof(FrameRingHeader, seq));
            const uint32_t seqBefore =
                std::atomic_ref<const uint32_t>(sharedSeq).load(
                    std::memory_order_acquire);
            if (seqBefore != header.seq || (seqBefore & 1u) != 0u) {
                continue;  // torn header or write in progress
            }
            if (header.magic != kMagic || header.version != kProtocolVersion) {
                return false;
            }
            if ((header.flags & kFlagActive) == 0) {
                return false;
            }
            const uint64_t payload =
                uint64_t{header.width} * header.height * kBytesPerPixel;
            if (header.width == 0 || header.height == 0 ||
                kHeaderSize + payload > m_viewSize) {
                return false;
            }
            frame.resize(static_cast<size_t>(payload));
            std::memcpy(frame.data(), m_view + kHeaderSize,
                        static_cast<size_t>(payload));
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint32_t seqAfter =
                std::atomic_ref<const uint32_t>(sharedSeq).load(
                    std::memory_order_acquire);
            if (seqAfter != seqBefore) {
                continue;  // torn: the writer moved on mid-copy
            }
            width = header.width;
            height = header.height;
            return true;
        }
        return false;
    }

 private:
#if defined(CUSTBACK_VCAM_GATE_DIAGNOSTICS)
    void TraceOpenFailure(const wchar_t* sectionName, DWORD error) {
        if (m_reportedOpenFailure && error == m_lastReportedOpenError) {
            return;
        }
        m_reportedOpenFailure = true;
        m_lastReportedOpenError = error;

        wchar_t message[256]{};
        if (SUCCEEDED(::StringCchPrintfW(
                message, ARRAYSIZE(message),
                L"CustbackVCam MIT-C1: OpenFileMappingW(\"%ls\") failed; "
                L"GetLastError=%lu\r\n",
                sectionName, error))) {
            ::OutputDebugStringW(message);
        }
    }

    bool m_reportedOpenFailure = false;
    DWORD m_lastReportedOpenError = ERROR_SUCCESS;
#endif

    HANDLE m_section = nullptr;
    const uint8_t* m_view = nullptr;
    SIZE_T m_viewSize = 0;
};

}  // namespace custback::vcam
