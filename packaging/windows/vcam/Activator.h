// COM activator for the Custback virtual camera media source (WIN-6.1).
//
// This is the class the installer registers under HKCU\Software\Classes and
// whose CLSID the shell passes to MFCreateVirtualCamera.  Frame Server
// CoCreates it and calls IMFActivate::ActivateObject to obtain the media
// source.  IMFActivate derives from IMFAttributes, so the full attribute
// surface is forwarded to an owned attribute store (Frame Server stamps
// configuration onto the activator before activation).

#pragma once

#include <mfapi.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <winrt/base.h>

namespace custback::vcam {

// Canonical CLSID of the virtual camera activator.  Must match, byte for
// byte: vcam_native.VCAM_CLSID (Python), VirtualCameraSession.cs (shell),
// and Package.wxs (install/uninstall registration).
// {7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}
inline constexpr wchar_t kActivatorClsidString[] =
    L"{7A4C1B2E-9D35-4E6A-8B1F-52C84D9A6E01}";
inline constexpr GUID kActivatorClsid = {
    0x7A4C1B2E,
    0x9D35,
    0x4E6A,
    {0x8B, 0x1F, 0x52, 0xC8, 0x4D, 0x9A, 0x6E, 0x01}};

struct Activator : winrt::implements<Activator, IMFActivate, IMFAttributes> {
    Activator();

    // IMFActivate
    IFACEMETHODIMP ActivateObject(REFIID riid, void** object) noexcept override;
    IFACEMETHODIMP ShutdownObject() noexcept override;
    IFACEMETHODIMP DetachObject() noexcept override;

    // IMFAttributes — forwarded to the owned store.
    IFACEMETHODIMP GetItem(REFGUID key, PROPVARIANT* value) noexcept override;
    IFACEMETHODIMP GetItemType(REFGUID key, MF_ATTRIBUTE_TYPE* type) noexcept override;
    IFACEMETHODIMP CompareItem(REFGUID key, REFPROPVARIANT value,
                               BOOL* result) noexcept override;
    IFACEMETHODIMP Compare(IMFAttributes* theirs,
                           MF_ATTRIBUTES_MATCH_TYPE matchType,
                           BOOL* result) noexcept override;
    IFACEMETHODIMP GetUINT32(REFGUID key, UINT32* value) noexcept override;
    IFACEMETHODIMP GetUINT64(REFGUID key, UINT64* value) noexcept override;
    IFACEMETHODIMP GetDouble(REFGUID key, double* value) noexcept override;
    IFACEMETHODIMP GetGUID(REFGUID key, GUID* value) noexcept override;
    IFACEMETHODIMP GetStringLength(REFGUID key, UINT32* length) noexcept override;
    IFACEMETHODIMP GetString(REFGUID key, LPWSTR value, UINT32 size,
                             UINT32* length) noexcept override;
    IFACEMETHODIMP GetAllocatedString(REFGUID key, LPWSTR* value,
                                      UINT32* length) noexcept override;
    IFACEMETHODIMP GetBlobSize(REFGUID key, UINT32* size) noexcept override;
    IFACEMETHODIMP GetBlob(REFGUID key, UINT8* buffer, UINT32 size,
                           UINT32* copied) noexcept override;
    IFACEMETHODIMP GetAllocatedBlob(REFGUID key, UINT8** buffer,
                                    UINT32* size) noexcept override;
    IFACEMETHODIMP GetUnknown(REFGUID key, REFIID riid,
                              void** object) noexcept override;
    IFACEMETHODIMP SetItem(REFGUID key, REFPROPVARIANT value) noexcept override;
    IFACEMETHODIMP DeleteItem(REFGUID key) noexcept override;
    IFACEMETHODIMP DeleteAllItems() noexcept override;
    IFACEMETHODIMP SetUINT32(REFGUID key, UINT32 value) noexcept override;
    IFACEMETHODIMP SetUINT64(REFGUID key, UINT64 value) noexcept override;
    IFACEMETHODIMP SetDouble(REFGUID key, double value) noexcept override;
    IFACEMETHODIMP SetGUID(REFGUID key, REFGUID value) noexcept override;
    IFACEMETHODIMP SetString(REFGUID key, LPCWSTR value) noexcept override;
    IFACEMETHODIMP SetBlob(REFGUID key, const UINT8* buffer,
                           UINT32 size) noexcept override;
    IFACEMETHODIMP SetUnknown(REFGUID key, IUnknown* object) noexcept override;
    IFACEMETHODIMP LockStore() noexcept override;
    IFACEMETHODIMP UnlockStore() noexcept override;
    IFACEMETHODIMP GetCount(UINT32* count) noexcept override;
    IFACEMETHODIMP GetItemByIndex(UINT32 index, GUID* key,
                                  PROPVARIANT* value) noexcept override;
    IFACEMETHODIMP CopyAllItems(IMFAttributes* destination) noexcept override;

 private:
    winrt::com_ptr<IMFAttributes> m_attributes;
    winrt::com_ptr<IMFMediaSource> m_source;
    winrt::slim_mutex m_lock;
};

}  // namespace custback::vcam
