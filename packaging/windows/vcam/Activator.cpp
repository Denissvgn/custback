#include "Activator.h"

#include <mferror.h>

#include "MediaSource.h"

namespace custback::vcam {

Activator::Activator() {
    winrt::check_hresult(MFCreateAttributes(m_attributes.put(), 4));
}

// ---- IMFActivate -----------------------------------------------------------

IFACEMETHODIMP Activator::ActivateObject(REFIID riid, void** object) noexcept try {
    if (object == nullptr) {
        return E_POINTER;
    }
    *object = nullptr;
    winrt::slim_lock_guard guard(m_lock);
    if (!m_source) {
        auto source = winrt::make_self<MediaSource>();
        winrt::check_hresult(source->RuntimeClassInitialize());
        m_source = source.as<IMFMediaSource>();
    }
    return m_source->QueryInterface(riid, object);
} catch (...) {
    return winrt::to_hresult();
}

IFACEMETHODIMP Activator::ShutdownObject() noexcept {
    winrt::slim_lock_guard guard(m_lock);
    if (m_source) {
        m_source->Shutdown();
        m_source = nullptr;
    }
    return S_OK;
}

IFACEMETHODIMP Activator::DetachObject() noexcept {
    winrt::slim_lock_guard guard(m_lock);
    m_source = nullptr;
    return S_OK;
}

// ---- IMFAttributes forwarding ---------------------------------------------

IFACEMETHODIMP Activator::GetItem(REFGUID key, PROPVARIANT* value) noexcept {
    return m_attributes->GetItem(key, value);
}
IFACEMETHODIMP Activator::GetItemType(REFGUID key,
                                      MF_ATTRIBUTE_TYPE* type) noexcept {
    return m_attributes->GetItemType(key, type);
}
IFACEMETHODIMP Activator::CompareItem(REFGUID key, REFPROPVARIANT value,
                                      BOOL* result) noexcept {
    return m_attributes->CompareItem(key, value, result);
}
IFACEMETHODIMP Activator::Compare(IMFAttributes* theirs,
                                  MF_ATTRIBUTES_MATCH_TYPE matchType,
                                  BOOL* result) noexcept {
    return m_attributes->Compare(theirs, matchType, result);
}
IFACEMETHODIMP Activator::GetUINT32(REFGUID key, UINT32* value) noexcept {
    return m_attributes->GetUINT32(key, value);
}
IFACEMETHODIMP Activator::GetUINT64(REFGUID key, UINT64* value) noexcept {
    return m_attributes->GetUINT64(key, value);
}
IFACEMETHODIMP Activator::GetDouble(REFGUID key, double* value) noexcept {
    return m_attributes->GetDouble(key, value);
}
IFACEMETHODIMP Activator::GetGUID(REFGUID key, GUID* value) noexcept {
    return m_attributes->GetGUID(key, value);
}
IFACEMETHODIMP Activator::GetStringLength(REFGUID key,
                                          UINT32* length) noexcept {
    return m_attributes->GetStringLength(key, length);
}
IFACEMETHODIMP Activator::GetString(REFGUID key, LPWSTR value, UINT32 size,
                                    UINT32* length) noexcept {
    return m_attributes->GetString(key, value, size, length);
}
IFACEMETHODIMP Activator::GetAllocatedString(REFGUID key, LPWSTR* value,
                                             UINT32* length) noexcept {
    return m_attributes->GetAllocatedString(key, value, length);
}
IFACEMETHODIMP Activator::GetBlobSize(REFGUID key, UINT32* size) noexcept {
    return m_attributes->GetBlobSize(key, size);
}
IFACEMETHODIMP Activator::GetBlob(REFGUID key, UINT8* buffer, UINT32 size,
                                  UINT32* copied) noexcept {
    return m_attributes->GetBlob(key, buffer, size, copied);
}
IFACEMETHODIMP Activator::GetAllocatedBlob(REFGUID key, UINT8** buffer,
                                           UINT32* size) noexcept {
    return m_attributes->GetAllocatedBlob(key, buffer, size);
}
IFACEMETHODIMP Activator::GetUnknown(REFGUID key, REFIID riid,
                                     void** object) noexcept {
    return m_attributes->GetUnknown(key, riid, object);
}
IFACEMETHODIMP Activator::SetItem(REFGUID key, REFPROPVARIANT value) noexcept {
    return m_attributes->SetItem(key, value);
}
IFACEMETHODIMP Activator::DeleteItem(REFGUID key) noexcept {
    return m_attributes->DeleteItem(key);
}
IFACEMETHODIMP Activator::DeleteAllItems() noexcept {
    return m_attributes->DeleteAllItems();
}
IFACEMETHODIMP Activator::SetUINT32(REFGUID key, UINT32 value) noexcept {
    return m_attributes->SetUINT32(key, value);
}
IFACEMETHODIMP Activator::SetUINT64(REFGUID key, UINT64 value) noexcept {
    return m_attributes->SetUINT64(key, value);
}
IFACEMETHODIMP Activator::SetDouble(REFGUID key, double value) noexcept {
    return m_attributes->SetDouble(key, value);
}
IFACEMETHODIMP Activator::SetGUID(REFGUID key, REFGUID value) noexcept {
    return m_attributes->SetGUID(key, value);
}
IFACEMETHODIMP Activator::SetString(REFGUID key, LPCWSTR value) noexcept {
    return m_attributes->SetString(key, value);
}
IFACEMETHODIMP Activator::SetBlob(REFGUID key, const UINT8* buffer,
                                  UINT32 size) noexcept {
    return m_attributes->SetBlob(key, buffer, size);
}
IFACEMETHODIMP Activator::SetUnknown(REFGUID key, IUnknown* object) noexcept {
    return m_attributes->SetUnknown(key, object);
}
IFACEMETHODIMP Activator::LockStore() noexcept {
    return m_attributes->LockStore();
}
IFACEMETHODIMP Activator::UnlockStore() noexcept {
    return m_attributes->UnlockStore();
}
IFACEMETHODIMP Activator::GetCount(UINT32* count) noexcept {
    return m_attributes->GetCount(count);
}
IFACEMETHODIMP Activator::GetItemByIndex(UINT32 index, GUID* key,
                                         PROPVARIANT* value) noexcept {
    return m_attributes->GetItemByIndex(index, key, value);
}
IFACEMETHODIMP Activator::CopyAllItems(IMFAttributes* destination) noexcept {
    return m_attributes->CopyAllItems(destination);
}

}  // namespace custback::vcam
