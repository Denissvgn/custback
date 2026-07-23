// COM entry points of the Custback virtual camera DLL (WIN-6.1).
//
// The DLL exports the classic in-proc server surface for the single Activator
// class.  Registration is declarative: the installer writes the per-user
// HKCU\Software\Classes\CLSID\{...}\InProcServer32 keys (Package.wxs) and
// removes them on uninstall — DllRegisterServer exists for developer loops
// only and writes the same per-user keys, never HKLM.

#include <mfapi.h>
#include <winrt/base.h>

#include "Activator.h"

namespace {

struct ClassFactory
    : winrt::implements<ClassFactory, IClassFactory> {
    IFACEMETHODIMP CreateInstance(IUnknown* outer, REFIID riid,
                                  void** object) noexcept try {
        if (object == nullptr) {
            return E_POINTER;
        }
        *object = nullptr;
        if (outer != nullptr) {
            return CLASS_E_NOAGGREGATION;
        }
        auto activator = winrt::make_self<custback::vcam::Activator>();
        return activator->QueryInterface(riid, object);
    } catch (...) {
        return winrt::to_hresult();
    }

    IFACEMETHODIMP LockServer(BOOL lock) noexcept {
        if (lock) {
            ++winrt::get_module_lock();
        } else {
            --winrt::get_module_lock();
        }
        return S_OK;
    }
};

std::wstring ModulePath(HINSTANCE module) {
    wchar_t path[MAX_PATH]{};
    ::GetModuleFileNameW(module, path, MAX_PATH);
    return path;
}

HINSTANCE g_module = nullptr;

}  // namespace

BOOL APIENTRY DllMain(HINSTANCE instance, DWORD reason, LPVOID /*reserved*/) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_module = instance;
        ::DisableThreadLibraryCalls(instance);
    }
    return TRUE;
}

_Check_return_ STDAPI DllGetClassObject(_In_ REFCLSID clsid, _In_ REFIID riid,
                                        _Outptr_ void** object) {
    if (object == nullptr) {
        return E_POINTER;
    }
    *object = nullptr;
    if (clsid != custback::vcam::kActivatorClsid) {
        return CLASS_E_CLASSNOTAVAILABLE;
    }
    return winrt::make_self<ClassFactory>()->QueryInterface(riid, object);
}

__control_entrypoint(DllExport) STDAPI DllCanUnloadNow(void) {
    return winrt::get_module_lock() ? S_FALSE : S_OK;
}

// Developer convenience only: per-user registration identical to the keys the
// installer manages.  Production install/uninstall stays with Package.wxs so
// removal is guaranteed by the MSI (WIN-5.7).
STDAPI DllRegisterServer(void) {
    const std::wstring keyPath =
        std::wstring(L"Software\\Classes\\CLSID\\") +
        custback::vcam::kActivatorClsidString + L"\\InProcServer32";
    HKEY key = nullptr;
    LONG status = ::RegCreateKeyExW(HKEY_CURRENT_USER, keyPath.c_str(), 0,
                                    nullptr, 0, KEY_WRITE, nullptr, &key,
                                    nullptr);
    if (status != ERROR_SUCCESS) {
        return HRESULT_FROM_WIN32(status);
    }
    const std::wstring module = ModulePath(g_module);
    status = ::RegSetValueExW(
        key, nullptr, 0, REG_SZ,
        reinterpret_cast<const BYTE*>(module.c_str()),
        static_cast<DWORD>((module.size() + 1) * sizeof(wchar_t)));
    if (status == ERROR_SUCCESS) {
        const wchar_t threading[] = L"Both";
        status = ::RegSetValueExW(key, L"ThreadingModel", 0, REG_SZ,
                                  reinterpret_cast<const BYTE*>(threading),
                                  sizeof(threading));
    }
    ::RegCloseKey(key);
    return status == ERROR_SUCCESS ? S_OK : HRESULT_FROM_WIN32(status);
}

STDAPI DllUnregisterServer(void) {
    const std::wstring keyPath =
        std::wstring(L"Software\\Classes\\CLSID\\") +
        custback::vcam::kActivatorClsidString;
    const LONG status =
        ::RegDeleteTreeW(HKEY_CURRENT_USER, keyPath.c_str());
    return (status == ERROR_SUCCESS || status == ERROR_FILE_NOT_FOUND)
               ? S_OK
               : HRESULT_FROM_WIN32(status);
}
