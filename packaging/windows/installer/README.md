# Build the Windows installer

The WiX sources package the desktop shell and frozen engine. Signed installers
are outside the initial public distribution scope. Build and review a complete
payload before distributing an installer.

From the repository root, supply the actual output directories and a signing
certificate available in the Windows certificate store:

```powershell
pwsh packaging/windows/installer/build.ps1 `
    -Version 0.4.0 `
    -EngineDir dist/custback `
    -ShellDir dist/shell `
    -CertThumbprint YOUR_CERTIFICATE_THUMBPRINT
```

`Package.wxs` defines the per-user application install and upgrade behavior.
`Bundle.wxs` handles runtime prerequisites; OBS Virtual Camera is detected and
is not redistributed. Model weights are not installer assets. External
libraries and redistributables retain their upstream licenses.

Ordinary uninstall removes application binaries while retaining user configs,
backgrounds, rigs, tokens, and logs. `REMOVEUSERDATA=1` explicitly opts into
removing managed user data. Review that choice before running an uninstall.

See [engine](../pyinstaller/README.md) and [shell](../shell/README.md) builds.
