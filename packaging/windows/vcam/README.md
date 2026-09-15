# Native Windows virtual camera

This C++ component provides an experimental Media Foundation virtual-camera
source for Windows 11. It is not a replacement for the documented OBS source
installation path until the complete application distribution supports it.

The engine produces frames in a private shared-memory ring. The desktop shell
owns the current-user, session-lifetime virtual camera. The media source reads
only that ring; it does not open a physical camera or create a network service.

Build `CustbackVCam.vcxproj` with the Visual Studio C++ toolchain and a compatible
Windows SDK for the target architecture. Pair the resulting DLL with a matching
[shell](../shell/README.md) and [engine](../pyinstaller/README.md) build.

Select `output.backend: native` explicitly in the operator-owned configuration.
The shell creates the camera only after the engine reports that backend active.
A native build alone does not establish physical camera, meeting-application,
installation, signing, or ARM64 compatibility.
