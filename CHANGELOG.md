# Changelog

## Unreleased — 0.4.0 preparation

- Document source installation for Windows, Ubuntu/Debian, and macOS.
- Require patched versions of image, multipart, and web dependencies; update
  optional vision and audio dependency profiles consistently.
- Reduce repeated metadata and array-header work in offline quality evaluation
  while retaining per-read file ownership and integrity checks.
- Add repository metadata, contribution guidance, private security reporting,
  and automated repository security checks.
- Keep package publishing separate from ordinary CI and manual qualification.

Compatibility rendering defaults remain in place. Experimental presets require
explicit selection. Registry packages and signed Windows installers have not
been published as part of this preparation work.
