# Changelog

## 0.4.0

- Shorten the quick start and preserve detailed usage in a separate user guide.
- Verify immutable core packages on Ubuntu, Windows, and macOS through a manual
  candidate workflow; retain the extended qualification suite separately.
- Document source installation for Windows, Ubuntu/Debian, and macOS.
- Require patched versions of image, multipart, and web dependencies; update
  optional vision and audio dependency profiles consistently.
- Reduce repeated metadata and array-header work in offline quality evaluation
  while retaining per-read file ownership and integrity checks.
- Add repository metadata, contribution guidance, private security reporting,
  and automated repository security checks.
- Keep package publishing separate from ordinary CI and manual qualification.
- Use fast required CI for the first source publication; retain the full matrix,
  performance checks, CodeQL, and Windows frozen builds as manual workflows.

Compatibility rendering defaults remain in place. Experimental presets require
explicit selection. Signed Windows installers are outside this release scope. See
[GitHub Releases](https://github.com/Denissvgn/custback/releases) for publication
status and available downloads.
