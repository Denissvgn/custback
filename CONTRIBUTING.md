# Contributing to Custback

Custback targets Windows, Ubuntu/Debian, and macOS. Contributions should keep
camera output, authenticated control, private storage, and recovery behavior
consistent across the supported installation paths.

## Development setup

Use Python 3.12 for a development environment with optional vision backends.
Core Python compatibility is declared in `pyproject.toml`.

```bash
git clone https://github.com/Denissvgn/custback.git
cd custback
git switch -c feature/your-change
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use
`.venv\Scripts\python.exe` for the Python commands. Install `.[dev,windows]`.
Activation is optional. npm launcher tests require a supported Node version
and run on Linux/macOS; use Node 22 for local development.

Optional backends belong in separate environments. Choose `rvm` for CPU or
`gpu` for CUDA; the Windows `directml` profile also provides the `onnxruntime`
module and must not be co-installed with either alternative. Supported avatar
installers keep the `mediapipe` and `audio2face` driver profiles separate.

## Verify changes

The initial publication uses a fast required CI set: Ruff, focused core/security
regressions, Node tests, dependency/secret scanning, and source startup checks
on Ubuntu, Windows, and macOS. The exact focused command is in
`.github/workflows/ci.yml`.

The full compatibility matrix, Pyright, optional backends, stress, performance,
and frozen Windows builds remain in the manually triggered **Full validation**
workflow. **CodeQL** is also manually triggered. These checks are deferred, not
deleted or represented as passing. Run them before expanding support claims or
qualifying distribution artifacts.

For complete local verification:

```bash
python -m pip check
python -m ruff check src tests examples scripts/release
python -m ruff format --check src tests examples scripts/release
python -m pyright
python -m pytest -q --durations=25
npm test
npm run release:check -- --quick
```

Run heavy checks one at a time on a development machine. Capture the actual
failure and runtime if a check exceeds its budget; do not remove coverage or
relax correctness thresholds to obtain a passing result. Camera, GPU, and
virtual-camera claims need evidence from the relevant devices and applications.
Synthetic tests cannot establish physical-device compatibility or visual quality.
The large cross-device qualification and two 720p wall-clock sanity checks have
unresolved timing failures on the initial development host; keep their thresholds
and resolve them as part of extended qualification.

The quick release check verifies metadata and package contents. Full release
qualification additionally verifies the exact built artifacts and authenticated
workflow evidence. It is not an ordinary local build command.

## Pull requests

- Use a branch and open a pull request against `main`.
- Explain the user-visible problem, the resulting behavior, and the checks run.
- Add a focused regression for behavior or security fixes. Preserve the API's
  authentication, privacy fallback, upload limits, and transactional recovery.
- Keep optional dependencies compatible with their separately supported profiles.
- Keep dependency bounds, workflow installs, and the reviewed package contracts
  in `scripts/release/verify-release.js` consistent.
- Package contents are explicitly checked. Update `MANIFEST.in`, `package.json`,
  and the relevant inventory checks when adding files that must ship.
- Keep internal plans, local wiki state, raw footage, credentials, and private
  diagnostic bundles out of commits and public CI artifacts.

## Bug reports and community expectations

Use the issue templates for reproducible bugs and feature requests. Include the
OS, Python version, installation method, relevant configuration, and redacted
diagnostics. Share only media you have permission to publish.

Be respectful, focus criticism on the work, and avoid harassment or personal
information. Read the [community conduct policy](https://github.com/Denissvgn/custback/blob/main/CODE_OF_CONDUCT.md).
The maintainer is [Denissvgn](https://github.com/Denissvgn). Public
issues are suitable for ordinary support and moderation requests that contain
no confidential details. Security vulnerabilities follow [SECURITY.md](SECURITY.md).

Registry uploads and release tags are maintainer operations. Passing a PR does
not automatically publish a package.
