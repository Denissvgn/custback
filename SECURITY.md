# Security policy

## Supported source

Security fixes are maintained on `main` for the 0.4 release line. Version 0.4.0
is being prepared for distribution; older experimental source snapshots are
not maintained releases. See the README for supported installation paths and
the status of downloadable packages.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/Denissvgn/custback/security/advisories/new).
Do not open a public issue containing exploit details, credentials, private
camera frames, or sensitive deployment information.

Include the affected version or commit, installation method, OS, prerequisites,
expected impact, and a minimal reproduction using synthetic or redacted data.
State whether the issue requires a management token, renderer token, or local
filesystem access. The maintainer will coordinate investigation and disclosure
through the private report; there is no guaranteed response-time commitment.

If a credential was exposed, revoke or rotate it promptly. Removing it from
the latest file does not remove it from Git history or previously shared logs.

## Deployment boundary

The control API binds to numeric loopback by default. Management and renderer
credentials serve different roles. Remote deployments require the documented
authenticated TLS configuration; do not disable the network-boundary checks.

Local diagnostic bundles may contain raw pixels. Store and share them only
through an appropriate private channel. Public bug reports should use synthetic
inputs and redacted diagnostics.
