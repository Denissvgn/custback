# Custback remediation plan

Status: **ELIGIBLE FOR QUALIFICATION — no open remediation blockers**

This plan converts the findings from the July 2026 repository review into an
implementation sequence. A completeness re-audit on 2026-07-16 found that six
previously resolved blockers do not yet satisfy their full acceptance contracts
and identified two additional stop-ship findings. Phase 1–4 completion records
below are retained as historical implementation records, but they no longer
authorize a release. Phase 5 closes the corrective backlog; Phase 6 performs
migration, artifact, stress, and two-host TLS validation.

The machine-readable registry now reports phase 6 with
`release_blocked=false`. All 28 registered blockers are resolved. Windows
production evidence is deferred and temporarily excluded from the blocker
registry and required-gate manifest.

## Target invariants

The remediated system must guarantee that:

- API users cannot select arbitrary outbound destinations or credential files.
- Plaintext transport is accepted only over numeric loopback.
- Remote mode never emits an unprotected camera frame; failure output is an
  input-independent opaque slate.
- A published configuration version always describes the resources actually
  rendering that version.
- Cancelled or stopped workers cannot continue using resources that have been
  replaced or closed.
- Staging and committed data share private permissions and aggregate quotas.
- Every advertised optional feature installs and runs from built artifacts.
- Wheel, sdist, and npm installs expose the same canonical commands and bundled
  configuration without access to the source checkout.
- npm reinstall and upgrade preserve the managed environment, rollback state,
  and explicitly selected extras.
- Release authorization is bound to one clean commit and the exact artifact
  digests that passed every required dynamic gate.

## Delivery roadmap

| Phase | Scope | Exit gate |
| --- | --- | --- |
| 0 | Freeze release and add deterministic regressions | Every known defect has a blocker ID and executable regression |
| 1 | Proxy, token, transport, and privacy containment | SSRF performs no I/O; remote plaintext fails; no raw startup output |
| 2 | Transactional avatar engine and worker lifecycle | Config/resources activate atomically; no worker survives teardown |
| 3 | Storage, merge semantics, segmentation, rendering, streaming | Concurrency, quota, media, and correctness gates pass |
| 4 | Audio2Face, npm, release, docs, and compliance | All extras install from artifacts; npm upgrades preserve state |
| 5 | Corrective security, lifecycle, storage, and packaging | Seven Phase 5 blockers pass |
| 6 | Migration and end-to-end validation | Clean artifacts, stress, upgrade, and two-host TLS gates pass |

The original Phase 0–4 sequence is retained below for traceability. Current
implementation order and parallelization rules are defined by the corrective PR
sequence after Phase 6. Phase 6 implementation is present in this revision, but
its dynamic release-candidate gates cannot authorize this dirty parent revision.

### Historical Phase 0 completion record

- At the Phase 0 freeze, the machine-readable registry contained 26 open
  blocker IDs.
- Every blocker points to an executable regression file that contains its ID;
  release verification checks those links before evaluating blocker status.
- Python remediation coverage initially recorded one passing loopback baseline
  and 30 strict expected failures. An unexpected pass remains a hard failure
  until the owning remediation removes the marker.
- npm remediation coverage records 11 executable TODO contracts.
- `release:check` and `prepack` fail closed while any blocker remains open, or
  if the registry is missing, malformed, inconsistent, or loses regression
  coverage.

## Blocker registry

The machine-readable source of truth is
`scripts/release/remediation-blockers.json`. `release:check` and `prepack`
must fail while any entry remains open. The registry is aligned with the
corrective implementation: all 28 registered entries are resolved.
A blocker is closed only in the same change that removes its strict
expected-failure marker and makes its acceptance test pass. Merely checking that
a regression file contains the blocker ID is not evidence that the acceptance
scenario executed.

| ID | Finding | Historical target phase |
| --- | --- | --- |
| SEC-01 | Avatar proxy authenticated SSRF and credential-file exfiltration | 1 |
| TOKEN-01 | Proxy client token auto-creation and upstream-auth misclassification | 1 |
| TRANS-01 | Non-loopback WS/HTTP/gRPC plaintext transports | 1 |
| PRIV-01 | Raw/near-raw frames can pass remote startup or fallback | 1 |
| A2F-01 | Audio2Face extra and protobuf contract are unusable | 4 |
| CFG-01 | Avatar config is published before component activation | 2 |
| CFG-02 | Nested merge-patch resets sibling values | 3 |
| LIFE-01 | Cancelled render work can outlive sessions and resource ownership | 2 |
| LIFE-02 | Blocked Audio2Face RPC/audio source survives close | 2 |
| STOR-01 | Avatar assets are created with non-private modes | 3 |
| STOR-02 | Staging, rig aggregate, and decoded-image limits are incomplete | 3 |
| SEG-01 | Model acquisition blocks the frame worker after API timeout | 3 |
| SEG-02 | Auto backend ignores a supplied `.tflite` model | 3 |
| SEG-03 | Non-finite masks can reach startup compositing/output | 3 |
| RENDER-01 | Straight alpha is multiplied twice | 3 |
| RENDER-02 | `follow_pose` is a nonfunctional API/UI setting | 3 |
| API-01 | Long-lived streams can exhaust the shared asyncio executor | 3 |
| NPM-01 | npm upgrades discard the managed venv, generations, and extras | 4 |
| PKG-01 | npm avatar binary and checkout-relative documentation disagree | 4 |
| PKG-02 | Prepack verification/message and tarball recipe are inconsistent | 4 |
| DEPLOY-01 | Remote control-plane token/TLS deployment is incomplete | 4 |
| MISC-01 | Failed camera-backdrop construction leaks its capture | 3 |
| MISC-02 | YAML, URL-port, and combined CLI validation are inconsistent | 3 |
| LICENSE-01 | Declared MIT artifacts contain no license text | 4 |
| PLATFORM-01 | Automatic Linux setup is Ubuntu-specific but scope is broader | 4 |
| HYGIENE-01 | Generated ONNX Runtime profiles are not rejected as stale | 4 |

### Corrective registry delta (2026-07-16)

| ID | Registry action | Finding | Target phase |
| --- | --- | --- | --- |
| SEC-02 | Add open | Hot camera-backdrop configuration permits authenticated SSRF | 5 |
| PRIV-01 | Reopen | Raw replay is accepted after fingerprint history expires | 5 |
| LIFE-01 | Reopen | A cancelled avatar session can publish a queued render | 5 |
| LIFE-02 | Reopen | Audio2Face loses RPC/channel ownership while its worker survives | 5 |
| STOR-02 | Reopen | Post-rename cleanup failure leaves an unowned inode/reservation gap | 5 |
| PKG-01 | Reopen | Canonical avatar/config-export commands are npm-wrapper-only | 5 |
| PKG-02 | Reopen | `npm pack --silent` emits non-filename text on stdout | 5 |
| REL-01 | Add open | Release automation can pass without Phase 6, Ruff, or stress evidence | 6 |

The corrective freeze therefore begins with eight open entries: six reopened
and two new. Existing passing behavior for the other 20 original blockers must
remain permanent regression coverage.

### Re-audit authority

The Phase 1–4 completion records describe what the earlier suites proved at the
time. Where a completion record conflicts with the corrective registry delta or
the Phase 5 acceptance tests below, the corrective requirement wins. Do not edit
the historical test counts to make them appear to cover scenarios that were not
executed.

## Phase 1 — security and privacy

### Immutable proxy targets and least-privilege credentials

- Make avatar URL, credential references, CA paths, and TLS settings
  startup-only. A PATCH must return `409 restart_required` without changing the
  version.
- Replace free-form browser destination editing with an optional hot
  `active_target` ID selected from operator-defined startup targets.
- Resolve one immutable `AvatarProxyTarget` at startup. Request handlers must
  not read token files or consult environment variables.
- Split server token provisioning (may create) from client token loading (must
  already exist). Bind each client credential to one normalized origin.
- Disable redirects, use `trust_env=False`, and allow-list exact proxied
  method/route pairs.
- Expose redacted public configuration models without token paths, private-key
  paths, or secret material.
- Translate upstream avatar `401/403` into `502 avatar_auth_failed` rather than
  expiring the core browser session.

Acceptance: the original attacker-URL plus core-token-path reproduction returns
409, performs zero file reads and zero network requests, and a missing proxy
token creates no file.

### Secure transports

- Share one outbound URL/TLS validator across source WS, avatar control HTTP,
  and Audio2Face gRPC.
- Permit `ws://`, `http://`, and insecure gRPC only for numeric loopback.
- Require verified `wss://`, `https://`, or `grpcs://` remotely; validate CA and
  client-certificate pairs at startup and never downgrade after TLS failure.
- Eagerly validate ports, userinfo, paths, queries, fragments, and unsafe
  address classes.
- Add a renderer-scoped credential accepted only on `/ws/frames`; do not give a
  remote renderer the core management token.

Acceptance: certificate failure sends no bearer token or PCM, all remote paths
use verified TLS, and renderer credentials cannot access REST/session routes.

### Central remote-output privacy firewall

- Remove the preflight privacy bypass. Test the output backend with a synthetic
  slate rather than a real capture.
- Route startup, preview, virtual-camera, repeated, remote, and fallback frames
  through one mandatory gate immediately before publication.
- Use a fixed opaque, input-independent slate for missing, stale, malformed,
  failed, or implausibly raw output. Do not call segmented background
  replacement privacy-safe because it preserves foreground pixels.
- Validate masks centrally and compare remote output against current/recent
  downsampled raw fingerprints to reject exact, JPEG-altered, and delayed raw
  echoes.

Acceptance: a recording sink observes every emitted frame; raw, near-raw,
all-foreground, invalid-mask, and delayed-echo cases all receive the same slate
for different camera inputs.

### Historical Phase 1 completion record (2026-07-16)

- `SEC-01`, `TOKEN-01`, `TRANS-01`, and `PRIV-01` are resolved and their
  strict expected-failure markers are now permanent passing regressions.
- The central mask/privacy work also resolved `SEG-03` ahead of Phase 3. The
  registry therefore has 21 open blockers remaining.
- The avatar proxy now holds one immutable startup target containing its
  normalized URL, existing client token, and verified CA/mTLS context. Exact
  method/path pairs are allow-listed; redirects and environment proxies are
  disabled; browser edits of outbound destinations were removed. No hot
  `active_target` selector is exposed while only one operator target exists.
- Core management, renderer-frame, and avatar-control credentials are
  separate. The renderer token is accepted only by the raw-frame WebSocket;
  public config responses and OpenAPI schemas omit token/trust/private-key
  paths.
- HTTP, WebSocket, and gRPC clients share one endpoint policy: plaintext is
  numeric-loopback-only, remote connections require verified TLS, and
  optional private CA/mTLS pairs are validated before use.
- Every remote-mode sink and preview publication crosses the same final gate.
  Startup probes, renderer outages, malformed output, invalid/all-foreground
  masks, and current/recent raw echoes produce a fixed input-independent slate.
- Final verification passed: 543 Python tests, with 17 strict expected
  failures retained for unresolved blockers, plus all four npm test files.
  The release verifier correctly remains fail-closed on the 21 open entries.

## Phase 2 — transactional avatar runtime and lifecycle

### Atomic activation

- Route avatar PATCH through a compare-and-swap activation coordinator rather
  than committing directly to `AvatarRuntime`.
- Introduce one render/activation lane that owns complete `_Components`
  generations. Embed the effective config and version in each generation.
- Pre-acquire assets away from the render lane, construct changed resources
  under an `ExitStack`, trial them on a copied/synthetic frame, then swap one
  complete generation at a frame boundary.
- Publish config/version only after the swap succeeds. Close replaced resources
  after acknowledgement and remove per-frame reconfiguration retries.
- Serialize deletion of active rigs/backgrounds with activation.

Acceptance: a failed driver/rig/background candidate leaves config, version, and
all live identities unchanged; concurrent patches linearize or conflict; no
status reports version N while version N-1 resources render.

### Owned render and Audio2Face lifecycles

- Replace shared `asyncio.to_thread()` rendering with a dedicated single-worker
  executor. Cancellation discards stale results but waits for terminal resource
  ownership before replacement/close.
- Track the active Audio2Face source, call, and channel under a lifecycle lock.
  Close by setting stop, interrupting the source, cancelling the call, closing
  the channel, and joining the worker. Make close idempotent and pacing waits
  interruptible.

Acceptance: blocked fake render/gRPC calls cannot overlap reconnect, observe
closed resources, or survive the shutdown deadline.

### Historical Phase 2 completion record (2026-07-16)

- `CFG-01`, `LIFE-01`, and `LIFE-02` are resolved and their strict
  expected-failure markers are now permanent passing regressions. The registry
  has 18 open blockers remaining.
- Avatar PATCH now pre-acquires driver assets away from a dedicated
  single-worker render/activation lane. That lane constructs changed resources
  under rollback ownership, trials a complete config-bearing generation, swaps
  it at a frame boundary, and publishes the matching version through one CAS.
- Candidate failure or conflict preserves the published config, version, and
  every live resource identity. Replaced resources close only after the
  successful acknowledgement; active rig/background deletion is serialized
  with activation and cannot leave a committed path missing.
- Session cancellation shields and drains the authoritative render future
  before reconnect or teardown, so no abandoned shared-executor render can use
  a replaced or closed generation.
- Audio2Face now owns its source, RPC, channel, and worker under one lifecycle
  lock. Close is ordered and idempotent (stop, source interrupt, RPC cancel,
  channel close, worker join), and reconnect/pacing waits are interruptible.

## Phase 3 — storage and runtime correctness

### Private, quota-owned storage

- Create managed directories as `0700` and files through exclusive no-follow
  opens as `0600`, followed by explicit `fchmod` after atomic rename.
- Replace bare staging paths with reservation objects that account for each
  chunk. Include committed data, compressed/extracted staging, and all in-flight
  bytes in quotas.
- Add aggregate rig count/bytes plus per-layer and total decoded-pixel limits.
- Validate media with Pillow before OpenCV; reject decompression warnings/errors,
  non-PNG rig layers, inconsistent dimensions, oversized manifests, and
  non-finite rig geometry.
- Provide a doctor/fix path for existing user-owned permissions.

Acceptance: permissive umasks still yield `0700/0600`; concurrent uploads never
exceed quota; failures leave no reservation/staging leak; bombs are rejected
before `cv2.imread`.

### Merge, segmentation, rendering, and streaming

- Add one RFC 7396 helper: recursively merge objects, replace arrays/scalars,
  define `null` reset semantics, deep-copy inputs, and retain unknown keys for
  Pydantic errors.
- Make custom-model suffix authoritative (`.onnx` -> RVM, `.tflite` ->
  MediaPipe). Acquire/build candidates on a dedicated executor; the frame
  thread performs only trial/CAS/swap.
- Validate mask shape, dtype, finiteness, and range at startup, trial, and
  runtime before compositing.
- Keep a documented straight-alpha representation, premultiply only around
  OpenCV transforms, and apply alpha once at final composition.
- Wire `follow_pose` through both rig implementations while retaining
  expressions.
- Replace one blocking worker per stream with latest-only async subscriptions,
  encode each sequence once, and enforce authenticated connection limits.
- Catch YAML errors, validate CLI overrides as one candidate, validate ports
  eagerly, and release failed camera-backdrop captures.

Acceptance includes analytical alpha cases, pose behavior, NaN/Inf masks,
custom-model routing, non-blocking model acquisition, executor saturation, and
concise CLI errors.

### Historical Phase 3 completion record (2026-07-16)

- `CFG-02`, `STOR-01`, `STOR-02`, `SEG-01`, `SEG-02`, `RENDER-01`,
  `RENDER-02`, `API-01`, `MISC-01`, and `MISC-02` are resolved and their
  strict expected-failure markers are permanent passing regressions. Together
  with the earlier `SEG-03` resolution, the registry now has 8 open Phase 4
  blockers.
- Core and avatar storage now use exact private modes, no-follow inode checks,
  chunk-owned reservations, concurrent aggregate quotas, decoded-pixel and
  manifest limits, complete Pillow decoding before OpenCV, and retryable
  cleanup ownership. Avatar storage also provides permission audit/repair CLI
  paths for existing user-owned assets.
- Core and avatar configuration share recursive merge-patch semantics and
  validate combined CLI overrides once. YAML/port/startup errors are concise;
  custom model suffixes select their authoritative backend; candidate
  acquisition and abandoned cleanup remain on bounded, deadline-owned lanes;
  and masks are validated before every compositing boundary.
- Rendering consistently uses straight alpha with premultiplied transforms,
  preserves transparent-color isolation, and honors `follow_pose` in both rig
  implementations. MJPEG/WebSocket delivery is event-loop-native, shares one
  bounded latest-only JPEG scheduler, enforces authenticated connection caps,
  and releases leases/upstream responses across the full ASGI lifecycle.
- Final verification passed 689 Python tests and all four npm test files. The
  release verifier remains fail-closed on the 8 unresolved Phase 4 entries.

## Phase 4 — Audio2Face, npm, release, and compliance

### Audio2Face contract

- Until repaired, hide/remove the advertised UI mode and availability claim.
- Replace the impossible dependency with the official
  `nvidia-audio2face-3d` package and a verified compatible
  `nvidia-ace`/`grpcio` intersection.
- Rewrite protocol imports and message construction against the published
  modules; probe the complete protocol, not merely the parent package.
- Separate protocol availability from microphone availability so WAV mode does
  not require PortAudio.
- Add the extra to npm capabilities if npm users are expected to use it.
- CI must install the extra from source and built wheel, run `pip check`,
  serialize real messages, parse a representative response, and exercise an
  in-process generated gRPC service.

Acceptance: the extra resolves in a fresh environment and no installed-binding
test skips the actual protocol contract.

### Durable npm runtime and intent

- Move the default venv and generations to a deterministic npm-prefix-scoped
  path outside the replaceable package tree. Keep `CUSTBACK_VENV` as the
  highest-priority override.
- Persist requested extras beside the stable target. Absent env preserves
  intent; explicitly empty env clears it; update intent only after promotion.
- Add `custback extras` and `custback rebuild --extras`.
- Rebuild rather than relocate legacy venvs because shebangs are absolute.
- Hash all shipped runtime/config inputs, including `config/avatar.yaml`.

Acceptance: simulated package-directory replacement preserves the venv, extras,
rollback generations, and failed-rebuild recovery.

### Packaging, documentation, compliance, and platform scope

- Make `custback avatar` canonical and add a `custback-avatar` npm alias for
  compatibility. Add a command to export the bundled avatar config.
- Make prepack shell-safe and truthful; verify the packlist without recursion
  and reserve built-artifact installation for `release:check`.
- Document distinct WSS renderer and HTTPS avatar-control trust planes, tokens,
  CA/certificate handling, firewall direction, rotation, and outage behavior.
- Confirm holder/year, add root MIT `LICENSE`, and assert identical inclusion in
  npm, wheel, and sdist.
- Refuse unsupported automatic Linux setup before mutation; advertise the
  tested Ubuntu/Debian scope until additional distro installers exist.
- Delete/ignore ONNX Runtime profile debris and make release checks reject it.
- Pin CI actions to commit SHAs and use `npm ci`.

### Historical Phase 4 completion record (2026-07-16)

- `A2F-01`, `NPM-01`, `PKG-01`, `PKG-02`, `DEPLOY-01`, `LICENSE-01`,
  `PLATFORM-01`, and `HYGIENE-01` are resolved. All 26 registry entries are now
  permanent passing regressions, with no open blocker and
  `release_blocked=false`.
- Audio2Face uses the verified official `nvidia-audio2face-3d==1.3.0`,
  `nvidia-ace==1.0.0`, `grpcio>=1.67,<1.67.2`, and compatible protobuf stack.
  Source and built-wheel installs serialize real messages, parse responses,
  exercise an in-process generated gRPC service, and keep WAV availability
  independent of microphone/PortAudio availability.
- npm environments and rollback generations now live under a stable
  prefix-scoped root. Extra intent survives package replacement, can be
  inspected or rebuilt explicitly, updates only after promotion, and covers
  the shipped avatar configuration in its source digest. `custback avatar` is
  canonical and the compatibility launcher/config export are packaged.
- The root MIT license is asserted byte-for-byte in npm, wheel, and sdist;
  deployment documentation separates the renderer and avatar-control trust
  planes; automatic Linux setup rejects unsupported distributions before
  mutation; CI uses reviewed action SHA pins and `npm ci`; generated ONNX
  Runtime profiles are ignored and rejected as stale release input.
- Full artifact verification is memory-bounded: it rejects Linux tmpfs/ramfs
  scratch by default, relocates all subprocess scratch to a configured
  disk-backed root, limits native builds to two jobs by default, and removes
  each optional-profile environment immediately on success or failure. This
  prevents the release matrix from accumulating multiple large venvs in the
  editor's cgroup.
- Final local verification passed 690 Python tests with one expected core-env
  skip for the optional Audio2Face bindings, all four npm test files, the quick
  metadata/packlist gate, and the full source/wheel/npm artifact gate. The full
  Python 3.14 artifact run completed in an isolated service in 17m40s with a
  3 GB peak and no swap; its scratch tree was removed on success.

## Phase 5 — corrective remediation

### Entry gate and regression freeze

- Change the registry to phase 5, set `release_blocked=true`, reopen the six
  existing IDs, and add `SEC-02` plus `REL-01`. The registry must contain 28
  unique entries, with eight open at the corrective freeze.
- Add one deterministic regression for each reproduced failure before changing
  production code. Use strict expected failures/TODOs only while the owning
  blocker is open; an unexpected pass is a failure until the marker is removed
  in the implementation change. Node TODO coverage must add an explicit
  unexpected-pass failure because the default Node TODO result is not strict.
- Record exact pytest node IDs or Node test names in the registry instead of
  treating a source-file substring match as executable coverage. Release CI must
  prove that every recorded test was collected and produced the registry-required
  open/resolved outcome; every resolved blocker must pass normally.
- Preserve all passing Phase 1–4 tests. The corrective work is not permission to
  weaken TLS policy, privacy fallbacks, quota accounting, or lifecycle deadlines.

Acceptance: the registry reports 20 resolved and eight open entries,
`release_blocked=true`, and both `prepack` and every release-check mode fail
before artifact publication. All eight new regressions execute and fail for the
documented reason on the pre-fix tree.

### Immutable live-backdrop authority (`SEC-02`)

- Remove free-form `background.camera_device` network destinations from the hot
  API contract. Numeric local camera indices may remain selectable; strings that
  can cause filesystem or network access are startup authority.
- Prefer operator-defined immutable backdrop targets with public, non-secret IDs.
  A hot request may select an ID, but it cannot supply or mutate a path, URL,
  credential, TLS setting, or OpenCV backend option.
- Normalize and validate a startup target before constructing any capture. Apply
  explicit scheme, port, address-class, DNS, redirect, proxy, and TLS policy. If
  OpenCV cannot guarantee verified remote transport, reject that remote scheme
  rather than silently delegating security to `cv2.VideoCapture`.
- Snapshot the approved target in the candidate resource. Unrelated PATCHes must
  not reread startup-only files or environment state.
- Keep redacted public configuration limited to target IDs and safe status.

Acceptance: PATCHes containing HTTP, RTSP, link-local metadata, loopback admin,
device-path, or credential-path strings return `409 restart_required` or a
validation error without changing the version and with zero calls to
`VideoCapture`, DNS, file open, or network connect. A configured target ID still
activates transactionally, and a failed target leaves the old backdrop identity
and version unchanged.

### Session-wide privacy firewall (`PRIV-01`)

- Replace the three-second permissive history eviction with session-wide replay
  protection. No expired or capacity-evicted raw fingerprint may turn into an
  allow decision.
- Use a bounded design that fails closed: for example, a session-keyed compact
  replay structure or robust per-frame tag. If its safe capacity is exhausted,
  emit the privacy slate and reset/re-authenticate the renderer session rather
  than evicting evidence into a permissive state.
- Detect exact, near-raw, JPEG-altered, delayed, capacity-pressure, and
  prior-session echoes. Clear/rekey evidence only at a linearized authenticated
  session boundary, after stale remote output has been invalidated.
- Continue sending every startup, repeated, preview, virtual-camera, fallback,
  and remote frame through the same last-mile gate.

Acceptance: with fake monotonic time advanced beyond the old window and with
history driven past its configured capacity, replaying any earlier exact or JPEG
raw frame produces the same input-independent slate for different camera inputs.
A recording sink proves that no publication path observes the replay.

### Linearizable render-session cancellation (`LIFE-01`)

- Give each renderer connection a session epoch/lease and include it in deferred
  render publications.
- Linearize cancellation by invalidating the lease under the same ownership
  mechanism used to authorize preview publication and render statistics. A
  generation-version check alone is insufficient.
- Make send completion, local publication, and accounting order explicit. Once
  cancellation wins a side effect's linearization point, later callbacks must be
  discard-only. A send that committed before cancellation may be counted exactly
  once, but it cannot authorize a later preview publication or `frames_rendered`
  update.
- Drain the authoritative lane future before reconnect, resource replacement, or
  executor teardown without converting a cancelled result back into a publishable
  one.

Acceptance: barriers at decode, render completion, WebSocket send, queued
publication, and publication callback make each side effect's winner
deterministic. In the reproduced ordering, WebSocket send returns, lane
publication blocks, stop invalidates the lease, and the callback is released;
the preview and `frames_rendered` remain unchanged. Completed-send accounting is
defined separately and occurs at most once. The stale result cannot affect a new
session even when the component version is unchanged.

### Terminal Audio2Face ownership (`LIFE-02`)

- Retain the source, RPC, channel, interruption operations, and worker as one
  owned session until the worker is terminal. A successful return from
  `cancel()` or `close()` is only an interruption attempt, not proof of terminal
  ownership.
- If the worker remains alive, preserve every handle for ordered retry and
  prohibit reconnect, component replacement, or a successful close result.
- Ensure every source read, pacing wait, RPC iteration, and reconnect wait has a
  bounded interruption path. If a native gRPC thread cannot be made killable
  within the shutdown deadline, isolate the streaming session in a subprocess
  that can be terminated and reaped rather than weakening the deadline.
- Keep close idempotent across partial interruption, retry, normal completion,
  and concurrent start/close races.

Acceptance: a fake RPC whose `cancel()` returns success without waking its
iterator, combined with a channel whose `close()` also returns, causes the first
close either to terminate/reap an isolated worker or to time out while retaining
the complete source/call/channel/worker generation and refusing restart. After
the iterator is released, repeated close joins and clears the same generation.
Repeated close is safe, and no Audio2Face worker remains after a successful
teardown. If subprocess isolation is used, forced termination and reaping have a
separate deterministic regression.

### Rename-aware storage cleanup (`STOR-02`)

- Represent upload/install ownership with a transaction that tracks the
  authoritative inode path through temporary, staged, and final names.
- Transfer cleanup ownership before each rename. A failed post-rename chmod,
  inode check, rollback rename, unlink, or recursive removal must retain the
  actual remaining path and its byte/file reservation in a retry queue.
- Remove `suppress`/`ignore_errors` cleanup branches that can discard ownership.
  Surface the primary error while keeping retryable cleanup metadata.
- Count committed, active, and cleanup-pending data exactly once. Add bounded
  startup recovery for crash-left owned staging records without deleting
  unmarked user data.
- Apply the same state machine to core uploads, avatar media, and rig directories.

Acceptance: fault injection at every rename/hardening/rollback/removal boundary,
including two simultaneous failures, leaves either no inode or a fully charged,
discoverable cleanup record. A later retry or restart removes it and releases the
reservation; no failed publication is visible or non-private.

### Installed avatar command surface (`PKG-01`)

- Make `custback avatar ...` dispatch in the Python entry point as well as the
  npm wrapper. Keep `custback-avatar` as a compatibility alias on both surfaces.
- Package the annotated avatar template as Python package data and export it via
  `importlib.resources` (or an equivalent installed-resource API), not a checkout
  or npm-tree-relative path.
- Support `custback avatar config export [PATH]` from wheel, sdist, editable, and
  npm installs with exclusive, mode-`0600` destination creation.
- Update the shipped template, examples, and documentation to call the canonical
  command while describing the alias only as compatibility behavior.

Acceptance: isolated installs of wheel, sdist, and npm tarball successfully run
`custback avatar --help`, start a hardware-free avatar smoke, export
byte-identical config, refuse overwrite, and run the alias. Tests must execute
installed launchers, not only call JavaScript helper functions.

### Shell-safe npm pack contract (`PKG-02`)

- Keep prepack diagnostics off stdout when npm is expected to print the tarball
  name. Send human diagnostics to stderr or suppress the success line in
  lifecycle mode.
- Make the documented command substitution capture exactly one non-empty
  filename with no whitespace or extra lines.
- Retain non-recursive `npm pack --dry-run --json --ignore-scripts` verification
  internally and full artifact installation only in the explicit release path.
- Replace the string-matching regression with a real subprocess test in an
  isolated pack destination, followed by installation of the captured path.

Acceptance: `TARBALL=$(npm pack --silent)` yields exactly
`custback-<version>.tgz`; the file exists, installs successfully, and prepack
emits no other stdout. Failure diagnostics remain actionable on stderr.

### Phase 5 exit gate

Phase 5 is complete only when `SEC-02`, `PRIV-01`, `LIFE-01`, `LIFE-02`,
`STOR-02`, `PKG-01`, and `PKG-02` are resolved by their executable acceptance
tests. Then advance the registry's top-level phase to 6; `REL-01` remains the
only open entry, retains blocker phase 6, and keeps `release_blocked=true` until
release enforcement is installed. Passing the ordinary unit suites is necessary
but not sufficient.

### Phase 5 completion record (2026-07-16)

- The reviewed registry contract contains 28 entries: 27 resolved and only
  phase-6 `REL-01` open, with `release_blocked=true`.
- `SEC-02` now limits hot backdrop selection to immutable, operator-owned local
  targets; unsafe source syntax is rejected before file, capture, DNS, or
  network I/O, and source paths are absent from public configuration.
- `PRIV-01` now maintains bounded session-wide replay evidence and fails closed
  on capacity exhaustion without resetting across reconnects or mode changes.
- `LIFE-01` linearizes renderer-session publication with stop/replace leases,
  while `LIFE-02` retains the complete Audio2Face generation until terminal
  worker ownership is proven.
- `STOR-02` uses durable, inode-bound rename transactions and quota ownership
  for core uploads, avatar media, and rigs, including restart-safe cleanup and
  no-follow recovery.
- `PKG-01` ships the canonical `custback avatar` surface and byte-identical
  packaged config export through wheel, sdist, alias, and npm installs.
- `PKG-02` keeps npm pack stdout shell-safe and proves the captured tarball is
  installable. Package smoke is a separate non-authorizing diagnostic; normal
  prepack and release checks still fail on `REL-01`.
- Final local verification: 727 Python tests passed with one optional-backend
  skip; 80 Node tests passed with the single intentional `REL-01` TODO. Exact
  pytest and Node registry regressions, installed wheel/sdist/npm smoke, and
  release fail-closed checks passed.

## Phase 6 — migration and end-to-end release validation

### Upgrade and migration matrix

- Build fixtures from the last released artifacts and every supported legacy
  on-disk format. Test clean install, in-place upgrade, interrupted upgrade,
  rollback, reinstall, and uninstall/reinstall without relying on the source
  checkout.
- Verify npm package-directory replacement preserves the prefix-scoped venv,
  explicit extras intent, active and rollback generations, and recovery journal.
- Exercise config migration for the new immutable backdrop-target model. Unsafe
  free-form remote targets must require explicit operator migration; they must
  never be silently re-enabled as hot API authority.
- Audit and repair existing core/avatar stores without data loss, following
  symlinks, widening permissions, or dropping quota ownership.
- Run migrations from wheel, sdist, and npm artifacts on every advertised OS and
  supported runtime where the artifact is published. The initial reviewed matrix
  is Python 3.10–3.14 on Ubuntu, Python 3.12 on macOS, Node 18/20/22 on Ubuntu,
  and Node 20 on macOS. Keep this finite matrix machine-readable; bound or update
  `engines.node` when changing supported Node lines.
- In that manifest, enumerate artifact and legacy-fixture IDs plus MediaPipe and
  RVM on Python 3.11/3.12, Audio2Face source/wheel contracts on Python 3.12,
  Linux x86-64 GPU dependency resolution, and a separate CUDA-hardware execution
  gate that does not infer execution from provider registration.

Acceptance: golden pre-upgrade fixtures produce the expected post-upgrade config,
assets, permissions, extras, and rollback state. Killing each migration at every
durable boundary converges to the old or new valid state on retry.

### Two-host WSS/HTTPS system test

- Run meeting-host and renderer-host processes in separate containers, network
  namespaces, or equivalent isolated hosts with distinct addresses and no shared
  secret directory.
- Run the isolated-network scenario for every release candidate and repeat it on
  two clean machines or VMs before production publication; a loopback-only
  substitution is not release evidence.
- Generate ephemeral independent CAs, server identities, and renderer/control
  tokens. Exercise the WSS frame plane and HTTPS control plane with real sockets,
  hostname verification, and the documented firewall direction.
- Test nominal rendering, renderer outage, stale output, wrong renderer token,
  wrong control token, untrusted/expired/wrong-host certificates, removed
  firewall paths, reconnect, and rotation. Never downgrade to plaintext.
- Attach a recording virtual-camera/preview sink and assert the fixed privacy
  slate throughout startup and every frame-plane failure. Assert control failures
  map to `avatar_unreachable`/`avatar_auth_failed` without invalidating the core
  browser session.
- Prove failed TLS handshakes transmit no bearer token, PCM, or camera frame to
  the untrusted peer.

Acceptance: the complete two-host matrix passes from packaged artifacts, and the
test tears down every process, socket, certificate, token, and temporary file on
success, assertion failure, timeout, and cancellation.

### Quality, stress, artifact, and clean-tree gates

- Build the wheel, sdist, and npm tarball once from a clean release commit,
  record their SHA-256 digests, and pass those exact files to every downstream
  job. The publish job must upload the qualified files without rebuilding them.
- Add Ruff to the development/release toolchain with a reviewed configuration
  and require both `ruff check src tests examples` and
  `ruff format --check src tests examples`, plus any Python release runners.
- Add deterministic bounded stress jobs for concurrent PATCH/activation,
  session cancellation/reconnect, stream connection caps, upload reservations,
  cleanup retries, and repeated Audio2Face shutdown. Run at least 100 iterations
  per race/fault family with recorded seeds and retain the failing seed.
- Run the full Python and npm suites plus `pip check` from clean source and built
  artifacts. Every advertised optional extra must install and execute its real
  smoke from a built wheel.
- Reject generated profiles, build outputs, caches, untracked release payloads,
  and missing delivery-critical files. `LICENSE`, deployment docs, protocol
  tests, and every reviewed payload must be committed before release.
- Bind every result to the exact commit and artifact digests. Stale evidence from
  another tree, a checked-in result, or a locally edited attestation must fail
  closed.

Acceptance: required CI jobs cover Python, Node, Ruff, stress, migrations,
artifacts, clean-tree checks, and two-host TLS. Re-running the gate on a changed
source or artifact digest invalidates the prior evidence. Each stress group ends
at quiescence with zero failures, timeouts, unexpected skips, surviving workers
or handles, leaked reservations, lost cleanup owners, or privacy violations.

### Release workflow integrity (`REL-01`)

- Make the production publish workflow depend on all required Phase 6 jobs. A
  green metadata/packlist prepack is not a production-readiness attestation.
- Add one aggregate release-gate job that evaluates every required dependency
  even when an earlier job fails and refuses publication unless all conclusions
  are successful for the same workflow run.
- Maintain one versioned machine-readable allow-list of required gate, CI-job,
  matrix, and scenario IDs. The workflow, aggregate job, evidence producer, and
  verifier must match that exact set; they cannot silently agree to omit a gate.
- Keep `prepack` non-recursive and fast, but make it fail whenever the blocker
  registry is open or required release metadata is inconsistent.
- Provide a non-authorizing qualification/test entry point so Phase 6 jobs can
  exercise their runners while the registry is open. Its evidence is diagnostic
  only and cannot satisfy the publish workflow or bypass registry checks. It may
  build an npm candidate with lifecycle scripts disabled, but final qualification
  must use the normal prepack path after every blocker closes.
- Make full `release:check` run the locally executable gates and verify
  commit-bound CI evidence for any true two-host/platform matrix that cannot run
  locally.
- Close `REL-01` only after the enforcement machinery and its fail-closed
  regressions are mandatory. Keep `release_blocked=true` while any registered
  entry remains open. Any later source, dependency, workflow, or artifact
  change requires a fresh qualification run and new artifacts.

`release_blocked=false` means that a revision is eligible to attempt final
qualification; it is not itself evidence that any artifact passed. Dynamic
evidence must be produced by the release workflow and must name the exact commit,
workflow run, platform matrix, scenario IDs, and publishable artifact digests.
Publication evidence must be authenticated by the CI provider or a reviewed
signed-attestation mechanism and tied to artifacts from that same workflow run;
a locally fabricated JSON file is invalid. Local `release:check` is diagnostic
unless it can verify that trusted provenance.

Acceptance: deleting or failing any required job/evidence, reopening a blocker,
changing a reviewed file, or substituting an artifact makes the publish job and
full release check fail. Only the exact fully validated commit can publish.

### Phase 6 implementation record (2026-07-16)

- Added a no-follow, journaled Python config/store migrator with byte-exact
  backups, explicit refusal of unsafe legacy remote targets, ownership-ledger
  preservation, idempotent retry, and failpoint coverage at every durable
  boundary.
- Added the pre-upgrade npm bridge required before npm replaces a 0.3 package
  directory. It relocates both direct and generational managed environments,
  preserves extras plus active/rollback generations, rewrites owned metadata,
  and converges after every durable boundary.
- Added one strict manifest for publishable artifacts, legacy fixtures,
  Python/Node/optional/runtime matrices, six 100-iteration stress families, 21
  TLS scenarios, required job IDs, the aggregate gate, and publish job.
- Added artifact-only migration qualification. Direct npm and PyPI registry
  checks found no published 0.3 `custback` artifacts, so the workflow rebuilds
  explicitly unpublished references once from reviewed commit
  `099001e9f25a3ea1d0c820b066a6318e4172d549`; reports and documentation never
  describe those source reconstructions as released bytes.
- Added the packaged two-host WSS/HTTPS harness with isolated container
  addresses, independent ephemeral CAs and secrets, real hostname verification,
  firewall removal, outage/reconnect/rotation/auth/certificate scenarios,
  privacy-slate recording, pre-TLS no-payload capture, and finally-safe cleanup.
- Added reviewed Ruff configuration, deterministic seeded stress tests,
  recursive clean-tree rejection, exact npm/sdist payload allow-lists, and
  build-once candidate digest recording.
- Added a tag/manual production workflow that consumes the same three candidate
  files everywhere. Migration and two-host matrix legs retain artifact-bound
  reports; same-run evidence rejects report/runtime/host/digest drift and
  is itself attested. Candidate and evidence attestations must verify before
  the aggregate gate, full release check, or publish job can succeed.
- Local unit, migration, stress, schema, workflow, and packaging checks verify
  the enforcement machinery. The true multi-platform, CUDA, two-clean-host,
  real-container, and signed-provenance gates remain dynamic release-candidate
  work; no success is claimed for them in this implementation record.
- `REL-01` is resolved and its acceptance regression now passes normally.
  No remediation blocker is currently open, so `release_blocked=false`.
- Windows production evidence remains deferred outside the blocker registry and
  required-gate manifest.

## Corrective PR sequence

1. Registry freeze: reopen/add eight entries and land strict failing regressions.
2. `SEC-02`: immutable operator-owned live-backdrop targets.
3. `PRIV-01`: session-wide, capacity-safe raw replay rejection.
4. `LIFE-01`: linearizable renderer-session publication lease.
5. `LIFE-02`: terminal/killable Audio2Face ownership.
6. `STOR-02`: rename-aware cleanup transactions and restart recovery.
7. `PKG-01`: installed Python/npm avatar command and packaged config export.
8. `PKG-02`: stdout-clean prepack and real command-substitution regression.
9. Migration fixtures, Ruff, stress, clean-tree, and artifact gates.
10. Packaged two-host WSS/HTTPS system test and required CI integration.
11. `REL-01`: exact-commit release candidate, registry close, and publish gate
    (**complete**).

Security/privacy, lifecycle, storage, and packaging PRs may proceed in parallel
after the registry freeze. Do not combine `LIFE-01`/`LIFE-02` ownership changes
with `STOR-02`.

## Final release gate

A release is permitted only when:

- the blocker registry contains no open entries, reports
  `release_blocked=false`, and the required-gate allow-list is exact;
- hot backdrop and proxy SSRF reproductions perform no file, DNS, capture, or
  network I/O;
- remote plaintext fails before connection;
- a recording sink proves every remote frame passes the privacy firewall,
  including raw replays beyond the old time window and under capacity pressure;
- invalid avatar patches leave version/resources unchanged;
- cancelled sessions cannot publish or incorrectly account queued renders, and
  no render or Audio2Face worker/handle survives a successful shutdown;
- concurrent staging, committed data, and cleanup-pending data stay private and
  within aggregate quota;
- decompression bombs fail before OpenCV;
- every optional extra installs and smokes from a built wheel;
- npm reinstall preserves venv, extras, and rollback state;
- installed npm, wheel, and sdist surfaces contain the license, canonical avatar
  command, compatibility alias, and exportable avatar config;
- `npm pack --silent` emits exactly one installable filename on stdout;
- migration, Python, npm, Ruff, stress, clean-tree, and artifact checks pass for
  the exact release commit;
- the publish job selects the already qualified wheel, sdist, and npm tarball by
  their recorded digests without rebuilding them; and
- the documented two-host WSS/HTTPS deployment passes, including renderer
  outage, certificate, rotation, privacy-slate, and token-failure behavior.
