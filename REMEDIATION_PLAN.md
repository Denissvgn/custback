# Custback remediation plan

Status: **Phase 1 complete — release blocked; Phase 2 is next**

This plan converts the findings from the July 2026 repository review into an
implementation sequence. The Phase 1 security/privacy gates are satisfied,
but the tree remains stop-ship while 21 later-phase blockers remain open,
including the Audio2Face dependency/protocol gate.

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
- npm reinstall and upgrade preserve the managed environment, rollback state,
  and explicitly selected extras.

## Delivery roadmap

| Phase | Scope | Exit gate |
| --- | --- | --- |
| 0 | Freeze release and add deterministic regressions | Every known defect has a blocker ID and executable regression |
| 1 | Proxy, token, transport, and privacy containment | SSRF performs no I/O; remote plaintext fails; no raw startup output |
| 2 | Transactional avatar engine and worker lifecycle | Config/resources activate atomically; no worker survives teardown |
| 3 | Storage, merge semantics, segmentation, rendering, streaming | Concurrency, quota, media, and correctness gates pass |
| 4 | Audio2Face, npm, release, docs, and compliance | All extras install from artifacts; npm upgrades preserve state |
| 5 | Migration and end-to-end validation | Clean artifacts and a two-host TLS smoke test pass |

Phases 1, 2, and 4 may run in parallel after Phase 0. Production fixes should
be split into reviewable PRs; do not combine the transactional render-engine
change with the storage-reservation migration.

### Phase 0 completion record

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
must fail while any entry remains open. A blocker is closed only in the same
change that removes its strict expected-failure marker and makes its acceptance
test pass.

| ID | Finding | Target phase |
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

### Phase 1 completion record (2026-07-16)

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

## Recommended PR sequence

1. Phase 0 blocker registry and regression harness.
2. Immutable proxy target, token split, redaction, and upstream-auth mapping.
3. Secure endpoint policy and central privacy firewall.
4. Shared merge-patch and validation fixes.
5. Transactional avatar engine and render lifecycle.
6. Audio2Face protocol, TLS, cancellation, and CI.
7. Private storage, reservations, quotas, and decoded-image limits.
8. Segmentation, alpha, pose, and async streaming.
9. Stable npm runtime and persistent extras.
10. Binaries, prepack, license, platform guards, and deployment documentation.
11. End-to-end TLS deployment and final artifact release gate.

## Final release gate

A release is permitted only when:

- the blocker registry contains no open entries;
- the SSRF reproduction performs no I/O;
- remote plaintext fails before connection;
- a recording sink proves every remote frame passes the privacy firewall;
- invalid avatar patches leave version/resources unchanged;
- no render or Audio2Face worker survives shutdown;
- concurrent staging plus committed data stays within quota;
- decompression bombs fail before OpenCV;
- every optional extra installs and smokes from a built wheel;
- npm reinstall preserves venv, extras, and rollback state;
- npm, wheel, and sdist contain the license and expected launchers;
- Python, npm, Ruff, stress, and clean-artifact checks pass; and
- the documented two-host WSS/HTTPS deployment passes, including renderer
  outage and token-failure behavior.
