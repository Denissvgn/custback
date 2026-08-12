# Experimental system profiles

Custback exposes a server-owned, versioned profile catalog for concrete
quality/cadence and camera-framing choices. These entries are
**Experimental** or **Locally screened**: both states make no portable quality
claim, require an explicit acknowledgement, and do not change schema-1
defaults. Evidence from one host may advance a row to `locally_screened`;
portable promotion remains blocked until a second hardware identity completes
the required capture, model, visual, performance, sink, privacy, restart, and
shutdown matrix.

## Selection and restart contract

The initial quality rows are:

| Profile | Capture | Transport | RVM policy |
| --- | --- | --- | --- |
| Performance | 960×540 at 30 fps | 960×540 at 30 fps | CUDA, ratio 0.40 |
| Balanced | 1280×720 at 15 fps | 1280×720 at 30 fps | CUDA, automatic ratio (0.40 at 720p) |
| Quality | 1280×720 at 15 fps | 1280×720 at 30 fps | CUDA, ratio 0.50 |
| Motion stable | 1280×720 at 15 fps | 1280×720 at 30 fps | Balanced plus motion-aware boundaries at 0.10 s / 720 px/s |

The stability-first rows use native RVM alpha, model foreground, zero mask
shift, zero light wrap, correction off, temporal wrap off, and legacy sRGB.
They require the built-in RVM model and proven CUDA execution. A custom model
makes the rows unavailable rather than being cleared. Unsupported canvases or
sinks stay visible with a reason; Custback never retains a profile label while
substituting MediaPipe, CPU, another canvas, or another sink. In particular,
Windows native output does not support the Performance row's 960×540 mode.
An active PyVirtualCam mode proves only that exact mode; it does not prove that
the platform backend accepts another canvas. A cross-canvas row therefore
stays disabled until the runtime has bounded exact `supported_output_modes`
evidence for that sink.

Camera framing is an independent axis:

- Fill center uses proportional `cover` at anchors 0.5/0.5.
- Headroom uses proportional `cover` at anchors 0.5/0.35.
- Show all uses `contain` at anchors 0.5/0.5.

These are static camera geometry policies, not face tracking or primary-person
selection. Backdrop fit and anchors remain independent, live, asset-specific
controls. `stretch` remains available only for compatibility.

Every initial profile owns restart-only leaves. Selecting one therefore saves
the complete concrete row for restart and applies none of its hot subset to
the old canvas. The UI distinguishes `Active`, `Saved for restart`,
`Configured but unavailable`, and `Custom`. It provides restart instructions
but has no bearer-only shutdown authority.

## Managed concrete preferences

Selections are stored as sparse concrete `AppConfig` values in
`profile-preferences.yaml` under Custback's platform configuration directory.
Profile IDs are never persisted. Startup precedence is:

1. built-in defaults or explicit operator YAML;
2. managed profile concrete values; and
3. explicit leaf CLI flags.

The operator YAML is never rewritten. CLI-owned leaves remain locked and an
incompatible selection is rejected. Resetting one profile axis removes only
that axis's managed leaves, revealing YAML/default values on the next restart.
Raw `PATCH /config` remains session-only.

The managed file and lock use no-follow/reparse-safe opens, owner checks,
owner-only permissions or DACLs, bounded YAML without aliases or duplicate
keys, cross-process locking, dual revision CAS, an fsynced same-directory
atomic replacement, and directory fsync. Invalid or insecure preferences fail
startup. Use
`--no-profile-preferences` only as an explicit recovery bypass; it also disables
profile persistence for that process.

## API

All routes use the existing API authentication boundary:

- `GET /profiles` returns the catalog/version/digest, evidence and bounded
  availability, active/desired matches, configuration and preference
  revisions, CLI locks, and restart-pending field names.
- `POST /profiles/apply` resolves reviewed server-side IDs. It requires
  `expected_config_version`, `expected_preferences_revision`, and
  `accept_experimental: true` for every non-qualified current entry.
- `POST /profiles/reset` clears selected axes using both revision checks.

Responses contain neither private configuration values nor filesystem paths.
The catalog's SHA-256 and every canonical concrete patch SHA-256 are bound in
the rollout ledger and checked by the release verifier.

## Capture-only screening

Stop the normal engine before probing because a camera generally cannot be
owned by both processes. Then run, for example:

```console
custback system-profile-probe \
  --profile performance \
  --profile balanced \
  --profile quality \
  --profile motion_stable \
  --accept-experimental \
  --output NEW_PRIVATE_DIR
```

The command opens one production capture reader at a time and closes it before
the next row. It records exact requested/negotiated/delivered geometry,
capture/read/normalization timing, cadence, reported controls, failures,
restarts, stalls, and per-row capture suitability in one owner-only,
pixel-free report. Ordinary open/read failures become row results; a reader
close failure aborts the remaining matrix. The probe never mutates preferences
and never infers model, compositor, output-sink, consumer, or portable
qualification from capture-only results.

At 30 fps the capture screen requires at least 27 unique fps. The current
15-fps rows use the stricter local gate of 14.5 unique fps. The gate uses the
lowest active, wall-completion, and full-window availability rate and also
requires a complete, sustained measurement window, exact requested versus
negotiated/delivered geometry, and no read failure, restart, geometry change,
stall, capture error, or close error. `--accept-experimental` is also required
for a `locally_screened` row because both evidence states remain explicitly
non-qualified. Full qualification still uses the existing private
replay/evaluation, RVM, visual, performance, platform, privacy, hot-patch,
restart, shutdown, and consumer-sink tools.

One-host screening must exercise every quality row without changing its
`quality_claim: false` state. For `motion_stable`, compare it directly with
Balanced on stationary hair/headphones, speech, fast turns, hands crossing the
face, entry/exit, occlusion, low contrast, low light, and dynamic video. The
existing gates remain authoritative: the declared cadence/service budget,
contour displacement p95 at most 1.5 px and 0.60× baseline, stationary area
drift at most 1%, trails at most 1.10× baseline with no prior-contour dominance
beyond one unique interval, and the existing core/background alpha, halo,
fine-detail, gradient, MSE, fallback, switch, restart, and shutdown gates. Only
ablate the motion ceiling from 720 to 480 px/s if the declared fast-motion gate
rejects 720.

The future **Gentle match** appearance row is deliberately absent from the
catalog. Its screening candidate is light wrap 0.08, temporal-bounded wrap at
0.12 s, color-correction strength 0.30, exposure limit 0.50 EV,
white-balance strength 0.30, and adaptation time 1.50 s. It may be added only
after separate edge-variation, settling, timing, and scene-cut evidence passes;
it must not silently alter the stable quality rows.

For live visual diagnosis, use the native preview's `d`/`D` views described in
[Local live matte diagnostics](matte-live-diagnostics.md). Alpha, silhouettes,
and temporal matte metrics remain private and are not added to public status.
