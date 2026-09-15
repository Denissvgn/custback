# Visual configuration and rollback

Use [config/default.yaml](../config/default.yaml) as the full configuration
reference. Current compatibility defaults are camera fit `stretch`, backdrop
fit `cover`, blend space `srgb_legacy`, and foreground color correction `off`.
The alternatives below are explicit options; they are not default changes.

## Configure the visual policies

Acquisition size and output canvas are different contracts. `camera.width` and
`camera.height` request a hardware mode; `output.width` and `output.height`
select the canonical output canvas when both are set. If both output values
are `null`, the requested camera dimensions are the canvas.

```yaml
schema_version: 1

camera:
  width: 1280
  height: 720
  fit_mode: cover       # cover | contain | stretch
  anchor_x: 0.5         # 0 = left, 1 = right
  anchor_y: 0.5         # 0 = top, 1 = bottom
  rotation: 0           # 0 | 90 | 180 | 270
  mirror: false

background:
  fit_mode: cover       # cover | contain | stretch
  anchor_x: 0.5
  anchor_y: 0.5

compositing:
  blend_space: linear_srgb   # srgb_legacy | linear_srgb
  color_correction:
    mode: auto               # off | auto
    strength: 0.5
    exposure_limit_ev: 0.85
    white_balance_strength: 0.5
    adaptation_time_s: 0.8

output:
  width: 1280
  height: 720
```

`cover` scales proportionally and crops overflow. Anchors choose which part is
retained. `contain` scales proportionally and adds opaque black bars. Anchors
position the content within those bars. `stretch` fills the canvas with
independent horizontal and vertical scales and can distort the subject.
Orientation is applied before mirror and fit. Downscaling uses area sampling;
upscaling uses linear sampling.

Camera geometry and output canvas changes are restart-only. Background fit,
background anchors, blend space, and correction settings are hot,
transactional changes. Do not mix a restart-only field with a hot field in one
PATCH: the server returns `409 restart_required` and applies nothing.

Examples for one policy at a time:

```bash
# Restart-only: edit the YAML, stop custback, and restart it.
# camera:
#   fit_mode: cover

# Hot: use the authenticated helper in user-guide.md.
auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
  -H 'content-type: application/merge-patch+json' \
  -d '{"background":{"fit_mode":"contain","anchor_y":0.25}}'

auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
  -H 'content-type: application/merge-patch+json' \
  -d '{"compositing":{"blend_space":"linear_srgb"}}'

auth_header | curl --config - -X PATCH http://127.0.0.1:8710/config \
  -H 'content-type: application/merge-patch+json' \
  -d '{"compositing":{"color_correction":{"mode":"auto","strength":0.5}}}'
```

After each change, inspect `GET /config`, `GET /status`, and a fresh preview
frame. Useful status fields include `camera_fit`, camera crop/pad/scale values,
`background_fit`, backdrop crop/pad/scale values, `color_correction_state`,
`color_correction_reason`, `color_correction_confidence`,
`color_correction_exposure_ev`, the three WB gains, and
`color_input_assumption`.

## Eligibility and input-color contract

Automatic correction is deliberately foreground-only and mode-limited:

| Background mode | Automatic correction |
| --- | --- |
| `image`, `video`, `camera` | Eligible when mask, target, clipping, and confidence gates pass |
| `passthrough` | Excluded; canonical raw pixels remain exact |
| `blur` | Excluded; both sides already derive from the same camera |
| `color` | Excluded; a solid color is not illumination evidence |
| `remote` | Excluded; returned frames are exact-canvas external output |

An eligible frame may still use identity or exposure-only correction.
Insufficient foreground coverage, clipped/black samples, or low exposure
confidence produces no new estimate. Insufficient neutral target samples
allows bounded exposure without white balance. During a short loss of
confidence, a reliable prior estimate can freeze and then decay to identity;
`GET /status` distinguishes warming, active, low-confidence hold, stale decay,
and bypass.

Logical camera and remote frames are exact-shape, full-range,
display-referred sRGB `uint8 BGR`. Remote output is never resized, rotated,
profile-converted, or automatically corrected; wrong-size or malformed remote
frames fail closed. Still images with a valid ICC profile are converted to
sRGB; untagged still images explicitly assume sRGB. Supported tagged SDR video
is normalized to full-range sRGB BGR. Untagged video uses the observable
schema-1 legacy BT.709/full/BT.709/sRGB assumption. The four
`background.video_color_*` overrides are only for an operator-owned local file
whose declarations are known; histogram-based matrix/range guessing is never
performed. Unsupported HDR or wide-gamut tags are rejected rather than
displayed under a guessed transform.

Camera-control reporting is read-only. `GET /status.camera_controls` reports
what OpenCV can observe for auto white balance, WB temperature, auto exposure,
exposure, gain, and gamma. Zero may mean either a legitimate value or an
unsupported property, so it is reported as `indeterminate-zero`. Custback does
not write these controls automatically; comparisons must account
for the camera's own exposure/WB loop.

## Deterministic upgrade behavior

A persisted mapping without `schema_version` has schema-1 semantics. An
explicit schema-1 document has the same semantics. Before model validation,
absent visual fields are materialized as:

```yaml
camera:
  fit_mode: stretch
  anchor_x: 0.5
  anchor_y: 0.5
  rotation: 0
background:
  fit_mode: cover
  anchor_x: 0.5
  anchor_y: 0.5
compositing:
  blend_space: srgb_legacy
  color_correction:
    mode: "off"
    strength: 0.5
    exposure_limit_ev: 0.85
    white_balance_strength: 0.5
    adaptation_time_s: 0.8
output:
  width: null
  height: null
```

The explicit `custback migrate --config ...` operation writes the schema
version and effective visual policy atomically while retaining a private
byte-exact backup. Loading an old file does not rewrite it. `PATCH /config`
cannot migrate schema semantics.

While schema 1 remains active, a negotiated camera frame whose aspect differs
from the output canvas emits one path-free upgrade note per capture lifetime:
legacy `stretch` preserves the old distortion, whereas explicit `cover`
selection crops proportionally. Pin `camera.fit_mode: stretch` to retain
the old framing, or preview an explicit `cover` selection and anchors before saving the changed configuration.

Later target-default schemas must not reinterpret a versionless or schema-1
omission. Each migration is explicit, atomic, and separately reviewed.

## Rollback

Rollback pins the changed field; it does not lower the highest recognized
schema version. Downgrading the schema parser could strand files already
written by a newer release.

Before changing a persisted file, stop custback and make an operator-owned
copy. After restarting, compare `GET /config`, `GET /status`, and a new preview
frame.

- Camera-cover rollback: set `camera.fit_mode: stretch` in YAML and restart.
  Recheck anchors only after the legacy frame is restored.
- Linear-compositing rollback: PATCH or persist
  `compositing.blend_space: srgb_legacy`. This is hot and versioned.
- Automatic-correction rollback: PATCH or persist
  `compositing.color_correction.mode: off`. Confirm
  `color_correction_active` becomes `false` on the next committed output.

If a mixed PATCH returns `409`, split it and restart for the camera/output
part. If activation returns `422` or `503`, the transaction retained the old
provider, state, pixels, and config version; fetch `GET /config` before
retrying.

## Troubleshooting

### The subject is cropped

`cover` intentionally crops overflow to preserve proportions. Use
`camera.anchor_x` / `anchor_y` or the corresponding background anchors to
retain the important region. Use `contain` when the whole source must remain
visible. Use `stretch` only when distortion is an explicit compatibility
choice.

### Black bars appear

`contain` creates deterministic opaque padding when source and canvas aspects
differ. Select `cover` to fill the canvas by cropping, or match the output
canvas aspect to the source. Bars are excluded from automatic color sampling.

### Correction says low confidence, warming, or stale

Use a supported `image`, `video`, or `camera` backdrop and keep a useful,
unclipped foreground mask in view. Low confidence is a safe identity/hold
outcome, not a reason to raise strength or widen the EV clamp. Check
`color_correction_reason`, confidence, mask/segmentation health, and whether
contain padding leaves enough valid target content.

### Exposure or color still pumps

The physical camera or live backdrop may run its own auto-exposure/WB loop.
Inspect `camera_controls`, test with correction `off`, and retain the
preserve-only control policy. Do not infer a camera property's meaning from a
zero value or attempt to lock controls through undocumented OpenCV writes.

### Tagged media is rejected

The decoder supports a bounded SDR tag set. Convert HDR/wide-gamut media to a
supported SDR sRGB/BT.709 deliverable with explicit declarations. For a trusted
local untagged or incorrectly tagged asset, set all known
`background.video_color_*` overrides explicitly and verify the input/output
color description in status. Do not choose overrides from the pixel
histogram.

### Untagged image or video looks wrong

Untagged images assume sRGB. Untagged video uses the documented schema-1
legacy assumption. Re-export with correct metadata when possible. Overrides
are a reproducibility tool for known local files, not an automatic detector.
