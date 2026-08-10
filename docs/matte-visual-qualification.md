# End-to-end matte visual qualification

Status: **MATTE-5.2 qualification harness and owner-controlled evidence
protocol**

This workflow decides whether one exact matte policy both improves
representative video and survives every local output boundary. It combines the
existing digest-bound matte metrics with same-generation route captures and a
human visual review. It does not select a preset, change a default, or convert
generated fixtures into evidence about real people, models, cameras, or
virtual-camera consumers.

The checked-in
[`matte-visual-qualification-local-template.json`](matte-visual-qualification-local-template.json)
contains no pixels and is not qualification evidence. Copy it outside the
repository and replace every `REPLACE_` value. Use absolute paths for private
bundles, annotations, and sidecars. Replace the example root with an absolute
owner-controlled path outside the checkout:

```console
PRIVATE_EVIDENCE_ROOT=/absolute/path/outside/repository/matte-visual
install -d -m 700 "$PRIVATE_EVIDENCE_ROOT"
install -m 600 docs/matte-visual-qualification-local-template.json \
  "$PRIVATE_EVIDENCE_ROOT/plan.json"
custback matte-visual-qualify \
  "$PRIVATE_EVIDENCE_ROOT/plan.json" \
  --output "$PRIVATE_EVIDENCE_ROOT/qualification-output"
```

The output directory must be new. It contains `qualification.json` and the
review companion `qualification.md`. Exit status `0` means the supplied real
evidence qualified, `1` means a structurally valid run remains pending or
failed, and `2` means the plan, sidecar, artifact, or privacy contract is
invalid. A retained report with status `pending` or `failed` is useful evidence
but is not approval.

The command is an offline verifier, not a recorder: it opens no camera, model,
network service, HighGUI window, or virtual-camera sink. Collect the private
bundles, route artifacts, and review sidecars first, then run the verifier over
those immutable inputs.

## Evidence classes and current disposition

The qualifier distinguishes evidence authority instead of inferring it from
favorable pixels:

| Authority | What it proves | Qualification effect |
| --- | --- | --- |
| `generated-fake` | Deterministic frame identity, route decoding, exact in-process parity, comparison arithmetic, and failure sensitivity | Always `pending`; cannot approve a candidate or preset |
| `local-observed` backed by `consented-local` or `licensed-local` provenance | Representative model output, physical preview and output routes, and human review | May qualify only when every required gate and coverage item passes |

Generated arrays, fake MediaPipe/RVM runtimes, fake HighGUI, fake
`pyvirtualcam`, and an injected Windows ring remain the fast deterministic
contract. They are diagnostic only. A real qualification still requires all
of the following:

- consented or licensed representative clips;
- held-frame captures from the production routes;
- physical HighGUI review;
- a real `pyvirtualcam` loopback recording;
- a Windows native virtual-camera capture where that route applies;
- a completed blinded or side-by-side human review;
- a real live-camera backdrop case; and
- observed 60 FPS and 1920×1080 cases on supported hardware.

Absence or inapplicability is recorded; another backend, lower cadence, fake
sink, projected timestamp sequence, or smaller canvas cannot silently fill a
real row. MATTE-5.3 separately decides performance and broad platform support.

The verifier rejects inconsistent generated/local labels and stale digest
bindings, but it cannot authenticate the external consent/license record or
infer from pixels alone that a clip and physical capture are genuine. Local
authority remains an evidence-owner and reviewer attestation backed by the
private source/capture record. Coordinately relabelling generated inputs as
local is not legitimate qualification even if someone also rewrites every
self-attested provenance field and digest.

## Exact candidate policy

The plan schema is `custback.matte-visual-qualification-plan`, version `1`.
Every candidate contains the full configured segmentation and compositing
objects. Every case adds the expected effective backend/device/RVM ratio and
whole typed matte policy because canvas and backdrop eligibility can change
that resolution. The qualifier checks the recorded effective policy rather
than trusting the candidate label. It also recomputes the policy with the
production resolver and rejects arbitrary or stale policy text, including a
template placeholder that was not replaced.

Every case also declares `baseline_expected` with the baseline's full
segmentation config, compositing config, and expected effective-policy object.
The verifier recomputes and audits that baseline contract just as it does the
candidate, and the report retains a path-free digest. This prevents an
undeclared weak baseline from manufacturing an improvement. It cannot decide
product history for the evidence owner: the owner remains responsible for
choosing the legitimate deployed compatibility baseline, recording it without
sabotage, and explaining any intentional difference beyond the selected
candidate controls.

Every declared candidate must be referenced and must independently complete
the full visual matrix, quality gates, seven-boundary aggregate, all case
reviews, and at least one preferred review. One exploratory candidate cannot
borrow another candidate's evidence.

MATTE-2.1, MATTE-2.2, and MATTE-2.4 introduced explicit, default-off or
compatibility-preserving candidates. A plan that claims them must name their
exact controls:

| Task | Qualification candidate | Checked-in compatibility behavior |
| --- | --- | --- |
| MATTE-2.1 | `segmentation.boundary_stabilization.mode: motion_aware` | `off`; the historical EMA remains independently configured |
| MATTE-2.2 | `segmentation.spatial_edge_refinement.mode: stable_guided` with `edge_refine: true` | `legacy_watershed`; RVM bypasses generic spatial refinement |
| MATTE-2.4 | `compositing.light_wrap_stabilization.mode: temporal_bounded` on video or camera backdrops with positive light wrap | `off`; the stateless wrap path remains active |

A model-only RVM row does not qualify `stable_guided`, because the RVM policy
correctly reports generic spatial refinement as inapplicable. Qualify that
candidate on a mask-producing backend where it is effective. Likewise, an
image or solid-color backdrop cannot qualify temporal light wrap, and a plan
with motion-aware stabilization off cannot claim MATTE-2.1. The report is
limited to the exact algorithms observed in its cases.

Each selected optional algorithm also has a complete task-specific gate:

- MATTE-2.1 needs effective stationary 1280×720 cells at 15, 30, and 60 FPS.
  In every cell, stationary contour displacement must be at most `1.5 px` and
  at most 60% of the declared baseline, subject-area drift p95 at most `0.01`,
  and previous-contour dominance at most one interval. The extra 720p15 and
  720p60 skeletons in the local template are required algorithm cells; the
  360p15 and 1080p60 taxonomy rows do not substitute for them.
- MATTE-2.2 needs effective cells at 640×360, 1280×720, and 1920×1080. Ground-
  truth alpha MSE and gradient error must both be evaluated and non-regressing,
  with at least one materially improved in every applicable case.
- MATTE-2.4 needs effective dynamic-video and live-camera cells. Edge shimmer
  must be materially improved while opaque-core deficit/confidence,
  background alpha, and halo area remain non-regressing in every applicable
  case.

The checked-in template selects all three as one combined candidate, so the
report truthfully labels its evidence scope `combined-selected-policy` and
sets `independent_optional_algorithm_causality_claimed` to `false`. Run
separate one-change ablations if a release decision needs to attribute the
improvement independently to one algorithm.

Reactions must be disabled for all matte metrics, contact sheets, and human
review. A later, separately labelled reaction pass may check final-sink parity,
but it cannot improve, obscure, or replace a matte result. For the current base
pipeline, the recorded base and final composites must remain identical or be
an explicit lossless alias.

## Required visual taxonomy

Coverage is aggregated from the plan's cases, but each tag must describe pixels
and timing actually present in its bound baseline and candidate bundles. Do not
attach every tag to one convenient clip. Retain enough cases to cover:

Appearance and source-condition labels cannot be inferred reliably from the
numeric arrays alone. The report therefore records
`coverage.appearance_and_source_condition_authority` as `generated-proxy` for
generated fixtures or `owner-and-reviewer-attested` for local evidence; the
latter remains a human evidence claim, not a classifier result.

| Axis | Required observations |
| --- | --- |
| Appearance | bald or short hair; long or fine hair; glasses; facial hair; headphones or another solid accessory; dark opaque clothing; light opaque clothing; skin-tone and lighting diversity |
| Motion | stationary; speech micro-motion; slow turn; fast turn; hand or prop crossing the face; entering or leaving frame |
| Source condition | bright; dim; compression noise; low contrast; clutter |
| Background | static image; dynamic video; person-free blur; solid color; live camera |
| Cadence | 15 FPS; 30 FPS; 60 FPS; irregular delivery; dropped inputs; restart/discontinuity |
| Canvas | 640×360; 1280×720; 1920×1080 where supported |

Motion claims must also have at least one matching candidate-annotation segment:

| Claimed motion | Accepted annotation segment kind |
| --- | --- |
| `stationary` | `stationary` |
| `speech_micro_motion`, `slow_turn` | `moving` |
| `fast_turn` | `fast_motion` |
| `hand_or_prop_crossing_face` | `occlusion` |
| `entering_frame`, `leaving_frame` | `moving` or `fast_motion` |

That coarse segment kind is a necessary machine check, not permission to
mislabel its contents; the operator and reviewer still verify the exact motion
claim in the bound pixels.

Cadence projection may test report arithmetic, but it does not establish RVM
recurrence at that cadence. A restart case must contain a real capture or
geometry generation change, and a dropped-input case must retain the original
sequence gap. A live-camera background is a separately approved local camera
target, not a video file relabelled as a camera.

The baseline and candidate in each case must have source, annotation, and
timing identity suitable for direct comparison. Use the same people, motion,
background presentation, frame order, timestamps, generations, and review
regions. If a model must be rerun for a candidate, prove the canonical raw
source identity rather than comparing unrelated recordings.

Baseline and candidate annotation provenance maps must be byte-for-byte
equivalent after parsing. Their `kind` must equal the plan provenance kind; for
local evidence, annotation `license` and human-review provenance `reference`
must both exactly equal the plan's `license_or_consent_reference`. Renaming a
generated plan or sidecar therefore cannot promote generated annotations.

## Same-generation boundary procedure

Each case references a `custback.matte-visual-boundary-evidence` version-1
sidecar. Its source identity binds:

- the matte replay manifest SHA-256;
- bundle sequence and capture sequence;
- capture and geometry generations; and
- the lossless reference final-composite SHA-256; and
- an `identity_region` with exact `x`, `y`, `width`, `height`, and
  `reference_region_sha256` fields.

Use a deliberately held frame with an asymmetric, machine-readable fiducial in
a textured background or opaque-interior region that stays clear of the
uncertain/contour edge band. Encode the target sequence/generation and make the
region large and high-contrast enough to survive the configured JPEG route.
Hold it long enough for paced and resampling sinks to present that exact
generation. The fiducial identifies the frame after JPEG or virtual-camera
transport; it must not be used as alpha ground truth or included in
subject-edge metrics.

Compute `reference_region_sha256` from the contiguous row-major BGR `uint8`
bytes of that exact rectangle in the candidate final composite. The qualifier
verifies that binding, then computes `identity_region_mae` from the same
rectangle in every decoded boundary. Exact seams require zero identity-region
error; transported artifacts use the plan's bounded full-frame MAE ceiling.
The rectangle must be at least 8×8, avoid the uncertain-alpha/contour edge
band, and have grayscale standard deviation of at least `4.0`; stable
background or opaque-interior placement is allowed. Its pixels must differ by
more than the transport tolerance from the same rectangle in every other
visually different replay final. Byte-identical repeated finals are exempt, but
a flat or generation-invariant crop is not frame-identity evidence.

For snapshot, MJPEG, and WebSocket in particular, keep the source generation
held until all three responses are captured and verify their decoded fiducial
against the sidecar's exact capture sequence and capture/geometry generations.
Matching width and height, nearby timestamps, or whichever frame was latest
after the hold was released are not valid pairing evidence.

Collect all seven boundary artifacts across the applicable route set:

1. lossless in-memory final composite;
2. lossless frame supplied to HighGUI before the status/help overlay;
3. `/video/snapshot.jpg` response;
4. one decoded `/video/mjpeg` part;
5. one output `/ws/frames?stream=output` JPEG message;
6. a lossless extracted frame from a real `pyvirtualcam` loopback recording;
7. a lossless extracted frame from the Windows native virtual-camera recording.

The three API artifacts remain the original JPEG bytes in the evidence set;
the qualifier decodes them. PNG is used for the exact and extracted loopback
frames. A captured descriptor has exactly `filename`, `sha256`, `bytes`, and
`media_type`; the filename is one relative component beside the sidecar. Use
`image/jpeg` for the three API boundaries and `image/png` for the other four.
Do not screenshot an API image and call it a transport capture.

Each artifact entry is explicitly `captured` with its digest-bound descriptor,
or `not_applicable` with no artifact and a bounded reason. Not-applicable does
not count as coverage: a qualification must include at least one real captured
artifact for every boundary across its cases. It exists only so an unsupported
mode is described honestly instead of being filled with another sink.

Every entry also records an exact `capture_method` and `platform`; a filename
or matching dimensions are not capture provenance. Local observed evidence
uses these methods:

| Boundary | Local `capture_method` | Generated diagnostic method |
| --- | --- | --- |
| `in_memory` | `pipeline-memory-tap` | `generated-memory-tap` |
| `highgui_pre_overlay` | `highgui-pre-overlay-tap` | `generated-highgui-pre-overlay` |
| `snapshot_jpeg` | `authenticated-http-snapshot` | `generated-http-snapshot` |
| `mjpeg_jpeg` | `authenticated-mjpeg-part` | `generated-mjpeg-part` |
| `websocket_jpeg` | `authenticated-output-websocket` | `generated-output-websocket` |
| `pyvirtualcam_loopback` | `pyvirtualcam-consumer-recording` | `generated-pyvirtualcam-double` |
| `windows_native_loopback` | `windows-native-consumer-recording` | `generated-native-ring-double` |

Local platforms are `linux`, `macos`, or `windows`, and a captured Windows
native entry must use `windows`. Generated diagnostics use platform
`generated`. A genuinely unsupported entry uses method `not-applicable`, a
null artifact, and a specific non-empty reason; it still does not cover that
boundary.

The verifier checks these method/platform values as bounded owner attestations;
it cannot cryptographically prove that a PNG came from a physical window or
that a loopback frame came from the named consumer or device. The path-free
report therefore records `local_boundary_method_and_platform_attested: true`
and `physical_capture_origin_cryptographically_proven: false` for local runs.
Keep the capture log and physical screen/consumer review with the private
evidence, and do not describe a passing pixel comparison alone as hardware
provenance.

The seven artifacts are a spatial parity check for one held processed
generation. They do not qualify output inter-send cadence, repeat behavior,
pacing, or sink scheduling; MATTE-5.3 owns those temporal transport and
performance gates. The report records scope
`same-generation-spatial-resize-and-contour-parity` and
`boundary_sink_cadence_qualified: false`. The representative clip evidence in
version 1 is the digest-bound baseline/candidate replay sequence, not an
unmodeled contact sheet or a single boundary still.

### Boundary sidecar example

The following is the complete strict sidecar shape for the template's
1920×1080@60 solid-color case on Windows. Replace every `REPLACE_` value and
the illustrative scalars. Artifact filenames are relative to the sidecar's
directory. This row honestly marks native output inapplicable because 60 FPS
is not a supported native mode; a separate 1280×720@30 or 1920×1080@30 case
must capture that boundary for aggregate qualification.

```json
{
  "schema": "custback.matte-visual-boundary-evidence",
  "version": 1,
  "case_id": "solid_color_1080p60",
  "authority": "local-observed",
  "source": {
    "bundle_manifest_sha256": "REPLACE_WITH_CANDIDATE_BUNDLE_MANIFEST_SHA256",
    "bundle_sequence": 42,
    "capture_sequence": 12042,
    "capture_generation": 7,
    "geometry_generation": 3,
    "reference_final_composite_sha256": "REPLACE_WITH_FINAL_COMPOSITE_SHA256",
    "identity_region": {
      "x": 16,
      "y": 16,
      "width": 64,
      "height": 64,
      "reference_region_sha256": "REPLACE_WITH_CONTIGUOUS_BGR_REGION_SHA256"
    }
  },
  "artifacts": {
    "in_memory": {
      "status": "captured",
      "artifact": {
        "filename": "in-memory.png",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/png"
      },
      "reason": "",
      "capture_method": "pipeline-memory-tap",
      "platform": "windows"
    },
    "highgui_pre_overlay": {
      "status": "captured",
      "artifact": {
        "filename": "highgui-pre-overlay.png",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/png"
      },
      "reason": "",
      "capture_method": "highgui-pre-overlay-tap",
      "platform": "windows"
    },
    "snapshot_jpeg": {
      "status": "captured",
      "artifact": {
        "filename": "snapshot.jpg",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/jpeg"
      },
      "reason": "",
      "capture_method": "authenticated-http-snapshot",
      "platform": "windows"
    },
    "mjpeg_jpeg": {
      "status": "captured",
      "artifact": {
        "filename": "mjpeg-part.jpg",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/jpeg"
      },
      "reason": "",
      "capture_method": "authenticated-mjpeg-part",
      "platform": "windows"
    },
    "websocket_jpeg": {
      "status": "captured",
      "artifact": {
        "filename": "websocket-output.jpg",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/jpeg"
      },
      "reason": "",
      "capture_method": "authenticated-output-websocket",
      "platform": "windows"
    },
    "pyvirtualcam_loopback": {
      "status": "captured",
      "artifact": {
        "filename": "pyvirtualcam-loopback.png",
        "sha256": "REPLACE_WITH_ARTIFACT_SHA256",
        "bytes": 123456,
        "media_type": "image/png"
      },
      "reason": "",
      "capture_method": "pyvirtualcam-consumer-recording",
      "platform": "windows"
    },
    "windows_native_loopback": {
      "status": "not_applicable",
      "artifact": null,
      "reason": "unsupported_exact_mode",
      "capture_method": "not-applicable",
      "platform": "windows"
    }
  }
}
```

HighGUI draws the operator overlay on a copy. Pixel parity therefore uses the
pre-overlay frame, while the physical review also confirms that the real
window presents the correct subject and generation. A screenshot with the
overlay is a review companion, not the metric-authoritative pre-overlay
artifact.

Windows native output supports only its advertised exact modes, currently
1280×720@30 and 1920×1080@30. Do not fabricate a 640×360 or 60 FPS native
capture. Cover those source/canvas/cadence observations in applicable cases
and retain a native boundary case at a supported exact mode. The report must
describe the limited route scope rather than generalize it.

## Boundary comparison gates

The plan records ratified maximum full-frame and subject-edge-band mean
absolute errors. Those values are evidence policy, not runtime tuning knobs.
For each boundary, the qualifier checks or reports:

- source and case identity;
- decoded dimensions and canonical canvas;
- fiducial/frame-identity agreement and `identity_region_mae`;
- full-frame BGR mean absolute error versus the in-memory reference;
- annotated edge-band BGR mean absolute error;
- exact equality for lossless in-process seams where required;
- independent agreement of the snapshot, MJPEG, and WebSocket JPEG decodes
  with the same held in-memory reference; and
- absence of an unexplained crop, pad, orientation change, or extra resize.

Whole-frame error alone is insufficient: a small shifted contour can disappear
inside a favorable average. Inspect edge-band error and fiducial geometry, and
reject a route that changes hair/accessory shape even if its global error stays
low. Codec tolerances must be ratified from the actual route; do not weaken the
matte gates to accommodate a poor consumer recording.

The bound candidate quality report must pass every applicable opaque-core,
background, hole, halo, ground-truth, registered-temporal, motion-trail,
fine-detail, and edge-colour absolute gate. The baseline is recomputed with the
same source and annotations to establish metric-by-metric non-regression and a
material improvement; it may intentionally reproduce the old defect. Boundary
pixels cannot rescue a failed candidate alpha report.

## Human review

Every case references a `custback.matte-visual-human-review` version-1
sidecar. Its bindings include the plan, both bundle manifests, both annotation
manifests, both recomputed quality-evidence digests, and boundary-evidence
digest. A review therefore cannot be moved to another baseline, candidate,
annotation set, quality result, or boundary capture after the fact. Choose one
declared method:

- `blinded`: randomize baseline/candidate labels before review and retain the
  mapping privately; record `assignment_sha256`, `reveal_sha256`, and
  `revealed_after_decisions: true` in the strict `blinding` object; or
- `side_by_side`: present synchronized, identically scaled baseline and
  candidate views, with `blinding: null`.

Record `reviewed_at` as an ISO-8601 UTC timestamp ending in `Z`. The blinded
assignment/reveal digests are audit anchors for the retained private records;
`revealed_after_decisions` is the reviewer's ordering attestation. The
qualifier checks distinct digests and that assertion, but does not open those
records or independently prove disclosure timing. Neither field makes a
generated review human evidence.

This is the complete strict blinded-review shape for that same case. The
review provenance reference must exactly equal the plan's private consent or
license reference. Compute `plan_sha256` and `boundary_evidence_sha256` from
the raw JSON bytes. Use each replay or annotation reader's verified manifest
digest for the four manifest fields. Run the same bound baseline and candidate
through `matte-evaluate` and copy each report's
`determinism.evidence_sha256`; the qualifier recomputes both values. Use
different real digests for the assignment and reveal records.

```json
{
  "schema": "custback.matte-visual-human-review",
  "version": 1,
  "case_id": "solid_color_1080p60",
  "provenance": {
    "kind": "consented-local",
    "reference": "REPLACE_WITH_PRIVATE_CONSENT_OR_LICENSE_RECORD_ID"
  },
  "bindings": {
    "plan_sha256": "REPLACE_WITH_PLAN_JSON_SHA256",
    "baseline_bundle_manifest_sha256": "REPLACE_WITH_BASELINE_MANIFEST_SHA256",
    "candidate_bundle_manifest_sha256": "REPLACE_WITH_CANDIDATE_MANIFEST_SHA256",
    "baseline_annotation_manifest_sha256": "REPLACE_WITH_BASELINE_ANNOTATION_MANIFEST_SHA256",
    "candidate_annotation_manifest_sha256": "REPLACE_WITH_CANDIDATE_ANNOTATION_MANIFEST_SHA256",
    "baseline_quality_evidence_sha256": "REPLACE_WITH_BASELINE_QUALITY_EVIDENCE_SHA256",
    "candidate_quality_evidence_sha256": "REPLACE_WITH_CANDIDATE_QUALITY_EVIDENCE_SHA256",
    "boundary_evidence_sha256": "REPLACE_WITH_BOUNDARY_SIDECAR_SHA256"
  },
  "method": "blinded",
  "blinding": {
    "assignment_sha256": "REPLACE_WITH_BLIND_ASSIGNMENT_SHA256",
    "reveal_sha256": "REPLACE_WITH_DISTINCT_REVEAL_SHA256",
    "revealed_after_decisions": true
  },
  "reviewer": "local-reviewer-01",
  "reviewed_at": "2026-08-10T12:00:00Z",
  "concerns": {
    "halo": "pass",
    "edge_shimmer": "pass",
    "ghost_trail": "pass",
    "cutout_sharpness": "pass",
    "hair_retention": "pass",
    "opaque_core_backdrop_leakage": "pass",
    "accessory_coverage": "pass",
    "motion_cadence": "pass"
  },
  "overall": "candidate_preferred",
  "notes": "REPLACE_WITH_BOUNDED_REVIEW_NOTES"
}
```

For a `side_by_side` review, set `method` to `side_by_side` and `blinding` to
`null`; every other key remains required.

Review the digest-bound, synchronized baseline and candidate replay sequences
for all representative motion segments at normal playback speed. The
`motion_cadence` decision must come from those sequences; boundary stills and
paused/contact-sheet inspection are supplements only. The report records that
review scope as `digest-bound-replay-clip-sequence`. Record one explicit
`pass` or `fail` outcome for each required concern key:

1. `halo`;
2. `edge_shimmer`;
3. `ghost_trail`;
4. `cutout_sharpness`;
5. `hair_retention`;
6. `opaque_core_backdrop_leakage`;
7. `accessory_coverage`;
8. `motion_cadence`.

The overall result is `candidate_preferred`, `candidate_acceptable`, or
`candidate_worse`. A worse candidate fails. An acceptable candidate still
must pass every numeric and coverage gate; human preference cannot waive a
hole, halo, leakage, cadence, or boundary failure. Aggregate qualification
also requires at least one `candidate_preferred` case; a matrix containing only
acceptable outcomes does not establish a material visual improvement.

## Evidence collection checklist

- [ ] Copy the template to an owner-controlled location and replace every
      placeholder.
- [ ] Record consent/license reference and a non-identifying qualification ID.
- [ ] Freeze the exact candidate and legitimate compatibility-baseline
      configurations and their code-resolved expected effective matte policies.
- [ ] Disable reactions and any unlisted post-base effect.
- [ ] Record model-backed full matte bundles and annotations for the complete
      taxonomy.
- [ ] Run `matte-evaluate`, applicable `matte-diagnose`/`matte-ablate`, and the
      RVM profile gate without projecting unavailable model evidence.
- [ ] Capture the held fiducial generation at every applicable boundary.
- [ ] Decode and inspect snapshot, MJPEG, and output WebSocket artifacts.
- [ ] Record real pyvirtualcam loopback and supported Windows native output.
- [ ] Perform the physical HighGUI and blinded/side-by-side review.
- [ ] Preserve the manifest-bound representative replay clips in the private
      evidence root. Optional contact sheets are review companions, not inputs
      the version-1 qualifier independently binds.
- [ ] Run `matte-visual-qualify`; retain its JSON and Markdown reports even
      when the status is pending or failed.
- [ ] Do not enable a preset or default until MATTE-5.3 qualifies the exact
      platform scope and the separate
      [MATTE-5.4 authority](matte-quality-rollout.md) advances from its current
      compatibility hold.

## Privacy and artifact handling

Raw frames, alpha masks, clean foregrounds, edge views, contact sheets,
loopback recordings, and even silhouettes can identify a person. Keep the
entire evidence tree outside the repository in a newly created owner-only
directory. Use owner-only files, avoid symlinks, and do not place private paths,
pixels, masks, model exceptions, or reviewer identity in public logs or
`/status`.

The checked-in template and final content-free summary may retain bounded IDs,
policy values, digests, numeric metrics, coverage outcomes, and review status.
They must retain `contains_private_footage_in_repository: false`. Do not upload
private evidence to CI artifacts. Generated CI evidence uses only repository-
owned arrays and fake runtimes and remains permanently non-authoritative.

Related contracts:

- [Matte replay bundles](matte-replay-bundle.md)
- [Matte quality metrics](matte-quality-metrics.md)
- [RVM profile qualification](matte-rvm-profiles.md)
- [Deterministic matte regression gate](matte-deterministic-regression-gate.md)
- [Matte performance qualification](matte-performance.md)
