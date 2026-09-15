# Matte configuration and rollback

Custback keeps compatibility defaults. Experimental profiles are explicit
choices and do not promise a particular image quality or sustainable frame rate.
See [system profiles](system-profiles.md) for selecting and restoring one.

## Backends and effective settings

| Backend | Behavior |
| --- | --- |
| RVM | Optional `rvm` (CPU) or `gpu` (CUDA) package extra; produces alpha and model foreground. |
| MediaPipe | Optional `mediapipe` extra; produces a confidence mask, not true alpha. |
| Heuristic | Core fallback; threshold applies to this backend. |
| Passthrough / remote | No local person matte; remote failure uses a fixed privacy slate. |

`segmentation.backend: auto` is a request. Inspect `segmentation_selection`,
the actual provider, and `matte_policy` in authenticated `GET /status` to see
what runs. Installing an optional backend can change what `auto` selects.
RVM and GPU extras are alternatives because their ONNX runtimes conflict.

Compatibility values include `rvm_downsample: 0.0`, `edge_refine: true`,
`mask_shift: 0`, `delegate: cpu`, spatial mode `legacy_watershed`, boundary
stabilization `off`, and light-wrap stabilization `off`. RVM neutralizes generic
mask-shift and edge-refinement controls that do not apply to its alpha output.
The resolved downsample ratio can differ from configured automatic selection.

`stable_guided` spatial refinement, `motion_aware` boundary stabilization, and
`temporal_bounded` light-wrap stabilization are opt-in controls. Use
[local diagnostics](matte-live-diagnostics.md) and change one option at a time.
The [operator guide](matte-operator-mitigations.md) provides apply, confirmation,
and rollback steps. A generated proxy does not establish real model behavior.

## Configuration and status

Versionless and explicit schema-1 configurations retain the same compatibility
semantics. Startup does not rewrite user files. Explicit `custback migrate --config PATH` writes atomically and retains a private backup. Runtime patches
apply transactionally; fetch the current config and status after any failure.

The `matte_rollout` status object reports the active compatibility decision,
experimental-profile state, and aggregate counters. Its sanitized rollout
telemetry contains no image data and is not a quality or hardware guarantee.
Reactions are outside matte-policy rollback; diagnose the base image with
reactions disabled. Avoid sharing full configs, device labels, tokens, or images.

The canonical compatibility patch digest is
`27638e419a0dcf5955d52e2eb4ead2dafbdca7f2bbe0535108aa7c56c1f2f60d`.

## One-patch rollback

`custback.config.legacy_matte_policy_patch()` is the code-owned schema-1
rollback authority. It returns a new detached merge patch on every call and
intentionally omits `schema_version`, model/cache paths, output, background,
API, and avatar settings. In JSON form the patch is:

```json
{
  "segmentation": {
    "backend": "auto",
    "delegate": "cpu",
    "rvm_downsample": 0.0,
    "threshold": 0.5,
    "mask_blur": 7,
    "edge_refine": true,
    "spatial_edge_refinement": {
      "mode": "legacy_watershed",
      "reference_short_edge_px": 720,
      "radius_at_reference_px": 8,
      "min_radius_px": 2,
      "max_radius_px": 12
    },
    "mask_shift": 0,
    "temporal_smoothing": 0.35,
    "boundary_stabilization": {
      "mode": "off",
      "time_constant_s": 0.1,
      "max_motion_px_per_s": 720.0
    }
  },
  "acceleration": {
    "mode": "auto",
    "provider": "auto",
    "device_id": 0
  },
  "compositing": {
    "light_wrap": 0.25,
    "use_model_foreground": true,
    "blend_space": "srgb_legacy",
    "light_wrap_stabilization": {
      "mode": "off",
      "time_constant_s": 0.12
    },
    "color_correction": {
      "mode": "off",
      "strength": 0.5,
      "exposure_limit_ev": 0.85,
      "white_balance_strength": 0.5,
      "adaptation_time_s": 0.8
    }
  }
}
```

Apply this object once through the authenticated transactional
`PATCH /config` API. Use `application/merge-patch+json`, not a sequence of
individual changes. A successful activation commits one new config version;
segmentation policy/backend changes rebuild and reset the matte timeline at
one generation boundary. A `409`, `422`, `503`, or activation timeout applies
nothing and keeps the previous live generation.

After success, fetch both endpoints again and verify:

- `/config` contains the complete patch and returns the new
  `X-Config-Version`;
- `/status.config_version` catches up to that same version;
- `matte_rollout` identifies the legacy rollback patch and records one
  successful rollback;
- `segmentation_selection` truthfully shows the backend/provider actually
  selected by `auto`, including any fallback;
- `matte_policy` reports schema-1 effective controls for that selected path;
- the segmentation generation and reset count advance together; and
- the first authoritative output is fresh, with no cross-generation flash.

The rollback does not lower the recognized schema, delete the user config,
remove an installed optional backend, or delete/redownload the model cache.
Because `auto` considers installed backends, the selected runtime tier after
rollback may differ from a host that never installed RVM; that fact must remain
visible in status. If package intent also needs to return to an earlier set of
extras, first complete the policy rollback, then run a separately reviewed
`custback rebuild --extras LIST`. Rebuild stages a candidate environment and
must leave the current runtime usable on failure. It still does not require
deleting the config or model cache.

Persist the rollback values to the operator-owned YAML only after the live
rollback succeeds and the result has been reviewed. Do not delete the YAML or
cache as a substitute for an explicit policy decision.
