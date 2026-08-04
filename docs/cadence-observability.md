# Visual cadence observability

Status: **MATTE-3.2 live status contract, version 1**

Custback has separate capture, segmentation, safe-base, optional post-base, and
output-send clocks. A near-target output FPS can therefore be made from repeated
safe bases while the camera, matte, and base composite update more slowly.
`GET /status`, the preview HUD, and the ready/shutdown records expose those
clocks separately. Final-output equality never reclassifies a unique capture
or base update; exact final-output repeats are a separate byte-equality signal.

The contract is scalar-only and constant-space. It contains no pixels, masks,
frame-derived hashes, wall-clock recording timestamps, device paths, or raw
monotonic timestamps.

## Counts and compatibility aliases

Run counters include the successful startup/preflight send. They advance only
at their named boundary:

| Status field | Boundary |
| --- | --- |
| `segmentation_update_count` | Successful authoritative segmentation of a unique capture. Candidate hot-activation trials are excluded. |
| `base_composite_update_count` | A unique accepted capture produced the safe base for a successful send. Pixel equality with an earlier capture does not turn it into reuse. |
| `base_composite_reuse_count` | A successful output opportunity found no unread capture and reused the current safe base. |
| `exact_final_output_repeat_count` | A successful final frame was byte-identical to the immediately preceding successful final frame. |
| `output_send_count` | Successful sink sends, whether they carry a new base or a reuse. |

Existing fields retain their compatibility meaning:

```text
frames_in              == base_composite_update_count
frames_out             == output_send_count
output_repeated_frames == base_composite_reuse_count
```

`capture_frames_read` counts successful capture-worker publications before
pipeline consumption. `capture_dropped_frames` counts latest-slot overwrites.
`capture_sequence_gap_count` counts accepted-sequence gap events, while
`capture_missing_input_count` sums the missing sequence values. An overwrite,
sequence gap, processing deadline miss, or output pacing event is correlated
evidence—not proof that it caused an application reuse.

The corresponding rolling rates are:

- `segmentation_update_fps`;
- `base_composite_update_fps`;
- `base_composite_reuse_fps`;
- `exact_final_output_repeat_fps`; and
- `output_send_fps`.

The cadence horizon is 2.0 seconds with a hard cap of 1,024 send events.
Interval summaries use at most the latest 256 derived intervals and
nearest-rank p50/p95. These bounds remain in force even if an injected source
or sink runs faster than its configured target.

Rolling send/base/segmentation/reuse/exact rates share the successful-send
window span. The first retained send establishes the interval baseline; event
classifications on the remaining sends are divided by that common span. This
makes visual-update FPS decay during repeat-only output instead of preserving
the last active update rate.

`base_composite_reuse_ratio` uses safe-base reuses over all successful sends.
`exact_final_output_repeat_ratio` uses exact repeats over comparable
consecutive-send transitions (the first send has no predecessor). With no
optional post-base stage, a safe-base reuse necessarily produces an exact
final-output repeat. The reverse is not true: a new pixel-identical capture
still increments the unique base and segmentation clocks and may also
increment the exact-repeat clock. A later effect can change final pixels while
the same base is reused, so the two fields must not be collapsed.

`cadence_mismatch_active` is the bounded rolling health decision used by both
operator UIs. When active and its rates are available, the preview warns:

```text
VISUAL UPDATES 15 FPS; OUTPUT REPEATS TO 30 FPS
```

## Relative timing and interval summaries

`last_unique_frame_age_ms` is derived from the process-monotonic instant when
the latest unique safe base finished guard/validation. It therefore measures
the age of the visual update that can actually be sent and continues to
increase across output reuses. It is distinct from `capture_frame_age_ms`,
which describes the capture worker's latest successful publication.

Only bounded interval summaries are public:

| Fields | Meaning |
| --- | --- |
| `capture_timestamp_delta_p50_ms`, `capture_timestamp_delta_p95_ms` | Deltas between successive accepted unique captures, using their carried capture-completion timestamps. Sequence gaps remain visible separately. |
| `output_send_delta_p50_ms`, `output_send_delta_p95_ms` | Deltas between successful sink-acceptance/submission events, before deliberate sink pacing. |
| `output_send_jitter_p50_ms`, `output_send_jitter_p95_ms` | Absolute send-interval error against the configured output interval. |
| `base_composite_delta_p50_ms`, `base_composite_delta_p95_ms` | Inter-arrival deltas between guarded/validated new safe bases, independent of later output pacing. |

Summaries remain unavailable until the bounded window has enough samples. No
raw timestamp is serialized.

## Deadlines, pacing, and recovery

The event classes are independent:

- `processing_deadline_misses` retains its compatibility scope: processing
  from the accepted frame through base construction, before the final guard,
  sink submission, and deliberate pacing;
- `serialized_new_frame_deadline_misses` covers the complete serialized
  unique-frame path through guard/validation, sink submission/copy, and
  application- or sink-owned pacing;
- `output_sink_pacing_events` counts successful sink-owned pacing events;
- `application_pacing_events` counts application-owned waits for sinks that do
  not pace themselves;
- `output_schedule_late_events` counts missed output schedule opportunities;
  and
- `output_sink_recovery_events` is reserved for explicit, successful sink
  recovery—not startup fallback and not an application repeat.

`timing_schema_version` versions the fixed `timing_ms` object. Each value is a
bounded scalar summary for one documented boundary; unavailable measurements
are null rather than inferred. The existing compatibility timing fields remain
available. In particular, `frame_processing_ms` is narrower than the complete
serialized new-frame duration. RVM and compositor substage owners publish only
inside their allocated timing namespaces and cannot relabel core boundaries.

Version 1 reserves these exact dotted keys:

| `timing_ms` key | Boundary |
| --- | --- |
| `capture.read` | Capture-worker blocking source read; device pacing, transfer, and backend decode are not separable at the OpenCV boundary. |
| `segmentation.total` | Complete authoritative segmentation/refinement stage. |
| `segmentation.preprocess`, `segmentation.inference`, `segmentation.postprocess` | Reserved backend-owned segmentation substages; null when the selected backend cannot report them. |
| `background.total` | Backdrop selection/decode/normalization used by the new base. |
| `color_correction.total` | Foreground/backdrop color analysis and preparation on the frame lane. |
| `compositor.total` | Complete base compositor stage. |
| `compositor.prepare`, `compositor.blend` | Compositor-owned preparation (including prepared light wrap) and render work for a newly processed local frame. They remain null until such a frame has been observed and are zero for an explicit local bypass. |
| `output.send_total` | Complete sink call, including sink-owned deliberate pacing where applicable. |
| `output.submission` | Sink validation/submission/copy before sink-owned pacing. |
| `output.sink_pacing_wait` | Deliberate wait owned by a pacing sink. |
| `output.application_pacing_wait` | Deliberate application wait used only for a non-pacing sink. |
| `output.schedule_lateness` | Nonnegative lateness against the ideal sink-submission deadline. |
| `pipeline.processing_only` | Compatibility processing boundary ending before final guard/validation and output. |
| `pipeline.new_frame_service` | Starts after any application-owned pacing wait, immediately before the successful capture read/poll, and ends at sink submission. |
| `pipeline.new_frame_serialized_loop` | Starts before any application-owned wait for that cycle and ends at sink completion, including capture/poll, final guard/validation, output work, and application- or sink-owned pacing. |

## Post-base extension isolation

Optional output-tick stages publish cadence provenance only under
`extensions.post_base`. The extension is versioned, typed, size-bounded, and
restricted to sanitized stage namespaces and scalar counters/rates. It cannot
replace or merge into capture, segmentation, matte, base-update, reuse,
exact-repeat, or send fields.

This lets a future reaction stage report output-only visual changes while a
15 FPS safe base is reused at a 30 FPS send cadence. Such changes never raise
`base_composite_update_fps` or `segmentation_update_fps`.

## Operator query

After allowing the rolling window to warm, inspect:

```bash
auth_header | curl --config - http://127.0.0.1:8710/status |
  jq '{
    capture: {
      fps: .capture_fps,
      sequence: .capture_sequence,
      gap_events: .capture_sequence_gap_count,
      missing: .capture_missing_input_count,
      overwritten: .capture_dropped_frames
    },
    updates: {
      segmentation_count: .segmentation_update_count,
      segmentation_fps: .segmentation_update_fps,
      base_count: .base_composite_update_count,
      base_fps: .base_composite_update_fps,
      last_unique_age_ms: .last_unique_frame_age_ms
    },
    reuse: {
      base_count: .base_composite_reuse_count,
      base_fps: .base_composite_reuse_fps,
      base_ratio: .base_composite_reuse_ratio,
      exact_final_count: .exact_final_output_repeat_count,
      exact_final_fps: .exact_final_output_repeat_fps,
      exact_final_ratio: .exact_final_output_repeat_ratio
    },
    output: {
      sends: .output_send_count,
      send_fps: .output_send_fps,
      sink_pacing: .output_sink_pacing_events,
      application_pacing: .application_pacing_events,
      schedule_late: .output_schedule_late_events,
      sink_recovery: .output_sink_recovery_events
    },
    deadlines: {
      processing: .processing_deadline_misses,
      serialized_new_frame: .serialized_new_frame_deadline_misses
    },
    mismatch: .cadence_mismatch_active,
    timing_schema_version,
    timing_ms,
    post_base: .extensions.post_base
  }'
```

Do not divide lifetime counts by `uptime_s` to recreate rolling rates. Use the
published rates, interval summaries, and a strict two-snapshot sidecar when
comparing against the standalone
[capture-only diagnostic](capture-cadence-diagnostics.md).
