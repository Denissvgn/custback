# Camera hardware-control characterization

- Status: Accepted preserve-only policy
- Date: 2026-07-30
- Backlog owner: VIS-3.4

## Decision

Custback observes a bounded set of camera properties once per successful
capture generation and never writes them. The only implemented hardware policy
is `preserve`.

No `lock_after_warmup` or `manual` policy is exposed. Generic OpenCV cannot
prove device support, ranges, units, or accepted values consistently enough
across V4L2, MSMF, and DSHOW. Its property path crosses the API backend,
operating-system driver, and device; a successful call or returned value is not
proof that the hardware accepted the same setting. Writing even the currently
reported value would therefore violate the side-effect-free capability-report
contract.

This result is deliberate: the software color harmonizer cannot form a fast
feedback loop with hardware exposure or white balance because it performs no
hardware writes.

## Runtime report

`CaptureHealth.camera_controls`, the frame-hub status, and the status API expose
one path-free object:

```json
{
  "policy": "preserve",
  "backend_family": "v4l2",
  "qualification": "unqualified",
  "writes_performed": false,
  "generation": 2,
  "properties": {
    "auto_white_balance": {"status": "reported", "value": 1.0},
    "white_balance_temperature": {
      "status": "indeterminate-zero",
      "value": 0.0
    },
    "auto_exposure": {"status": "reported", "value": 0.75},
    "exposure": {"status": "reported", "value": -5.0},
    "gain": {"status": "indeterminate-zero", "value": 0.0},
    "gamma": {"status": "unavailable", "value": null}
  }
}
```

The observations have intentionally narrow meanings:

- `reported`: OpenCV returned a finite non-zero value. The value is
  backend-defined and is not promoted to a portable unit or writable setting.
- `indeterminate-zero`: OpenCV returned zero. Zero is both OpenCV's
  unsupported-property sentinel and a valid value for several controls, so
  support is not guessed.
- `unavailable`: the OpenCV constant was absent, the backend raised while
  reading it, or it returned a non-finite value.

The report is read once after the first valid frame establishes a generation.
It carries that generation number, contains no device identifier or path, and
is replaced after reconnect rather than combined with stale observations.
Synthetic capture reports `qualification: not-applicable` with no properties.

## Backend characterization

| Backend family | Read behavior | Write qualification | Custback policy |
| --- | --- | --- | --- |
| V4L2 | Generic OpenCV values are observable, but it does not expose the V4L2 query-control flags, range, step, menu, or unit contract used to prove support. Auto-exposure encodings and exposure units remain driver/backend dependent at this boundary. | Unqualified | Preserve; observe only |
| MSMF | Generic OpenCV does not expose the Windows `MediaCapture` control capability/range objects used by native applications to establish support. | Unqualified | Preserve; observe only |
| DSHOW | Generic OpenCV property values remain driver and backend dependent and do not prove accepted device state. | Unqualified | Preserve; observe only |
| Other OpenCV backend | No backend-specific contract has been ratified. | Unqualified | Preserve; observe only |

A future write policy requires a separate backend adapter with authoritative
capability/range discovery, an opt-in restart-safe configuration, write
readback verification, representative-device evidence for that exact backend,
and a software-harmonizer reset after every confirmed transition. Until those
conditions are met, status must continue to say `unqualified`.

## Executable evidence

`tests/test_capture_geometry.py` proves:

- every requested property is read exactly once per capture generation;
- no requested control property is passed to `VideoCapture.set`;
- finite non-zero, ambiguous zero, and non-finite results retain their exact
  truthful states;
- V4L2, MSMF, and DSHOW are classified independently;
- reconnect replaces the report with the new generation; and
- synthetic capture remains explicitly not applicable.

The report is diagnostic evidence only. Live-device and cross-platform
qualification remains part of VIS-4.2 and cannot be inferred from unit-test
doubles.

## Primary external contracts

- OpenCV `VideoCaptureProperties`:
  <https://docs.opencv.org/master/d4/d15/group__videoio__flags__base.html>
- Linux V4L2 user controls:
  <https://docs.kernel.org/userspace-api/media/v4l/control.html>
- Linux V4L2 camera controls:
  <https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html>
- Windows manual capture controls:
  <https://learn.microsoft.com/windows/apps/develop/camera/capture-device-controls-for-photo-and-video-capture>
