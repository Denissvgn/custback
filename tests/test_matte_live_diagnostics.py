"""Deterministic contracts for private, native matte diagnostic views."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.matte_live_diagnostics as matte_live_mod
from custback.compositor import PreparedLightWrap, composite
from custback.matte_diagnostics import MatteCaptureMetadata, MatteFrameEvidence
from custback.matte_live_diagnostics import (
    DiagnosticView,
    LocalMatteDiagnosticFrame,
    LocalMatteDiagnosticMonitor,
    diagnostic_view_names,
)


def _evidence(
    sequence: int,
    *,
    source: np.ndarray,
    raw_alpha: np.ndarray | None = None,
    refined_alpha: np.ndarray | None = None,
    foreground: np.ndarray | None = None,
    backdrop: np.ndarray | None = None,
    base_composite: np.ndarray | None = None,
    prepared_light_wrap: PreparedLightWrap | None = None,
    capture_monotonic_ns: int | None = None,
    capture_generation: int = 2,
    geometry_generation: int = 3,
) -> MatteFrameEvidence:
    return MatteFrameEvidence(
        metadata=MatteCaptureMetadata(
            bundle_sequence=sequence,
            capture_sequence=sequence,
            capture_monotonic_ns=(
                sequence * 33_333_333
                if capture_monotonic_ns is None
                else capture_monotonic_ns
            ),
            timestamp_source="test",
            capture_generation=capture_generation,
            geometry_generation=geometry_generation,
        ),
        raw_frame=source,
        raw_mask=raw_alpha,
        refined_mask=refined_alpha,
        clean_foreground=foreground,
        backdrop_frame=backdrop,
        base_composite=base_composite,
        prepared_light_wrap=prepared_light_wrap,
        effective_controls={
            "blend_space": "srgb_legacy",
            "light_wrap": 0.35,
            "use_model_foreground": foreground is not None,
        },
        timings_ms={
            "backend_inference_ms": 4.25,
            "refinement_ms": 0.5,
        },
        compositor_substages_ms={
            "foreground_selection_ms": 0.1,
            "light_wrap_ms": 0.2,
        },
        segmentation_diagnostics={"backend": "test"},
    )


def _complete_tracks(
    height: int = 16,
    width: int = 20,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    source = (
        np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3) % 251
    ).astype(np.uint8)
    raw_alpha = np.linspace(
        0.0,
        1.0,
        height * width,
        dtype=np.float32,
    ).reshape(height, width)
    refined_alpha = np.clip(raw_alpha * 0.9 + 0.05, 0.0, 1.0).astype(np.float32)
    foreground = np.ascontiguousarray(np.flip(source, axis=1))
    backdrop = np.full_like(source, 37)
    base = composite(
        source,
        backdrop,
        refined_alpha,
        edge_foreground=foreground,
        light_wrap=0.35,
    )
    return source, raw_alpha, refined_alpha, foreground, backdrop, base


def _render(
    view: DiagnosticView,
    evidence: MatteFrameEvidence,
    *,
    status: Mapping[str, object] | None = None,
) -> LocalMatteDiagnosticFrame:
    monitor = LocalMatteDiagnosticMonitor()
    try:
        monitor.select(view)
        assert monitor.submit(
            evidence,
            status=(
                {"segmentation_generation": 7, "matte_reset_count": 2}
                if status is None
                else status
            ),
        )
        frame, _sequence = monitor.get(timeout=2.0)
        assert frame is not None
        return frame
    finally:
        monitor.close()


def _nonzero_support(pixels: np.ndarray) -> np.ndarray:
    return np.any(pixels != 0, axis=2)


def test_diagnostic_catalogue_contains_every_required_view_in_cycle_order() -> None:
    assert diagnostic_view_names() == (
        "raw_camera",
        "raw_alpha",
        "refined_alpha",
        "clean_foreground",
        "backdrop",
        "alpha_over_source",
        "uncertain_boundary",
        "opaque_core_deficit",
        "foreground_holes",
        "exterior_halo",
        "model_foreground_only",
        "light_wrap_only",
        "final_composite_contribution",
        "instability",
    )


def test_every_diagnostic_view_has_a_detached_bgr8_frame() -> None:
    tracks = _complete_tracks()
    source, raw_alpha, refined_alpha, foreground, backdrop, base = tracks
    inputs = tracks
    monitor = LocalMatteDiagnosticMonitor()
    last_sequence = -1
    try:
        for capture_sequence, view in enumerate(diagnostic_view_names(), start=1):
            assert monitor.select(view) == view
            assert monitor.submit(
                _evidence(
                    capture_sequence,
                    source=source,
                    raw_alpha=raw_alpha,
                    refined_alpha=refined_alpha,
                    foreground=foreground,
                    backdrop=backdrop,
                    base_composite=base,
                ),
                status={"segmentation_generation": 7, "matte_reset_count": 2},
            )
            frame, last_sequence = monitor.get(
                last_sequence=last_sequence,
                timeout=2.0,
            )

            assert frame is not None
            assert frame.view == view
            assert frame.available
            assert frame.unavailable_reason == ""
            assert frame.pixels.shape == source.shape
            assert frame.pixels.dtype == np.uint8
            assert frame.pixels.flags.c_contiguous
            assert not frame.pixels.flags.writeable
            assert all(
                not np.shares_memory(frame.pixels, input_array)
                for input_array in inputs
            )
    finally:
        monitor.close()


def test_raw_and_refined_alpha_views_map_values_exactly_to_gray_bgr() -> None:
    source = np.zeros((2, 4, 3), dtype=np.uint8)
    raw_alpha = np.asarray(
        [[0.0, 0.1, 0.5, 1.0], [0.05, 0.501, 0.95, 0.999]],
        dtype=np.float32,
    )
    refined_alpha = np.flip(raw_alpha, axis=1).copy()
    evidence = _evidence(
        1,
        source=source,
        raw_alpha=raw_alpha,
        refined_alpha=refined_alpha,
    )

    raw_frame = _render("raw_alpha", evidence)
    refined_frame = _render("refined_alpha", evidence)

    expected_raw = np.repeat(
        np.rint(raw_alpha * 255.0).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    expected_refined = np.repeat(
        np.rint(refined_alpha * 255.0).astype(np.uint8)[..., None],
        3,
        axis=2,
    )
    np.testing.assert_array_equal(raw_frame.pixels, expected_raw)
    np.testing.assert_array_equal(refined_frame.pixels, expected_refined)


def test_exact_raster_tracks_and_alpha_over_source_pixels() -> None:
    source, raw_alpha, refined_alpha, foreground, backdrop, base = _complete_tracks()
    evidence = _evidence(
        1,
        source=source,
        raw_alpha=raw_alpha,
        refined_alpha=refined_alpha,
        foreground=foreground,
        backdrop=backdrop,
        base_composite=base,
    )

    raw_camera = _render("raw_camera", evidence)
    clean_foreground = _render("clean_foreground", evidence)
    exact_backdrop = _render("backdrop", evidence)
    alpha_over_source = _render("alpha_over_source", evidence)

    np.testing.assert_array_equal(raw_camera.pixels, source)
    np.testing.assert_array_equal(clean_foreground.pixels, foreground)
    np.testing.assert_array_equal(exact_backdrop.pixels, backdrop)
    alpha_gray = np.rint(refined_alpha * 255.0).astype(np.uint8)
    alpha_heat = cv2.applyColorMap(alpha_gray, cv2.COLORMAP_TURBO)
    expected_overlay = np.rint(
        source.astype(np.float32) * 0.45 + alpha_heat.astype(np.float32) * 0.55
    ).astype(np.uint8)
    np.testing.assert_array_equal(alpha_over_source.pixels, expected_overlay)


def test_uncertain_boundary_uses_strict_alpha_thresholds() -> None:
    source = np.zeros((1, 7, 3), dtype=np.uint8)
    alpha = np.asarray(
        [[0.0, 0.05, 0.051, 0.5, 0.949, 0.95, 1.0]],
        dtype=np.float32,
    )

    frame = _render(
        "uncertain_boundary",
        _evidence(1, source=source, refined_alpha=alpha),
    )

    expected_support = np.asarray([[False, False, True, True, True, False, False]])
    np.testing.assert_array_equal(_nonzero_support(frame.pixels), expected_support)


@pytest.mark.parametrize(
    ("view", "reason"),
    [
        ("clean_foreground", "did not provide a clean foreground"),
        ("model_foreground_only", "did not provide a clean foreground"),
    ],
)
def test_rvm_foreground_views_are_honestly_unavailable(
    view: DiagnosticView,
    reason: str,
) -> None:
    source, raw_alpha, refined_alpha, _foreground, backdrop, base = _complete_tracks()
    frame = _render(
        view,
        _evidence(
            1,
            source=source,
            raw_alpha=raw_alpha,
            refined_alpha=refined_alpha,
            backdrop=backdrop,
            base_composite=base,
        ),
    )

    assert not frame.available
    assert reason in frame.unavailable_reason
    np.testing.assert_array_equal(
        frame.pixels,
        np.full(source.shape, 24, dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("view", "alpha", "expected_support"),
    [
        (
            "opaque_core_deficit",
            np.pad(
                np.full((5, 5), 0.75, dtype=np.float32),
                2,
            ),
            np.pad(
                np.ones((3, 3), dtype=bool),
                3,
            ),
        ),
        (
            "foreground_holes",
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.pad(
                np.ones((5, 5), dtype=bool),
                2,
            ),
        ),
        (
            "exterior_halo",
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0, 0],
                    [0, 0, 0.25, 1, 1, 1, 0.25, 0, 0],
                    [0, 0, 0.25, 1, 1, 1, 0.25, 0, 0],
                    [0, 0, 0.25, 1, 1, 1, 0.25, 0, 0],
                    [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=bool,
            ),
        ),
    ],
)
def test_proxy_heatmaps_mark_only_the_inferred_support(
    view: DiagnosticView,
    alpha: np.ndarray,
    expected_support: np.ndarray,
) -> None:
    source = np.zeros((*alpha.shape, 3), dtype=np.uint8)

    frame = _render(
        view,
        _evidence(1, source=source, refined_alpha=alpha),
    )

    np.testing.assert_array_equal(
        _nonzero_support(frame.pixels),
        expected_support,
    )


def test_exterior_halo_excludes_enclosed_soft_foreground_holes() -> None:
    alpha = np.zeros((11, 11), dtype=np.float32)
    alpha[1:10, 1:10] = 0.25
    alpha[2:9, 2:9] = 1.0
    alpha[4:7, 4:7] = 0.25
    source = np.zeros((*alpha.shape, 3), dtype=np.uint8)

    holes = _render(
        "foreground_holes",
        _evidence(1, source=source, refined_alpha=alpha),
    )
    halo = _render(
        "exterior_halo",
        _evidence(2, source=source, refined_alpha=alpha),
    )
    hole_support = _nonzero_support(holes.pixels)
    halo_support = _nonzero_support(halo.pixels)

    assert bool(np.any(hole_support))
    assert bool(np.any(halo_support))
    assert not bool(np.any(hole_support & halo_support))
    assert bool(np.all(hole_support[4:7, 4:7]))
    assert not bool(np.any(halo_support[4:7, 4:7]))


def test_submission_freezes_inputs_and_publishes_read_only_detached_pixels() -> None:
    source, raw_alpha, refined_alpha, foreground, backdrop, base = _complete_tracks()
    originals = tuple(
        array.copy()
        for array in (
            source,
            raw_alpha,
            refined_alpha,
            foreground,
            backdrop,
            base,
        )
    )
    monitor = LocalMatteDiagnosticMonitor()
    try:
        monitor.select("raw_camera")
        assert monitor.submit(
            _evidence(
                1,
                source=source,
                raw_alpha=raw_alpha,
                refined_alpha=refined_alpha,
                foreground=foreground,
                backdrop=backdrop,
                base_composite=base,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 2,
                "config_version": 11,
            },
        )
        frame, _sequence = monitor.get(timeout=2.0)
        assert frame is not None

        np.testing.assert_array_equal(frame.pixels, originals[0])
        for input_array, original in zip(
            (source, raw_alpha, refined_alpha, foreground, backdrop, base),
            originals,
            strict=True,
        ):
            np.testing.assert_array_equal(input_array, original)
            assert not np.shares_memory(frame.pixels, input_array)
        assert not frame.pixels.flags.writeable
    finally:
        monitor.close()


def test_counterfactual_views_hold_inputs_and_separate_compositor_features() -> None:
    source, raw_alpha, refined_alpha, foreground, backdrop, base = _complete_tracks()
    evidence = _evidence(
        1,
        source=source,
        raw_alpha=raw_alpha,
        refined_alpha=refined_alpha,
        foreground=foreground,
        backdrop=backdrop,
        base_composite=base,
    )

    model_only = _render("model_foreground_only", evidence)
    wrap_only = _render("light_wrap_only", evidence)
    final_delta = _render("final_composite_contribution", evidence)

    plain = composite(source, backdrop, refined_alpha)
    np.testing.assert_array_equal(
        model_only.pixels,
        composite(
            source,
            backdrop,
            refined_alpha,
            edge_foreground=foreground,
        ),
    )
    np.testing.assert_array_equal(
        wrap_only.pixels,
        composite(
            source,
            backdrop,
            refined_alpha,
            light_wrap=0.35,
        ),
    )
    magnitude = (
        np.mean(
            np.abs(base.astype(np.float32) - plain.astype(np.float32)),
            axis=2,
            dtype=np.float32,
        )
        / 255.0
    )
    expected_delta = cv2.applyColorMap(
        np.rint(np.clip(magnitude, 0.0, 1.0) * 255.0).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    np.testing.assert_array_equal(final_delta.pixels, expected_delta)
    assert not np.array_equal(model_only.pixels, wrap_only.pixels)


def test_light_wrap_view_uses_the_exact_stabilized_prepared_sample() -> None:
    source, raw_alpha, refined_alpha, foreground, backdrop, _base = _complete_tracks()
    prepared_pixels = np.full(source.shape, 211.0, dtype=np.float32)
    prepared_pixels.setflags(write=False)
    prepared = PreparedLightWrap(
        pixels_bgr=prepared_pixels,
        blend_space="srgb_legacy",
        stabilized=True,
    )
    base = composite(
        source,
        backdrop,
        refined_alpha,
        edge_foreground=foreground,
        light_wrap=0.35,
        prepared_light_wrap=prepared,
    )
    evidence = _evidence(
        1,
        source=source,
        raw_alpha=raw_alpha,
        refined_alpha=refined_alpha,
        foreground=foreground,
        backdrop=backdrop,
        base_composite=base,
        prepared_light_wrap=prepared,
    )

    wrap_only = _render("light_wrap_only", evidence)
    expected = composite(
        source,
        backdrop,
        refined_alpha,
        light_wrap=0.35,
        prepared_light_wrap=prepared,
    )

    np.testing.assert_array_equal(wrap_only.pixels, expected)
    assert not np.array_equal(
        wrap_only.pixels,
        composite(
            source,
            backdrop,
            refined_alpha,
            light_wrap=0.35,
        ),
    )
    assert not np.shares_memory(wrap_only.pixels, prepared_pixels)


def _instability_pair(
    *,
    source: np.ndarray,
    first_alpha: np.ndarray,
    second_alpha: np.ndarray,
    first_foreground: np.ndarray | None,
    second_foreground: np.ndarray | None,
    first_backdrop: np.ndarray | None = None,
    second_backdrop: np.ndarray | None = None,
) -> LocalMatteDiagnosticFrame:
    monitor = LocalMatteDiagnosticMonitor()
    first_backdrop = (
        np.full_like(source, 19) if first_backdrop is None else first_backdrop
    )
    second_backdrop = first_backdrop if second_backdrop is None else second_backdrop
    status = {"segmentation_generation": 7, "matte_reset_count": 2}
    try:
        monitor.select("instability")
        assert monitor.submit(
            _evidence(
                1,
                source=source,
                raw_alpha=first_alpha,
                refined_alpha=first_alpha,
                foreground=first_foreground,
                backdrop=first_backdrop,
                base_composite=composite(
                    source,
                    first_backdrop,
                    first_alpha,
                    edge_foreground=first_foreground,
                    light_wrap=0.35,
                ),
            ),
            status=status,
        )
        first, sequence = monitor.get(timeout=2.0)
        assert first is not None
        assert monitor.submit(
            _evidence(
                2,
                source=source,
                raw_alpha=second_alpha,
                refined_alpha=second_alpha,
                foreground=second_foreground,
                backdrop=second_backdrop,
                base_composite=composite(
                    source,
                    second_backdrop,
                    second_alpha,
                    edge_foreground=second_foreground,
                    light_wrap=0.35,
                ),
            ),
            status=status,
        )
        second, _sequence = monitor.get(sequence, timeout=2.0)
        assert second is not None
        return second
    finally:
        monitor.close()


def test_instability_channels_separate_alpha_from_downstream_edge_colour() -> None:
    rng = np.random.default_rng(9482)
    source = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    first_alpha = np.full((64, 64), 0.4, dtype=np.float32)
    moved_alpha = first_alpha.copy()
    moved_alpha[20:44, 20:44] = 0.8
    stable_foreground = np.full_like(source, 70)

    alpha_only = _instability_pair(
        source=source,
        first_alpha=first_alpha,
        second_alpha=moved_alpha,
        first_foreground=stable_foreground,
        second_foreground=stable_foreground,
    )

    assert alpha_only.available
    assert int(alpha_only.pixels[..., 2].max()) > 0
    assert int(alpha_only.pixels[..., :2].max()) == 0
    assert alpha_only.temporal.edge_colour_abs_diff == 0.0

    edge_only = _instability_pair(
        source=source,
        first_alpha=first_alpha,
        second_alpha=first_alpha,
        first_foreground=np.full_like(source, 30),
        second_foreground=np.full_like(source, 220),
    )

    assert edge_only.available
    assert int(edge_only.pixels[..., 2].max()) <= 1
    assert int(edge_only.pixels[..., 0].max()) > 0
    np.testing.assert_array_equal(
        edge_only.pixels[..., 0],
        edge_only.pixels[..., 1],
    )
    assert edge_only.temporal.edge_colour_abs_diff is not None
    assert edge_only.temporal.edge_colour_abs_diff > 0.0


def test_edge_colour_channel_retains_stateless_wrap_motion() -> None:
    rng = np.random.default_rng(4502)
    source = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    alpha = np.full((64, 64), 0.5, dtype=np.float32)
    first_backdrop = np.zeros_like(source)
    first_backdrop[:, :32] = (20, 40, 80)
    second_backdrop = np.zeros_like(source)
    second_backdrop[:, :32] = (180, 210, 240)

    frame = _instability_pair(
        source=source,
        first_alpha=alpha,
        second_alpha=alpha,
        first_foreground=None,
        second_foreground=None,
        first_backdrop=first_backdrop,
        second_backdrop=second_backdrop,
    )

    assert frame.available
    assert frame.temporal.edge_colour_state == "ready"
    assert frame.temporal.edge_colour_abs_diff is not None
    assert frame.temporal.edge_colour_abs_diff > 0.0
    assert int(frame.pixels[..., 2].max()) <= 1
    assert int(frame.pixels[..., 0].max()) > 0


def test_low_texture_registration_is_explicitly_unavailable() -> None:
    source = np.full((64, 64, 3), 127, dtype=np.uint8)
    first_alpha = np.zeros((64, 64), dtype=np.float32)
    second_alpha = first_alpha.copy()
    second_alpha[20:44, 20:44] = 1.0

    frame = _instability_pair(
        source=source,
        first_alpha=first_alpha,
        second_alpha=second_alpha,
        first_foreground=None,
        second_foreground=None,
    )

    assert not frame.available
    assert frame.temporal.history_state == "ready"
    assert frame.temporal.registration_state == "low-confidence"
    assert frame.temporal.registration_response is not None
    assert frame.temporal.registration_response < 0.10
    assert frame.temporal.registration_dx_px is None
    assert frame.temporal.registration_dy_px is None
    assert frame.temporal.raw_alpha_abs_diff is not None
    assert frame.temporal.raw_alpha_compensated_abs_diff is None
    assert frame.temporal.refined_alpha_abs_diff is not None
    assert frame.temporal.refined_alpha_compensated_abs_diff is None
    assert "low confidence" in frame.unavailable_reason


def test_temporal_metrics_compensate_translation_and_reset_history() -> None:
    rng = np.random.default_rng(1234)
    height = width = 64
    source = rng.integers(
        0,
        256,
        size=(height, width, 3),
        dtype=np.uint8,
    )
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[17:46, 19:43] = 0.4
    alpha[18:45, 20:42] = 1.0
    translation = np.asarray(
        [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    shifted_source = cv2.warpAffine(
        source,
        translation,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    shifted_alpha = cv2.warpAffine(
        alpha,
        translation,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    backdrop = np.zeros_like(source)
    monitor = LocalMatteDiagnosticMonitor()
    last_sequence = -1
    try:
        monitor.select("instability")
        assert monitor.submit(
            _evidence(
                10,
                source=source,
                raw_alpha=alpha,
                refined_alpha=alpha,
                backdrop=backdrop,
                base_composite=source,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 2,
                "config_version": 11,
            },
        )
        first, last_sequence = monitor.get(last_sequence, timeout=2.0)
        assert first is not None
        assert first.temporal.history_state == "warming"
        assert not first.available

        assert monitor.submit(
            _evidence(
                11,
                source=shifted_source,
                raw_alpha=shifted_alpha,
                refined_alpha=shifted_alpha,
                backdrop=backdrop,
                base_composite=shifted_source,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 2,
                "config_version": 11,
            },
        )
        second, last_sequence = monitor.get(last_sequence, timeout=2.0)
        assert second is not None
        temporal = second.temporal
        assert second.available
        assert temporal.history_state == "ready"
        assert temporal.capture_sequence_delta == 1
        assert temporal.capture_timestamp_delta_ms == pytest.approx(33.333333)
        assert temporal.registration_state == "ready"
        assert temporal.registration_response is not None
        assert temporal.registration_response >= 0.1
        assert temporal.registration_overlap_fraction is not None
        assert temporal.registration_overlap_fraction < 1.0
        assert temporal.registration_dx_px == pytest.approx(3.0, abs=0.1)
        assert temporal.registration_dy_px == pytest.approx(2.0, abs=0.1)
        assert temporal.raw_alpha_abs_diff is not None
        assert temporal.raw_alpha_compensated_abs_diff is not None
        assert temporal.refined_alpha_abs_diff is not None
        assert temporal.refined_alpha_compensated_abs_diff is not None
        assert temporal.raw_alpha_compensated_abs_diff < temporal.raw_alpha_abs_diff
        assert (
            temporal.refined_alpha_compensated_abs_diff
            < temporal.refined_alpha_abs_diff
        )
        assert not bool(np.any(second.pixels[:2, :, 2]))
        assert not bool(np.any(second.pixels[:, :3, 2]))

        assert monitor.submit(
            _evidence(
                12,
                source=shifted_source,
                raw_alpha=shifted_alpha,
                refined_alpha=shifted_alpha,
                backdrop=backdrop,
                base_composite=shifted_source,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 3,
                "config_version": 11,
            },
        )
        reset, last_sequence = monitor.get(last_sequence, timeout=2.0)
        assert reset is not None
        assert reset.temporal.history_state == "reset"
        assert reset.temporal.capture_sequence_delta == 1
        assert reset.temporal.raw_alpha_abs_diff is None
        assert reset.temporal.refined_alpha_abs_diff is None
        assert not reset.available

        monitor.clear_history()
        assert monitor.accepting
        assert monitor.selected_view == "instability"
        assert monitor.latest() is None
        assert monitor.submit(
            _evidence(
                13,
                source=shifted_source,
                raw_alpha=shifted_alpha,
                refined_alpha=shifted_alpha,
                backdrop=backdrop,
                base_composite=shifted_source,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 3,
                "config_version": 11,
            },
        )
        warmed, last_sequence = monitor.get(last_sequence, timeout=2.0)
        assert warmed is not None
        assert warmed.temporal.history_state == "warming"
        assert not warmed.available

        assert monitor.submit(
            _evidence(
                14,
                source=shifted_source,
                raw_alpha=shifted_alpha,
                refined_alpha=shifted_alpha,
                backdrop=backdrop,
                base_composite=shifted_source,
            ),
            status={
                "segmentation_generation": 7,
                "matte_reset_count": 3,
                "config_version": 12,
            },
        )
        config_reset, last_sequence = monitor.get(last_sequence, timeout=2.0)
        assert config_reset is not None
        assert config_reset.temporal.history_state == "reset"
        assert config_reset.temporal.registration_state == "reset"
        assert not config_reset.available
    finally:
        monitor.close()


def test_inactive_monitor_fast_path_does_not_touch_evidence_and_clears() -> None:
    monitor = LocalMatteDiagnosticMonitor()
    poison_evidence = cast(MatteFrameEvidence, object())
    poison_status = cast(Mapping[str, object], object())
    try:
        assert not monitor.accepting
        assert monitor.selected_view is None
        assert not monitor.submit(poison_evidence, status=poison_status)
        assert monitor.latest() is None

        source, raw_alpha, refined_alpha, foreground, backdrop, base = (
            _complete_tracks()
        )
        monitor.select("raw_camera")
        assert monitor.submit(
            _evidence(
                1,
                source=source,
                raw_alpha=raw_alpha,
                refined_alpha=refined_alpha,
                foreground=foreground,
                backdrop=backdrop,
                base_composite=base,
            ),
            status={"segmentation_generation": 7, "matte_reset_count": 2},
        )
        frame, _sequence = monitor.get(timeout=2.0)
        assert frame is not None
        assert monitor.latest() is frame

        monitor.deactivate()
        assert not monitor.accepting
        assert monitor.selected_view is None
        assert monitor.latest() is None
        assert not monitor.submit(poison_evidence, status=poison_status)
    finally:
        monitor.close()


def test_deactivate_releases_idle_worker_private_rasters() -> None:
    source = np.full((24, 32, 3), 73, dtype=np.uint8)
    monitor = LocalMatteDiagnosticMonitor()
    try:
        monitor.select("raw_camera")
        assert monitor.submit(
            _evidence(1, source=source),
            status={"segmentation_generation": 0, "matte_reset_count": 0},
        )
        frame, _sequence = monitor.get(timeout=2.0)
        assert frame is not None
        latest_sample = monitor._latest_sample
        assert latest_sample is not None
        frozen_source_ref = weakref.ref(latest_sample.raw_frame)
        rendered_pixels_ref = weakref.ref(frame.pixels)
        del latest_sample
        del frame

        monitor.deactivate()
        deadline = time.monotonic() + 2.0
        while (
            frozen_source_ref() is not None or rendered_pixels_ref() is not None
        ) and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)

        assert frozen_source_ref() is None
        assert rendered_pixels_ref() is None
    finally:
        monitor.close()


def test_close_waits_for_an_inflight_private_render(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    original_render = matte_live_mod.render_diagnostic_view

    def blocked_render(*args, **kwargs):
        started.set()
        assert release.wait(2.0)
        return original_render(*args, **kwargs)

    monkeypatch.setattr(matte_live_mod, "render_diagnostic_view", blocked_render)
    monitor = LocalMatteDiagnosticMonitor()
    monitor.select("raw_camera")
    source = np.full((24, 32, 3), 91, dtype=np.uint8)
    assert monitor.submit(
        _evidence(1, source=source),
        status={"segmentation_generation": 0, "matte_reset_count": 0},
    )
    assert started.wait(2.0)

    closer = threading.Thread(
        target=lambda: (monitor.close(), closed.set()),
        daemon=True,
    )
    closer.start()
    try:
        assert not closed.wait(0.05)
        release.set()
        assert closed.wait(2.0)
        closer.join(timeout=2.0)
        assert not closer.is_alive()
        assert monitor._worker is not None
        assert not monitor._worker.is_alive()
    finally:
        release.set()
        monitor.close()
