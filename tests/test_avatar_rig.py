"""Avatar rigs and frame composition, hardware-free."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import custback.avatar.rig as rig_mod
import custback.backgrounds as backgrounds_mod
from custback.avatar.config import (
    AVATAR_PARTS,
    BUILTIN_AVATARS,
    AppearanceConfig,
    AvatarBackgroundConfig,
)
from custback.avatar.renderer import blurred_room, compose_avatar, create_avatar_backdrop
from custback.avatar.rig import (
    AVATAR_PRESETS,
    BuiltinRig,
    LayeredRig,
    RigError,
    alpha_over,
    apply_head_pose,
    create_rig,
    sketch_filter,
)
from custback.avatar.state import FaceState

ALL_PARTS = frozenset(AVATAR_PARTS)


def neutral(**channels):
    state = FaceState.neutral()
    for name, value in channels.items():
        state.set_channel(name, value)
    return state


def test_alpha_over_retains_straight_color_on_transparent_base():
    base = np.zeros((1, 1, 4), dtype=np.uint8)
    layer = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)

    alpha_over(base, layer)

    np.testing.assert_array_equal(base[0, 0], layer[0, 0])


def test_alpha_over_matches_porter_duff_for_two_translucent_layers():
    base = np.array([[[100, 20, 10, 128]]], dtype=np.uint8)
    layer = np.array([[[20, 40, 200, 128]]], dtype=np.uint8)
    source_alpha = 128.0 / 255.0
    destination_alpha = 128.0 / 255.0
    expected_alpha = source_alpha + destination_alpha * (1.0 - source_alpha)
    expected_rgb = (
        layer[0, 0, :3] * source_alpha
        + base[0, 0, :3] * destination_alpha * (1.0 - source_alpha)
    ) / expected_alpha

    alpha_over(base, layer)

    np.testing.assert_allclose(base[0, 0, :3], expected_rgb, atol=1.0)
    assert base[0, 0, 3] == pytest.approx(expected_alpha * 255.0, abs=1.0)


def test_layer_alpha_is_applied_once_at_final_avatar_composition():
    sprite = np.zeros((10, 10, 4), dtype=np.uint8)
    layer = np.full((10, 10, 4), (200, 100, 50, 128), dtype=np.uint8)
    alpha_over(sprite, layer)

    rendered = compose_avatar(
        sprite,
        np.zeros((20, 20, 3), dtype=np.uint8),
        AppearanceConfig(scale=0.5),
    )

    covered = rendered[np.any(rendered > 0, axis=2)]
    assert covered.size > 0
    np.testing.assert_allclose(covered.max(axis=0), (100, 50, 25), atol=2)


def test_head_pose_transform_preserves_straight_color_at_soft_edges():
    layer = np.zeros((41, 41, 4), dtype=np.uint8)
    authored = np.array([30, 120, 220], dtype=np.uint8)
    layer[10:31, 10:31, :3] = authored
    layer[10:31, 10:31, 3] = 255

    transformed = apply_head_pose(
        layer,
        yaw=0.2,
        pitch=0.1,
        roll=0.23,
        pivot=(20.0, 20.0),
        sway_px=(4.0, 3.0),
    )

    soft_edge = (transformed[..., 3] > 8) & (transformed[..., 3] < 247)
    assert soft_edge.any()
    expected = np.broadcast_to(authored, transformed[..., :3][soft_edge].shape)
    np.testing.assert_allclose(transformed[..., :3][soft_edge], expected, atol=2)


def test_sketch_filter_ignores_rgb_hidden_by_zero_alpha():
    first = np.zeros((41, 41, 4), dtype=np.uint8)
    first[10:31, 10:31] = (40, 120, 220, 255)
    second = first.copy()
    transparent = second[..., 3] == 0
    second[transparent, :3] = (255, 15, 190)

    sketch_filter(first)
    sketch_filter(second)

    np.testing.assert_array_equal(first, second)


def test_builtin_rig_renders_bgra_sprite():
    sprite = BuiltinRig().render(neutral(), ALL_PARTS)
    assert sprite.shape == (BuiltinRig.HEIGHT, BuiltinRig.WIDTH, 4)
    assert sprite.dtype == np.uint8
    assert (sprite[..., 3] > 0).sum() > 10_000  # substantial avatar coverage


def test_builtin_parts_are_individually_selectable():
    rig = BuiltinRig()
    full = rig.render(neutral(), ALL_PARTS)
    head_only = rig.render(neutral(), frozenset({"head", "eyes", "mouth"}))
    assert (head_only[..., 3] > 0).sum() < (full[..., 3] > 0).sum()
    # Without the torso the bottom rows are fully transparent.
    assert head_only[-40:, :, 3].max() == 0
    assert full[-40:, :, 3].max() > 0


def test_builtin_blink_and_jaw_change_pixels():
    rig = BuiltinRig()
    base = rig.render(neutral(), ALL_PARTS)
    blink = rig.render(neutral(eyeBlinkLeft=1.0, eyeBlinkRight=1.0), ALL_PARTS)
    talk = rig.render(neutral(jawOpen=0.8), ALL_PARTS)
    assert not np.array_equal(base, blink)
    assert not np.array_equal(base, talk)
    # The blink affects the eye band, not the mouth band.
    eye_band = slice(210, 270)
    mouth_band = slice(310, 400)
    assert not np.array_equal(base[eye_band], blink[eye_band])
    assert np.array_equal(base[mouth_band], blink[mouth_band])
    assert not np.array_equal(base[mouth_band], talk[mouth_band])


def test_builtin_head_pose_moves_head_but_not_torso():
    rig = BuiltinRig()
    state = FaceState.neutral()
    state.roll = 0.2
    state.yaw = 0.3
    posed = rig.render(state, ALL_PARTS)
    base = rig.render(FaceState.neutral(), ALL_PARTS)
    assert not np.array_equal(base, posed)
    torso_only = frozenset({"torso"})
    assert np.array_equal(rig.render(state, torso_only), rig.render(FaceState.neutral(), torso_only))


def _assert_follow_pose_switch_preserves_expression(rig):
    state = neutral(jawOpen=0.8, eyeBlinkLeft=1.0, eyeBlinkRight=1.0)
    state.yaw = 0.4
    state.pitch = 0.15
    state.roll = 0.2
    expression_only = FaceState(
        present=True,
        blendshapes=dict(state.blendshapes),
    )

    fixed = rig.render(state, ALL_PARTS, follow_pose=False)
    expected_fixed = rig.render(expression_only, ALL_PARTS, follow_pose=True)
    following = rig.render(state, ALL_PARTS, follow_pose=True)
    neutral_fixed = rig.render(FaceState.neutral(), ALL_PARTS, follow_pose=False)

    assert np.array_equal(fixed, expected_fixed)
    assert not np.array_equal(fixed, following)
    assert not np.array_equal(fixed, neutral_fixed)


def test_builtin_follow_pose_switch_keeps_expression_channels_active():
    _assert_follow_pose_switch_preserves_expression(BuiltinRig())


def test_create_rig_builtin_and_missing_directory(tmp_path):
    assert isinstance(create_rig("builtin"), BuiltinRig)
    with pytest.raises(RigError, match="does not exist"):
        create_rig(str(tmp_path / "missing"))


def test_builtin_avatar_presets_cover_config_names():
    assert set(AVATAR_PRESETS) == set(BUILTIN_AVATARS)


def test_builtin_avatars_render_distinct_characters():
    sprites = {
        name: BuiltinRig(avatar=name).render(neutral(), ALL_PARTS)
        for name in BUILTIN_AVATARS
    }
    names = list(sprites)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            assert not np.array_equal(sprites[first], sprites[second]), (first, second)


def test_builtin_rejects_unknown_avatar_and_style():
    with pytest.raises(RigError, match="unknown builtin avatar"):
        BuiltinRig(avatar="zorp")
    with pytest.raises(RigError, match="unknown avatar style"):
        BuiltinRig(style="anime")


def test_builtin_styles_change_the_render():
    cartoon = BuiltinRig(style="cartoon").render(neutral(), ALL_PARTS)
    realistic = BuiltinRig(style="realistic").render(neutral(), ALL_PARTS)
    sketch = BuiltinRig(style="sketch").render(neutral(), ALL_PARTS)
    assert not np.array_equal(cartoon, realistic)
    assert not np.array_equal(cartoon, sketch)
    # The pencil filter recolors but keeps the exact silhouette...
    assert np.array_equal(sketch[..., 3], cartoon[..., 3])
    # ...and is near-grayscale (faint warm paper tint only).
    opaque = sketch[..., 3] > 0
    spread = sketch[..., 2][opaque].astype(int) - sketch[..., 0][opaque].astype(int)
    assert np.abs(spread).max() <= 12


def test_create_rig_forwards_avatar_and_style():
    rig = create_rig("builtin", avatar="robin", style="sketch")
    assert isinstance(rig, BuiltinRig)
    assert rig.avatar == "robin"
    assert rig.style == "sketch"


def test_builtin_framing_windows_are_ordered_shots():
    rig = BuiltinRig()
    assert rig.framing_window("full") == (0.0, 1.0)
    bust_top, bust_bottom = rig.framing_window("bust")
    close_top, close_bottom = rig.framing_window("closeup")
    assert bust_top == 0.0 and 0.7 <= bust_bottom <= 0.9
    assert close_bottom < bust_bottom  # closeup is the tighter shot
    assert rig.framing_window("unknown") == (0.0, 1.0)


def _write_layer(path, size, color, region):
    layer = np.zeros((*size, 4), dtype=np.uint8)
    y, x = region
    layer[y, x] = (*color, 255)
    assert cv2.imwrite(str(path), layer)


@pytest.fixture()
def rig_dir(tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    size = (200, 100)
    _write_layer(directory / "torso.png", size, (255, 0, 0), (slice(150, 200), slice(20, 80)))
    _write_layer(directory / "head.png", size, (0, 255, 0), (slice(20, 150), slice(25, 75)))
    _write_layer(directory / "eyes.png", size, (0, 0, 255), (slice(60, 70), slice(35, 65)))
    _write_layer(directory / "eyes_closed.png", size, (0, 0, 128), (slice(64, 68), slice(35, 65)))
    _write_layer(directory / "mouth.png", size, (255, 255, 0), (slice(110, 120), slice(40, 60)))
    _write_layer(directory / "mouth_open.png", size, (0, 255, 255), (slice(105, 125), slice(40, 60)))
    return directory


def test_layered_rig_renders_and_reports_available_parts(rig_dir):
    rig = LayeredRig(rig_dir)
    assert rig.parts == ("torso", "head", "mouth", "eyes")
    sprite = rig.render(neutral(), ALL_PARTS)
    assert sprite.shape == (200, 100, 4)
    assert (sprite[..., 3] > 0).any()
    torso_only = rig.render(neutral(), frozenset({"torso"}))
    assert (torso_only[:140, :, 3] == 0).all()


def test_layered_rig_expression_variants(rig_dir):
    rig = LayeredRig(rig_dir)
    base = rig.render(neutral(), ALL_PARTS)
    blink = rig.render(neutral(eyeBlinkLeft=1.0, eyeBlinkRight=1.0), ALL_PARTS)
    talk = rig.render(neutral(jawOpen=1.0), ALL_PARTS)
    assert not np.array_equal(base, blink)
    assert not np.array_equal(base, talk)


def test_layered_follow_pose_switch_keeps_expression_variants_active(rig_dir):
    _assert_follow_pose_switch_preserves_expression(LayeredRig(rig_dir))


def test_layered_rig_manifest_controls_head_parts(rig_dir):
    (rig_dir / "rig.yaml").write_text(
        "pivot: [50, 120]\nsway: [10, 5]\nhead_parts: [eyes, mouth]\n"
    )
    rig = LayeredRig(rig_dir)
    state = FaceState.neutral()
    state.roll = 0.3
    # Parts excluded from head_parts stay static under head pose...
    head_only = frozenset({"head"})
    assert np.array_equal(
        rig.render(state, head_only), rig.render(FaceState.neutral(), head_only)
    )
    # ...while listed parts follow it.
    eyes_only = frozenset({"eyes"})
    assert not np.array_equal(
        rig.render(state, eyes_only), rig.render(FaceState.neutral(), eyes_only)
    )


def test_layered_rig_rejects_bad_manifests_and_layers(rig_dir, tmp_path):
    (rig_dir / "rig.yaml").write_text("pivot: [50]\n")
    with pytest.raises(RigError, match="pivot"):
        LayeredRig(rig_dir)
    (rig_dir / "rig.yaml").write_text("unexpected: 1\n")
    with pytest.raises(RigError, match="unknown rig.yaml keys"):
        LayeredRig(rig_dir)
    (rig_dir / "rig.yaml").unlink()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RigError, match="no part layers"):
        LayeredRig(empty)

    _write_layer(rig_dir / "hair.png", (50, 50), (1, 2, 3), (slice(0, 10), slice(0, 10)))
    with pytest.raises(RigError, match="does not match"):
        LayeredRig(rig_dir)


@pytest.mark.parametrize(
    ("manifest", "field"),
    [
        ("pivot: [.nan, 120]\n", "pivot"),
        ("pivot: [" + "9" * 1000 + ", 120]\n", "pivot"),
        ("sway: [10, .inf]\n", "sway"),
        ("framing:\n  bust: [0.1, .nan]\n", "framing"),
        (
            "framing:\n  bust: [0.1, " + "9" * 1000 + "]\n",
            "framing",
        ),
    ],
)
def test_layered_rig_rejects_non_finite_geometry(rig_dir, manifest, field):
    (rig_dir / "rig.yaml").write_text(manifest)
    with pytest.raises(RigError, match=field):
        LayeredRig(rig_dir)


def test_layered_rig_rejects_opaque_layers(tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    solid = np.zeros((10, 10, 3), dtype=np.uint8)
    assert cv2.imwrite(str(directory / "head.png"), solid)
    with pytest.raises(RigError, match="alpha"):
        LayeredRig(directory)


def test_direct_rig_rejects_invalid_png_before_opencv(monkeypatch, tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    (directory / "head.png").write_bytes(b"not a png")
    monkeypatch.setattr(
        rig_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail("OpenCV saw an invalid layer"),
    )

    with pytest.raises(RigError, match="invalid PNG"):
        LayeredRig(directory)


def test_direct_rig_rejects_layer_pixel_bomb_before_opencv(
    monkeypatch, tmp_path
):
    directory = tmp_path / "rig"
    directory.mkdir()
    _write_layer(
        directory / "head.png",
        (9, 9),
        (1, 2, 3),
        (slice(0, 9), slice(0, 9)),
    )
    monkeypatch.setattr(
        rig_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail("OpenCV saw an oversized layer"),
    )

    with pytest.raises(RigError, match="exceeds 64 pixels"):
        LayeredRig(directory, rig_layer_max_pixels=64)


def test_direct_rig_rejects_total_pixels_before_opencv(monkeypatch, tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    for name in ("head", "torso"):
        _write_layer(
            directory / f"{name}.png",
            (9, 9),
            (1, 2, 3),
            (slice(0, 9), slice(0, 9)),
        )
    monkeypatch.setattr(
        rig_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail("OpenCV saw an oversized rig"),
    )

    with pytest.raises(RigError, match="decoded layers exceed 100 pixels"):
        LayeredRig(
            directory,
            rig_layer_max_pixels=100,
            rig_total_max_pixels=100,
        )


def test_direct_rig_rejects_non_uint8_opencv_decode(monkeypatch, tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    _write_layer(
        directory / "head.png",
        (4, 4),
        (1, 2, 3),
        (slice(0, 4), slice(0, 4)),
    )
    monkeypatch.setattr(
        rig_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: np.zeros((4, 4, 4), dtype=np.uint16),
    )

    with pytest.raises(RigError, match="PNG with alpha"):
        LayeredRig(directory)


def test_direct_rig_rejects_oversized_manifest_before_opencv(
    monkeypatch, tmp_path
):
    directory = tmp_path / "rig"
    directory.mkdir()
    _write_layer(
        directory / "head.png",
        (4, 4),
        (1, 2, 3),
        (slice(0, 4), slice(0, 4)),
    )
    (directory / "rig.yaml").write_text("#" * 17)
    monkeypatch.setattr(
        rig_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail("OpenCV ran before manifest cap"),
    )

    with pytest.raises(RigError, match="rig.yaml exceeds 16 bytes"):
        LayeredRig(directory, rig_manifest_max_bytes=16)


def test_direct_rig_maps_recursive_manifest_failure_to_rig_error(tmp_path):
    directory = tmp_path / "rig"
    directory.mkdir()
    _write_layer(
        directory / "head.png",
        (4, 4),
        (1, 2, 3),
        (slice(0, 4), slice(0, 4)),
    )
    (directory / "rig.yaml").write_text("[" * 2_000 + "]" * 2_000)

    with pytest.raises(RigError, match="invalid rig.yaml"):
        LayeredRig(directory)


def test_layered_rig_framing_manifest(rig_dir):
    assert LayeredRig(rig_dir).framing_window("bust") == (0.0, 0.8)
    (rig_dir / "rig.yaml").write_text("framing:\n  bust: [0.1, 0.6]\n")
    rig = LayeredRig(rig_dir)
    assert rig.framing_window("bust") == (0.1, 0.6)
    assert rig.framing_window("full") == (0.0, 1.0)  # others keep defaults
    (rig_dir / "rig.yaml").write_text("framing:\n  waist: [0.1, 0.6]\n")
    with pytest.raises(RigError, match="unknown framing"):
        LayeredRig(rig_dir)
    (rig_dir / "rig.yaml").write_text("framing:\n  bust: [0.9, 0.2]\n")
    with pytest.raises(RigError, match="fractions"):
        LayeredRig(rig_dir)


def test_layered_rig_sketch_style(rig_dir):
    base = LayeredRig(rig_dir).render(neutral(), ALL_PARTS)
    sketch = LayeredRig(rig_dir, style="sketch").render(neutral(), ALL_PARTS)
    assert not np.array_equal(base, sketch)
    assert np.array_equal(base[..., 3], sketch[..., 3])
    with pytest.raises(RigError, match="unknown avatar style"):
        LayeredRig(rig_dir, style="anime")


def test_compose_avatar_framing_window_crops_sprite():
    sprite = np.zeros((100, 50, 4), dtype=np.uint8)
    sprite[:50, :, 1] = 255  # top half green (head), bottom half red
    sprite[50:, :, 2] = 255
    sprite[..., 3] = 255
    backdrop = np.zeros((100, 100, 3), dtype=np.uint8)
    out = compose_avatar(
        sprite, backdrop, AppearanceConfig(), framing_window=(0.0, 0.5)
    )
    # Only the framed top half is in shot, filling the frame height.
    assert (out[..., 1] == 255).any()
    assert not (out[..., 2] == 255).any()
    full = compose_avatar(sprite, backdrop, AppearanceConfig())
    assert (full[..., 2] == 255).any()


def test_compose_avatar_places_sprite_bottom_center():
    sprite = np.zeros((100, 50, 4), dtype=np.uint8)
    sprite[..., 1] = 255
    sprite[..., 3] = 255
    backdrop = np.zeros((120, 200, 3), dtype=np.uint8)
    appearance = AppearanceConfig(scale=0.5)  # avatar height = 60 px
    out = compose_avatar(sprite, backdrop, appearance)
    assert out.shape == backdrop.shape
    green = out[..., 1] == 255
    ys, xs = np.nonzero(green)
    assert ys.max() == 119 and ys.min() == 60  # anchored to the bottom edge
    assert abs((xs.min() + xs.max()) / 2 - 99.5) <= 1.5  # centered


def test_compose_avatar_offsets_and_cropping():
    sprite = np.zeros((100, 50, 4), dtype=np.uint8)
    sprite[..., 2] = 255
    sprite[..., 3] = 255
    backdrop = np.zeros((100, 100, 3), dtype=np.uint8)
    shifted = compose_avatar(
        sprite, backdrop, AppearanceConfig(scale=0.5, offset_x=0.9, offset_y=-0.5)
    )
    ys, xs = np.nonzero(shifted[..., 2] == 255)
    assert xs.max() == 99  # cropped at the right edge
    assert ys.min() < 30  # lifted off the bottom
    off_frame = compose_avatar(
        sprite, backdrop, AppearanceConfig(scale=0.5, offset_y=-1.0, offset_x=-1.0)
    )
    assert off_frame.shape == backdrop.shape  # partial/total cropping is safe


def test_compose_avatar_alpha_blends_soft_edges():
    sprite = np.full((10, 10, 4), (0, 0, 255, 128), dtype=np.uint8)
    backdrop = np.zeros((20, 20, 3), dtype=np.uint8)
    out = compose_avatar(sprite, backdrop, AppearanceConfig(scale=0.5))
    blended = out[..., 2][out[..., 2] > 0]
    assert blended.size > 0
    assert 100 <= blended.max() <= 140  # ~50% alpha, not opaque


def test_create_avatar_backdrop_modes(tmp_path):
    color = create_avatar_backdrop(AvatarBackgroundConfig(mode="color", color=(1, 2, 3)))
    frame = color.frame(32, 16)
    assert frame.shape == (16, 32, 3)
    assert (frame == (1, 2, 3)).all()
    assert create_avatar_backdrop(AvatarBackgroundConfig(mode="blur")) is None

    image_path = tmp_path / "bg.png"
    assert cv2.imwrite(str(image_path), np.full((8, 8, 3), 9, dtype=np.uint8))
    image = create_avatar_backdrop(
        AvatarBackgroundConfig(mode="image", image_path=str(image_path))
    )
    assert image.frame(32, 16).shape == (16, 32, 3)


def test_avatar_image_backdrop_uses_configured_pixel_cap_before_opencv(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "oversized.png"
    assert cv2.imwrite(
        str(image_path), np.full((9, 9, 3), 9, dtype=np.uint8)
    )
    monkeypatch.setattr(
        backgrounds_mod.cv2,
        "imread",
        lambda *_args, **_kwargs: pytest.fail(
            "OpenCV must not see an avatar backdrop over its configured cap"
        ),
    )

    with pytest.raises(ValueError, match="exceeds 64 pixels"):
        create_avatar_backdrop(
            AvatarBackgroundConfig(mode="image", image_path=str(image_path)),
            image_max_pixels=64,
        )


def test_blurred_room_preserves_geometry():
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[10:14, 14:18] = 255
    blurred = blurred_room(frame, 31)
    assert blurred.shape == frame.shape
    assert blurred.max() < 255  # actually blurred
