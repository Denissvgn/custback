"""The control page: served only behind auth and internally consistent."""

import json
import re
import shutil
import subprocess

import pytest

pytest.importorskip("fastapi")

from custback.api.webui import WEBUI_HTML


def test_page_is_self_contained():
    # The API's CSP allows only same-origin assets plus inline CSS/JS.
    for attribute in ("src", "href"):
        assert not re.search(
            rf'{attribute}=["\'](?:https?:)?//', WEBUI_HTML, re.IGNORECASE
        )
    assert not re.search(r"url\(\s*[\"']?(?:https?:)?//", WEBUI_HTML, re.IGNORECASE)
    assert not re.search(r"fetch\(\s*[\"']https?://", WEBUI_HTML, re.IGNORECASE)
    assert not re.search(r"<script\b[^>]*\bsrc=", WEBUI_HTML, re.IGNORECASE)
    assert not re.search(
        r"<link\b[^>]*\brel=[\"']stylesheet[\"']", WEBUI_HTML, re.IGNORECASE
    )
    assert "@import" not in WEBUI_HTML
    assert not re.search(r"<[^>]+\son[a-z]+\s*=", WEBUI_HTML, re.IGNORECASE)
    assert "eval(" not in WEBUI_HTML
    assert "new Function" not in WEBUI_HTML
    assert "<title>custback control</title>" in WEBUI_HTML
    assert "oklch(" in WEBUI_HTML


def test_page_targets_both_control_planes():
    for path in (
        "/config",
        "/status",
        "/backgrounds",
        "/background/",
        "/video/mjpeg",
        "/video/snapshot.jpg",
        "/avatar/config",
        "/avatar/status",
        "/avatar/avatars",
        "/avatar/rigs",
        "/avatar/backgrounds",
        "/avatar/video/mjpeg",
        "/avatar/video/snapshot.jpg",
        "/auth/session",
        "/docs",
        "/openapi.json",
    ):
        assert path in WEBUI_HTML, path
    for operation in (
        'api("PATCH", "/config"',
        'api("PATCH", "/avatar/config"',
        'api("POST", "/background/" + kind',
        'api("POST", "/avatar/backgrounds/" + kind',
        'await api("DELETE", path)',
        '? "/backgrounds/" + encodeURIComponent(entry.name)',
        ': "/avatar/backgrounds/" + encodeURIComponent(entry.name)',
        'api("POST", "/avatar/rigs?name="',
        'api("DELETE", "/avatar/rigs/"',
        '"/avatar/avatars/" + name + "/thumbnail.jpg',
        '"/avatar/rigs/" + encodeURIComponent(rig.name)',
    ):
        assert operation in WEBUI_HTML, operation


def test_every_scripted_element_id_exists_in_the_markup():
    static_ids = re.findall(r'id="([\w-]+)"', WEBUI_HTML)
    assert len(static_ids) == len(set(static_ids)), "markup contains duplicate ids"
    ids = set(static_ids)
    ids |= set(re.findall(r'rigInput\.id = "([\w-]+)"', WEBUI_HTML))
    referenced = set(re.findall(r'\$\("([\w-]+)"\)', WEBUI_HTML))
    missing = referenced - ids
    assert not missing, f"script references unknown element ids: {sorted(missing)}"
    for attribute in (
        "for",
        "aria-describedby",
        "aria-errormessage",
        "aria-labelledby",
        "aria-controls",
    ):
        for value in re.findall(rf'{attribute}="([\w -]+)"', WEBUI_HTML):
            for target in value.split():
                assert target in ids, f'{attribute} points to unknown id "{target}"'


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_inline_javascript_parses():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        ["node", "--check"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_avatar_enable_toggle_patches_remote_mode():
    toggle = WEBUI_HTML.split("// -- avatar enable toggle", 1)[1].split(
        "// -- background panel", 1
    )[0]
    assert 'mode: "remote"' in toggle
    assert "await patchCore(patch)" in toggle
    assert "remote_fallback_mode" in WEBUI_HTML
    assert 'LOCAL_MODES.includes(mode) ? mode : "blur"' in WEBUI_HTML
    assert "adjusted.remote_fallback_mode = adjusted.mode" in WEBUI_HTML
    assert 'patchBackground({mode: "camera", camera_target:' in WEBUI_HTML
    assert 'mode === "passthrough"' in WEBUI_HTML
    assert "Turn avatar output off before showing the original room" in WEBUI_HTML
    assert "uploadWouldDisableAvatar" in WEBUI_HTML
    assert "The camera upload endpoint activates new media immediately" in WEBUI_HTML


def test_provider_setup_never_patches_an_arbitrary_outbound_target():
    for element_id in ("avatar-url", "avatar-connect", "a2f-url", "a2f-set"):
        assert f'id="{element_id}"' not in WEBUI_HTML
        assert f'$("{element_id}")' not in WEBUI_HTML
    assert 'id="provider-url"' in WEBUI_HTML
    assert 'id="provider-source-url"' in WEBUI_HTML
    assert 'id="provider-copy"' in WEBUI_HTML
    assert 'data-provider="local"' in WEBUI_HTML
    assert 'data-provider="remote"' in WEBUI_HTML
    assert "Remote providers require HTTPS" in WEBUI_HTML
    assert "restart config" in WEBUI_HTML
    assert "operator-owned" in WEBUI_HTML
    assert "patchCore({avatar: {url" not in WEBUI_HTML
    assert "patchAvatar({source: {url" not in WEBUI_HTML
    assert "token_file" in WEBUI_HTML
    assert 'type="password"' not in WEBUI_HTML
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    planner = script.split("// -- provider restart planner", 1)[1].split(
        "// -- camera quality", 1
    )[0]
    assert "api(" not in planner
    assert "fetch(" not in planner
    assert "navigator.clipboard.writeText" in planner
    provider_validator = planner.split("function validateRemoteProvider", 1)[1].split(
        "function validateRemoteSource", 1
    )[0]
    source_validator = planner.split("function validateRemoteSource", 1)[1].split(
        "function connectableCoreHost", 1
    )[0]
    for validator, protocol in (
        (provider_validator, 'parsed.protocol !== "https:"'),
        (source_validator, 'parsed.protocol !== "wss:"'),
    ):
        assert protocol in validator
        for guard in (
            "parsed.username",
            "parsed.password",
            'parsed.pathname !== "/"',
            "parsed.search",
            "parsed.hash",
        ):
            assert guard in validator
    compact = re.sub(r"\s+", "", planner)
    assert "patchCore({avatar:{url" not in compact
    assert "patchAvatar({source:{url" not in compact
    assert "# CORE HOST" in WEBUI_HTML
    assert "# AVATAR HOST" in WEBUI_HTML
    assert "ready = Boolean(controlUrl && sourceUrl)" in planner
    assert '$("provider-copy").disabled = !ready' in planner
    assert '"http://127.0.0.1:8711"' in planner
    assert 'localProtocol + "://" + coreHost + ":" + corePort' in planner
    assert 'avatar:\\n  url: \\"" + shownControl' in planner
    assert 'source:\\n  url: \\"" + shownSource' in planner
    assert "checks URL shape only" in WEBUI_HTML
    assert "docs/remote-deployment.md" in WEBUI_HTML
    assert "avatar.token_file" in WEBUI_HTML
    assert "source.token_file" in WEBUI_HTML
    assert "providerSourceTouched" in planner


def test_polling_and_external_refreshes_are_race_guarded():
    assert "let corePollBusy = false" in WEBUI_HTML
    assert "let avatarPollBusy = false" in WEBUI_HTML
    assert "if (corePollBusy) return" in WEBUI_HTML
    assert "if (avatarPollBusy) return" in WEBUI_HTML
    assert "pollCoreStatus();\n  pollAvatarStatus();" in WEBUI_HTML
    assert "!state.coreRefreshPending" in WEBUI_HTML
    assert "!state.avatarRefreshPending" in WEBUI_HTML
    assert "state.avatarStatusOutage = true" in WEBUI_HTML
    assert "const recovering = state.avatarStatusOutage" in WEBUI_HTML
    assert "state.avatarStatusOutage = false" in WEBUI_HTML
    assert (
        "loadCore().then(() => {\n        state.coreVersion = observedVersion"
        in WEBUI_HTML
    )
    assert "loadAvatar().then((loaded) => {\n        if (!loaded) return;" in WEBUI_HTML
    assert "state.avatarVersion = observedVersion" in WEBUI_HTML
    assert "status.run_id !== state.runId" in WEBUI_HTML
    assert "location.reload();" in WEBUI_HTML


def test_interface_exposes_the_existing_user_controls():
    for element_id in (
        "preview-source",
        "preview-snapshot",
        "avatar-enabled",
        "bg-scope",
        "bg-modes",
        "bg-camera-target",
        "bg-color",
        "bg-blur",
        "bg-upload",
        "bg-upload-hint",
        "provider-kind",
        "provider-url",
        "provider-source-url",
        "provider-copy",
        "avatar-tiles",
        "avatar-modes",
        "avatar-style",
        "avatar-framing",
        "avatar-scale",
        "avatar-x",
        "avatar-y",
        "avatar-smoothing",
        "avatar-follow",
        "avatar-parts",
        "avatar-max-fps",
        "avatar-jpeg-quality",
        "quality-backend",
        "quality-delegate",
        "quality-threshold",
        "quality-rvm-downsample",
        "quality-mask-blur",
        "quality-mask-shift",
        "quality-smoothing",
        "quality-light-wrap",
        "quality-edge-refine",
        "quality-model-foreground",
        "quality-color-auto",
        "quality-color-strength",
        "quality-color-status",
        "quality-background-fit",
        "quality-camera-fit",
        "quality-background-anchor-x",
        "quality-background-anchor-y",
        "quality-exposure-limit",
        "quality-wb-strength",
        "quality-adaptation-time",
        "quality-blend-space",
        "quality-camera-rotation",
        "quality-output-width",
        "quality-output-height",
        "core-diagnostics",
        "core-diagnostics-all",
        "avatar-diagnostics",
        "avatar-diagnostics-all",
        "core-remote-timeout",
        "avatar-connect-timeout",
        "avatar-read-timeout",
        "effective-config",
        "signout",
    ):
        assert f'id="{element_id}"' in WEBUI_HTML, element_id


def test_workbench_navigation_and_accessibility_hooks_are_present():
    views = set(re.findall(r'data-view="([\w-]+)"', WEBUI_HTML))
    panels = set(re.findall(r'data-panel="([\w-]+)"', WEBUI_HTML))
    assert views == panels == {"background", "avatar", "quality", "system"}
    assert 'aria-label="Control areas"' in WEBUI_HTML
    assert 'role="tablist"' in WEBUI_HTML
    assert WEBUI_HTML.count('role="tab"') == 4
    assert WEBUI_HTML.count('role="tabpanel"') == 4
    for view in views:
        tab_match = re.search(rf'<button[^>]+id="tab-{view}"[^>]*>', WEBUI_HTML)
        panel_match = re.search(rf'<div[^>]+id="view-{view}"[^>]*>', WEBUI_HTML)
        assert tab_match is not None
        assert panel_match is not None
        tab = tab_match.group()
        panel = panel_match.group()
        assert f'aria-controls="view-{view}"' in tab
        assert f'aria-labelledby="tab-{view}"' in panel
    assert WEBUI_HTML.count('aria-selected="true"') == 1
    assert WEBUI_HTML.count('tabindex="0"') == 1
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert f'event.key === "{key}"' in WEBUI_HTML
    assert 'aria-live="polite"' in WEBUI_HTML
    assert ":focus-visible" in WEBUI_HTML
    assert "prefers-reduced-motion:reduce" in WEBUI_HTML


def test_visual_controls_are_accessible_responsive_and_restart_marked():
    for element_id in (
        "quality-color-auto",
        "quality-color-strength",
        "quality-background-fit",
        "quality-camera-fit",
        "quality-background-anchor-x",
        "quality-background-anchor-y",
        "quality-exposure-limit",
        "quality-wb-strength",
        "quality-adaptation-time",
        "quality-blend-space",
        "quality-camera-rotation",
        "quality-output-width",
        "quality-output-height",
    ):
        assert re.search(rf'<label[^>]+for="{element_id}"', WEBUI_HTML)
    assert 'id="quality-color-status" role="status"' in WEBUI_HTML
    assert 'aria-live="polite"' in WEBUI_HTML
    assert "Backdrop horizontal focal point" in WEBUI_HTML
    assert "Backdrop vertical focal point" in WEBUI_HTML
    assert WEBUI_HTML.count('class="restart-tag"') >= 2
    assert "Restart required" in WEBUI_HTML
    assert "@media (max-width:23.5rem)" in WEBUI_HTML
    assert "@media (min-width:40rem)" in WEBUI_HTML
    assert ".field-grid.two{grid-template-columns:" in WEBUI_HTML


def test_visual_controls_emit_minimal_merge_patches():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    quality = script.split("// -- camera quality", 1)[1].split(
        "// -- diagnostics and safe runtime settings", 1
    )[0]

    assert "patchCoreControl({background: {fit_mode: event.target.value}})" in quality
    assert "patchCoreControl({camera: {fit_mode: event.target.value}})" in quality
    assert (
        "patchCoreControl({background: {[field]: parseFloat(event.target.value)}})"
        in quality
    )
    assert (
        "patchCoreControl({compositing: {color_correction: {\n"
        '    mode: event.target.checked ? "auto" : "off",' in quality
    )
    assert (
        "patchCoreControl({compositing: {color_correction: {\n"
        "      [field]: parseFloat(event.target.value)," in quality
    )
    assert (
        "patchCoreControl({compositing: {blend_space: event.target.value}})" in quality
    )
    assert (
        "patchCoreControl({camera: {rotation: parseInt(event.target.value, 10)}})"
        in quality
    )
    assert "patchCoreControl({output: {width, height}})" in quality


def test_core_control_failure_restores_effective_config():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    helper = script.split("async function patchCoreControl", 1)[1].split(
        "async function patchAvatar", 1
    )[0]
    assert "[409, 422, 503].includes(err.status)" in helper
    assert 'state.core = await api("GET", "/config")' in helper
    assert "renderAll();" in helper
    assert helper.index('state.core = await api("GET", "/config")') < helper.index(
        "renderAll();"
    )
    assert helper.index("renderAll();") < helper.index("throw err;")

    quality = script.split("// -- camera quality", 1)[1].split(
        "// -- diagnostics and safe runtime settings", 1
    )[0]
    assert "patchCore(" not in quality
    assert quality.count("patchCoreControl(") >= 10


def test_color_diagnostic_never_mislabels_held_or_stale_state_as_active():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    summary = script.split("function colorCorrectionSummary", 1)[1].split(
        "function renderColorCorrectionStatus", 1
    )[0]
    for state in (
        "disabled",
        "mode-excluded",
        "warming",
        "low-confidence",
        "stale-decay",
        "scene-cut",
        "active",
    ):
        assert f'"{state}"' in summary
    assert 'phase !== "active" || !active' in summary
    assert "Low confidence · the previous correction is being held" in summary
    assert (
        "Stale estimate · the previous correction is fading toward neutral" in summary
    )
    assert '"Active" + (details.length' in summary
    assert "color_correction_effective_mode" in summary
    assert "color_correction_confidence" in summary
    assert "color_correction_exposure_ev" in summary
    assert 'String(effective).includes("white-balance")' in summary


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_color_diagnostic_runtime_semantics():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    title_case = (
        "function titleCase"
        + script.split("function titleCase", 1)[1].split("function setView", 1)[0]
    )
    summary = (
        "function colorCorrectionSummary"
        + script.split("function colorCorrectionSummary", 1)[1].split(
            "function renderColorCorrectionStatus", 1
        )[0]
    )
    cases = r"""
const base = {
  color_correction_mode: "auto",
  color_correction_effective_mode: "exposure",
  color_correction_active: true,
  color_correction_confidence: 0.82,
  color_correction_exposure_ev: 0.35,
  color_correction_warming: false,
  color_correction_stale: false,
};
const result = {
  disabled: colorCorrectionSummary({...base, color_correction_mode: "off",
    color_correction_state: "disabled"})[0],
  excluded: colorCorrectionSummary({...base, color_correction_state: "mode-excluded"})[0],
  warming: colorCorrectionSummary({...base, color_correction_state: "warming"})[0],
  sceneCut: colorCorrectionSummary({...base, color_correction_state: "scene-cut",
    color_correction_warming: true})[0],
  frozen: colorCorrectionSummary({...base, color_correction_state: "low-confidence"})[0],
  stale: colorCorrectionSummary({...base, color_correction_state: "stale-decay",
    color_correction_stale: true})[0],
  identity: colorCorrectionSummary({...base, color_correction_state: "active",
    color_correction_active: false, color_correction_effective_mode: "identity"})[0],
  active: colorCorrectionSummary({...base, color_correction_state: "active"})[0],
};
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node"],
        input=title_case + summary + cases,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["disabled"].startswith("Off")
    assert values["excluded"].startswith("Bypassed")
    assert values["warming"].startswith("Warming up")
    assert values["sceneCut"].startswith("Scene changed")
    assert values["frozen"].startswith("Low confidence")
    assert values["stale"].startswith("Stale estimate")
    assert values["identity"].startswith("Ready")
    assert not values["frozen"].startswith("Active")
    assert not values["stale"].startswith("Active")
    assert values["active"] == "Active · +0.35 EV · 82% confidence"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_core_control_runtime_restores_rejected_changes():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    api_error = (
        "class ApiError"
        + script.split("class ApiError", 1)[1].split("async function api", 1)[0]
    )
    helper = (
        "async function patchCoreControl"
        + script.split("async function patchCoreControl", 1)[1].split(
            "async function patchAvatar", 1
        )[0]
    )
    harness = r"""
const state = {core: {value: "old"}};
const calls = [];
let renderCount = 0;
let rejectedStatus = 409;
async function patchCore() {
  throw new ApiError(rejectedStatus, "rejected", "rejected");
}
async function api(method, path) {
  calls.push([method, path]);
  return {value: "effective-" + rejectedStatus};
}
function renderAll() { renderCount += 1; }
(async () => {
  const results = {};
  for (const status of [409, 422, 503]) {
    rejectedStatus = status;
    try { await patchCoreControl({camera: {fit_mode: "cover"}}); } catch (err) {}
    results[status] = state.core.value;
  }
  process.stdout.write(JSON.stringify({results, calls, renderCount}));
})().catch((err) => { console.error(err); process.exit(1); });
"""
    result = subprocess.run(
        ["node"],
        input=api_error + helper + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["results"] == {
        "409": "effective-409",
        "422": "effective-422",
        "503": "effective-503",
    }
    assert values["calls"] == [["GET", "/config"]] * 3
    assert values["renderCount"] == 3


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_system_diagnostics_render_source_color_and_camera_controls():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    for key in (
        "background_video_decoder_backend",
        "background_video_color_status",
        "background_video_input_color",
        "background_video_output_color",
        "background_video_color_assumed_fields",
        "background_video_color_overridden_fields",
        "camera_controls",
    ):
        assert key in script
    assert "if (Array.isArray(value))" in script
    assert 'coreRows.push(["Video colour", ...videoColorSummary(status)])' in script

    title_case = (
        "function titleCase"
        + script.split("function titleCase", 1)[1].split("function setView", 1)[0]
    )
    formatters = (
        "function formatDuration"
        + script.split("function formatDuration", 1)[1].split(
            "function diagnosticTone", 1
        )[0]
    )
    video_summary = (
        "function videoColorSummary"
        + script.split("function videoColorSummary", 1)[1].split(
            "function renderDiagnostics", 1
        )[0]
    )
    harness = r"""
const controls = formatDiagnostic("camera_controls", {
  policy: "preserve",
  backend_family: "v4l2",
  qualification: "read-only-qualified",
  writes_performed: false,
  generation: 3,
  properties: {
    exposure: {status: "reported", value: 0.5},
    white_balance: {status: "ambiguous-zero", value: 0},
  },
});
const result = {
  assumed: formatDiagnostic("background_video_color_assumed_fields",
    ["matrix", "range"]),
  noOverrides: formatDiagnostic("background_video_color_overridden_fields", []),
  controls,
  tagged: videoColorSummary({
    background_video_decoder_backend: "PYAV_FFMPEG",
    background_video_color_status: "tagged",
    background_video_input_color: "bt709/limited/bt709/srgb",
    background_video_output_color: "srgb-full-bgr",
  }),
  legacy: videoColorSummary({
    background_video_decoder_backend: "OPENCV",
    background_video_color_status: "legacy-opencv-assumption",
    background_video_input_color: "opencv-opaque/assumed-srgb-full",
    background_video_output_color: "srgb-full-bgr",
  }),
};
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node"],
        input=title_case + formatters + video_summary + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["assumed"] == "Matrix, Range"
    assert values["noOverrides"] == "None"
    assert (
        "Preserve · V4l2 · Read Only Qualified · no writes · generation 3"
        in values["controls"]
    )
    assert "Exposure: Reported 0.5" in values["controls"]
    assert "White Balance: Ambiguous Zero 0" in values["controls"]
    assert values["tagged"] == [
        "Tagged · PYAV FFMPEG · bt709/limited/bt709/srgb → srgb-full-bgr",
        "good",
    ]
    assert values["legacy"][1] == "warn"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_geometry_summary_distinguishes_crop_pad_and_stretch():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    title_case = (
        "function titleCase"
        + script.split("function titleCase", 1)[1].split("function setView", 1)[0]
    )
    summary = (
        "function geometrySummary"
        + script.split("function geometrySummary", 1)[1].split(
            "function videoColorSummary", 1
        )[0]
    )
    harness = r"""
const base = {
  capture_delivered_width: 640,
  capture_delivered_height: 480,
  capture_oriented_width: 640,
  capture_oriented_height: 480,
  output_width: 1280,
  output_height: 720,
  camera_pad_left: 0,
  camera_pad_top: 0,
  camera_pad_right: 0,
  camera_pad_bottom: 0,
};
const result = {
  cover: geometrySummary({...base, camera_fit: "cover",
    camera_scale_x: 2, camera_scale_y: 2,
    camera_crop_left: 0, camera_crop_top: 120,
    camera_crop_right: 1280, camera_crop_bottom: 840}),
  contain: geometrySummary({...base, camera_fit: "contain",
    camera_scale_x: 1.5, camera_scale_y: 1.5,
    camera_crop_left: 0, camera_crop_top: 0,
    camera_crop_right: 960, camera_crop_bottom: 720,
    camera_pad_left: 160, camera_pad_right: 160}),
  stretch: geometrySummary({...base, camera_fit: "stretch",
    camera_scale_x: 2, camera_scale_y: 1.5,
    camera_crop_left: 0, camera_crop_top: 0,
    camera_crop_right: 1280, camera_crop_bottom: 720}),
};
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node"],
        input=title_case + summary + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert " / crop → " in values["cover"]
    assert " / pad → " in values["contain"]
    assert " / no crop → " in values["stretch"]
