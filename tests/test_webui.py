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
    # Matte silhouettes remain native-preview-only. The remotely bindable
    # management page must never acquire a mask image route by accident.
    assert "/video/matte" not in WEBUI_HTML
    assert "/diagnostics/matte" not in WEBUI_HTML


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
    assert "const snapshot = await loadCoreConfig();" in WEBUI_HTML
    assert "if (snapshot.version < state.coreVersion) return false;" in WEBUI_HTML
    assert "state.coreVersion = snapshot.version;" in WEBUI_HTML
    assert "refreshCoreConfig().then(() => {\n        renderAll();" in WEBUI_HTML
    assert "loadAvatar().then((loaded) => {\n        if (!loaded) return;" in WEBUI_HTML
    assert "state.avatarVersion = observedVersion" in WEBUI_HTML
    assert "status.run_id !== state.runId" in WEBUI_HTML
    assert "location.reload();" in WEBUI_HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_core_config_refresh_keeps_body_version_pair_and_rejects_stale_completion():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    loader = (
        "async function loadCoreConfig"
        + script.split("async function loadCoreConfig", 1)[1].split(
            "async function loadAvatar", 1
        )[0]
    )
    harness = r"""
class ApiError extends Error {
  constructor(status, code, message) {
    super(message); this.status = status; this.code = code;
  }
}
function plainObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
const state = {
  core: {value: "initial"},
  coreVersion: 1,
  coreFiles: {files: []},
};
let nextConfig;
let backgroundsFail = false;
async function api(method, path, body, contentType, responseInfo) {
  if (path === "/config") {
    const snapshot = await nextConfig;
    responseInfo.configVersion = snapshot.version;
    return snapshot.config;
  }
  if (path === "/backgrounds" && backgroundsFail) throw new Error("files failed");
  return {files: ["ok"]};
}
let resolveOld;
nextConfig = new Promise((resolve) => { resolveOld = resolve; });
const oldRefresh = refreshCoreConfig();
commitCoreSnapshot({config: {value: "new patch"}, version: 4});
resolveOld({config: {value: "old GET"}, version: 3});
(async () => {
  await oldRefresh;
  const afterOld = {value: state.core.value, version: state.coreVersion};

  backgroundsFail = true;
  nextConfig = Promise.resolve({config: {value: "new GET"}, version: 5});
  let filesFailed = false;
  try { await loadCore(); } catch (err) { filesFailed = true; }
  const afterFilesFailure = {
    value: state.core.value, version: state.coreVersion, filesFailed,
  };

  nextConfig = Promise.resolve({config: {value: "unversioned"}, version: null});
  let invalidCode = "";
  try { await loadCoreConfig(); } catch (err) { invalidCode = err.code; }
  process.stdout.write(JSON.stringify({
    afterOld, afterFilesFailure, invalidCode,
  }));
})().catch((err) => { console.error(err); process.exit(1); });
"""
    result = subprocess.run(
        ["node"],
        input=loader + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values == {
        "afterOld": {"value": "new patch", "version": 4},
        "afterFilesFailure": {
            "value": "new GET",
            "version": 5,
            "filesFailed": True,
        },
        "invalidCode": "invalid_response",
    }
    assert 'response.headers.get("x-config-version")' in script
    assert "responseInfo.configVersion = Number.isSafeInteger" in script


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
    assert "const snapshot = await loadCoreConfig();" in helper
    assert "commitCoreSnapshot(snapshot);" in helper
    assert "renderAll();" in helper
    assert helper.index("commitCoreSnapshot(snapshot);") < helper.index("renderAll();")
    assert helper.index("renderAll();") < helper.index("throw err;")

    quality = script.split("// -- camera quality", 1)[1].split(
        "// -- diagnostics and safe runtime settings", 1
    )[0]
    assert "patchCore(" not in quality
    assert quality.count("patchCoreControl(") >= 10


def test_matte_presets_are_versioned_evidence_gated_and_atomic():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    quality = script.split("// -- camera quality", 1)[1].split(
        "// -- diagnostics and safe runtime settings", 1
    )[0]

    assert 'schema: "custback.matte-quality-presets"' in script
    assert "version: 1" in script
    assert 'evidenceStatus: "not_qualified"' in script
    assert "presets: Object.freeze({})" in script
    assert "checked-in evidence has not qualified portable model-backed profiles" in (
        script
    )
    assert "matte-legacy-v1 policy as one atomic patch" in script
    assert "not require deleting configuration or the model cache" in script
    for name in ("performance", "balanced", "quality"):
        button = re.search(
            rf'<button[^>]+data-quality-preset="{name}"[^>]*>', WEBUI_HTML
        )
        assert button is not None
        assert "disabled" in button.group()
        assert f">{name.title()}<" in WEBUI_HTML
    assert 'data-quality-preset="custom"' in WEBUI_HTML
    assert 'id="quality-preset-help" role="status"' in WEBUI_HTML

    # A future qualified definition expands to one detached concrete patch and
    # uses the same atomic activation/rollback helper as individual controls.
    assert "return JSON.parse(JSON.stringify(definition.patch));" in quality
    assert 'withBusy(button, "Applying…", () => patchCoreControl(patch))' in quality
    assert 'api("PATCH", "/config"' not in quality


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_matte_preset_expansion_matches_concrete_values_without_alias_state():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    helpers = (
        "function plainObject"
        + script.split("function plainObject", 1)[1].split(
            "function renderQualityPresets", 1
        )[0]
    )
    harness = r"""
const MATTE_PRESET_NAMES = ["performance", "balanced", "quality"];
const catalog = {
  schema: "custback.matte-quality-presets",
  version: 1,
  evidenceStatus: "qualified",
  presets: {
    balanced: {patch: {
      segmentation: {rvm_downsample: 0.5, mask_shift: 0},
      compositing: {light_wrap: 0.1},
    }},
  },
};
const config = {
  segmentation: {backend: "auto", rvm_downsample: 0.5, mask_shift: 0},
  compositing: {light_wrap: 0.1, use_model_foreground: true},
};
const patch = qualityPresetPatch("balanced", catalog);
patch.segmentation.rvm_downsample = 0.9;
process.stdout.write(JSON.stringify({
  detached: catalog.presets.balanced.patch.segmentation.rvm_downsample,
  matched: matchingQualityPreset(config, catalog),
  custom: matchingQualityPreset({...config,
    segmentation: {...config.segmentation, rvm_downsample: 0.4}}, catalog),
  badVersion: qualityPresetPatch("balanced", {...catalog, version: 2}),
  badEvidence: qualityPresetPatch("balanced",
    {...catalog, evidenceStatus: "not_qualified"}),
}));
"""
    result = subprocess.run(
        ["node"],
        input="const MATTE_PRESET_CATALOG = {};\n" + helpers + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values == {
        "detached": 0.5,
        "matched": "balanced",
        "custom": "custom",
        "badVersion": None,
        "badEvidence": None,
    }


def test_matte_controls_expose_runtime_truth_and_lifecycle():
    for element_id in (
        "quality-runtime",
        "quality-backend-policy",
        "quality-delegate-policy",
        "quality-threshold-policy",
        "quality-rvm-downsample-policy",
        "quality-mask-blur-policy",
        "quality-mask-shift-policy",
        "quality-smoothing-policy",
        "quality-light-wrap-policy",
        "quality-edge-refine-policy",
        "quality-model-foreground-policy",
    ):
        assert f'id="{element_id}"' in WEBUI_HTML
    for element_id in (
        "quality-delegate",
        "quality-threshold",
        "quality-rvm-downsample",
        "quality-mask-blur",
        "quality-mask-shift",
        "quality-smoothing",
        "quality-light-wrap",
        "quality-edge-refine",
        "quality-model-foreground",
    ):
        control = re.search(
            rf'<(?:input|select)[^>]+id="{element_id}"[^>]*>', WEBUI_HTML
        )
        assert control is not None
        assert "disabled" in control.group()

    assert "<summary>Advanced matte controls</summary>" in WEBUI_HTML
    assert WEBUI_HTML.count("Rebuild + reset") >= 7
    assert "applies live without a matte reset" in WEBUI_HTML.lower()
    assert "configured " in WEBUI_HTML
    assert "effective " in WEBUI_HTML

    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    quality = script.split("// -- camera quality", 1)[1].split(
        "// -- diagnostics and safe runtime settings", 1
    )[0]
    for key in (
        "segmentation_selection",
        "matte_policy",
        "matte_rollout",
        "qualityTier",
        "activeDevice",
        "activeProvider",
        "segmentation_update_fps",
        "base_composite_update_fps",
        "cadence_mismatch_active",
    ):
        assert key in script
    for element_id, policy_key in (
        ("quality-threshold", "threshold"),
        ("quality-rvm-downsample", "rvm_downsample_ratio"),
        ("quality-mask-blur", "mask_blur"),
        ("quality-mask-shift", "mask_shift"),
        ("quality-smoothing", "temporal_smoothing"),
        ("quality-light-wrap", "light_wrap"),
        ("quality-edge-refine", "edge_refine"),
        ("quality-model-foreground", "use_model_foreground"),
    ):
        assert f'["{element_id}", "{policy_key}"' in quality
    assert "status.config_version !== configVersion" in quality
    assert 'policy.backend_kind !== "null_passthrough"' in quality
    assert 'selection.selectedBackend.toLowerCase() === "mediapipe"' in quality
    assert "renderMatteQualityRuntime();" in quality
    assert "renderMatteControlPolicy();" in quality


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_matte_rollout_status_is_exact_version_aligned_and_visible():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    helpers = (
        "function currentMatteRollout"
        + script.split("function currentMatteRollout", 1)[1].split(
            "function renderMatteQualityRuntime", 1
        )[0]
    )
    harness = r"""
const rollout = {
  schema: "custback.matte-rollout-status",
  version: 1,
  stage: "compatibility_hold",
  decision: "held_pending_physical_qualification",
  configured_schema_version: 1,
  config_version: 4,
  qualified_default_active: false,
  preset_catalog_version: 1,
  preset_evidence_status: "not_qualified",
  legacy_policy_available: true,
  legacy_policy_active: true,
  rollback_patch_id: "matte-legacy-v1",
  patch_attempts: 3,
  patch_in_flight: 0,
  patch_successes: 2,
  patch_failures: 1,
  legacy_rollbacks: 1,
  last_outcome: "rollback",
};
const status = {config_version: 4, matte_rollout: rollout};
process.stdout.write(JSON.stringify({
  current: currentMatteRollout(status, 4),
  rows: matteRolloutRows(status, 4),
  stale: currentMatteRollout(status, 3),
  nestedStale: currentMatteRollout(
    {...status, matte_rollout: {...rollout, config_version: 3}}, 4),
  unknown: currentMatteRollout(
    {...status, matte_rollout: {...rollout, raw_error: "/private/path"}}, 4),
  inconsistent: currentMatteRollout(
    {...status, matte_rollout: {...rollout, patch_attempts: 4}}, 4),
}));
"""
    result = subprocess.run(
        ["node"],
        input=helpers + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["current"]["rollback_patch_id"] == "matte-legacy-v1"
    assert values["rows"] == [
        [
            "Matte rollout",
            "Compatibility hold · physical qualification pending · "
            "presets unavailable · legacy policy active",
            "warn",
        ],
        [
            "Matte rollout changes",
            "3 attempted · 2 succeeded · 1 failed · 0 in flight · 1 rolled back",
            "warn",
        ],
    ]
    assert values["stale"] is None
    assert values["nestedStale"] is None
    assert values["unknown"] is None
    assert values["inconsistent"] is None


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_backend_aware_matte_control_matrix_and_stale_status_gate():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    plain_object = (
        "function plainObject"
        + script.split("function plainObject", 1)[1].split(
            "function qualityPresetPatch", 1
        )[0]
    )
    policy = (
        "function currentMattePolicy"
        + script.split("function currentMattePolicy", 1)[1].split(
            "function updateFpsText", 1
        )[0]
    )
    harness = r"""
const ids = [
  "quality-threshold", "quality-rvm-downsample", "quality-mask-blur",
  "quality-mask-shift", "quality-smoothing", "quality-light-wrap",
  "quality-edge-refine", "quality-model-foreground",
];
const elements = {
  "quality-backend-policy": {},
  "quality-delegate": {},
  "quality-delegate-policy": {},
};
for (const id of ids) {
  elements[id] = {};
  elements[id + "-policy"] = {};
}
function $(id) { return elements[id]; }
function titleCase(value) { return String(value); }
function backendDisplayName(value) { return String(value); }
function deviceDisplayName(value) { return String(value).toUpperCase(); }
function segmentationSelection(status) { return status.selection; }
const state = {
  coreVersion: 7,
  core: {segmentation: {backend: "auto", delegate: "cpu"}},
  status: null,
};
const c = (configured, effective, state, reason) => ({
  configured, effective, state, reason,
});
const common = {
  mask_shift: c(0, 0, "bypassed", "configured-off"),
  light_wrap: c(0.25, 0.25, "effective", "configured-active"),
};
function status(kind, selected, controls) {
  return {
    config_version: 7,
    matte_policy: {
      schema: "custback.matte-policy",
      version: 1,
      backend_kind: kind,
      effective: {},
      controls,
    },
    selection: {
      structured: true,
      selectedBackend: selected,
      activeDevice: "cpu",
      fallbackActive: false,
    },
  };
}
function disabled() {
  return Object.fromEntries([...ids, "quality-delegate"].map(
    (id) => [id, elements[id].disabled]
  ));
}

state.status = status("true_alpha_recurrent", "rvm", {
  rvm_downsample_ratio: c(0, 0.4, "effective", "runtime-auto-ratio"),
  threshold: c(0.5, null, "inapplicable",
    "rvm-native-alpha-is-never-hard-thresholded"),
  mask_blur: c(7, 0, "bypassed",
    "rvm-native-alpha-bypasses-generic-blur"),
  edge_refine: c(true, false, "bypassed",
    "rvm-native-alpha-bypasses-generic-edge-refinement"),
  temporal_smoothing: c(0.35, 0, "bypassed",
    "rvm-recurrence-bypasses-generic-ema"),
  use_model_foreground: c(true, true, "effective", "configured-active"),
  ...common,
});
renderMatteControlPolicy();
const rvm = disabled();
const rvmBlurHelp = elements["quality-mask-blur-policy"].textContent;

state.status = status("confidence_mask_video", "mediapipe", {
  rvm_downsample_ratio: c(0, null, "inapplicable",
    "selected-backend-does-not-use-rvm-ratio"),
  threshold: c(0.5, null, "inapplicable",
    "mediapipe-confidence-mask-does-not-use-threshold"),
  mask_blur: c(7, 7, "effective", "configured-active"),
  edge_refine: c(true, true, "effective", "configured-active"),
  temporal_smoothing: c(0.35, 0.35, "effective", "configured-active"),
  use_model_foreground: c(true, false, "inapplicable",
    "selected-backend-does-not-produce-clean-foreground"),
  ...common,
});
renderMatteControlPolicy();
const mediapipe = disabled();

state.status = status("null_passthrough", "mediapipe",
  Object.fromEntries([
    ["rvm_downsample_ratio", c(0, null, "inapplicable",
      "null-or-passthrough-has-no-rvm-inference")],
    ["threshold", c(0.5, null, "inapplicable",
      "null-or-passthrough-has-no-mask-threshold")],
    ["mask_blur", c(7, 0, "inapplicable",
      "null-or-passthrough-has-no-matte-refiner")],
    ["edge_refine", c(true, false, "inapplicable",
      "null-or-passthrough-has-no-matte-refiner")],
    ["mask_shift", c(0, 0, "inapplicable",
      "null-or-passthrough-has-no-matte-refiner")],
    ["temporal_smoothing", c(0.35, 0, "inapplicable",
      "null-or-passthrough-has-no-temporal-matte")],
    ["use_model_foreground", c(true, false, "inapplicable",
      "null-or-passthrough-has-no-model-foreground")],
    ["light_wrap", c(0.25, 0, "inapplicable",
      "null-or-passthrough-has-no-soft-edge-composite")],
  ]));
renderMatteControlPolicy();
const passthrough = disabled();

state.coreVersion = 8;
renderMatteControlPolicy();
const stale = disabled();

process.stdout.write(JSON.stringify({
  rvm, mediapipe, passthrough, stale, rvmBlurHelp,
}));
"""
    result = subprocess.run(
        ["node"],
        input=plain_object + policy + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["rvm"] == {
        "quality-threshold": True,
        "quality-rvm-downsample": False,
        "quality-mask-blur": True,
        "quality-mask-shift": False,
        "quality-smoothing": True,
        "quality-light-wrap": False,
        "quality-edge-refine": True,
        "quality-model-foreground": False,
        "quality-delegate": True,
    }
    assert values["mediapipe"] == {
        "quality-threshold": True,
        "quality-rvm-downsample": True,
        "quality-mask-blur": False,
        "quality-mask-shift": False,
        "quality-smoothing": False,
        "quality-light-wrap": False,
        "quality-edge-refine": False,
        "quality-model-foreground": True,
        "quality-delegate": False,
    }
    assert all(values["passthrough"].values())
    assert all(values["stale"].values())
    assert "configured 7; effective 0" in values["rvmBlurHelp"]
    assert "RVM preserves native soft alpha" in values["rvmBlurHelp"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_quality_runtime_does_not_call_selected_backend_active_in_passthrough():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    runtime = (
        "function updateFpsText"
        + script.split("function updateFpsText", 1)[1].split(
            "function colorCorrectionSummary", 1
        )[0]
    )
    harness = r"""
const state = {coreVersion: 4, status: null};
let rendered = [];
function $(id) { return {id}; }
function currentMattePolicy(status) { return status && status.matte_policy; }
function segmentationSelection(status) { return status.selection; }
function backendDisplayName(value) { return String(value).toUpperCase(); }
function deviceDisplayName(value) { return String(value).toUpperCase(); }
function titleCase(value) { return String(value); }
function mattePolicySummary() {
  return ["Effective matte policy", "opaque passthrough", ""];
}
function segmentationFallbackRows() { return []; }
function renderDiagnosticList(node, rows) { rendered = rows; }
const selection = {
  selectedBackend: "rvm", qualityTier: "matting", activeDevice: "cuda",
  activeProvider: "cuda", fallbackActive: false,
};
const base = {
  config_version: 4,
  segmentation_update_fps: 0,
  base_composite_update_fps: 29.8,
  cadence_mismatch_active: false,
  selection,
};
state.status = {...base, matte_policy: {
  backend_kind: "null_passthrough", passthrough: true,
}};
renderMatteQualityRuntime();
const passthrough = rendered;
state.status = {...base, matte_policy: {
  backend_kind: "true_alpha_recurrent", passthrough: false,
}};
renderMatteQualityRuntime();
process.stdout.write(JSON.stringify({passthrough, active: rendered}));
"""
    result = subprocess.run(
        ["node"],
        input=runtime + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["passthrough"][0] == [
        "Matte path",
        "No active matte · passthrough output",
        "",
    ]
    assert values["passthrough"][1][0] == "Selected backend (bypassed)"
    assert all(row[0] != "Active backend" for row in values["passthrough"])
    assert values["active"][0] == ["Active backend", "RVM · matting tier", "good"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_configured_off_is_editable_but_backend_bypass_is_not():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    plain_object = (
        "function plainObject"
        + script.split("function plainObject", 1)[1].split(
            "function qualityPresetPatch", 1
        )[0]
    )
    helpers = (
        "function matteReasonCopy"
        + script.split("function matteReasonCopy", 1)[1].split(
            "const MATTE_CONTROL_BINDINGS", 1
        )[0]
    )
    harness = r"""
function titleCase(value) { return String(value); }
const result = {
  off: matteControlPresentation({
    configured: 0, effective: 0, state: "bypassed", reason: "configured-off",
  }, formatPolicyShift),
  overridden: matteControlPresentation({
    configured: 7, effective: 0, state: "bypassed",
    reason: "rvm-native-alpha-bypasses-generic-blur",
  }, String),
  threshold: matteControlPresentation({
    configured: 0.5, effective: 0.4, state: "effective",
    reason: "heuristic-score-cutoff",
  }, (value) => formatPolicyNumber(value)),
  awaiting: matteControlPresentation({
    configured: 0, effective: null, state: "effective",
    reason: "awaiting-first-rvm-inference",
  }, formatPolicyRatio),
};
process.stdout.write(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node"],
        input=plain_object + helpers + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["off"]["editable"] is True
    assert values["overridden"]["editable"] is False
    assert "configured 7; effective 0" in values["overridden"]["text"]
    assert values["threshold"]["editable"] is True
    assert "configured 0.50; effective 0.40" in values["threshold"]["text"]
    assert values["awaiting"]["editable"] is True
    assert "effective pending first frame" in values["awaiting"]["text"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_mask_blur_slider_emits_only_distinct_canonical_kernel_values():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    canonical = (
        "function canonicalMaskBlur"
        + script.split("function canonicalMaskBlur", 1)[1].split(
            "function mattePolicyValue", 1
        )[0]
    )
    harness = r"""
const previousValues = [0, 1, 3, 7, 149, 151];
const moves = [];
for (const previous of previousValues) {
  if (previous > 0) {
    moves.push([previous, previous - 1,
      canonicalMaskBlur(previous - 1, previous)]);
  }
  if (previous < 151) {
    moves.push([previous, previous + 1,
      canonicalMaskBlur(previous + 1, previous)]);
  }
}
process.stdout.write(JSON.stringify({
  moves,
  directDown: canonicalMaskBlur(8, 9),
  directUp: canonicalMaskBlur(10, 9),
}));
"""
    result = subprocess.run(
        ["node"],
        input=canonical + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    for previous, _raw, emitted in values["moves"]:
        assert emitted == 0 or emitted % 2 == 1
        assert emitted != previous
    assert values["directDown"] == 7
    assert values["directUp"] == 11
    assert "dataset.canonicalValue = String(segmentation.mask_blur)" in script
    assert "mask_blur: updateMaskBlur(event)" in script
    assert 'setAttribute("aria-valuetext", formatted)' in script


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
const state = {core: {value: "old"}, coreVersion: 0};
const calls = [];
let renderCount = 0;
let rejectedStatus = 409;
async function patchCore() {
  throw new ApiError(rejectedStatus, "rejected", "rejected");
}
async function loadCoreConfig() {
  calls.push(["GET", "/config"]);
  return {
    config: {value: "effective-" + rejectedStatus},
    version: rejectedStatus,
  };
}
function commitCoreSnapshot(snapshot) {
  state.core = snapshot.config;
  state.coreVersion = snapshot.version;
}
function renderAll() { renderCount += 1; }
(async () => {
  const results = {};
  const versions = {};
  for (const status of [409, 422, 503]) {
    rejectedStatus = status;
    try { await patchCoreControl({camera: {fit_mode: "cover"}}); } catch (err) {}
    results[status] = state.core.value;
    versions[status] = state.coreVersion;
  }
  process.stdout.write(JSON.stringify({results, versions, calls, renderCount}));
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
    assert values["versions"] == {"409": 409, "422": 422, "503": 503}
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
  reuseRatio: formatDiagnostic("base_composite_reuse_ratio", 564 / 1105),
  timing: formatDiagnostic("timing_ms", {"output.submission": 2.9}),
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
    assert values["reuseRatio"] == "51.0%"
    assert values["timing"] == '{"output.submission":2.9}'
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
def test_system_diagnostics_report_backend_selection_and_effective_policy():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    for key in (
        "segmentation_selection",
        "requested_backend",
        "selected_backend",
        "quality_tier",
        "selection_mode",
        "fallback_category",
        "fallback_reason",
        "guidance",
        "active_device",
        "active_provider",
        "matte_policy",
        "matte_rollout",
        "rvm_downsample_ratio",
        "raw_alpha_mode",
    ):
        assert key in script
    assert '["Subject detection", ...segmentationSummary(status)]' in script
    assert "coreRows.push(...segmentationFallbackRows(status))" in script
    assert "if (mattePolicy) coreRows.push(mattePolicy)" in script
    assert "coreRows.push(...matteRolloutRows(status, state.coreVersion))" in script

    title_case = (
        "function titleCase"
        + script.split("function titleCase", 1)[1].split("function setView", 1)[0]
    )
    helpers = (
        "function boundedStatusText"
        + script.split("function boundedStatusText", 1)[1].split(
            "function formatDiagnostic", 1
        )[0]
    )
    harness = r"""
const downgraded = {
  segmentation_backend: "MediaPipeSegmenter",
  segmentation_device: "cpu",
  segmentation_selection: {
    requested_backend: "auto",
    selected_backend: "mediapipe",
    quality_tier: "segmentation",
    selection_mode: "automatic",
    fallback_active: true,
    fallback_category: "runtime-not-installed",
    fallback_reason: "RVM unavailable: runtime not installed",
    guidance: "Install the RVM runtime profile and restart.",
    active_device: "cpu",
    active_provider: "cpu",
    attempts: [],
  },
  matte_policy: {
    effective: {
      raw_alpha_mode: "native_soft_alpha",
      rvm_downsample_ratio: 0.25,
      edge_refine: false,
      residual_temporal_mode: "model_only",
      light_wrap: 0.1,
    },
  },
};
const explicit = {
  segmentation_fallback_active: true,
  segmentation_fallback_reason: "legacy-misclassification",
  segmentation_selection: {
    requested_backend: "mediapipe",
    selected_backend: "mediapipe",
    quality_tier: "segmentation",
    selection_mode: "explicit",
    fallback_active: false,
    fallback_category: "none",
    fallback_reason: "",
    guidance: "",
    active_device: "cpu",
    active_provider: "cpu",
    attempts: [],
  },
};
const legacy = {
  segmentation_backend: "RVMSegmenter",
  segmentation_device: "cpu",
  segmentation_fallback_active: true,
  segmentation_fallback_reason: "ml-backend-unavailable",
};
process.stdout.write(JSON.stringify({
  downgradedSummary: segmentationSummary(downgraded),
  downgradedRows: segmentationFallbackRows(downgraded),
  policy: mattePolicySummary(downgraded),
  explicitSummary: segmentationSummary(explicit),
  explicitRows: segmentationFallbackRows(explicit),
  legacySummary: segmentationSummary(legacy),
  legacyRows: segmentationFallbackRows(legacy),
}));
"""
    result = subprocess.run(
        ["node"],
        input=title_case + helpers + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert values["downgradedSummary"] == [
        "MediaPipe · Segmentation tier · device CPU · "
        "provider CPU · Automatic selection",
        "warn",
    ]
    assert values["downgradedRows"] == [
        [
            "Backend downgrade",
            "Auto → MediaPipe · Runtime Not Installed · "
            "RVM unavailable: runtime not installed · "
            "Install the RVM runtime profile and restart.",
            "warn",
        ]
    ]
    assert values["policy"] == [
        "Effective matte policy",
        "Native Soft Alpha · RVM ratio 0.25 · edge refine off · "
        "temporal Model Only · light wrap 0.1",
        "",
    ]
    assert values["explicitSummary"] == [
        "MediaPipe · Segmentation tier · device CPU · "
        "provider CPU · Explicit selection",
        "",
    ]
    assert values["explicitRows"] == []
    assert values["legacySummary"] == ["RVMSegmenter · Cpu", "warn"]
    assert values["legacyRows"] == [
        [
            "Backend downgrade",
            "Preferred backend unavailable · ml-backend-unavailable",
            "warn",
        ]
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_system_diagnostics_distinguish_visual_cadence_and_exact_repeat_equality():
    script = WEBUI_HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    for key in (
        "segmentation_update_fps",
        "base_composite_update_fps",
        "base_composite_reuse_ratio",
        "exact_final_output_repeat_ratio",
        "output_send_fps",
        "last_unique_frame_age_ms",
        "serialized_new_frame_deadline_misses",
        "output_sink_pacing_events",
        "output_sink_recovery_events",
        "application_pacing_events",
        "output_schedule_late_events",
        "cadence_mismatch_active",
        "timing_schema_version",
        "timing_ms",
        "extensions",
    ):
        assert key in script

    cadence_rows = (
        "function visualCadenceRows"
        + script.split("function visualCadenceRows", 1)[1].split(
            "function geometrySummary", 1
        )[0]
    )
    harness = r"""
const rows = visualCadenceRows({
  segmentation_update_count: 541,
  segmentation_update_fps: 14.9,
  base_composite_update_count: 541,
  base_composite_update_fps: 15.0,
  base_composite_reuse_count: 564,
  base_composite_reuse_fps: 14.7,
  base_composite_reuse_ratio: 564 / 1105,
  exact_final_output_repeat_count: 564,
  exact_final_output_repeat_fps: 14.7,
  exact_final_output_repeat_ratio: 564 / 1104,
  cadence_mismatch_active: true,
});
process.stdout.write(JSON.stringify(rows));
"""
    result = subprocess.run(
        ["node"],
        input=cadence_rows + harness,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows == [
        ["Visual updates", "15.0 fps · 541 total", "warn"],
        ["Segmentation updates", "14.9 fps · 541 total", "warn"],
        ["Safe-base reuse", "14.7 fps · 564 total · 51.0%", "warn"],
        ["Exact final-output repeats", "14.7 fps · 564 total · 51.1%", "warn"],
    ]


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
