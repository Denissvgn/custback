"""The control page: served only behind auth and internally consistent."""

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
