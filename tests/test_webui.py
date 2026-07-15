"""The control page: served only behind auth and internally consistent."""

import re

import pytest

pytest.importorskip("fastapi")

from custback.api.webui import WEBUI_HTML


def test_page_is_self_contained():
    # The API's CSP allows only same-origin assets plus inline CSS/JS.
    assert "http://" not in WEBUI_HTML.replace("http://127.0.0.1:8711", "")
    assert "https://" not in WEBUI_HTML
    assert "<title>custback control</title>" in WEBUI_HTML


def test_page_targets_both_control_planes():
    for path in (
        "/config",
        "/status",
        "/backgrounds",
        "/background/",
        "/video/mjpeg",
        "/avatar/config",
        "/avatar/status",
        "/avatar/avatars",
        "/avatar/rigs",
        "/avatar/backgrounds",
        "/avatar/video/mjpeg",
    ):
        assert path in WEBUI_HTML, path


def test_every_scripted_element_id_exists_in_the_markup():
    ids = set(re.findall(r'id="([\w-]+)"', WEBUI_HTML))
    ids |= set(re.findall(r'rigInput\.id = "([\w-]+)"', WEBUI_HTML))
    referenced = set(re.findall(r'\$\("([\w-]+)"\)', WEBUI_HTML))
    missing = referenced - ids
    assert not missing, f"script references unknown element ids: {sorted(missing)}"


def test_avatar_enable_toggle_patches_remote_mode():
    assert '{background: {mode: "remote"}}' in WEBUI_HTML
    assert "remote_fallback_mode" in WEBUI_HTML


def test_outbound_avatar_targets_are_not_browser_editable():
    for element_id in ("avatar-url", "avatar-connect", "a2f-url", "a2f-set"):
        assert f'id="{element_id}"' not in WEBUI_HTML
        assert f'$("{element_id}")' not in WEBUI_HTML
    assert "operator-owned startup settings" in WEBUI_HTML
