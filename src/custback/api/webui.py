"""The custback control page served at ``GET /``.

One self-contained HTML document (inline CSS/JS, no external assets, CSP
friendly) that drives both control planes through this origin only:
custback's own API directly, and the avatar service through the
``/avatar/*`` proxy. Authentication is the existing browser session — the
unauthenticated shell in :data:`custback.api.server.LOGIN_HTML` handles
sign-in before this page is ever served.

Living in a Python module keeps the page inside the reviewed ``src/**/*.py``
release payload without new packaging rules.
"""

from __future__ import annotations

WEBUI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>custback control</title>
<style>
:root{
  --bg:#111417;--panel:#1a1f24;--panel-2:#22282e;--line:#2e363d;
  --text:#e8ecef;--muted:#9aa7b0;--accent:#7fb8e6;--accent-2:#2b5f8a;
  --ok:#69c284;--warn:#e0b060;--err:#e07a6a;--radius:10px;
}
*{box-sizing:border-box}
body{margin:0;font:15px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--text)}
header{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;
  padding:.7rem 1.2rem;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font-size:1.05rem;margin:0;letter-spacing:.04em}
header .chips{display:flex;gap:.5rem;flex-wrap:wrap;margin-left:auto}
.chip{font-size:.78rem;padding:.15rem .55rem;border-radius:999px;
  border:1px solid var(--line);color:var(--muted);white-space:nowrap}
.chip.on{color:var(--ok);border-color:var(--ok)}
.chip.off{color:var(--muted)}
.chip.bad{color:var(--err);border-color:var(--err)}
main{display:grid;grid-template-columns:minmax(340px,7fr) minmax(320px,5fr);
  gap:1rem;padding:1rem;max-width:1500px;margin:0 auto}
@media (max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);
  border-radius:var(--radius);padding:1rem;margin-bottom:1rem}
.card h2{margin:.1rem 0 .8rem;font-size:.95rem;color:var(--accent);
  text-transform:uppercase;letter-spacing:.08em}
.row{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;margin:.45rem 0}
.row label{min-width:7.5rem;color:var(--muted);font-size:.86rem}
.grow{flex:1}
select,input[type=text],input[type=url]{background:var(--panel-2);color:var(--text);
  border:1px solid var(--line);border-radius:6px;padding:.4rem .55rem;font:inherit}
input[type=range]{flex:1;accent-color:var(--accent)}
input[type=color]{width:2.6rem;height:1.9rem;border:1px solid var(--line);
  border-radius:6px;background:var(--panel-2);padding:2px}
button{background:var(--panel-2);color:var(--text);border:1px solid var(--line);
  border-radius:6px;padding:.42rem .8rem;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent-2);border-color:var(--accent-2)}
button:disabled{opacity:.45;cursor:default}
.value{min-width:3.2rem;text-align:right;color:var(--muted);font-size:.84rem;
  font-variant-numeric:tabular-nums}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:.35rem .8rem}
.seg button.active{background:var(--accent-2);color:#fff}
.switch{position:relative;width:2.9rem;height:1.55rem;flex:none}
.switch input{opacity:0;width:0;height:0}
.switch span{position:absolute;inset:0;border-radius:999px;background:var(--panel-2);
  border:1px solid var(--line);transition:.15s}
.switch span::after{content:"";position:absolute;top:2px;left:3px;width:1.1rem;
  height:1.1rem;border-radius:50%;background:var(--muted);transition:.15s}
.switch input:checked+span{background:var(--accent-2);border-color:var(--accent-2)}
.switch input:checked+span::after{left:1.5rem;background:#fff}
#preview-box{position:relative;background:#000;border-radius:var(--radius);
  overflow:hidden;border:1px solid var(--line);min-height:220px}
#preview{display:block;width:100%;min-height:220px;object-fit:contain}
#preview-msg{position:absolute;inset:0;display:flex;align-items:center;
  justify-content:center;color:var(--muted);pointer-events:none}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(108px,1fr));gap:.6rem}
.tile{position:relative;border:2px solid var(--line);border-radius:8px;
  overflow:hidden;cursor:pointer;background:var(--panel-2);padding:0;text-align:center}
.tile img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover}
.tile .name{display:block;font-size:.74rem;color:var(--muted);
  padding:.2rem .3rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tile.active{border-color:var(--accent)}
.tile .del{position:absolute;top:2px;right:2px;background:rgba(0,0,0,.55);
  border:0;border-radius:5px;color:#eee;font-size:.8rem;line-height:1;
  padding:.15rem .4rem;display:none}
.tile:hover .del{display:block}
.tile.upload{display:flex;align-items:center;justify-content:center;
  aspect-ratio:16/11;color:var(--muted);border-style:dashed;font-size:.85rem}
.modes{display:grid;gap:.45rem}
.mode{display:flex;gap:.55rem;align-items:baseline;padding:.5rem .6rem;
  border:1px solid var(--line);border-radius:8px;cursor:pointer}
.mode input{accent-color:var(--accent)}
.mode.active{border-color:var(--accent)}
.mode.unavailable{opacity:.5;cursor:default}
.mode small{display:block;color:var(--muted)}
.parts{display:flex;gap:.65rem;flex-wrap:wrap}
.parts label{min-width:0;display:flex;gap:.25rem;align-items:center;
  font-size:.84rem;color:var(--text)}
.hint{color:var(--muted);font-size:.82rem;margin:.3rem 0}
.notice{border:1px solid var(--warn);border-radius:8px;color:var(--warn);
  padding:.6rem .8rem;font-size:.86rem;margin:.5rem 0}
#toasts{position:fixed;right:1rem;bottom:1rem;display:grid;gap:.5rem;z-index:50;
  max-width:26rem}
.toast{background:var(--panel-2);border:1px solid var(--line);border-left:4px solid
  var(--accent);border-radius:8px;padding:.55rem .8rem;font-size:.86rem;
  box-shadow:0 4px 14px rgba(0,0,0,.4)}
.toast.err{border-left-color:var(--err)}
.toast.warn{border-left-color:var(--warn)}
#session-banner{display:none;background:var(--err);color:#fff;text-align:center;
  padding:.5rem}
#session-banner button{margin-left:.8rem;background:rgba(0,0,0,.25);border:0;color:#fff}
.hidden{display:none !important}
footer{color:var(--muted);font-size:.78rem;text-align:center;padding:0 1rem 1.2rem}
footer a{color:var(--accent)}
</style>
</head>
<body>
<div id="session-banner">Session expired.<button onclick="location.reload()">Sign in again</button></div>
<header>
  <h1>custback</h1>
  <label class="row" style="margin:0;gap:.45rem">
    <span style="color:var(--muted);font-size:.86rem">Avatar</span>
    <span class="switch"><input type="checkbox" id="avatar-enabled"><span></span></span>
  </label>
  <div class="chips">
    <span class="chip" id="chip-mode">mode: —</span>
    <span class="chip" id="chip-fps">— fps</span>
    <span class="chip" id="chip-remote">avatar: —</span>
    <span class="chip" id="chip-driver">driver: —</span>
  </div>
</header>
<main>
<section>
  <div class="card">
    <h2>Preview</h2>
    <div class="row" style="margin-top:0">
      <div class="seg" id="preview-source">
        <button data-src="output" class="active">Camera output</button>
        <button data-src="avatar">Avatar render</button>
      </div>
      <span class="hint grow" id="preview-hint">what the meeting sees</span>
    </div>
    <div id="preview-box">
      <img id="preview" alt="live preview">
      <div id="preview-msg">connecting…</div>
    </div>
  </div>
  <div class="card">
    <h2>Background</h2>
    <div class="row" style="margin-top:0">
      <div class="seg" id="bg-scope">
        <button data-scope="camera" class="active">Camera</button>
        <button data-scope="avatar">Avatar scene</button>
      </div>
      <span class="hint grow" id="bg-scope-hint"></span>
    </div>
    <div class="row"><label>Mode</label><div class="seg" id="bg-modes"></div></div>
    <div class="row" id="bg-color-row"><label>Color</label>
      <input type="color" id="bg-color"></div>
    <div class="row" id="bg-blur-row"><label>Blur</label>
      <input type="range" id="bg-blur" min="3" max="151" step="2">
      <span class="value" id="bg-blur-value"></span></div>
    <div class="row"><label>Gallery</label>
      <button id="bg-upload-btn">Upload image / video…</button>
      <input type="file" id="bg-upload" class="hidden"
        accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.webm,.mov,.mkv,.gif,.avi"></div>
    <div class="tiles" id="bg-tiles"></div>
  </div>
</section>
<section>
  <div class="card" id="avatar-card">
    <h2>Avatar</h2>
    <div id="avatar-setup" class="hidden">
      <div class="notice" id="avatar-setup-msg"></div>
      <p class="hint">The avatar control endpoint and its credential are
      operator-owned startup settings. Configure <code>avatar.url</code> and
      <code>avatar.token_file</code> (plus TLS trust/mTLS files for HTTPS),
      then restart custback.</p>
    </div>
    <div id="avatar-controls" class="hidden">
      <div class="tiles" id="avatar-tiles"></div>
      <div class="row"><label>Follows</label></div>
      <div class="modes" id="avatar-modes"></div>
      <div class="row"><label>Style</label><div class="seg" id="avatar-style"></div></div>
      <div class="row"><label>Framing</label><div class="seg" id="avatar-framing"></div></div>
      <div class="row"><label>Size</label>
        <input type="range" id="avatar-scale" min="0.1" max="3" step="0.05">
        <span class="value" id="avatar-scale-value"></span></div>
      <div class="row"><label>Position</label>
        <input type="range" id="avatar-x" min="-1" max="1" step="0.02" title="horizontal">
        <input type="range" id="avatar-y" min="-1" max="1" step="0.02" title="vertical">
      </div>
      <div class="row"><label>Smoothing</label>
        <input type="range" id="avatar-smoothing" min="0" max="0.95" step="0.05">
        <span class="value" id="avatar-smoothing-value"></span></div>
      <div class="row"><label>Parts</label><div class="parts" id="avatar-parts"></div></div>
      <div class="row"><label>Head follows pose</label>
        <span class="switch"><input type="checkbox" id="avatar-follow"><span></span></span>
      </div>
    </div>
  </div>
</section>
</main>
<footer>API docs at <a href="/docs">/docs</a> · hot changes apply live;
greyed options need a service restart or a missing extra.</footer>
<div id="toasts"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const LOCAL_MODES = ["blur", "image", "video", "color", "camera"];
const CORE_BG_MODES = ["blur", "color", "image", "video", "camera", "passthrough"];
const AVATAR_BG_MODES = ["color", "image", "video", "blur"];
const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];

const state = {
  core: null,            // core /config body
  coreVersion: -1,
  avatar: null,          // avatar /config body (via proxy)
  avatarVersion: -1,
  avatarInfo: null,      // /avatar/avatars body
  avatarState: "loading",// ok | unconfigured | unreachable | loading
  coreFiles: {files: [], directory: ""},
  avatarFiles: {files: []},
  bgScope: "camera",
  previewSource: "output",
  scopeFollowsToggle: true,
};

function toast(message, kind) {
  const node = document.createElement("div");
  node.className = "toast" + (kind ? " " + kind : "");
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), 6000);
}

class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function api(method, path, body, contentType) {
  const init = {method, headers: {}};
  if (body !== undefined && body !== null) {
    if (body instanceof Blob || body instanceof FormData) {
      init.body = body;
      if (contentType) init.headers["content-type"] = contentType;
    } else {
      init.headers["content-type"] = "application/json";
      init.body = JSON.stringify(body);
    }
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    throw new ApiError(0, "network", "custback is not reachable");
  }
  if (response.status === 401) {
    $("session-banner").style.display = "block";
    throw new ApiError(401, "unauthorized", "session expired");
  }
  if (!response.ok) {
    let detail = {};
    try { detail = (await response.json()).detail || {}; } catch (err) { /* text body */ }
    let message = detail.message || response.statusText;
    if (detail.code === "restart_required") {
      message = "restart required to apply: " + (detail.fields || []).join(", ");
    }
    throw new ApiError(response.status, detail.code || "error", message);
  }
  if (response.status === 204) return null;
  const kind = response.headers.get("content-type") || "";
  return kind.includes("json") ? response.json() : response;
}

function reportError(err) {
  if (err instanceof ApiError && err.status === 401) return;
  toast(err.message || String(err), "err");
}

// -- config patching ---------------------------------------------------------

async function patchCore(patch) {
  const body = await api("PATCH", "/config", patch);
  state.core = body.config;
  state.coreVersion = body.config_version;
  renderAll();
}

async function patchAvatar(patch) {
  const body = await api("PATCH", "/avatar/config", patch);
  state.avatar = body.config;
  state.avatarVersion = body.config_version;
  renderAll();
}

// -- data loading -------------------------------------------------------------

async function loadCore() {
  state.core = await api("GET", "/config");
  state.coreFiles = await api("GET", "/backgrounds");
}

async function loadAvatar() {
  try {
    state.avatar = await api("GET", "/avatar/config");
    state.avatarInfo = await api("GET", "/avatar/avatars");
    state.avatarFiles = await api("GET", "/avatar/backgrounds");
    state.avatarState = "ok";
  } catch (err) {
    state.avatar = null;
    state.avatarState = err.code === "avatar_unconfigured"
      ? "unconfigured" : "unreachable";
    if (err.status === 401) throw err;
  }
}

async function refreshAll() {
  try { await loadCore(); } catch (err) { reportError(err); }
  await loadAvatar().catch(reportError);
  renderAll();
}

// -- header / status ----------------------------------------------------------

function applyStatus(status, avatarStatus) {
  if (status) {
    $("chip-mode").textContent = "mode: " + status.mode;
    $("chip-fps").textContent = status.fps.toFixed(0) + " fps";
    const remote = $("chip-remote");
    if (status.mode !== "remote") {
      remote.textContent = "avatar: off";
      remote.className = "chip off";
    } else if (status.remote_connected && !status.remote_fallback_active) {
      remote.textContent = "avatar: live";
      remote.className = "chip on";
    } else {
      remote.textContent = "avatar: fallback (" + status.remote_fallback_mode + ")";
      remote.className = "chip bad";
    }
    if (status.config_version !== state.coreVersion && state.coreVersion >= 0) {
      refreshAll();
    }
    state.coreVersion = status.config_version;
  }
  const driver = $("chip-driver");
  if (avatarStatus) {
    driver.textContent = "driver: " + (avatarStatus.driver_backend || "—");
    driver.className = "chip" + (avatarStatus.connected ? " on" : "");
    if (avatarStatus.config_version !== state.avatarVersion
        && state.avatarVersion >= 0) {
      loadAvatar().then(renderAll).catch(reportError);
    }
    state.avatarVersion = avatarStatus.config_version;
  } else {
    driver.textContent = "driver: —";
    driver.className = "chip off";
  }
}

async function poll() {
  let status = null, avatarStatus = null;
  try { status = await api("GET", "/status"); } catch (err) { /* offline */ }
  if (state.avatarState === "ok" || state.avatarState === "unreachable") {
    try {
      avatarStatus = await api("GET", "/avatar/status");
      if (state.avatarState !== "ok") { await loadAvatar(); renderAll(); }
    } catch (err) { /* stays unreachable */ }
  }
  applyStatus(status, avatarStatus);
}

// -- preview -------------------------------------------------------------------

let previewRetry = null;
function setPreview() {
  const source = state.previewSource;
  const path = source === "avatar" ? "/avatar/video/mjpeg" : "/video/mjpeg";
  $("preview-hint").textContent = source === "avatar"
    ? "raw avatar render (via the avatar service)"
    : "what the meeting sees";
  $("preview-msg").textContent = "connecting…";
  $("preview").src = path + "?t=" + Date.now();
}
$("preview").addEventListener("load", () => { $("preview-msg").textContent = ""; });
$("preview").addEventListener("error", () => {
  $("preview-msg").textContent = "stream unavailable — retrying…";
  clearTimeout(previewRetry);
  previewRetry = setTimeout(setPreview, 2500);
});
$("preview-source").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  state.previewSource = button.dataset.src;
  for (const other of $("preview-source").children) {
    other.classList.toggle("active", other === button);
  }
  setPreview();
});

// -- avatar enable toggle --------------------------------------------------------

$("avatar-enabled").addEventListener("change", async (event) => {
  const on = event.target.checked;
  try {
    if (!state.core) throw new ApiError(0, "no_config", "config not loaded yet");
    const mode = state.core.background.mode;
    if (on) {
      const patch = {background: {mode: "remote"}};
      if (LOCAL_MODES.includes(mode)) patch.background.remote_fallback_mode = mode;
      await patchCore(patch);
      if (state.scopeFollowsToggle) setScope("avatar");
      toast("Avatar output enabled — the vcam now shows the avatar service");
    } else {
      const fallback = state.core.background.remote_fallback_mode || "blur";
      await patchCore({background: {mode: fallback}});
      if (state.scopeFollowsToggle) setScope("camera");
      toast("Avatar output disabled — back to the " + fallback + " background");
    }
  } catch (err) {
    reportError(err);
    renderAll();
  }
});

// -- background panel -------------------------------------------------------------

function setScope(scope) {
  state.bgScope = scope;
  for (const button of $("bg-scope").children) {
    button.classList.toggle("active", button.dataset.scope === scope);
  }
  renderBackground();
}
$("bg-scope").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  state.scopeFollowsToggle = false;
  setScope(button.dataset.scope);
});

function bgConfig() {
  return state.bgScope === "camera"
    ? (state.core && state.core.background)
    : (state.avatar && state.avatar.background);
}

async function patchBackground(values) {
  if (state.bgScope === "camera") await patchCore({background: values});
  else await patchAvatar({background: values});
}

function bgrToHex(color) {
  const [b, g, r] = color;
  return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}
function hexToBgr(hex) {
  return [
    parseInt(hex.slice(5, 7), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(1, 3), 16),
  ];
}

function renderBackground() {
  const cameraScope = state.bgScope === "camera";
  $("bg-scope-hint").textContent = cameraScope
    ? "backdrop behind you (used while the avatar is off)"
    : "scene behind the avatar";
  const config = bgConfig();
  const modesBox = $("bg-modes");
  modesBox.textContent = "";
  if (!config) {
    $("bg-tiles").textContent = "";
    if (!cameraScope) {
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = "connect the avatar service to edit its scene";
      $("bg-tiles").appendChild(hint);
    }
    $("bg-color-row").classList.add("hidden");
    $("bg-blur-row").classList.add("hidden");
    return;
  }
  const modes = cameraScope ? CORE_BG_MODES : AVATAR_BG_MODES;
  for (const mode of modes) {
    const button = document.createElement("button");
    button.textContent = mode;
    button.classList.toggle("active", config.mode === mode);
    const needsFile = mode === "image" || mode === "video";
    const hasFile = needsFile
      && (mode === "image" ? config.image_path : config.video_path);
    button.addEventListener("click", () => {
      if (needsFile && !hasFile) {
        toast("Pick a file from the gallery below to use the "
          + mode + " background", "warn");
        return;
      }
      patchBackground({mode}).catch(reportError);
    });
    modesBox.appendChild(button);
  }
  $("bg-color-row").classList.toggle("hidden", config.mode !== "color");
  $("bg-color").value = bgrToHex(config.color || [60, 46, 32]);
  const blurry = config.mode === "blur"
    || (cameraScope && config.mode === "remote"
        && config.remote_fallback_mode === "blur");
  $("bg-blur-row").classList.toggle("hidden", !blurry);
  $("bg-blur").value = config.blur_strength;
  $("bg-blur-value").textContent = config.blur_strength;
  renderBackgroundTiles(config, cameraScope);
}

function coreFileEntries() {
  const directory = state.coreFiles.directory || "";
  return (state.coreFiles.files || []).map((name) => ({
    name,
    kind: IMAGE_EXTS.some((ext) => name.toLowerCase().endsWith(ext))
      ? "image" : "video",
    path: directory + "/" + name,
    thumbnail: "/backgrounds/" + encodeURIComponent(name) + "/thumbnail.jpg",
  }));
}

function avatarFileEntries() {
  return (state.avatarFiles.files || []).map((media) => ({
    name: media.name,
    kind: media.kind,
    path: media.path,
    thumbnail: "/avatar/backgrounds/" + encodeURIComponent(media.name)
      + "/thumbnail.jpg",
  }));
}

function renderBackgroundTiles(config, cameraScope) {
  const box = $("bg-tiles");
  box.textContent = "";
  const entries = cameraScope ? coreFileEntries() : avatarFileEntries();
  for (const entry of entries) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.role = "button";
    const active = (entry.kind === "image"
      ? config.image_path : config.video_path) === entry.path
      && config.mode === entry.kind;
    tile.classList.toggle("active", active);
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = entry.thumbnail;
    img.alt = entry.name;
    tile.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = entry.name;
    tile.appendChild(label);
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "×";
    del.title = "delete " + entry.name;
    del.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        const path = cameraScope
          ? "/backgrounds/" + encodeURIComponent(entry.name)
          : "/avatar/backgrounds/" + encodeURIComponent(entry.name);
        await api("DELETE", path);
        await (cameraScope
          ? api("GET", "/backgrounds").then((body) => { state.coreFiles = body; })
          : api("GET", "/avatar/backgrounds").then((body) => { state.avatarFiles = body; }));
        renderBackground();
      } catch (err) { reportError(err); }
    });
    tile.appendChild(del);
    tile.addEventListener("click", () => {
      const values = entry.kind === "image"
        ? {mode: "image", image_path: entry.path}
        : {mode: "video", video_path: entry.path};
      patchBackground(values).catch(reportError);
    });
    box.appendChild(tile);
  }
}

$("bg-upload-btn").addEventListener("click", () => $("bg-upload").click());
$("bg-upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  const isImage = IMAGE_EXTS.some((ext) => file.name.toLowerCase().endsWith(ext));
  const kind = isImage ? "image" : "video";
  try {
    toast("Uploading " + file.name + "…");
    if (state.bgScope === "camera") {
      const form = new FormData();
      form.append("file", file, file.name);
      await api("POST", "/background/" + kind, form);
      state.coreFiles = await api("GET", "/backgrounds");
    } else {
      await api("POST", "/avatar/backgrounds/" + kind
        + "?name=" + encodeURIComponent(file.name), file,
        file.type || "application/octet-stream");
      state.avatarFiles = await api("GET", "/avatar/backgrounds");
    }
    toast(file.name + " uploaded");
    renderBackground();
  } catch (err) { reportError(err); }
});

// -- avatar panel -----------------------------------------------------------------

function segButtons(container, options, current, onPick) {
  container.textContent = "";
  for (const option of options) {
    const button = document.createElement("button");
    button.textContent = option;
    button.classList.toggle("active", option === current);
    button.addEventListener("click", () => onPick(option));
    container.appendChild(button);
  }
}

function renderAvatarTiles() {
  const box = $("avatar-tiles");
  box.textContent = "";
  const info = state.avatarInfo;
  const appearance = state.avatar.appearance;
  for (const name of info.avatars) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.role = "button";
    tile.classList.toggle("active",
      appearance.rig === "builtin" && appearance.avatar === name);
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/avatar/avatars/" + name + "/thumbnail.jpg?style="
      + appearance.style;
    img.alt = name;
    tile.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;
    tile.appendChild(label);
    tile.addEventListener("click", () => {
      patchAvatar({appearance: {rig: "builtin", avatar: name}}).catch(reportError);
    });
    box.appendChild(tile);
  }
  for (const rig of info.rigs || []) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.role = "button";
    const active = appearance.rig === rig.name
      || appearance.rig.endsWith("/" + rig.name);
    tile.classList.toggle("active", active);
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/avatar/rigs/" + encodeURIComponent(rig.name)
      + "/thumbnail.jpg?v=" + rig.size_bytes;
    img.alt = rig.name;
    tile.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = rig.name + " (custom)";
    tile.appendChild(label);
    if (!active) {
      const del = document.createElement("button");
      del.className = "del";
      del.textContent = "×";
      del.title = "delete rig " + rig.name;
      del.addEventListener("click", async (event) => {
        event.stopPropagation();
        try {
          await api("DELETE", "/avatar/rigs/" + encodeURIComponent(rig.name));
          state.avatarInfo = await api("GET", "/avatar/avatars");
          renderAvatar();
        } catch (err) { reportError(err); }
      });
      tile.appendChild(del);
    }
    tile.addEventListener("click", () => {
      patchAvatar({appearance: {rig: rig.name}}).catch(reportError);
    });
    box.appendChild(tile);
  }
  const upload = document.createElement("button");
  upload.className = "tile upload";
  upload.textContent = "+ rig (.zip)";
  upload.title = "Upload a PNG-layer rig: <part>.png files plus optional "
    + "rig.yaml, zipped";
  upload.addEventListener("click", () => $("rig-upload").click());
  box.appendChild(upload);
}

const rigInput = document.createElement("input");
rigInput.type = "file";
rigInput.accept = ".zip";
rigInput.id = "rig-upload";
rigInput.className = "hidden";
document.body.appendChild(rigInput);
rigInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  const name = file.name.replace(/\.zip$/i, "").toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-").replace(/^[-_]+|[-_]+$/g, "")
    .slice(0, 32) || "rig";
  try {
    toast("Uploading rig " + name + "…");
    await api("POST", "/avatar/rigs?name=" + encodeURIComponent(name),
      file, "application/zip");
    state.avatarInfo = await api("GET", "/avatar/avatars");
    renderAvatar();
    toast("Rig " + name + " installed — click its tile to use it");
  } catch (err) { reportError(err); }
});

function renderAvatarModes() {
  const box = $("avatar-modes");
  box.textContent = "";
  const backend = state.avatar.driver.backend;
  for (const mode of state.avatarInfo.modes) {
    const selectable = mode.available && mode.configured;
    const row = document.createElement("label");
    row.className = "mode" + (selectable ? "" : " unavailable")
      + (mode.active ? " active" : "");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "avatar-mode";
    radio.checked = mode.backend === backend;
    radio.disabled = !selectable;
    radio.addEventListener("change", () => {
      patchAvatar({driver: {backend: mode.backend}}).catch((err) => {
        reportError(err);
        renderAvatarModes();
      });
    });
    row.appendChild(radio);
    const text = document.createElement("span");
    text.className = "grow";
    text.append(mode.label);
    const small = document.createElement("small");
    small.textContent = !mode.available
      ? mode.reason || mode.description
      : (!mode.configured
        ? "configure the operator-owned endpoint and restart the avatar service"
        : mode.description);
    text.appendChild(small);
    row.appendChild(text);
    box.appendChild(row);
  }
}

function renderAvatar() {
  const setup = $("avatar-setup");
  const controls = $("avatar-controls");
  if (state.avatarState !== "ok" || !state.avatar) {
    setup.classList.remove("hidden");
    controls.classList.add("hidden");
    $("avatar-setup-msg").textContent = state.avatarState === "unreachable"
      ? "The avatar service is configured but not answering. Is custback-avatar running?"
      : "No avatar service is configured. Update the startup configuration and restart custback.";
    return;
  }
  setup.classList.add("hidden");
  controls.classList.remove("hidden");
  const appearance = state.avatar.appearance;
  renderAvatarTiles();
  renderAvatarModes();
  segButtons($("avatar-style"), state.avatarInfo.styles, appearance.style,
    (style) => patchAvatar({appearance: {style}}).catch(reportError));
  segButtons($("avatar-framing"), state.avatarInfo.framings, appearance.framing,
    (framing) => patchAvatar({appearance: {framing}}).catch(reportError));
  $("avatar-scale").value = appearance.scale;
  $("avatar-scale-value").textContent = Number(appearance.scale).toFixed(2);
  $("avatar-x").value = appearance.offset_x;
  $("avatar-y").value = appearance.offset_y;
  $("avatar-smoothing").value = state.avatar.driver.smoothing;
  $("avatar-smoothing-value").textContent =
    Number(state.avatar.driver.smoothing).toFixed(2);
  $("avatar-follow").checked = appearance.follow_pose;
  const parts = $("avatar-parts");
  parts.textContent = "";
  for (const part of state.avatarInfo.parts) {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = appearance.parts.includes(part);
    box.addEventListener("change", () => {
      const selected = state.avatarInfo.parts.filter((name) =>
        name === part ? box.checked
          : appearance.parts.includes(name));
      patchAvatar({appearance: {parts: selected}}).catch((err) => {
        reportError(err);
        renderAvatar();
      });
    });
    label.appendChild(box);
    label.append(part);
    parts.appendChild(label);
  }
}

for (const [id, section, field, parse] of [
  ["avatar-scale", "appearance", "scale", parseFloat],
  ["avatar-x", "appearance", "offset_x", parseFloat],
  ["avatar-y", "appearance", "offset_y", parseFloat],
  ["avatar-smoothing", "driver", "smoothing", parseFloat],
]) {
  $(id).addEventListener("change", (event) => {
    patchAvatar({[section]: {[field]: parse(event.target.value)}})
      .catch(reportError);
  });
}
$("avatar-scale").addEventListener("input", (event) => {
  $("avatar-scale-value").textContent =
    Number(event.target.value).toFixed(2);
});
$("avatar-smoothing").addEventListener("input", (event) => {
  $("avatar-smoothing-value").textContent =
    Number(event.target.value).toFixed(2);
});
$("avatar-follow").addEventListener("change", (event) => {
  patchAvatar({appearance: {follow_pose: event.target.checked}})
    .catch(reportError);
});
$("bg-color").addEventListener("change", (event) => {
  patchBackground({mode: "color", color: hexToBgr(event.target.value)})
    .catch(reportError);
});
$("bg-blur").addEventListener("input", (event) => {
  $("bg-blur-value").textContent = event.target.value;
});
$("bg-blur").addEventListener("change", (event) => {
  patchBackground({blur_strength: parseInt(event.target.value, 10)})
    .catch(reportError);
});

// -- top-level render -----------------------------------------------------------

function renderAll() {
  if (state.core) {
    $("avatar-enabled").checked = state.core.background.mode === "remote";
  }
  renderAvatar();
  renderBackground();
}

refreshAll().then(() => {
  if (state.core && state.core.background.mode === "remote"
      && state.scopeFollowsToggle) {
    setScope("avatar");
  }
  setPreview();
  poll();
  setInterval(poll, 3000);
});
</script>
</body>
</html>
"""
