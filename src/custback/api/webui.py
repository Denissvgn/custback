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
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>custback control</title>
<style>
/* Hallmark · pre-emit critique: P5 H5 E4 S5 R5 V4 */
/* Hallmark · genre: modern-minimal · macrostructure: Workbench · theme: Cobalt
 * tone: technical · anchor hue: cobalt · enrichment: none · nav: N1b · footer: Ft2
 * audience: regular users · use: camera/background/avatar setup
 * contrast: pass (40–41) · slop: pass (42–45) · honest: pass (46)
 * chrome: pass (47) · tokens: pass (48) · responsive: pass (49)
 * icons: pass (30) · mobile: pass (34, 49, 50–57)
 */
:root{
  --color-paper:oklch(98.5% 0.004 250);
  --color-paper-2:oklch(96% 0.008 250);
  --color-paper-3:oklch(92% 0.012 250);
  --color-ink:oklch(20% 0.018 258);
  --color-ink-2:oklch(31% 0.018 257);
  --color-muted:oklch(43% 0.016 257);
  --color-rule:oklch(82% 0.012 250);
  --color-rule-strong:oklch(62% 0.026 252);
  --color-accent:oklch(48% 0.22 256);
  --color-accent-hover:oklch(42% 0.2 256);
  --color-accent-soft:oklch(93% 0.035 256);
  --color-accent-ink:oklch(98.5% 0.004 250);
  --color-focus:oklch(46% 0.2 256);
  --color-success:oklch(42% 0.13 148);
  --color-success-soft:oklch(94% 0.035 148);
  --color-success-on-dark:oklch(80% 0.14 148);
  --color-warning:oklch(45% 0.12 78);
  --color-warning-soft:oklch(94% 0.04 78);
  --color-warning-on-dark:oklch(82% 0.11 78);
  --color-danger:oklch(46% 0.18 25);
  --color-danger-soft:oklch(94% 0.04 25);
  --color-danger-on-dark:oklch(80% 0.13 25);
  --color-graphite:oklch(20% 0.016 260);
  --color-graphite-2:oklch(25% 0.018 260);
  --color-graphite-rule:oklch(38% 0.018 258);
  --color-graphite-text:oklch(94% 0.008 250);
  --color-graphite-muted:oklch(72% 0.012 252);
  --color-overlay:oklch(12% 0.012 260 / 0.78);
  --color-shadow:oklch(20% 0.018 258 / 0.08);
  --color-transparent:oklch(0% 0 0 / 0);
  --font-display:"Cascadia Code","IBM Plex Mono","DejaVu Sans Mono","Liberation Mono",monospace;
  --font-body:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --space-3xs:.125rem;--space-2xs:.25rem;--space-xs:.5rem;--space-sm:.75rem;
  --space-md:1rem;--space-lg:1.5rem;--space-xl:2.5rem;--space-2xl:4rem;
  --space-3xl:6rem;
  --text-xs:.75rem;--text-sm:.875rem;--text-base:1rem;--text-md:1.25rem;
  --text-lg:1.5625rem;--text-xl:1.953rem;--text-2xl:2.441rem;
  --ease-out:cubic-bezier(.16,1,.3,1);--ease-in:cubic-bezier(.7,0,.84,0);
  --ease-in-out:cubic-bezier(.65,0,.35,1);
  --dur-micro:120ms;--dur-short:220ms;--dur-long:420ms;
  --rule-thin:1px;--rule-focus:2px;--radius-control:.375rem;
  --radius-panel:.625rem;--radius-round:999px;
  --z-base:1;--z-raised:10;--z-dropdown:100;--z-sticky:200;
  --z-modal:400;--z-toast:500;--z-tooltip:600;
}
*{box-sizing:border-box}
html,body{overflow-x:clip}
html{background:var(--color-paper)}
body{margin:0;min-width:20rem;background:var(--color-paper);color:var(--color-ink-2);
  font:400 var(--text-base)/1.55 var(--font-body);padding-bottom:env(safe-area-inset-bottom)}
button,input,select{font:inherit}
button,a,input,select,summary{font-family:var(--font-body)}
button,a,.tile-pick,.mode{touch-action:manipulation}
button,.button-link,.nav-tab{min-height:2.75rem;white-space:nowrap}
button,.button-link{display:inline-flex;align-items:center;justify-content:center;gap:var(--space-xs);
  border:var(--rule-thin) solid var(--color-rule-strong);border-radius:var(--radius-control);
  background:var(--color-paper);color:var(--color-ink);padding:var(--space-xs) var(--space-sm);
  text-decoration:none;cursor:pointer;transition:background-color var(--dur-short) var(--ease-out),
  border-color var(--dur-short) var(--ease-out),color var(--dur-short) var(--ease-out),
  transform var(--dur-micro) var(--ease-out)}
button:active,.button-link:active{transform:translateY(1px)}
button:disabled,[aria-disabled="true"]{opacity:.52;cursor:not-allowed;transform:none}
button[data-state="loading"]{cursor:wait}
button[data-state="loading"]::before{content:"";width:.8rem;height:.8rem;border-radius:var(--radius-round);
  border:var(--rule-focus) solid var(--color-paper-3);border-block-start-color:var(--color-accent);
  animation:spin 700ms linear infinite}
button[data-state="error"]{border-color:var(--color-danger);color:var(--color-danger)}
button[data-state="success"]{border-color:var(--color-success);color:var(--color-success)}
button.primary{background:var(--color-accent);border-color:var(--color-accent);
  color:var(--color-accent-ink)}
a{color:var(--color-accent);text-underline-offset:.2em}
:focus-visible{outline:var(--rule-focus) solid var(--color-focus);outline-offset:var(--space-2xs)}
.primary:focus-visible{outline-color:var(--color-ink)}
h1,h2,h3,p{margin-block:0}
h1,h2,h3{color:var(--color-ink);font-family:var(--font-display);font-style:normal;
  overflow-wrap:anywhere;min-width:0}
h1{font-size:clamp(var(--text-lg),4vw,var(--text-2xl));line-height:1.08;letter-spacing:-.04em}
h2{font-size:var(--text-lg);line-height:1.18;letter-spacing:-.025em}
h3{font-size:var(--text-md);line-height:1.25;letter-spacing:-.015em}
code,pre,.tnum{font-family:var(--font-display);font-variant-numeric:tabular-nums}
.visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0,0,0,0);white-space:nowrap;border:0}
[hidden],.hidden{display:none !important}
.grow{flex:1;min-width:0}
.muted,.hint{color:var(--color-muted);font-size:var(--text-sm)}
.hint{max-width:68ch}

#session-banner{display:none;position:relative;z-index:var(--z-toast);padding:var(--space-sm) var(--space-md);
  background:var(--color-danger);color:var(--color-accent-ink);text-align:center}
#session-banner button{min-height:2rem;margin-inline-start:var(--space-sm);border-color:var(--color-accent-ink);
  background:var(--color-transparent);color:var(--color-accent-ink)}
#session-banner button:focus-visible{outline-color:var(--color-accent-ink)}
.appbar{position:sticky;inset-block-start:0;z-index:var(--z-sticky);display:grid;
  grid-template-areas:"brand actions" "nav nav";grid-template-columns:minmax(0,1fr) auto;
  gap:var(--space-sm) var(--space-md);align-items:center;padding:var(--space-sm) max(var(--space-md),env(safe-area-inset-left));
  border-block-end:var(--rule-thin) solid var(--color-rule);background:var(--color-paper)}
.brand{grid-area:brand;display:flex;align-items:baseline;gap:var(--space-xs);min-width:0}
.brand strong{font:700 var(--text-md)/1 var(--font-display);letter-spacing:-.04em;color:var(--color-ink)}
.brand span{display:none;color:var(--color-muted);font-size:var(--text-xs);white-space:nowrap}
.primary-nav{grid-area:nav;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:var(--space-2xs);min-width:0}
.nav-tab{border-color:var(--color-transparent);background:var(--color-transparent);color:var(--color-muted);
  min-width:0;padding-inline:var(--space-2xs);font-size:var(--text-xs);font-weight:600}
.nav-tab.active{border-color:var(--color-rule);background:var(--color-paper-2);color:var(--color-ink)}
.header-actions{grid-area:actions;display:flex;align-items:center;justify-content:flex-end;gap:var(--space-xs)}
.avatar-master{display:flex;align-items:center;gap:var(--space-xs);color:var(--color-ink);
  font-size:var(--text-sm);white-space:nowrap;cursor:pointer}
#avatar-state-label{display:none;color:var(--color-muted)}
.switch{position:relative;display:inline-block;width:2.75rem;height:1.55rem;flex:none}
.switch input{position:absolute;inset:0;z-index:var(--z-raised);width:100%;height:100%;margin:0;
  opacity:0;cursor:pointer}
.switch span{position:absolute;inset:0;border:var(--rule-thin) solid var(--color-rule-strong);
  border-radius:var(--radius-round);background:var(--color-paper-3);
  transition:background-color var(--dur-micro) var(--ease-in-out),border-color var(--dur-micro) var(--ease-in-out)}
.switch span::after{content:"";position:absolute;inset-block-start:.18rem;inset-inline-start:.2rem;
  width:1.05rem;height:1.05rem;border-radius:var(--radius-round);background:var(--color-muted);
  transition:transform var(--dur-micro) var(--ease-in-out),background-color var(--dur-micro) var(--ease-in-out)}
.switch input:checked+span{border-color:var(--color-accent);background:var(--color-accent)}
.switch input:checked+span::after{background:var(--color-accent-ink);transform:translateX(1.15rem)}
.switch input:active+span{border-color:var(--color-accent-hover)}
.switch input:checked:active+span{background:var(--color-accent-hover)}
.switch input:focus-visible+span{outline:var(--rule-focus) solid var(--color-focus);outline-offset:var(--space-2xs)}
#signout{min-height:2.25rem;border-color:var(--color-transparent);padding-inline:var(--space-xs);
  background:var(--color-transparent);color:var(--color-muted);font-size:var(--text-xs)}

.workbench{display:grid;grid-template-columns:minmax(0,1fr);gap:var(--space-lg);
  width:min(100%,92rem);margin-inline:auto;padding:var(--space-md) max(var(--space-md),env(safe-area-inset-left)) var(--space-xl)}
.preview-column{min-width:0;overflow:hidden;border-radius:var(--radius-panel);
  background:var(--color-graphite);color:var(--color-graphite-text)}
.preview-head{display:flex;align-items:flex-start;justify-content:space-between;gap:var(--space-md);
  padding:var(--space-lg)}
.preview-head h1{color:var(--color-graphite-text)}
.preview-head p{max-width:50ch;margin-block-start:var(--space-xs);color:var(--color-graphite-muted);
  font-size:var(--text-sm)}
.live-badge{display:inline-flex;align-items:center;gap:var(--space-xs);flex:none;min-height:2rem;
  border:var(--rule-thin) solid var(--color-graphite-rule);border-radius:var(--radius-round);
  padding:var(--space-2xs) var(--space-sm);color:var(--color-graphite-muted);font-size:var(--text-xs);
  white-space:nowrap}
.status-dot{width:.5rem;height:.5rem;border-radius:var(--radius-round);background:var(--color-warning-on-dark)}
.live-badge.on .status-dot{background:var(--color-success-on-dark)}
.live-badge.bad .status-dot{background:var(--color-danger-on-dark)}
#preview-box{position:relative;overflow:hidden;border-block:var(--rule-thin) solid var(--color-graphite-rule);
  background:var(--color-graphite-2);aspect-ratio:16/9}
#preview{display:block;width:100%;height:100%;object-fit:contain;background:var(--color-graphite)}
#preview.unavailable{visibility:hidden}
#preview-msg{position:absolute;inset:0;display:grid;place-items:center;padding:var(--space-lg);
  background:var(--color-overlay);color:var(--color-graphite-muted);text-align:center;pointer-events:none}
.preview-toolbar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
  gap:var(--space-sm);padding:var(--space-md) var(--space-lg);border-block-end:var(--rule-thin) solid var(--color-graphite-rule)}
.preview-toolbar .seg{border-color:var(--color-graphite-rule)}
.preview-toolbar .seg button{background:var(--color-graphite);color:var(--color-graphite-muted)}
.preview-toolbar .seg button.active{background:var(--color-graphite-text);color:var(--color-graphite)}
.preview-toolbar .button-link{border-color:var(--color-graphite-rule);background:var(--color-graphite);
  color:var(--color-graphite-text)}
.preview-column :focus-visible{outline-color:var(--color-graphite-text)}
.preview-status{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));margin:0;padding:0}
.preview-metric{min-width:0;padding:var(--space-sm) var(--space-md);border-block-end:var(--rule-thin) solid var(--color-graphite-rule)}
.preview-metric:nth-child(odd){border-inline-end:var(--rule-thin) solid var(--color-graphite-rule)}
.preview-metric dt{color:var(--color-graphite-muted);font-size:var(--text-xs)}
.preview-metric dd{margin:var(--space-2xs) 0 0;color:var(--color-graphite-text);font:600 var(--text-sm)/1.2 var(--font-display);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip.on{color:var(--color-success)}
.chip.bad{color:var(--color-danger)}
.preview-column .chip.on{color:var(--color-success-on-dark)}
.preview-column .chip.bad{color:var(--color-danger-on-dark)}

.control-column{min-width:0}
.panel-head{display:grid;gap:var(--space-xs);padding-block:var(--space-xs) var(--space-lg);
  border-block-end:var(--rule-thin) solid var(--color-rule)}
.panel-head p{max-width:60ch;color:var(--color-muted)}
.control-section{display:grid;gap:var(--space-md);padding-block:var(--space-xl);
  border-block-end:var(--rule-thin) solid var(--color-rule)}
.control-section:last-child{border-block-end:0}
.section-copy{display:grid;gap:var(--space-xs)}
.section-copy p{color:var(--color-muted);font-size:var(--text-sm);max-width:65ch}
.field-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:var(--space-md)}
.field{display:grid;align-content:start;gap:var(--space-xs);min-width:0}
.field>label,.field>legend{color:var(--color-ink);font-size:var(--text-sm);font-weight:650}
.field small{display:block;min-height:1lh;color:var(--color-muted);font-size:var(--text-xs)}
fieldset.field{margin:0;padding:0;border:0}
select,input[type="text"],input[type="url"],input[type="number"]{width:100%;min-width:0;height:2.75rem;
  border:var(--rule-thin) solid var(--color-rule-strong);border-radius:var(--radius-control);
  outline:var(--rule-focus) solid var(--color-transparent);outline-offset:var(--rule-thin);
  background:var(--color-paper);color:var(--color-ink);padding-inline:var(--space-sm)}
select:focus-visible,input[type="text"]:focus-visible,input[type="url"]:focus-visible,
input[type="number"]:focus-visible{outline-color:var(--color-focus)}
input:disabled,select:disabled{opacity:.55;cursor:not-allowed;background:var(--color-paper-2)}
.switch input:disabled{opacity:0}
.switch input:disabled+span{opacity:.55;cursor:not-allowed}
.avatar-master:has(input:disabled){cursor:not-allowed}
input[aria-invalid="true"]{border-color:var(--color-danger)}
input[type="range"]{width:100%;min-height:2.75rem;margin:0;accent-color:var(--color-accent)}
input[type="color"]{width:3.5rem;height:2.75rem;padding:var(--space-2xs);border:var(--rule-thin) solid var(--color-rule-strong);
  border-radius:var(--radius-control);background:var(--color-paper)}
input[type="checkbox"],input[type="radio"]{width:1.15rem;height:1.15rem;accent-color:var(--color-accent)}
.range-line{display:grid;grid-template-columns:minmax(0,1fr) 4.5rem;gap:var(--space-sm);align-items:center}
.value{min-width:0;color:var(--color-muted);font:500 var(--text-sm)/1 var(--font-display);
  font-variant-numeric:tabular-nums;text-align:end;white-space:nowrap}
.seg{display:flex;flex-wrap:wrap;align-items:stretch;gap:var(--space-2xs)}
.seg button{flex:1 1 auto;min-width:0;border-color:var(--color-rule);background:var(--color-paper-2);
  color:var(--color-muted);padding-inline:var(--space-sm)}
.seg button.active{border-color:var(--color-accent);background:var(--color-accent-soft);color:var(--color-accent)}
.modes{display:grid;gap:var(--space-xs)}
.mode{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:start;gap:var(--space-sm);
  min-height:3.5rem;padding:var(--space-sm);border:var(--rule-thin) solid var(--color-rule);
  border-radius:var(--radius-control);background:var(--color-paper);cursor:pointer}
.mode.active{border-color:var(--color-accent);background:var(--color-accent-soft)}
.mode.unavailable{opacity:.56;cursor:not-allowed}
.mode small{display:block;margin-block-start:var(--space-2xs);color:var(--color-muted);font-size:var(--text-xs)}
.parts{display:flex;flex-wrap:wrap;gap:var(--space-xs) var(--space-md)}
.parts label{display:flex;align-items:center;gap:var(--space-xs);min-height:2.75rem;color:var(--color-ink);font-size:var(--text-sm)}
.inline-toggle{display:flex;align-items:center;justify-content:space-between;gap:var(--space-md);
  min-height:3.5rem;padding:var(--space-sm) 0}
.inline-toggle label{display:grid;gap:var(--space-2xs);color:var(--color-ink);font-size:var(--text-sm)}
.inline-toggle small{color:var(--color-muted);font-size:var(--text-xs);font-weight:400}
.tiles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--space-sm)}
.tiles>.hint{grid-column:1/-1;max-width:45ch}
.tile{position:relative;min-width:0;overflow:hidden;border:var(--rule-thin) solid var(--color-rule);
  border-radius:var(--radius-control);background:var(--color-paper)}
.tile.active{border-color:var(--color-accent);outline:var(--rule-thin) solid var(--color-accent)}
.tile-pick{display:block;width:100%;min-height:0;padding:0;border:0;border-radius:0;background:var(--color-paper);color:var(--color-ink);
  text-align:start}
.tile-pick img{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:cover;background:var(--color-paper-2)}
.tile .name{display:block;overflow:hidden;padding:var(--space-xs) var(--space-sm);color:var(--color-ink-2);
  font-size:var(--text-xs);text-overflow:ellipsis;white-space:nowrap}
.tile .del{position:absolute;inset-block-start:var(--space-2xs);inset-inline-end:var(--space-2xs);
  min-height:2.25rem;padding-inline:var(--space-xs);border-color:var(--color-danger);
  background:var(--color-paper);color:var(--color-danger);font-size:var(--text-xs)}
.tile.upload{min-height:7rem;border-style:dashed;background:var(--color-paper-2);color:var(--color-muted)}
.notice{padding:var(--space-sm) var(--space-md);border:var(--rule-thin) solid var(--color-warning);
  border-radius:var(--radius-control);background:var(--color-warning-soft);color:var(--color-warning);font-size:var(--text-sm)}
.notice.err{border-color:var(--color-danger);background:var(--color-danger-soft);color:var(--color-danger)}
.notice.good{border-color:var(--color-success);background:var(--color-success-soft);color:var(--color-success)}
.notice.neutral{border-color:var(--color-rule);background:var(--color-paper-2);color:var(--color-ink-2)}
.restart-tag{display:inline-flex;align-items:center;min-height:1.5rem;margin-inline-start:var(--space-xs);
  padding:0 var(--space-xs);border:var(--rule-thin) solid var(--color-warning);border-radius:var(--radius-round);
  background:var(--color-warning-soft);color:var(--color-warning);font-size:var(--text-xs);
  font-weight:650;white-space:nowrap;vertical-align:middle}
.advanced-grid{display:grid;gap:var(--space-lg);padding-block-start:var(--space-sm)}
.advanced-group{display:grid;gap:var(--space-md)}
.advanced-group+.advanced-group{padding-block-start:var(--space-md);border-block-start:var(--rule-thin) solid var(--color-rule)}
.provider-planner{display:grid;gap:var(--space-md);padding:var(--space-md);border:var(--rule-thin) solid var(--color-rule);
  border-radius:var(--radius-panel);background:var(--color-paper-2)}
.provider-status{display:flex;align-items:center;justify-content:space-between;gap:var(--space-md);font-size:var(--text-sm)}
.provider-status strong{color:var(--color-ink)}
.config-output{max-height:16rem;overflow:auto;margin:0;padding:var(--space-md);
  border-block:var(--rule-thin) solid var(--color-rule);background:var(--color-paper);color:var(--color-ink);
  font-size:var(--text-xs);line-height:1.6;white-space:pre-wrap;overflow-wrap:anywhere}
details{min-width:0}
summary{min-height:2.75rem;color:var(--color-ink);font-weight:650;cursor:pointer}
.diagnostic-list{display:grid;margin:0;border-block-start:var(--rule-thin) solid var(--color-rule)}
.diagnostic-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(7rem,auto);gap:var(--space-md);
  padding:var(--space-sm) 0;border-block-end:var(--rule-thin) solid var(--color-rule)}
.diagnostic-row dt{min-width:0;color:var(--color-muted);font-size:var(--text-sm);overflow-wrap:anywhere}
.diagnostic-row dd{min-width:0;margin:0;color:var(--color-ink);font:500 var(--text-sm)/1.4 var(--font-display);
  text-align:end;overflow-wrap:anywhere}
.diagnostic-row dd.good{color:var(--color-success)}
.diagnostic-row dd.warn{color:var(--color-warning)}
.diagnostic-row dd.bad{color:var(--color-danger)}
.action-row{display:flex;flex-wrap:wrap;align-items:center;gap:var(--space-sm)}

#toasts{position:fixed;inset-inline:max(var(--space-md),env(safe-area-inset-left));inset-block-end:max(var(--space-md),env(safe-area-inset-bottom));
  z-index:var(--z-toast);display:grid;justify-items:end;gap:var(--space-xs);pointer-events:none}
.toast{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:var(--space-sm);width:min(100%,28rem);padding:var(--space-sm) var(--space-md);border:var(--rule-thin) solid var(--color-rule-strong);
  border-radius:var(--radius-control);background:var(--color-paper);color:var(--color-ink);box-shadow:0 1px 2px var(--color-shadow);
  font-size:var(--text-sm);pointer-events:auto}
.toast.err{border-color:var(--color-danger);background:var(--color-danger-soft);color:var(--color-danger)}
.toast.warn{border-color:var(--color-warning);background:var(--color-warning-soft);color:var(--color-warning)}
.toast-close{min-height:2rem;padding:0 var(--space-xs);border-color:var(--color-transparent);background:var(--color-transparent);color:currentColor}
.foot-line{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:var(--space-sm);width:min(100%,92rem);
  margin-inline:auto;padding:var(--space-lg) max(var(--space-md),env(safe-area-inset-left));
  border-block-start:var(--rule-thin) solid var(--color-rule);color:var(--color-muted);font-size:var(--text-xs)}
.foot-line a{line-height:1;white-space:nowrap}

@media (hover:hover) and (pointer:fine){
  button:hover,.button-link:hover{border-color:var(--color-accent);background:var(--color-accent-soft);color:var(--color-accent)}
  button.primary:hover{border-color:var(--color-accent-hover);background:var(--color-accent-hover);color:var(--color-accent-ink)}
  .switch input:not(:disabled):hover+span{border-color:var(--color-accent)}
  .tile-pick:hover{background:var(--color-paper-2)}
}
@media (pointer:coarse){button,.button-link,.nav-tab{min-height:3rem}.tile .del{min-height:2.75rem}}
@media (max-width:23.5rem){
  .primary-nav{grid-template-columns:1.35fr repeat(3,minmax(0,1fr))}
  .preview-head{display:grid}.live-badge{justify-self:start}
}
@media (min-width:40rem){
  .brand span,#avatar-state-label{display:inline}
  .workbench{padding:var(--space-lg)}
  .preview-status{grid-template-columns:repeat(4,minmax(0,1fr))}
  .preview-metric{border-block-end:0;border-inline-end:var(--rule-thin) solid var(--color-graphite-rule)}
  .preview-metric:last-child{border-inline-end:0}
  .field-grid.two{grid-template-columns:repeat(2,minmax(0,1fr))}
  .field-grid.uneven{grid-template-columns:minmax(0,1.25fr) minmax(0,.75fr)}
  .tiles{grid-template-columns:repeat(4,minmax(0,1fr))}
}
@media (min-width:56rem){
  .appbar{grid-template-areas:"brand nav actions";grid-template-columns:minmax(10rem,1fr) auto minmax(12rem,1fr);
    padding-inline:max(var(--space-lg),env(safe-area-inset-left))}
  .primary-nav{display:flex;justify-content:center}
  .nav-tab{padding-inline:var(--space-sm);font-size:var(--text-sm)}
}
@media (min-width:68rem){
  .workbench{grid-template-columns:minmax(0,7fr) minmax(22rem,5fr);align-items:start;gap:var(--space-xl);padding-block:var(--space-xl)}
  .preview-column{position:sticky;inset-block-start:6.5rem}
  .tiles{grid-template-columns:repeat(3,minmax(0,1fr))}
}
@media (min-width:90rem){.workbench{gap:var(--space-2xl)}.tiles{grid-template-columns:repeat(4,minmax(0,1fr))}}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;
    scroll-behavior:auto !important;transition-duration:.01ms !important}
  button[data-state="loading"]::before{animation:none}
}
@keyframes spin{to{transform:rotate(1turn)}}
</style>
</head>
<body>
<div id="session-banner" role="alert">Your session expired.<button type="button" id="session-reload">Sign in again</button></div>
<header class="appbar">
  <div class="brand"><strong>custback</strong><span>camera control</span></div>
  <nav class="primary-nav" aria-label="Control areas" role="tablist">
    <button type="button" class="nav-tab active" id="tab-background" data-view="background" role="tab" aria-controls="view-background" aria-selected="true" tabindex="0">Background</button>
    <button type="button" class="nav-tab" id="tab-avatar" data-view="avatar" role="tab" aria-controls="view-avatar" aria-selected="false" tabindex="-1">Avatar</button>
    <button type="button" class="nav-tab" id="tab-quality" data-view="quality" role="tab" aria-controls="view-quality" aria-selected="false" tabindex="-1">Quality</button>
    <button type="button" class="nav-tab" id="tab-system" data-view="system" role="tab" aria-controls="view-system" aria-selected="false" tabindex="-1">System</button>
  </nav>
  <div class="header-actions">
    <label class="avatar-master" for="avatar-enabled">Avatar <span id="avatar-state-label">Off</span>
      <span class="switch"><input type="checkbox" id="avatar-enabled" aria-label="Use avatar output"><span></span></span>
    </label>
    <button type="button" id="signout">Sign out</button>
  </div>
</header>
<main class="workbench">
  <section class="preview-column" aria-labelledby="preview-title">
    <div class="preview-head">
      <div><h1 id="preview-title">Camera output</h1><p id="preview-hint">This is what your meeting receives.</p></div>
      <span class="live-badge" id="live-status" role="status" aria-live="polite" aria-atomic="true"><span class="status-dot"></span><span id="live-status-text">Connecting</span></span>
    </div>
    <div id="preview-box">
      <img id="preview" alt="Live processed camera preview" width="1280" height="720">
      <div id="preview-msg">Connecting to the camera stream…</div>
    </div>
    <div class="preview-toolbar">
      <div class="seg" id="preview-source" role="group" aria-label="Preview source">
        <button type="button" data-src="output" class="active" aria-pressed="true">Final output</button>
        <button type="button" data-src="avatar" aria-pressed="false">Avatar only</button>
      </div>
      <a class="button-link" id="preview-snapshot" href="/video/snapshot.jpg" download="custback-snapshot.jpg">Save snapshot</a>
    </div>
    <dl class="preview-status">
      <div class="preview-metric"><dt>Output</dt><dd class="chip" id="chip-mode">—</dd></div>
      <div class="preview-metric"><dt>Frame rate</dt><dd class="chip" id="chip-fps">— fps</dd></div>
      <div class="preview-metric"><dt>Avatar link</dt><dd class="chip" id="chip-remote">—</dd></div>
      <div class="preview-metric"><dt>Following</dt><dd class="chip" id="chip-driver">—</dd></div>
    </dl>
  </section>

  <section class="control-column">
    <div class="view-panel" id="view-background" data-panel="background" role="tabpanel" aria-labelledby="tab-background">
      <header class="panel-head"><h2>Choose a background</h2><p>Set the scene behind your camera, or prepare a separate scene for the avatar.</p></header>
      <section class="control-section">
        <fieldset class="field"><legend>Apply to</legend>
          <div class="seg" id="bg-scope">
            <button type="button" data-scope="camera" class="active" aria-pressed="true">My camera</button>
            <button type="button" data-scope="avatar" aria-pressed="false">Avatar scene</button>
          </div>
          <small id="bg-scope-hint"></small>
        </fieldset>
        <fieldset class="field"><legend>Background type</legend><div class="seg" id="bg-modes"></div></fieldset>
        <div class="field" id="bg-camera-row" hidden>
          <label for="bg-camera-target">Approved camera</label>
          <select id="bg-camera-target"></select>
          <small id="bg-camera-hint">Only sources approved by the operator are shown.</small>
        </div>
        <div class="field" id="bg-color-row" hidden><label for="bg-color">Background colour</label>
          <input type="color" id="bg-color"><small>Used as a solid, distraction-free backdrop.</small></div>
        <div class="field" id="bg-blur-row" hidden><label for="bg-blur">Blur strength</label>
          <div class="range-line"><input type="range" id="bg-blur" min="3" max="151" step="2"><span class="value" id="bg-blur-value"></span></div>
          <small>Higher values hide more room detail.</small></div>
      </section>
      <section class="control-section">
        <div class="section-copy"><h3>Background library</h3><p>Choose an existing image or video, or add media from this device.</p></div>
        <div class="action-row"><button type="button" id="bg-upload-btn">Upload media</button>
          <input type="file" id="bg-upload" class="hidden" accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.webm,.mov,.mkv,.gif,.avi"></div>
        <p class="hint" id="bg-upload-hint"></p>
        <div class="tiles" id="bg-tiles"></div>
      </section>
    </div>

    <div class="view-panel" id="view-avatar" data-panel="avatar" role="tabpanel" aria-labelledby="tab-avatar" hidden>
      <header class="panel-head"><h2>Configure the avatar</h2><p>Choose who appears, decide whether they follow your camera, and tune the framing.</p></header>
      <section class="control-section">
        <div class="section-copy"><h3>Avatar provider</h3><p>Use a local avatar service, or prepare endpoint fragments for an already-provisioned remote avatar host.</p></div>
        <div class="provider-planner">
          <div class="provider-status"><span>Currently configured</span><strong id="provider-current">Checking…</strong></div>
          <fieldset class="field"><legend>Provider location</legend>
            <div class="seg" id="provider-kind">
              <button type="button" data-provider="local" class="active" aria-pressed="true">Local service</button>
              <button type="button" data-provider="remote" aria-pressed="false">Remote server</button>
            </div>
          </fieldset>
          <div class="field" id="provider-url-row" hidden>
            <label for="provider-url">Remote control URL</label>
            <input type="url" id="provider-url" placeholder="https://avatar.example:8711" inputmode="url" aria-describedby="provider-url-help" aria-errormessage="provider-url-help">
            <small id="provider-url-help" role="status" aria-live="polite" aria-atomic="true">Remote providers require HTTPS and a root URL with no path or credentials.</small>
          </div>
          <div class="field" id="provider-source-row" hidden>
            <label for="provider-source-url">Core frame URL</label>
            <input type="url" id="provider-source-url" placeholder="wss://camera.example:8710" inputmode="url" aria-describedby="provider-source-help" aria-errormessage="provider-source-help">
            <small id="provider-source-help" role="status" aria-live="polite" aria-atomic="true">The remote avatar server uses this secure WebSocket root to receive camera frames from custback.</small>
          </div>
          <pre class="config-output" id="provider-config-output">avatar:
  url: "http://127.0.0.1:8711"
source:
  url: "ws://127.0.0.1:8710"</pre>
          <div class="action-row"><button type="button" id="provider-copy">Copy restart config</button></div>
          <p class="hint" id="provider-config-help">These are non-secret endpoint fragments. Provider credentials, TLS trust, network binds, and allowed origins remain operator-owned and never enter the browser.</p>
        </div>
      </section>
      <section class="control-section" id="avatar-setup" hidden>
        <div class="notice" id="avatar-setup-msg"></div>
        <p class="hint">Also configure <code>avatar.token_file</code> on the core host, <code>source.token_file</code> on the avatar host, and any required TLS trust files before restarting both services.</p>
      </section>
      <div id="avatar-controls" hidden>
        <section class="control-section">
          <div class="section-copy"><h3>Choose an avatar</h3><p>Built-in characters and installed PNG-layer rigs appear together.</p></div>
          <div class="tiles" id="avatar-tiles"></div>
        </section>
        <section class="control-section">
          <div class="section-copy"><h3>How the avatar follows you</h3><p>Use camera movement, voice animation, an automatic choice, or a steady idle pose.</p></div>
          <div class="modes" id="avatar-modes"></div>
        </section>
        <section class="control-section">
          <div class="section-copy"><h3>Appearance and framing</h3><p>These changes apply to the avatar render immediately.</p></div>
          <fieldset class="field"><legend>Visual style</legend><div class="seg" id="avatar-style"></div></fieldset>
          <fieldset class="field"><legend>Framing</legend><div class="seg" id="avatar-framing"></div></fieldset>
          <div class="field"><label for="avatar-scale">Avatar size</label>
            <div class="range-line"><input type="range" id="avatar-scale" min="0.1" max="3" step="0.05"><span class="value" id="avatar-scale-value"></span></div></div>
          <div class="field-grid two">
            <div class="field"><label for="avatar-x">Horizontal position</label><div class="range-line"><input type="range" id="avatar-x" min="-1" max="1" step="0.02"><span class="value" id="avatar-x-value"></span></div></div>
            <div class="field"><label for="avatar-y">Vertical position</label><div class="range-line"><input type="range" id="avatar-y" min="-1" max="1" step="0.02"><span class="value" id="avatar-y-value"></span></div></div>
          </div>
          <div class="field"><label for="avatar-smoothing">Movement smoothing</label>
            <div class="range-line"><input type="range" id="avatar-smoothing" min="0" max="0.95" step="0.05"><span class="value" id="avatar-smoothing-value"></span></div>
            <small>Higher values reduce jitter but react more slowly.</small></div>
          <div class="inline-toggle"><label for="avatar-follow">Head follows camera pose<small>Turn this off to keep the head facing forward.</small></label>
            <span class="switch"><input type="checkbox" id="avatar-follow"><span></span></span></div>
        </section>
        <section class="control-section">
          <details><summary>Advanced avatar controls</summary>
            <div class="field-grid two">
              <fieldset class="field"><legend>Visible parts</legend><div class="parts" id="avatar-parts"></div></fieldset>
              <div class="field"><label for="avatar-max-fps">Maximum render rate</label><div class="range-line"><input type="range" id="avatar-max-fps" min="1" max="240" step="1"><span class="value" id="avatar-max-fps-value"></span></div></div>
              <div class="field"><label for="avatar-jpeg-quality">Stream image quality</label><div class="range-line"><input type="range" id="avatar-jpeg-quality" min="30" max="100" step="1"><span class="value" id="avatar-jpeg-quality-value"></span></div></div>
            </div>
          </details>
        </section>
      </div>
    </div>

    <div class="view-panel" id="view-quality" data-panel="quality" role="tabpanel" aria-labelledby="tab-quality" hidden>
      <header class="panel-head"><h2>Tune camera quality</h2><p>Match foreground colour, frame the scene, and adjust subject separation. Defaults are a good starting point for most cameras.</p></header>
      <section class="control-section">
        <div class="section-copy"><h3>Colour match</h3><p>Gently adapt the camera foreground to supported image, video, or secondary-camera backgrounds.</p></div>
        <div class="inline-toggle"><label for="quality-color-auto">Automatic colour correction<small>Applies only when the pipeline has a reliable background estimate.</small></label>
          <span class="switch"><input type="checkbox" id="quality-color-auto"><span></span></span></div>
        <div class="field"><label for="quality-color-strength">Correction strength</label>
          <div class="range-line"><input type="range" id="quality-color-strength" min="0" max="1" step="0.05" aria-describedby="quality-color-strength-help"><span class="value" id="quality-color-strength-value"></span></div>
          <small id="quality-color-strength-help">Lower values preserve more of the camera's original exposure and white balance.</small></div>
        <p class="notice" id="quality-color-status" role="status" aria-live="polite" aria-atomic="true">Waiting for colour-correction status…</p>
      </section>
      <section class="control-section">
        <div class="section-copy"><h3>Scene framing</h3><p>Choose how the camera and backdrop fill the output. Backdrop focal points keep the important area in view when cropping.</p></div>
        <div class="field-grid two">
          <div class="field"><label for="quality-background-fit">Background fit</label>
            <select id="quality-background-fit"><option value="cover">Fill and crop</option><option value="contain">Fit with padding</option><option value="stretch">Stretch to fill</option></select>
            <small>Applies immediately to image, video, and secondary-camera backgrounds.</small></div>
          <div class="field"><label for="quality-camera-fit">Camera fit <span class="restart-tag">Restart required</span></label>
            <select id="quality-camera-fit" aria-describedby="quality-camera-fit-help"><option value="cover">Fill and crop</option><option value="contain">Fit with padding</option><option value="stretch">Stretch to fill</option></select>
            <small id="quality-camera-fit-help">Camera geometry is fixed when capture starts. A rejected change is restored to the effective setting.</small></div>
          <div class="field"><label for="quality-background-anchor-x">Backdrop horizontal focal point</label>
            <div class="range-line"><input type="range" id="quality-background-anchor-x" min="0" max="1" step="0.05" aria-describedby="quality-background-anchor-x-help"><span class="value" id="quality-background-anchor-x-value"></span></div>
            <small id="quality-background-anchor-x-help">Left to right; affects cropped backdrops.</small></div>
          <div class="field"><label for="quality-background-anchor-y">Backdrop vertical focal point</label>
            <div class="range-line"><input type="range" id="quality-background-anchor-y" min="0" max="1" step="0.05" aria-describedby="quality-background-anchor-y-help"><span class="value" id="quality-background-anchor-y-value"></span></div>
            <small id="quality-background-anchor-y-help">Top to bottom; affects cropped backdrops.</small></div>
        </div>
      </section>
      <section class="control-section">
        <div class="section-copy"><h3>Subject detection</h3><p>The backend finds you in each frame before custback replaces the room.</p></div>
        <div class="field-grid two">
          <div class="field"><label for="quality-backend">Detection backend</label>
            <select id="quality-backend"><option value="auto">Automatic</option><option value="rvm">RVM matting</option><option value="mediapipe">MediaPipe</option><option value="heuristic">Basic heuristic</option><option value="none">Disabled</option></select>
            <small>Automatic uses the best installed option.</small></div>
          <div class="field"><label for="quality-delegate">Processor</label>
            <select id="quality-delegate"><option value="cpu">CPU</option><option value="gpu">GPU</option></select>
            <small>GPU is available only for automatic or MediaPipe detection.</small></div>
          <div class="field"><label for="quality-threshold">Subject threshold</label><div class="range-line"><input type="range" id="quality-threshold" min="0" max="1" step="0.01"><span class="value" id="quality-threshold-value"></span></div></div>
          <div class="field"><label for="quality-rvm-downsample">RVM detail scale</label><div class="range-line"><input type="range" id="quality-rvm-downsample" min="0" max="1" step="0.05"><span class="value" id="quality-rvm-downsample-value"></span></div><small>Zero lets the model choose automatically.</small></div>
        </div>
      </section>
      <section class="control-section">
        <div class="section-copy"><h3>Edges and motion</h3><p>Use these controls when hair edges flicker, the mask trails, or the subject looks cut out.</p></div>
        <div class="field-grid two">
          <div class="field"><label for="quality-mask-blur">Mask softness</label><div class="range-line"><input type="range" id="quality-mask-blur" min="0" max="151" step="1"><span class="value" id="quality-mask-blur-value"></span></div></div>
          <div class="field"><label for="quality-mask-shift">Mask expansion</label><div class="range-line"><input type="range" id="quality-mask-shift" min="-20" max="20" step="1"><span class="value" id="quality-mask-shift-value"></span></div></div>
          <div class="field"><label for="quality-smoothing">Temporal smoothing</label><div class="range-line"><input type="range" id="quality-smoothing" min="0" max="0.95" step="0.01"><span class="value" id="quality-smoothing-value"></span></div></div>
          <div class="field"><label for="quality-light-wrap">Light wrap</label><div class="range-line"><input type="range" id="quality-light-wrap" min="0" max="1" step="0.01"><span class="value" id="quality-light-wrap-value"></span></div></div>
        </div>
        <div class="inline-toggle"><label for="quality-edge-refine">Refine subject edges<small>Improves the boundary around hair and shoulders.</small></label><span class="switch"><input type="checkbox" id="quality-edge-refine"><span></span></span></div>
        <div class="inline-toggle"><label for="quality-model-foreground">Use model foreground<small>Uses the model's colour output to reduce edge spill.</small></label><span class="switch"><input type="checkbox" id="quality-model-foreground"><span></span></span></div>
      </section>
      <section class="control-section">
        <details><summary>Advanced colour and canvas controls</summary>
          <div class="advanced-grid">
            <div class="advanced-group">
              <div class="section-copy"><h3>Correction limits</h3><p>These bounded expert controls apply live. Keep conservative values unless a calibrated workflow calls for more.</p></div>
              <div class="field-grid two">
                <div class="field"><label for="quality-exposure-limit">Exposure limit</label>
                  <div class="range-line"><input type="range" id="quality-exposure-limit" min="0" max="1" step="0.05"><span class="value" id="quality-exposure-limit-value"></span></div><small>Maximum exposure adjustment in EV.</small></div>
                <div class="field"><label for="quality-wb-strength">White-balance strength</label>
                  <div class="range-line"><input type="range" id="quality-wb-strength" min="0" max="1" step="0.05"><span class="value" id="quality-wb-strength-value"></span></div><small>Restrains red, green, and blue gain matching.</small></div>
                <div class="field"><label for="quality-adaptation-time">Adaptation time</label>
                  <div class="range-line"><input type="range" id="quality-adaptation-time" min="0.05" max="10" step="0.05"><span class="value" id="quality-adaptation-time-value"></span></div><small>Seconds used to smooth reliable estimates.</small></div>
                <div class="field"><label for="quality-blend-space">Blend-space compatibility</label>
                  <select id="quality-blend-space"><option value="srgb_legacy">Legacy sRGB</option><option value="linear_srgb">Linear sRGB</option></select><small>Linear sRGB is photometrically correct; legacy mode preserves older output.</small></div>
              </div>
            </div>
            <div class="advanced-group">
              <div class="section-copy"><h3>Capture canvas <span class="restart-tag">Restart required</span></h3><p>These values are fixed when the pipeline starts. The API rejects live changes and this page restores the effective configuration.</p></div>
              <div class="field-grid two">
                <div class="field"><label for="quality-camera-rotation">Camera rotation</label>
                  <select id="quality-camera-rotation" aria-describedby="quality-camera-rotation-help"><option value="0">0°</option><option value="90">90° clockwise</option><option value="180">180°</option><option value="270">270° clockwise</option></select>
                  <small id="quality-camera-rotation-help">Applied before camera fit and mirroring.</small></div>
                <div class="field"><label for="quality-output-width">Output width</label>
                  <input type="number" id="quality-output-width" min="16" max="7680" step="1" inputmode="numeric" aria-describedby="quality-output-size-help"></div>
                <div class="field"><label for="quality-output-height">Output height</label>
                  <input type="number" id="quality-output-height" min="16" max="7680" step="1" inputmode="numeric" aria-describedby="quality-output-size-help"></div>
              </div>
              <p class="hint" id="quality-output-size-help">Set both output dimensions, or leave both blank to inherit the requested camera size.</p>
            </div>
          </div>
        </details>
      </section>
    </div>

    <div class="view-panel" id="view-system" data-panel="system" role="tabpanel" aria-labelledby="tab-system" hidden>
      <header class="panel-head"><h2>System status</h2><p>Inspect the live pipeline, provider connection, and browser-safe effective configuration.</p></header>
      <section class="control-section">
        <div class="section-copy"><h3>Camera pipeline</h3><p>Live measurements update every three seconds.</p></div>
        <dl class="diagnostic-list" id="core-diagnostics"></dl>
        <details><summary>All camera diagnostics</summary><dl class="diagnostic-list" id="core-diagnostics-all"></dl></details>
      </section>
      <section class="control-section" id="avatar-diagnostics-section">
        <div class="section-copy"><h3>Avatar service</h3><p>Control API reachability, camera-frame input, following driver, and render health.</p></div>
        <dl class="diagnostic-list" id="avatar-diagnostics"></dl>
        <details><summary>All avatar diagnostics</summary><dl class="diagnostic-list" id="avatar-diagnostics-all"></dl></details>
      </section>
      <section class="control-section">
        <div class="section-copy"><h3>Connection timing</h3><p>These safe timeout changes apply immediately; provider addresses and credentials remain restart-only.</p></div>
        <div class="field-grid two">
          <div class="field"><label for="core-remote-timeout">Remote frame timeout</label><input type="number" id="core-remote-timeout" min="1" max="60000" step="1"><small>Milliseconds before the privacy fallback appears.</small></div>
          <div class="field"><label for="avatar-connect-timeout">Provider connect timeout</label><input type="number" id="avatar-connect-timeout" min="0.5" max="60" step="0.5"><small>Seconds allowed to open the provider connection.</small></div>
          <div class="field"><label for="avatar-read-timeout">Provider read timeout</label><input type="number" id="avatar-read-timeout" min="1" max="600" step="1"><small>Seconds allowed for non-stream responses.</small></div>
        </div>
      </section>
      <section class="control-section">
        <details><summary>Browser-safe effective configuration</summary><pre class="config-output" id="effective-config">Loading…</pre></details>
        <div class="action-row"><a class="button-link" href="/docs">API guide</a><a class="button-link" href="/openapi.json">OpenAPI JSON</a></div>
      </section>
    </div>
  </section>
</main>
<footer class="foot-line"><span>custback · live changes apply immediately unless marked restart-only</span><span><a href="/docs">API guide</a> · <a href="/openapi.json">OpenAPI</a></span></footer>
<div id="toasts"></div>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const LOCAL_MODES = ["blur", "image", "video", "color", "camera"];
const CORE_BG_MODES = ["blur", "color", "image", "video", "camera", "passthrough"];
const AVATAR_BG_MODES = ["color", "image", "video", "blur"];
const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const MODE_LABELS = {
  blur: "Blur", color: "Colour", image: "Image", video: "Video",
  camera: "Extra camera", passthrough: "Original room",
};
const FOLLOW_LABELS = {
  auto: "Automatic",
  motion: "Follow camera",
  voice: "Follow voice",
  presence: "Do not follow",
};
const FOLLOW_DESCRIPTIONS = {
  auto: "Uses camera face tracking when available; otherwise keeps an idle pose.",
  motion: "Uses face movement from your camera.",
  voice: "Animates from the configured Audio2Face voice service.",
  presence: "Keeps a steady idle pose without following camera or voice.",
};

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
  status: null,
  avatarStatus: null,
  runId: null,
  activeView: "background",
  providerKind: "local",
  providerInitialized: false,
  providerSourceTouched: false,
  coreRefreshPending: false,
  avatarRefreshPending: false,
  avatarStatusOutage: false,
};

function toast(message, kind) {
  const node = document.createElement("div");
  node.className = "toast" + (kind ? " " + kind : "");
  node.setAttribute("role", kind === "err" ? "alert" : "status");
  node.setAttribute("aria-atomic", "true");
  const copy = document.createElement("span");
  copy.textContent = message;
  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-close";
  close.textContent = "Dismiss";
  close.setAttribute("aria-label", "Dismiss message");
  node.append(copy, close);
  $("toasts").appendChild(node);
  let timer = null;
  const stopTimer = () => clearTimeout(timer);
  const startTimer = () => {
    if (kind === "err") return;
    clearTimeout(timer);
    timer = setTimeout(() => node.remove(), 2500);
  };
  if (kind !== "err") timer = setTimeout(() => node.remove(), 6000);
  node.addEventListener("mouseenter", stopTimer);
  node.addEventListener("mouseleave", startTimer);
  node.addEventListener("focusin", stopTimer);
  node.addEventListener("focusout", startTimer);
  close.addEventListener("click", () => node.remove());
  return node;
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

async function withBusy(button, label, action) {
  const previous = button.textContent;
  button.disabled = true;
  button.dataset.state = "loading";
  button.textContent = label;
  try {
    return await action();
  } catch (err) {
    button.dataset.state = "error";
    throw err;
  } finally {
    button.disabled = false;
    button.textContent = previous;
    setTimeout(() => { delete button.dataset.state; }, 600);
  }
}

function titleCase(value) {
  return String(value || "—").replace(/[_-]/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function setView(view) {
  state.activeView = view;
  for (const tab of document.querySelectorAll(".nav-tab")) {
    const active = tab.dataset.view === view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const panel of document.querySelectorAll("[data-panel]")) {
    const active = panel.dataset.panel === view;
    panel.hidden = !active;
  }
}

const primaryNav = document.querySelector(".primary-nav");
primaryNav.addEventListener("click", (event) => {
  const tab = event.target.closest("[data-view]");
  if (tab) setView(tab.dataset.view);
});
primaryNav.addEventListener("keydown", (event) => {
  const tabs = [...primaryNav.querySelectorAll("[role=tab]")];
  const current = tabs.indexOf(document.activeElement);
  if (current < 0) return;
  let next = null;
  if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
  if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = tabs.length - 1;
  if (next === null) return;
  event.preventDefault();
  tabs[next].focus();
  setView(tabs[next].dataset.view);
});
$("session-reload").addEventListener("click", () => location.reload());
$("signout").addEventListener("click", async (event) => {
  try {
    await withBusy(event.currentTarget, "Signing out…", () =>
      api("DELETE", "/auth/session"));
    location.reload();
  } catch (err) { reportError(err); }
});

// -- config patching ---------------------------------------------------------

async function patchCore(patch) {
  const body = await api("PATCH", "/config", patch);
  state.core = body.config;
  state.coreVersion = body.config_version;
  renderAll();
}

async function patchCoreControl(patch) {
  try {
    return await patchCore(patch);
  } catch (err) {
    let refreshError = null;
    if (err instanceof ApiError && [409, 422, 503].includes(err.status)) {
      try {
        state.core = await api("GET", "/config");
      } catch (caught) {
        refreshError = caught;
      }
    }
    // PATCH is atomic. Render either the freshly fetched effective config or
    // the last known effective config so an optimistic browser control cannot
    // retain a value that the pipeline rejected.
    renderAll();
    if (refreshError instanceof ApiError && refreshError.status === 401) {
      throw refreshError;
    }
    throw err;
  }
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
    return true;
  } catch (err) {
    state.avatar = null;
    state.avatarState = err.code === "avatar_unconfigured"
      ? "unconfigured" : "unreachable";
    if (err.status === 401) throw err;
    return false;
  }
}

async function refreshAll() {
  try { await loadCore(); } catch (err) { reportError(err); }
  await loadAvatar().catch(reportError);
  renderAll();
}

// -- header / status ----------------------------------------------------------

function applyStatus(status, avatarStatus) {
  if (status && state.runId && status.run_id !== state.runId) {
    location.reload();
    return;
  }
  if (status) state.runId = status.run_id;
  state.status = status;
  state.avatarStatus = avatarStatus;
  if (status) {
    $("chip-mode").textContent = status.mode === "remote"
      ? "Avatar" : (MODE_LABELS[status.mode] || titleCase(status.mode));
    $("chip-fps").textContent = status.fps.toFixed(0) + " fps";
    const remote = $("chip-remote");
    if (status.mode !== "remote") {
      remote.textContent = "Off";
      remote.className = "chip off";
    } else if (status.remote_connected && !status.remote_fallback_active) {
      remote.textContent = "Live";
      remote.className = "chip on";
    } else {
      remote.textContent = "Fallback: "
        + (MODE_LABELS[status.remote_fallback_mode] || status.remote_fallback_mode);
      remote.className = "chip bad";
    }
    $("live-status").className = "live-badge on";
    if ($("live-status-text").textContent !== "Live") {
      $("live-status-text").textContent = "Live";
    }
    if (status.config_version !== state.coreVersion
        && !state.coreRefreshPending) {
      state.coreRefreshPending = true;
      const observedVersion = status.config_version;
      loadCore().then(() => {
        state.coreVersion = observedVersion;
        renderAll();
      }).catch(reportError).finally(() => { state.coreRefreshPending = false; });
    }
  }
  const driver = $("chip-driver");
  if (avatarStatus) {
    const matching = state.avatarInfo && state.avatarInfo.modes.find((mode) =>
      mode.backend === avatarStatus.driver_backend);
    driver.textContent = matching
      ? (FOLLOW_LABELS[matching.id] || matching.label)
      : titleCase(avatarStatus.driver_backend);
    driver.className = "chip" + (avatarStatus.connected ? " on" : "");
    if (avatarStatus.config_version !== state.avatarVersion
        && !state.avatarRefreshPending) {
      state.avatarRefreshPending = true;
      const observedVersion = avatarStatus.config_version;
      loadAvatar().then((loaded) => {
        if (!loaded) return;
        state.avatarVersion = observedVersion;
        renderAll();
      }).catch(reportError).finally(() => { state.avatarRefreshPending = false; });
    }
  } else {
    driver.textContent = "—";
    driver.className = "chip off";
  }
  renderDiagnostics();
}

let corePollBusy = false;
let avatarPollBusy = false;
async function pollCoreStatus() {
  if (corePollBusy) return;
  corePollBusy = true;
  try {
    let status = null;
    try { status = await api("GET", "/status"); } catch (err) { /* offline */ }
    if (!status) {
      $("live-status").className = "live-badge bad";
      if ($("live-status-text").textContent !== "Unavailable") {
        $("live-status-text").textContent = "Unavailable";
      }
    }
    applyStatus(status, state.avatarStatus);
  } finally { corePollBusy = false; }
}

async function pollAvatarStatus() {
  if (avatarPollBusy) return;
  avatarPollBusy = true;
  try {
    let avatarStatus = null;
    if (state.avatarState === "ok" || state.avatarState === "unreachable") {
      try {
        avatarStatus = await api("GET", "/avatar/status");
        const recovering = state.avatarStatusOutage;
        state.avatarStatusOutage = false;
        if (state.avatarState !== "ok" || recovering) {
          const loaded = await loadAvatar();
          if (loaded) state.avatarVersion = avatarStatus.config_version;
          renderAll();
        }
      } catch (err) { state.avatarStatusOutage = true; }
    }
    applyStatus(state.status, avatarStatus);
  } finally { avatarPollBusy = false; }
}

function poll() {
  pollCoreStatus();
  pollAvatarStatus();
}

// -- preview -------------------------------------------------------------------

let previewRetry = null;
function setPreview() {
  const source = state.previewSource;
  const path = source === "avatar" ? "/avatar/video/mjpeg" : "/video/mjpeg";
  $("preview-hint").textContent = source === "avatar"
    ? "The avatar render before it enters the final camera output."
    : "This is what your meeting receives.";
  $("preview-title").textContent = source === "avatar" ? "Avatar render" : "Camera output";
  $("preview-snapshot").href = source === "avatar"
    ? "/avatar/video/snapshot.jpg" : "/video/snapshot.jpg";
  $("preview-snapshot").download = source === "avatar"
    ? "custback-avatar-snapshot.jpg" : "custback-snapshot.jpg";
  $("preview-msg").textContent = "Connecting to the "
    + (source === "avatar" ? "avatar" : "camera") + " stream…";
  $("preview-msg").hidden = false;
  $("preview").classList.add("unavailable");
  $("preview").src = path + "?t=" + Date.now();
}
$("preview").addEventListener("load", () => {
  $("preview").classList.remove("unavailable");
  $("preview-msg").textContent = "";
  $("preview-msg").hidden = true;
});
$("preview").addEventListener("error", () => {
  $("preview").classList.add("unavailable");
  $("preview-msg").hidden = false;
  $("preview-msg").textContent = "The stream is unavailable. Retrying…";
  clearTimeout(previewRetry);
  previewRetry = setTimeout(setPreview, 2500);
});
$("preview-source").addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button) return;
  state.previewSource = button.dataset.src;
  for (const other of $("preview-source").children) {
    const active = other === button;
    other.classList.toggle("active", active);
    other.setAttribute("aria-pressed", String(active));
  }
  setPreview();
});

// -- avatar enable toggle --------------------------------------------------------

$("avatar-enabled").addEventListener("change", async (event) => {
  const on = event.target.checked;
  event.target.disabled = true;
  try {
    if (!state.core) throw new ApiError(0, "no_config", "config not loaded yet");
    if (on && state.avatarState === "unconfigured") {
      setView("avatar");
      throw new ApiError(0, "avatar_unconfigured",
        "Configure a local or remote avatar provider, then restart custback.");
    }
    const mode = state.core.background.mode;
    if (on) {
      const patch = {background: {
        mode: "remote",
        remote_fallback_mode: LOCAL_MODES.includes(mode) ? mode : "blur",
      }};
      await patchCore(patch);
      if (state.scopeFollowsToggle) setScope("avatar");
    } else {
      const fallback = state.core.background.remote_fallback_mode || "blur";
      await patchCore({background: {mode: fallback}});
      if (state.scopeFollowsToggle) setScope("camera");
    }
  } catch (err) {
    reportError(err);
    renderAll();
  } finally {
    event.target.disabled = false;
  }
});

// -- background panel -------------------------------------------------------------

function setScope(scope) {
  state.bgScope = scope;
  for (const button of $("bg-scope").children) {
    const active = button.dataset.scope === scope;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
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
  if (state.bgScope === "camera") {
    const adjusted = {...values};
    if (state.core && state.core.background.mode === "remote"
        && "mode" in adjusted) {
      adjusted.remote_fallback_mode = adjusted.mode;
      delete adjusted.mode;
    }
    await patchCore({background: adjusted});
  } else await patchAvatar({background: values});
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
    ? "Used behind your camera whenever avatar output is off."
    : "Rendered behind the selected avatar.";
  const config = bgConfig();
  const modesBox = $("bg-modes");
  modesBox.textContent = "";
  if (!config) {
    $("bg-tiles").textContent = "";
    if (!cameraScope) {
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = "Connect the avatar service to edit its scene.";
      $("bg-tiles").appendChild(hint);
    }
    $("bg-color-row").hidden = true;
    $("bg-blur-row").hidden = true;
    $("bg-camera-row").hidden = true;
    $("bg-upload-btn").disabled = true;
    $("bg-upload-hint").textContent = cameraScope
      ? "Camera configuration is unavailable." : "Connect the avatar service before uploading its scene media.";
    return;
  }
  const uploadWouldDisableAvatar = cameraScope && config.mode === "remote";
  $("bg-upload-btn").disabled = uploadWouldDisableAvatar;
  $("bg-upload-btn").title = uploadWouldDisableAvatar
    ? "Turn avatar output off before uploading camera background media" : "";
  $("bg-upload-hint").textContent = uploadWouldDisableAvatar
    ? "Turn avatar output off before uploading. The camera upload endpoint activates new media immediately."
    : (cameraScope ? "" : "Scene uploads are stored by the selected avatar provider.");
  const modes = cameraScope ? CORE_BG_MODES : AVATAR_BG_MODES;
  const selectedMode = cameraScope && config.mode === "remote"
    ? config.remote_fallback_mode : config.mode;
  for (const mode of modes) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = MODE_LABELS[mode] || titleCase(mode);
    const active = selectedMode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    const needsFile = mode === "image" || mode === "video";
    const hasFile = needsFile
      && (mode === "image" ? config.image_path : config.video_path);
    const hasCamera = cameraScope && (config.camera_source_configured
      || (config.camera_targets || []).length > 0);
    if (mode === "camera" && !hasCamera) {
      button.disabled = true;
      button.title = "No operator-approved camera source is configured";
    }
    if (cameraScope && config.mode === "remote" && mode === "passthrough") {
      button.disabled = true;
      button.title = "Turn avatar output off before showing the original room";
    }
    button.addEventListener("click", () => {
      if (needsFile && !hasFile) {
        toast("Pick a file from the gallery below to use the "
          + mode + " background", "warn");
        return;
      }
      const values = {mode};
      if (mode === "camera" && !config.camera_source_configured
          && (config.camera_targets || []).length) {
        values.camera_target = config.camera_targets[0];
      }
      patchBackground(values).catch(reportError);
    });
    modesBox.appendChild(button);
  }
  $("bg-color-row").hidden = selectedMode !== "color";
  $("bg-color").value = bgrToHex(config.color || [60, 46, 32]);
  const blurry = selectedMode === "blur";
  $("bg-blur-row").hidden = !blurry;
  $("bg-blur").value = config.blur_strength;
  $("bg-blur-value").textContent = config.blur_strength;
  const targetRow = $("bg-camera-row");
  const targetSelect = $("bg-camera-target");
  targetRow.hidden = !(cameraScope && selectedMode === "camera");
  targetSelect.textContent = "";
  const operatorDevice = config.camera_source_configured && !config.camera_target;
  if (operatorDevice) {
    const option = document.createElement("option");
    option.textContent = "Operator-configured camera";
    option.disabled = true;
    option.selected = true;
    targetSelect.appendChild(option);
    $("bg-camera-hint").textContent = "This legacy source is fixed at startup; approved named cameras require an operator configuration change.";
  } else {
    for (const target of config.camera_targets || []) {
      const option = document.createElement("option");
      option.value = target;
      option.textContent = titleCase(target);
      option.selected = target === config.camera_target;
      targetSelect.appendChild(option);
    }
    if (!(config.camera_targets || []).length) {
      const option = document.createElement("option");
      option.textContent = "No approved cameras";
      option.disabled = true;
      option.selected = true;
      targetSelect.appendChild(option);
    }
    $("bg-camera-hint").textContent = "Only sources approved by the operator are shown.";
  }
  targetSelect.disabled = operatorDevice || !(config.camera_targets || []).length;
  renderBackgroundTiles(config, cameraScope, selectedMode);
}

$("bg-camera-target").addEventListener("change", (event) => {
  patchBackground({mode: "camera", camera_target: event.target.value})
    .catch((err) => { reportError(err); renderBackground(); });
});

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

function renderBackgroundTiles(config, cameraScope, selectedMode) {
  const box = $("bg-tiles");
  box.textContent = "";
  const entries = cameraScope ? coreFileEntries() : avatarFileEntries();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No saved backgrounds yet. Upload an image or video to add one.";
    box.appendChild(empty);
  }
  for (const entry of entries) {
    const tile = document.createElement("article");
    tile.className = "tile";
    const active = (entry.kind === "image"
      ? config.image_path : config.video_path) === entry.path
      && selectedMode === entry.kind;
    tile.classList.toggle("active", active);
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "tile-pick";
    pick.setAttribute("aria-pressed", String(active));
    pick.setAttribute("aria-label", "Use " + entry.name + " as the "
      + (cameraScope ? "camera" : "avatar") + " background");
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = entry.thumbnail;
    img.alt = entry.name;
    img.width = 320;
    img.height = 180;
    pick.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = entry.name;
    pick.appendChild(label);
    pick.addEventListener("click", () => {
      const values = entry.kind === "image"
        ? {mode: "image", image_path: entry.path}
        : {mode: "video", video_path: entry.path};
      patchBackground(values).catch(reportError);
    });
    tile.appendChild(pick);
    const del = document.createElement("button");
    del.type = "button";
    del.className = "del";
    del.textContent = "Remove";
    del.setAttribute("aria-label", "Remove " + entry.name);
    del.disabled = active;
    if (active) del.title = "Choose another background before removing this file";
    del.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!confirm("Remove " + entry.name + "? This file cannot be restored.")) return;
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
    box.appendChild(tile);
  }
}

$("bg-upload-btn").addEventListener("click", () => $("bg-upload").click());
$("bg-upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  if (state.bgScope === "camera" && state.core
      && state.core.background.mode === "remote") {
    toast("Turn avatar output off before uploading camera background media.", "warn");
    return;
  }
  const isImage = IMAGE_EXTS.some((ext) => file.name.toLowerCase().endsWith(ext));
  const kind = isImage ? "image" : "video";
  const cameraUpload = state.bgScope === "camera";
  const button = $("bg-upload-btn");
  const avatarToggle = $("avatar-enabled");
  let pending = null;
  if (cameraUpload) avatarToggle.disabled = true;
  try {
    pending = toast("Uploading " + file.name + "…");
    await withBusy(button, "Uploading…", async () => {
      if (cameraUpload) {
        state.core = await api("GET", "/config");
        if (state.core.background.mode === "remote") {
          throw new ApiError(409, "avatar_active",
            "Turn avatar output off before uploading camera background media.");
        }
        const form = new FormData();
        form.append("file", file, file.name);
        await api("POST", "/background/" + kind, form);
        await loadCore();
      } else {
        await api("POST", "/avatar/backgrounds/" + kind
          + "?name=" + encodeURIComponent(file.name), file,
          file.type || "application/octet-stream");
        state.avatarFiles = await api("GET", "/avatar/backgrounds");
      }
    });
    renderAll();
  } catch (err) { reportError(err); renderAll(); }
  finally {
    if (pending) pending.remove();
    if (cameraUpload) avatarToggle.disabled = false;
  }
});

// -- avatar panel -----------------------------------------------------------------

function segButtons(container, options, current, onPick) {
  container.textContent = "";
  for (const option of options) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = titleCase(option);
    const active = option === current;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
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
    const tile = document.createElement("article");
    tile.className = "tile";
    const active = appearance.rig === "builtin" && appearance.avatar === name;
    tile.classList.toggle("active", active);
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "tile-pick";
    pick.setAttribute("aria-pressed", String(active));
    pick.setAttribute("aria-label", "Use avatar " + titleCase(name));
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/avatar/avatars/" + name + "/thumbnail.jpg?style="
      + appearance.style;
    img.alt = name;
    img.width = 320;
    img.height = 180;
    pick.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = name;
    pick.appendChild(label);
    pick.addEventListener("click", () => {
      patchAvatar({appearance: {rig: "builtin", avatar: name}}).catch(reportError);
    });
    tile.appendChild(pick);
    box.appendChild(tile);
  }
  for (const rig of info.rigs || []) {
    const tile = document.createElement("article");
    tile.className = "tile";
    const active = appearance.rig === rig.name
      || appearance.rig.endsWith("/" + rig.name);
    tile.classList.toggle("active", active);
    const pick = document.createElement("button");
    pick.type = "button";
    pick.className = "tile-pick";
    pick.setAttribute("aria-pressed", String(active));
    pick.setAttribute("aria-label", "Use custom avatar " + rig.name);
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = "/avatar/rigs/" + encodeURIComponent(rig.name)
      + "/thumbnail.jpg?v=" + rig.size_bytes;
    img.alt = rig.name;
    img.width = 320;
    img.height = 180;
    pick.appendChild(img);
    const label = document.createElement("span");
    label.className = "name";
    label.textContent = rig.name + " (custom)";
    pick.appendChild(label);
    pick.addEventListener("click", () => {
      patchAvatar({appearance: {rig: rig.name}}).catch(reportError);
    });
    tile.appendChild(pick);
    if (!active) {
      const del = document.createElement("button");
      del.type = "button";
      del.className = "del";
      del.textContent = "Remove";
      del.setAttribute("aria-label", "Remove custom avatar " + rig.name);
      del.addEventListener("click", async (event) => {
        event.stopPropagation();
        if (!confirm("Remove " + rig.name + "? This rig cannot be restored.")) return;
        try {
          await api("DELETE", "/avatar/rigs/" + encodeURIComponent(rig.name));
          state.avatarInfo = await api("GET", "/avatar/avatars");
          renderAvatar();
        } catch (err) { reportError(err); }
      });
      tile.appendChild(del);
    }
    box.appendChild(tile);
  }
  const upload = document.createElement("button");
  upload.type = "button";
  upload.className = "tile upload";
  upload.textContent = "Install rig (.zip)";
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
  let pending = null;
  try {
    pending = toast("Installing rig " + name + "…");
    await api("POST", "/avatar/rigs?name=" + encodeURIComponent(name),
      file, "application/zip");
    state.avatarInfo = await api("GET", "/avatar/avatars");
    renderAvatar();
  } catch (err) { reportError(err); }
  finally { if (pending) pending.remove(); }
});

function renderAvatarModes() {
  const box = $("avatar-modes");
  box.textContent = "";
  const backend = state.avatar.driver.backend;
  for (const mode of state.avatarInfo.modes) {
    const selectable = mode.available && mode.configured;
    const row = document.createElement("label");
    row.className = "mode" + (selectable ? "" : " unavailable")
      + (mode.backend === backend ? " active" : "");
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
    text.append(FOLLOW_LABELS[mode.id] || mode.label);
    const small = document.createElement("small");
    small.textContent = !mode.available
      ? mode.reason || mode.description
      : (!mode.configured
        ? "Configure the required service, then restart the avatar provider."
        : (FOLLOW_DESCRIPTIONS[mode.id] || mode.description));
    text.appendChild(small);
    row.appendChild(text);
    box.appendChild(row);
  }
}

function renderAvatar() {
  const setup = $("avatar-setup");
  const controls = $("avatar-controls");
  if (state.avatarState !== "ok" || !state.avatar) {
    setup.hidden = false;
    controls.hidden = true;
    $("avatar-setup-msg").textContent = state.avatarState === "unreachable"
      ? "The configured avatar provider is not answering. Check that it is running and reachable."
      : "No avatar provider is configured. Choose Local or Remote above and apply the restart configuration.";
    return;
  }
  setup.hidden = true;
  controls.hidden = false;
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
  $("avatar-x-value").textContent = Number(appearance.offset_x).toFixed(2);
  $("avatar-y").value = appearance.offset_y;
  $("avatar-y-value").textContent = Number(appearance.offset_y).toFixed(2);
  $("avatar-smoothing").value = state.avatar.driver.smoothing;
  $("avatar-smoothing-value").textContent =
    Number(state.avatar.driver.smoothing).toFixed(2);
  $("avatar-follow").checked = appearance.follow_pose;
  $("avatar-max-fps").value = state.avatar.render.max_fps;
  $("avatar-max-fps-value").textContent = state.avatar.render.max_fps + " fps";
  $("avatar-jpeg-quality").value = state.avatar.render.jpeg_quality;
  $("avatar-jpeg-quality-value").textContent = state.avatar.render.jpeg_quality + "%";
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
    label.append(titleCase(part));
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
      .catch((err) => { reportError(err); renderAvatar(); });
  });
}
for (const [id, valueId] of [
  ["avatar-scale", "avatar-scale-value"],
  ["avatar-x", "avatar-x-value"],
  ["avatar-y", "avatar-y-value"],
  ["avatar-smoothing", "avatar-smoothing-value"],
]) {
  $(id).addEventListener("input", (event) => {
    $(valueId).textContent = Number(event.target.value).toFixed(2);
  });
}
$("avatar-follow").addEventListener("change", (event) => {
  patchAvatar({appearance: {follow_pose: event.target.checked}})
    .catch((err) => { reportError(err); renderAvatar(); });
});
$("avatar-max-fps").addEventListener("input", (event) => {
  $("avatar-max-fps-value").textContent = event.target.value + " fps";
});
$("avatar-max-fps").addEventListener("change", (event) => {
  patchAvatar({render: {max_fps: parseInt(event.target.value, 10)}})
    .catch((err) => { reportError(err); renderAvatar(); });
});
$("avatar-jpeg-quality").addEventListener("input", (event) => {
  $("avatar-jpeg-quality-value").textContent = event.target.value + "%";
});
$("avatar-jpeg-quality").addEventListener("change", (event) => {
  patchAvatar({render: {jpeg_quality: parseInt(event.target.value, 10)}})
    .catch((err) => { reportError(err); renderAvatar(); });
});
$("bg-color").addEventListener("change", (event) => {
  patchBackground({mode: "color", color: hexToBgr(event.target.value)})
    .catch((err) => { reportError(err); renderBackground(); });
});
$("bg-blur").addEventListener("input", (event) => {
  $("bg-blur-value").textContent = event.target.value;
});
$("bg-blur").addEventListener("change", (event) => {
  patchBackground({blur_strength: parseInt(event.target.value, 10)})
    .catch((err) => { reportError(err); renderBackground(); });
});

// -- provider restart planner -------------------------------------------------

function providerLabel(url) {
  if (!url) return "Not configured";
  try {
    const parsed = new URL(url);
    const hostname = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, "");
    const octets = hostname.split(".").map(Number);
    const loopbackV4 = octets.length === 4 && octets[0] === 127
      && octets.every((part) => Number.isInteger(part) && part >= 0 && part <= 255);
    const local = hostname === "::1" || hostname === "localhost"
      || hostname.endsWith(".localhost") || loopbackV4;
    return local ? "Local service" : "Remote server";
  } catch (err) { return "Configured"; }
}

function validateRemoteProvider(showError) {
  const input = $("provider-url");
  const help = $("provider-url-help");
  const raw = input.value.trim();
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "https:" || parsed.username || parsed.password
        || (parsed.pathname !== "/" && parsed.pathname !== "")
        || parsed.search || parsed.hash) {
      throw new Error("invalid provider root");
    }
    input.setAttribute("aria-invalid", "false");
    help.textContent = "Remote providers require HTTPS and a root URL with no path or credentials.";
    return parsed.origin;
  } catch (err) {
    if (showError) {
      input.setAttribute("aria-invalid", "true");
      help.textContent = "Use an HTTPS root such as https://avatar.example:8711, with no path or credentials.";
    }
    return null;
  }
}

function validateRemoteSource(showError) {
  const input = $("provider-source-url");
  const help = $("provider-source-help");
  const raw = input.value.trim();
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "wss:" || parsed.username || parsed.password
        || (parsed.pathname !== "/" && parsed.pathname !== "")
        || parsed.search || parsed.hash) {
      throw new Error("invalid frame root");
    }
    input.setAttribute("aria-invalid", "false");
    help.textContent = "The remote avatar server uses this secure WebSocket root to receive camera frames from custback.";
    return parsed.origin;
  } catch (err) {
    if (showError) {
      input.setAttribute("aria-invalid", "true");
      help.textContent = "Use a WSS root such as wss://camera.example:8710, with no path or credentials.";
    }
    return null;
  }
}

function connectableCoreHost(host, secure) {
  let value = String(host || "127.0.0.1").trim().replace(/^\[|\]$/g, "");
  if (value === "0.0.0.0") value = "127.0.0.1";
  if (value === "::") value = "::1";
  if (!secure && (value === "localhost" || value.endsWith(".localhost"))) {
    value = "127.0.0.1";
  }
  return value.includes(":") ? "[" + value + "]" : value;
}

function renderProviderPlanner(showError) {
  const remote = state.providerKind === "remote";
  $("provider-url-row").hidden = !remote;
  $("provider-source-row").hidden = !remote;
  for (const button of $("provider-kind").children) {
    const active = button.dataset.provider === state.providerKind;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  const configuredControl = state.core && state.core.avatar.url;
  const currentIsLocal = providerLabel(configuredControl) === "Local service";
  let controlUrl = currentIsLocal
    ? configuredControl : "http://127.0.0.1:8711";
  const corePort = state.core ? state.core.api.port : 8710;
  const localProtocol = state.core && state.core.api.tls_certfile ? "wss" : "ws";
  const coreHost = connectableCoreHost(
    state.core && state.core.api.host, localProtocol === "wss");
  let sourceUrl = currentIsLocal && state.avatar && state.avatar.source.url
    ? state.avatar.source.url : localProtocol + "://" + coreHost + ":" + corePort;
  if (remote) {
    controlUrl = validateRemoteProvider(Boolean(showError));
    sourceUrl = validateRemoteSource(Boolean(showError));
  }
  const ready = Boolean(controlUrl && sourceUrl);
  const shownControl = controlUrl || "https://avatar.example:8711";
  const shownSource = sourceUrl || "wss://camera.example:8710";
  const preface = remote
    ? "# ENDPOINT FRAGMENTS ONLY — complete docs/remote-deployment.md first\n"
    : "# LOCAL ENDPOINT FRAGMENTS\n";
  $("provider-config-output").textContent = preface
    + "# CORE HOST — merge into custback.yaml\n"
    + "avatar:\n  url: \"" + shownControl + "\"\n\n"
    + "# AVATAR HOST — merge into avatar.yaml\n"
    + "source:\n  url: \"" + shownSource + "\"";
  $("provider-config-help").textContent = remote
    ? "The browser checks URL shape only; custback validates destination safety at startup. Complete docs/remote-deployment.md first: configure API binds, allowed origins, TLS, avatar.token_file on the core host, and source.token_file on the avatar host. Then save these endpoint fragments and restart both services."
    : "Save the first fragment in the core config and the second in the local avatar config. Configure avatar.token_file and source.token_file, then restart both services.";
  $("provider-copy").disabled = !ready;
  if (state.core) $("provider-current").textContent = providerLabel(state.core.avatar.url);
  return ready ? {controlUrl, sourceUrl} : null;
}

$("provider-kind").addEventListener("click", (event) => {
  const button = event.target.closest("[data-provider]");
  if (!button) return;
  state.providerKind = button.dataset.provider;
  renderProviderPlanner(false);
  if (state.providerKind === "remote") $("provider-url").focus();
});
$("provider-url").addEventListener("input", () => renderProviderPlanner(false));
$("provider-url").addEventListener("blur", () => {
  validateRemoteProvider(true);
  renderProviderPlanner(false);
});
$("provider-source-url").addEventListener("input", () => {
  state.providerSourceTouched = true;
  renderProviderPlanner(false);
});
$("provider-source-url").addEventListener("blur", () => {
  validateRemoteSource(true);
  renderProviderPlanner(false);
});
$("provider-copy").addEventListener("click", async (event) => {
  const plan = renderProviderPlanner(true);
  if (!plan) return;
  const button = event.currentTarget;
  try {
    await navigator.clipboard.writeText($("provider-config-output").textContent);
    button.dataset.state = "success";
    button.textContent = "Copied";
    setTimeout(() => {
      button.textContent = "Copy restart config";
      delete button.dataset.state;
    }, 2500);
  } catch (err) {
    button.dataset.state = "error";
    button.textContent = "YAML selected";
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($("provider-config-output"));
    selection.removeAllRanges();
    selection.addRange(range);
    toast("YAML selected — press Ctrl/Cmd+C to copy.", "warn");
    setTimeout(() => {
      button.textContent = "Copy restart config";
      delete button.dataset.state;
    }, 4000);
  }
});

// -- camera quality -----------------------------------------------------------

function formatAnchor(value, axis) {
  const numeric = Number(value);
  const percent = Math.round(numeric * 100) + "%";
  if (numeric === 0) return percent + (axis === "x" ? " · left" : " · top");
  if (numeric === 0.5) return percent + " · centre";
  if (numeric === 1) return percent + (axis === "x" ? " · right" : " · bottom");
  return percent;
}

function colorCorrectionSummary(status) {
  if (!status || status.color_correction_state === undefined) {
    return ["Waiting for detailed colour-correction status…", "warn"];
  }
  const phase = status.color_correction_state;
  const configured = status.color_correction_mode;
  const active = status.color_correction_active === true;
  if (configured === "off" || phase === "disabled") {
    return ["Off · disabled in the effective configuration", ""];
  }
  if (phase === "mode-excluded") {
    return ["Bypassed · this output mode does not use colour correction", ""];
  }
  if (phase === "scene-cut") {
    return ["Scene changed · the previous estimate was cleared", "warn"];
  }
  if (phase === "warming" || status.color_correction_warming === true) {
    return ["Warming up · no fresh correction is available yet", "warn"];
  }
  if (phase === "low-confidence") {
    return [
      active
        ? "Low confidence · the previous correction is being held"
        : "Low confidence · correction is currently bypassed",
      "warn",
    ];
  }
  if (phase === "stale-decay" || status.color_correction_stale === true) {
    return [
      active
        ? "Stale estimate · the previous correction is fading toward neutral"
        : "Stale estimate · correction has returned to neutral",
      "warn",
    ];
  }
  const effective = status.color_correction_effective_mode;
  if (effective === "bypass") {
    return [
      "Bypassed · " + titleCase(status.color_correction_reason || "pipeline fallback"),
      "warn",
    ];
  }
  if (phase !== "active" || !active
      || ["off", "bypass", "identity"].includes(effective)) {
    return ["Ready · no foreground adjustment is currently applied", ""];
  }
  const details = [];
  const exposure = Number(status.color_correction_exposure_ev);
  if (Number.isFinite(exposure)) {
    details.push((exposure >= 0 ? "+" : "") + exposure.toFixed(2) + " EV");
  }
  if (String(effective).includes("white-balance")) details.push("WB active");
  const confidence = Number(status.color_correction_confidence);
  if (Number.isFinite(confidence)) {
    details.push(Math.round(Math.max(0, Math.min(1, confidence)) * 100) + "% confidence");
  }
  return ["Active" + (details.length ? " · " + details.join(" · ") : ""), "good"];
}

function renderColorCorrectionStatus() {
  const [message, tone] = colorCorrectionSummary(state.status);
  $("quality-color-status").textContent = message;
  $("quality-color-status").className = "notice " + (tone || "neutral");
}

function renderQuality() {
  if (!state.core) return;
  const segmentation = state.core.segmentation;
  const compositing = state.core.compositing;
  const correction = compositing.color_correction;
  const background = state.core.background;
  const camera = state.core.camera;
  const output = state.core.output;
  $("quality-color-auto").checked = correction.mode === "auto";
  $("quality-color-strength").disabled = correction.mode !== "auto";
  $("quality-background-fit").value = background.fit_mode;
  $("quality-camera-fit").value = camera.fit_mode;
  $("quality-camera-rotation").value = String(camera.rotation);
  $("quality-blend-space").value = compositing.blend_space;
  $("quality-output-width").value = output.width ?? "";
  $("quality-output-height").value = output.height ?? "";
  $("quality-output-width").setAttribute("aria-invalid", "false");
  $("quality-output-height").setAttribute("aria-invalid", "false");
  $("quality-backend").value = segmentation.backend;
  $("quality-delegate").value = segmentation.delegate;
  const gpu = $("quality-delegate").querySelector('option[value="gpu"]');
  gpu.disabled = ["rvm", "heuristic", "none"].includes(segmentation.backend);
  for (const [id, value, output, formatter] of [
    ["quality-color-strength", correction.strength, "quality-color-strength-value", (v) => Math.round(Number(v) * 100) + "%"],
    ["quality-background-anchor-x", background.anchor_x, "quality-background-anchor-x-value", (v) => formatAnchor(v, "x")],
    ["quality-background-anchor-y", background.anchor_y, "quality-background-anchor-y-value", (v) => formatAnchor(v, "y")],
    ["quality-exposure-limit", correction.exposure_limit_ev, "quality-exposure-limit-value", (v) => Number(v).toFixed(2) + " EV"],
    ["quality-wb-strength", correction.white_balance_strength, "quality-wb-strength-value", (v) => Math.round(Number(v) * 100) + "%"],
    ["quality-adaptation-time", correction.adaptation_time_s, "quality-adaptation-time-value", (v) => Number(v).toFixed(2) + " s"],
    ["quality-threshold", segmentation.threshold, "quality-threshold-value", (v) => Number(v).toFixed(2)],
    ["quality-rvm-downsample", segmentation.rvm_downsample, "quality-rvm-downsample-value", (v) => Number(v) === 0 ? "Auto" : Number(v).toFixed(2)],
    ["quality-mask-blur", segmentation.mask_blur, "quality-mask-blur-value", String],
    ["quality-mask-shift", segmentation.mask_shift, "quality-mask-shift-value", (v) => (Number(v) > 0 ? "+" : "") + v],
    ["quality-smoothing", segmentation.temporal_smoothing, "quality-smoothing-value", (v) => Number(v).toFixed(2)],
    ["quality-light-wrap", compositing.light_wrap, "quality-light-wrap-value", (v) => Number(v).toFixed(2)],
  ]) {
    $(id).value = value;
    $(output).textContent = formatter(value);
  }
  $("quality-edge-refine").checked = segmentation.edge_refine;
  $("quality-model-foreground").checked = compositing.use_model_foreground;
  renderColorCorrectionStatus();
}

$("quality-color-auto").addEventListener("change", (event) => {
  patchCoreControl({compositing: {color_correction: {
    mode: event.target.checked ? "auto" : "off",
  }}}).catch(reportError);
});
$("quality-background-fit").addEventListener("change", (event) => {
  patchCoreControl({background: {fit_mode: event.target.value}}).catch(reportError);
});
$("quality-camera-fit").addEventListener("change", (event) => {
  patchCoreControl({camera: {fit_mode: event.target.value}}).catch(reportError);
});
$("quality-blend-space").addEventListener("change", (event) => {
  patchCoreControl({compositing: {blend_space: event.target.value}}).catch(reportError);
});
$("quality-camera-rotation").addEventListener("change", (event) => {
  patchCoreControl({camera: {rotation: parseInt(event.target.value, 10)}})
    .catch(reportError);
});

for (const [id, output, field, axis] of [
  ["quality-background-anchor-x", "quality-background-anchor-x-value", "anchor_x", "x"],
  ["quality-background-anchor-y", "quality-background-anchor-y-value", "anchor_y", "y"],
]) {
  $(id).addEventListener("input", (event) => {
    $(output).textContent = formatAnchor(event.target.value, axis);
  });
  $(id).addEventListener("change", (event) => {
    patchCoreControl({background: {[field]: parseFloat(event.target.value)}})
      .catch(reportError);
  });
}

for (const [id, output, field, formatter] of [
  ["quality-color-strength", "quality-color-strength-value", "strength", (v) => Math.round(Number(v) * 100) + "%"],
  ["quality-exposure-limit", "quality-exposure-limit-value", "exposure_limit_ev", (v) => Number(v).toFixed(2) + " EV"],
  ["quality-wb-strength", "quality-wb-strength-value", "white_balance_strength", (v) => Math.round(Number(v) * 100) + "%"],
  ["quality-adaptation-time", "quality-adaptation-time-value", "adaptation_time_s", (v) => Number(v).toFixed(2) + " s"],
]) {
  $(id).addEventListener("input", (event) => {
    $(output).textContent = formatter(event.target.value);
  });
  $(id).addEventListener("change", (event) => {
    patchCoreControl({compositing: {color_correction: {
      [field]: parseFloat(event.target.value),
    }}}).catch(reportError);
  });
}

function commitOutputCanvas() {
  const rawWidth = $("quality-output-width").value.trim();
  const rawHeight = $("quality-output-height").value.trim();
  const incomplete = (rawWidth === "") !== (rawHeight === "");
  $("quality-output-width").setAttribute("aria-invalid", String(incomplete));
  $("quality-output-height").setAttribute("aria-invalid", String(incomplete));
  if (incomplete) {
    toast("Set both output dimensions, or leave both blank.", "err");
    return;
  }
  const width = rawWidth === "" ? null : parseInt(rawWidth, 10);
  const height = rawHeight === "" ? null : parseInt(rawHeight, 10);
  patchCoreControl({output: {width, height}}).catch(reportError);
}
$("quality-output-width").addEventListener("change", commitOutputCanvas);
$("quality-output-height").addEventListener("change", commitOutputCanvas);

$("quality-backend").addEventListener("change", (event) => {
  const backend = event.target.value;
  const patch = {segmentation: {backend}};
  if (["rvm", "heuristic", "none"].includes(backend)
      && state.core.segmentation.delegate === "gpu") patch.segmentation.delegate = "cpu";
  patchCoreControl(patch).catch(reportError);
});
$("quality-delegate").addEventListener("change", (event) => {
  patchCoreControl({segmentation: {delegate: event.target.value}})
    .catch(reportError);
});

for (const [id, output, section, field, parse, format] of [
  ["quality-threshold", "quality-threshold-value", "segmentation", "threshold", parseFloat, (v) => Number(v).toFixed(2)],
  ["quality-rvm-downsample", "quality-rvm-downsample-value", "segmentation", "rvm_downsample", parseFloat, (v) => Number(v) === 0 ? "Auto" : Number(v).toFixed(2)],
  ["quality-mask-blur", "quality-mask-blur-value", "segmentation", "mask_blur", (v) => parseInt(v, 10), String],
  ["quality-mask-shift", "quality-mask-shift-value", "segmentation", "mask_shift", (v) => parseInt(v, 10), (v) => (Number(v) > 0 ? "+" : "") + v],
  ["quality-smoothing", "quality-smoothing-value", "segmentation", "temporal_smoothing", parseFloat, (v) => Number(v).toFixed(2)],
  ["quality-light-wrap", "quality-light-wrap-value", "compositing", "light_wrap", parseFloat, (v) => Number(v).toFixed(2)],
]) {
  $(id).addEventListener("input", (event) => {
    $(output).textContent = format(event.target.value);
  });
  $(id).addEventListener("change", (event) => {
    patchCoreControl({[section]: {[field]: parse(event.target.value)}})
      .catch(reportError);
  });
}
$("quality-edge-refine").addEventListener("change", (event) => {
  patchCoreControl({segmentation: {edge_refine: event.target.checked}})
    .catch(reportError);
});
$("quality-model-foreground").addEventListener("change", (event) => {
  patchCoreControl({compositing: {use_model_foreground: event.target.checked}})
    .catch(reportError);
});

// -- diagnostics and safe runtime settings -----------------------------------

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return hours + " h " + minutes + " min";
  return minutes ? minutes + " min" : total + " s";
}

function cameraControlsSummary(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return "—";
  const parts = [
    titleCase(value.policy || "unknown policy"),
    titleCase(value.backend_family || "unknown backend"),
    titleCase(value.qualification || "unqualified"),
    value.writes_performed ? "writes observed" : "no writes",
  ];
  if (Number.isInteger(value.generation)) {
    parts.push("generation " + value.generation);
  }
  const properties = value.properties;
  if (properties && typeof properties === "object" && !Array.isArray(properties)) {
    for (const [name, observation] of Object.entries(properties)) {
      if (!observation || typeof observation !== "object") continue;
      const reading = observation.value === null || observation.value === undefined
        ? "" : " " + new Intl.NumberFormat().format(observation.value);
      parts.push(titleCase(name) + ": " + titleCase(observation.status) + reading);
    }
  }
  return parts.join(" · ");
}

function formatDiagnostic(key, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (key === "camera_controls") return cameraControlsSummary(value);
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => titleCase(item)).join(", ") : "None";
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") {
    if (key.endsWith("_ms")) return value.toFixed(1) + " ms";
    if (key.endsWith("_s")) return formatDuration(value);
    if (key.endsWith("_ev")) return (value >= 0 ? "+" : "") + value.toFixed(2) + " EV";
    if (key.includes("_wb_gain_") || key.endsWith("_scale_x") || key.endsWith("_scale_y")) {
      return value.toFixed(3);
    }
    if (key.endsWith("_confidence")) return value.toFixed(2);
    if (key.endsWith("_rotation")) return value + "°";
    if (key.includes("fps") || key.endsWith("_pct") || key.endsWith("_ratio")) {
      return value.toFixed(1);
    }
    return new Intl.NumberFormat().format(value);
  }
  const enumKeys = new Set([
    "mode", "segmentation_backend", "segmentation_device", "output_backend",
    "capture_backend", "remote_fallback_mode", "background_video_timing_mode",
    "driver_backend", "driver_device",
    "acceleration_mode", "acceleration_requested_provider",
    "acceleration_active_provider", "acceleration_state",
    "color_correction_mode", "color_correction_effective_mode",
    "color_correction_state", "color_correction_reason",
    "camera_fit", "background_fit", "color_input_assumption",
    "background_video_decoder_backend", "background_video_color_status",
  ]);
  return enumKeys.has(key) ? titleCase(value) : String(value);
}

function diagnosticTone(key, value) {
  if (key === "color_correction_state") {
    if (value === "active") return "good";
    if (["warming", "low-confidence", "stale-decay", "scene-cut"].includes(value)) {
      return "warn";
    }
    return "";
  }
  if ((key.includes("failure") || key.includes("miss") || key.includes("dropped")
      || key.includes("restart")) && Number(value) > 0) return "bad";
  if ((key.includes("fallback") || key.includes("stalled")) && value === true) return "warn";
  if ((key === "connected" || key === "remote_connected") && value === true) return "good";
  if ((key === "connected" || key === "remote_connected") && value === false) return "bad";
  return "";
}

function diagnosticLabel(key) {
  const labels = {
    fps: "Output frame rate", mode: "Output mode", capture_fps: "Capture frame rate",
    capture_frame_age_ms: "Latest frame age", frame_processing_ms: "Frame processing",
    segmentation_backend: "Segmentation backend", segmentation_device: "Segmentation device",
    acceleration_mode: "Acceleration policy", acceleration_requested_provider: "Requested provider",
    acceleration_active_provider: "Active provider", acceleration_state: "Acceleration state",
    acceleration_fallback_active: "GPU fallback active", acceleration_fallback_reason: "GPU fallback reason",
    acceleration_fallback_count: "GPU fallback count", acceleration_device_id: "Accelerator device",
    acceleration_last_transition_ms: "Acceleration transition age",
    capture_dropped_frames: "Dropped camera frames", processing_deadline_misses: "Processing deadline misses",
    capture_delivered_width: "Delivered camera width", capture_delivered_height: "Delivered camera height",
    capture_oriented_width: "Oriented camera width", capture_oriented_height: "Oriented camera height",
    capture_normalized_width: "Normalized camera width", capture_normalized_height: "Normalized camera height",
    capture_generation: "Capture generation", camera_fit: "Camera fit", camera_rotation: "Camera rotation",
    camera_mirror: "Camera mirror", camera_scale_x: "Camera horizontal scale",
    camera_scale_y: "Camera vertical scale", background_fit: "Background fit",
    background_scale_x: "Background horizontal scale", background_scale_y: "Background vertical scale",
    camera_controls: "Camera-control qualification",
    background_video_decoder_backend: "Video decoder backend",
    background_video_color_status: "Video colour status",
    background_video_input_color: "Declared video input colour",
    background_video_output_color: "Normalized video output colour",
    background_video_color_assumed_fields: "Assumed video colour fields",
    background_video_color_overridden_fields: "Overridden video colour fields",
    color_correction_mode: "Configured colour correction",
    color_correction_active: "Colour transform applied",
    color_correction_effective_mode: "Effective colour correction",
    color_correction_state: "Colour correction state",
    color_correction_reason: "Colour correction reason",
    color_correction_confidence: "Colour estimate confidence",
    color_correction_exposure_ev: "Exposure correction",
    color_correction_wb_gain_r: "Red gain", color_correction_wb_gain_g: "Green gain",
    color_correction_wb_gain_b: "Blue gain", color_correction_warming: "Correction warming",
    color_correction_stale: "Correction stale", color_input_assumption: "Input colour assumption",
    uptime_s: "Uptime", connected: "Camera feed connected", driver_backend: "Following driver",
    face_present: "Face detected", render_ms: "Avatar render time", render_failures: "Render failures",
    output_width: "Output width", output_height: "Output height",
  };
  return labels[key] || titleCase(key);
}

function renderDiagnosticList(container, rows) {
  container.textContent = "";
  for (const [label, value, tone] of rows) {
    const row = document.createElement("div");
    row.className = "diagnostic-row";
    const term = document.createElement("dt");
    term.textContent = label;
    const detail = document.createElement("dd");
    detail.textContent = value;
    if (tone) detail.className = tone;
    row.append(term, detail);
    container.appendChild(row);
  }
}

function allDiagnosticRows(object) {
  if (!object) return [["Status", "Waiting for data", "warn"]];
  return Object.entries(object).map(([key, value]) => [
    diagnosticLabel(key), formatDiagnostic(key, value), diagnosticTone(key, value),
  ]);
}

function geometrySummary(status) {
  if (!status || !status.camera_fit) return "Waiting for geometry status";
  const sourceWidth = status.capture_delivered_width;
  const sourceHeight = status.capture_delivered_height;
  const outputWidth = status.output_width;
  const outputHeight = status.output_height;
  const source = sourceWidth && sourceHeight
    ? sourceWidth + " × " + sourceHeight : "camera";
  const output = outputWidth && outputHeight
    ? outputWidth + " × " + outputHeight : "output";
  const scaledWidth = Number(status.capture_oriented_width)
    * Number(status.camera_scale_x);
  const scaledHeight = Number(status.capture_oriented_height)
    * Number(status.camera_scale_y);
  const crop = Number.isFinite(scaledWidth) && Number.isFinite(scaledHeight)
    && scaledWidth > 0 && scaledHeight > 0
    && (
      Number(status.camera_crop_left) > 0
      || Number(status.camera_crop_top) > 0
      || Number(status.camera_crop_right) < Math.round(scaledWidth)
      || Number(status.camera_crop_bottom) < Math.round(scaledHeight)
    );
  const pad = ["left", "top", "right", "bottom"].some((edge) =>
    Number(status["camera_pad_" + edge]) > 0);
  const operation = crop ? "crop" : (pad ? "pad" : "no crop");
  return source + " → " + titleCase(status.camera_fit) + " / " + operation
    + " → " + output;
}

function videoColorSummary(status) {
  const colorStatus = status && status.background_video_color_status;
  if (!colorStatus) return ["Not applicable", ""];
  const parts = [titleCase(colorStatus)];
  if (status.background_video_decoder_backend) {
    parts.push(titleCase(status.background_video_decoder_backend));
  }
  if (status.background_video_input_color && status.background_video_output_color) {
    parts.push(status.background_video_input_color + " → "
      + status.background_video_output_color);
  }
  const warning = String(colorStatus).includes("legacy")
    || String(colorStatus).includes("assumption");
  return [parts.join(" · "), warning ? "warn" : "good"];
}

function renderDiagnostics() {
  const status = state.status;
  renderColorCorrectionStatus();
  if (status) {
    const dimensions = status.capture_width && status.capture_height
      ? status.capture_width + " × " + status.capture_height : "Negotiating";
    const coreRows = [
      ["Camera", dimensions + " · " + status.capture_fps.toFixed(1) + " fps", status.capture_stalled ? "bad" : "good"],
      ["Output", status.fps.toFixed(1) + " / " + status.output_target_fps + " fps", status.output_fallback_active ? "warn" : ""],
      ["Subject detection", titleCase(status.segmentation_backend) + " · " + titleCase(status.segmentation_device), status.segmentation_fallback_active ? "warn" : ""],
    ];
    // Acceleration is only meaningful for the RVM/ONNX Runtime backend; other
    // segmenters report an empty active provider. Show the truthful post-
    // fallback provider, not the requested one.
    if (status.acceleration_active_provider) {
      const accelValue = status.acceleration_fallback_active
        ? titleCase(status.acceleration_active_provider) + " (fell back from GPU)"
        : titleCase(status.acceleration_active_provider);
      coreRows.push(["Acceleration", accelValue, status.acceleration_fallback_active ? "warn" : ""]);
    }
    if (status.background_video_color_status) {
      coreRows.push(["Video colour", ...videoColorSummary(status)]);
    }
    coreRows.push(
      ["Geometry", geometrySummary(status), ""],
      ["Colour correction", ...colorCorrectionSummary(status)],
      ["Frame processing", formatDiagnostic("frame_processing_ms", status.frame_processing_ms), ""],
      ["Dropped camera frames", new Intl.NumberFormat().format(status.capture_dropped_frames), status.capture_dropped_frames ? "bad" : ""],
      ["Uptime", formatDuration(status.uptime_s), ""],
    );
    renderDiagnosticList($("core-diagnostics"), coreRows);
  } else {
    renderDiagnosticList($("core-diagnostics"), [["Camera pipeline", "Unavailable", "bad"]]);
  }
  renderDiagnosticList($("core-diagnostics-all"), allDiagnosticRows(status));

  const avatarStatus = state.avatarStatus;
  if (avatarStatus) {
    const size = avatarStatus.output_width && avatarStatus.output_height
      ? avatarStatus.output_width + " × " + avatarStatus.output_height : "Waiting for frames";
    renderDiagnosticList($("avatar-diagnostics"), [
      ["Control API", state.avatarState === "ok" ? "Reachable" : "Unavailable", state.avatarState === "ok" ? "good" : "bad"],
      ["Camera feed", avatarStatus.connected ? "Connected" : "Disconnected", avatarStatus.connected ? "good" : "bad"],
      ["Following driver", titleCase(avatarStatus.driver_backend), ""],
      ["Face detected", avatarStatus.face_present ? "Yes" : "No", avatarStatus.face_present ? "good" : "warn"],
      ["Avatar output", size, ""],
      ["Render time", formatDiagnostic("render_ms", avatarStatus.render_ms), ""],
      ["Render failures", new Intl.NumberFormat().format(avatarStatus.render_failures), avatarStatus.render_failures ? "bad" : ""],
    ]);
  } else {
    renderDiagnosticList($("avatar-diagnostics"), [["Avatar provider", state.avatarState === "unconfigured" ? "Not configured" : "Unavailable", "warn"]]);
  }
  renderDiagnosticList($("avatar-diagnostics-all"), allDiagnosticRows(avatarStatus));
}

function renderSystemConfig() {
  if (!state.core) return;
  $("core-remote-timeout").value = state.core.api.remote_timeout_ms;
  $("avatar-connect-timeout").value = state.core.avatar.connect_timeout_s;
  $("avatar-read-timeout").value = state.core.avatar.read_timeout_s;
  $("effective-config").textContent = JSON.stringify({
    core: state.core,
    avatar_service: state.avatar || "unavailable",
  }, null, 2);
}

$("core-remote-timeout").addEventListener("change", (event) => {
  patchCore({api: {remote_timeout_ms: parseInt(event.target.value, 10)}})
    .catch((err) => { reportError(err); renderSystemConfig(); });
});
$("avatar-connect-timeout").addEventListener("change", (event) => {
  patchCore({avatar: {connect_timeout_s: parseFloat(event.target.value)}})
    .catch((err) => { reportError(err); renderSystemConfig(); });
});
$("avatar-read-timeout").addEventListener("change", (event) => {
  patchCore({avatar: {read_timeout_s: parseFloat(event.target.value)}})
    .catch((err) => { reportError(err); renderSystemConfig(); });
});

// -- top-level render -----------------------------------------------------------

function renderAll() {
  if (state.core) {
    const enabled = state.core.background.mode === "remote";
    $("avatar-enabled").checked = enabled;
    $("avatar-state-label").textContent = enabled ? "On" : "Off";
    $("provider-current").textContent = providerLabel(state.core.avatar.url);
    if (!state.providerInitialized) {
      const currentUrl = state.core.avatar.url;
      const currentRemote = providerLabel(currentUrl) === "Remote server";
      state.providerKind = currentRemote ? "remote" : "local";
      if (currentRemote) {
        $("provider-url").value = currentUrl;
        if (state.avatar && state.avatar.source.url) {
          $("provider-source-url").value = state.avatar.source.url;
        }
      }
      state.providerInitialized = true;
    }
    const currentRemote = providerLabel(state.core.avatar.url) === "Remote server";
    if (currentRemote && !state.providerSourceTouched
        && !$("provider-source-url").value
        && state.avatar && state.avatar.source.url) {
      $("provider-source-url").value = state.avatar.source.url;
    }
  }
  renderProviderPlanner(false);
  renderAvatar();
  renderBackground();
  renderQuality();
  renderSystemConfig();
  renderDiagnostics();
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
