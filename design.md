<!-- Hallmark · pre-emit critique: P5 H5 E5 S5 R5 V4 -->
# Design — custback

Locked design system. Future Hallmark runs read this file first; pages defer
to it. Amend intentionally — this file is the rule.

## System

- Genre · modern-minimal
- Macrostructure · Workbench
- Theme · Cobalt, adapted for offline system fonts
- Axes · light paper / technical mono display / cool-cobalt accent
- Audience · regular users operating camera and avatar controls
- Voice · technical, direct, and understandable; explain consequences at the control

## Tokens (canonical · `tokens.css` is the source of truth)

```css
:root {
  --color-paper: oklch(98.5% 0.004 250);
  --color-paper-2: oklch(96% 0.008 250);
  --color-ink: oklch(20% 0.018 258);
  --color-ink-2: oklch(31% 0.018 257);
  --color-rule: oklch(82% 0.012 250);
  --color-accent: oklch(48% 0.22 256);
  --color-accent-ink: oklch(98.5% 0.004 250);
  --color-focus: oklch(46% 0.2 256);

  --font-display: "Cascadia Code", "IBM Plex Mono", "DejaVu Sans Mono", "Liberation Mono", monospace;
  --font-body: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;

  /* Full colour, 4-pt spacing, type, rule, and z-index sets live in tokens.css. */
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --dur-micro: 120ms; --dur-short: 220ms; --dur-long: 420ms;
  --radius-control: 0.375rem; --radius-panel: 0.625rem; --radius-round: 999px;
}
```

## CTA voice

- Primary · cobalt fill, accent ink, control radius, `--space-xs` × `--space-sm` padding.
- Secondary · paper fill with strong-rule outline and the same geometry.
- Labels · short verb–noun phrases; clickable text never wraps.

## Motion stance

- Motion-cut: short colour-state transitions and a 1 px active press only.
- Reduced motion disables spatial animation and the loading spinner.

## Notes

- Keep live output dominant: dark preview, light controls, N1b tabs, Ft2 footer.
- Stack preview before controls on narrow screens; split the Workbench on desktop.
- Separate sections with hairline rules and surface shifts, not nested card stacks.
- Use silent success for visible changes and persistent notices for failures.
- Controls are at least 44 px with immediate 3:1 focus rings and no wrapped labels.
- Support 320, 375, 414, and 768 px without horizontal scrolling.
- Mark restart-only settings explicitly; never imply provider credentials are browser-saved.

## Exports

`tokens.css` in this project is the source of truth. For Tailwind v4 `@theme`,
DTCG `tokens.json`, or shadcn/ui variables, extend this section using Hallmark's
canonical export mappings.
