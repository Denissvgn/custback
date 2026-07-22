'use strict';

// WIN-01 release-gate slot (WINDOWS_DECISIONS.md Part B, applied by WIN-5.8).
//
// This is the Windows analogue of the REL-01 gate in release-check.test.js: an
// open remediation blocker whose acceptance test (a TODO until the machinery
// lands) asserts the target end state — a Windows release must not publish
// without exact Windows evidence — and whose guard proves that today the
// installed publish machinery does not yet require it.
//
// The machinery this TODO waits on (a windows-latest evidence source, WIN-1.8;
// the frozen-engine/shell/installer build jobs; and the manifest-driven
// conditional that keeps a Linux-only release unblocked) does not exist yet, so
// mutating the consumed required-gates.json arrays or release.yml jobs now would
// break the live Linux pipeline. The slot is therefore held open here until
// that machinery exists, exactly as WIN-0.2 specified.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const release = require('../../../scripts/release/verify-release');

const root = path.resolve(__dirname, '..', '..', '..');

// The Windows evidence the target gate must require (WIN-0.2 item 3).
const WINDOWS_REQUIRED_GATES = [
  'windows-core',
  'windows-storage-ntfs',
  'windows-camera-obs',
  'windows-acceleration',
  'windows-installer-clean-vm',
];
const WINDOWS_JOB_IDS = [
  'windows-core',
  'windows-artifact-build',
  'windows-artifact-validation',
  'windows-installer',
];
const WINDOWS_INSTALLER_ARTIFACT = 'windows-installer';

function requiredGatesManifest() {
  return JSON.parse(
    fs.readFileSync(
      path.join(root, 'scripts', 'release', 'required-gates.json'), 'utf8',
    ),
  );
}

function releaseWorkflowSource() {
  return fs.readFileSync(
    path.join(root, '.github', 'workflows', 'release.yml'), 'utf8',
  );
}

test(
  'WIN-01: Windows production publish requires exact Windows release evidence',
  {
    todo:
      'WIN-01 stays open until WIN-1.8 provides a windows-latest evidence ' +
      'source and the frozen-engine/shell/installer build machinery exists',
  },
  () => {
    const manifest = requiredGatesManifest();
    const workflow = releaseWorkflowSource();
    const artifactIds = manifest.artifacts.map((entry) => entry.id);

    // Target: the manifest declares the Windows evidence gates, the installer
    // artifact, and the build jobs, and release.yml runs them on windows-latest.
    for (const gate of WINDOWS_REQUIRED_GATES) {
      assert.ok(
        manifest.required_gates.includes(gate),
        `required-gates.json must require ${gate}`,
      );
    }
    for (const jobId of WINDOWS_JOB_IDS) {
      assert.ok(
        manifest.workflow.required_job_ids.includes(jobId),
        `required-gates.json workflow must require ${jobId}`,
      );
    }
    assert.ok(
      artifactIds.includes(WINDOWS_INSTALLER_ARTIFACT),
      'required-gates.json must define the windows-installer artifact',
    );
    assert.match(
      workflow, /runs-on:\s*windows-latest/,
      'release.yml must build/validate on windows-latest',
    );

    // Scoping (WIN-0.2 item 4): the Windows gates are required only when a
    // Windows artifact is part of the release manifest, so a Linux/macOS-only
    // release is never blocked by absent Windows evidence.
    assert.ok(
      /windows[_-]artifact[_-]requested|manifest[_-]driven|conditional/i.test(workflow),
      'release.yml must scope Windows gates to a requested Windows artifact',
    );
  },
);

test(
  'WIN-01: open registry still blocks the installed Windows publish machinery',
  () => {
    // The open blocker keeps the whole release blocked (never a Windows-only
    // bypass around REL-01, CC-3): the aggregate check names WIN-01.
    assert.throws(() => release.verifyNoReleaseBlockers(root), /WIN-01/);

    // Prove the machinery gap is real *today*: no Windows evidence gate, no
    // Windows installer artifact, and no Windows build jobs are wired yet, so a
    // hypothetical Windows publish would not be required to carry any Windows
    // evidence. This is the known failure the TODO above will close.
    const manifest = requiredGatesManifest();
    const startsWithWindows = (value) => value.startsWith('windows');
    assert.ok(!manifest.required_gates.some(startsWithWindows));
    assert.ok(!manifest.workflow.required_job_ids.some(startsWithWindows));
    assert.ok(!manifest.artifacts.some((entry) => entry.id === WINDOWS_INSTALLER_ARTIFACT));
    assert.doesNotMatch(releaseWorkflowSource(), /runs-on:\s*windows-latest/);
  },
);
