'use strict';

// Deferred Windows release-evidence contract.
//
// This non-blocking TODO records the target end state: a Windows release must
// not publish without exact Windows evidence. It is intentionally not registered
// as a remediation blocker while Windows production publishing is deferred.
//
// The machinery this TODO waits on (a windows-latest evidence source, WIN-1.8;
// the frozen-engine/shell/installer build jobs; and the manifest-driven
// conditional that keeps a Linux-only release unblocked) does not exist yet.

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
  'deferred Windows production publish requires exact Windows release evidence',
  {
    todo:
      'deferred until WIN-1.8 provides a windows-latest evidence ' +
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
  'Windows evidence is currently outside the release blocker registry',
  () => {
    assert.doesNotThrow(() => release.verifyNoReleaseBlockers(root));
    assert.equal(
      release.remediationBlockers(root).some((entry) => entry.id === 'WIN-01'),
      false,
    );

    // Keep the deferred gap explicit: no Windows evidence gate, Windows
    // installer artifact, or Windows release job is wired yet.
    const manifest = requiredGatesManifest();
    const startsWithWindows = (value) => value.startsWith('windows');
    assert.ok(!manifest.required_gates.some(startsWithWindows));
    assert.ok(!manifest.workflow.required_job_ids.some(startsWithWindows));
    assert.ok(!manifest.artifacts.some((entry) => entry.id === WINDOWS_INSTALLER_ARTIFACT));
    assert.doesNotMatch(releaseWorkflowSource(), /runs-on:\s*windows-latest/);
  },
);
