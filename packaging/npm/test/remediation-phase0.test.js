'use strict';

/**
 * Phase 0 regression specifications for the packaging/remediation work.
 *
 * These tests deliberately describe the desired release invariants before the
 * production fixes land. Known failures are marked TODO so the existing npm
 * suite remains usable while each remediation item is implemented. Remove the
 * corresponding `todo` option when an item is fixed.
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const installer = require('../install');
const release = require('../../../scripts/release/verify-release');

const root = path.resolve(__dirname, '..', '..', '..');

function read(relative) {
  return fs.readFileSync(path.join(root, relative), 'utf8');
}

function json(relative) {
  return JSON.parse(read(relative));
}

test(
  'NPM-01: the default managed venv and rollback generations survive npm package replacement',
  { todo: 'move the default managed runtime outside the replaceable npm package root' },
  () => {
    const sources = [read('packaging/npm/install.js'), read('packaging/npm/custback.js')];
    for (const source of sources) {
      assert.equal(
        /DEFAULT_VENV\s*=\s*path\.join\(PKG_ROOT,\s*['"]\.venv['"]\s*\)/.test(source),
        false,
        'the default venv must not be stored below the replaceable package root',
      );
    }
  },
);

test(
  'NPM-01: an absent CUSTBACK_EXTRAS preserves persisted intent while an explicit empty value clears it',
  { todo: 'persist install intent and distinguish an absent environment variable from an empty one' },
  () => {
    const source = read('packaging/npm/install.js');
    const collapsesAbsentAndEmpty =
      /parseExtras\(process\.env\.CUSTBACK_EXTRAS\s*\|\|\s*['"]{2}\)/.test(source);
    const distinguishesPresence =
      /hasOwnProperty(?:\.call)?\([^\n]*CUSTBACK_EXTRAS/.test(source) ||
      /process\.env\.CUSTBACK_EXTRAS\s*!==\s*undefined/.test(source);
    const persistsIntent = /(?:install|extras)[-_ ]intent/i.test(source);

    assert.deepEqual(
      { collapsesAbsentAndEmpty, distinguishesPresence, persistsIntent },
      { collapsesAbsentAndEmpty: false, distinguishesPresence: true, persistsIntent: true },
    );
  },
);

test(
  'NPM-01: config/avatar.yaml participates in the managed-environment source digest',
  { todo: 'include every shipped configuration template in sourceDigest()' },
  (t) => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-avatar-digest-'));
    t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));

    fs.mkdirSync(path.join(fixture, 'config'), { recursive: true });
    fs.writeFileSync(path.join(fixture, 'package.json'), '{"name":"custback"}\n');
    fs.writeFileSync(path.join(fixture, 'pyproject.toml'), '[project]\nname = "custback"\n');
    fs.writeFileSync(path.join(fixture, 'config', 'default.yaml'), 'mode: blur\n');
    fs.writeFileSync(path.join(fixture, 'config', 'avatar.yaml'), 'driver: vision\n');

    const before = installer.sourceDigest(fixture);
    fs.writeFileSync(path.join(fixture, 'config', 'avatar.yaml'), 'driver: audio2face\n');
    const after = installer.sourceDigest(fixture);

    assert.notEqual(after, before, 'changing a shipped avatar template must invalidate the venv');
  },
);

test(
  'PKG-01: npm exposes both documented custback and custback-avatar launch surfaces',
  { todo: 'publish the compatibility avatar bin and keep package-lock/docs in sync' },
  () => {
    const pkg = json('package.json');
    const lock = json('package-lock.json');
    const readme = read('README.md');
    const expectedBins = {
      custback: 'packaging/npm/custback.js',
      'custback-avatar': 'packaging/npm/custback.js',
    };

    assert.deepEqual(pkg.bin, expectedBins);
    assert.deepEqual(lock.packages[''].bin, expectedBins);
    assert.match(readme, /\bcustback avatar\b/);
    assert.match(readme, /\bcustback-avatar\b/);
  },
);

test(
  'PKG-02: the documented npm pack command is safe for shell command substitution',
  { todo: 'silence lifecycle chatter in the documented tarball capture command' },
  () => {
    const tarballCommands = read('README.md').split('\n')
      .filter((line) => line.includes('TARBALL='));
    assert.ok(tarballCommands.length > 0, 'README must document local tarball installation');
    assert.equal(
      tarballCommands.every((line) => line.includes('TARBALL=$(npm pack --silent)')),
      true,
      `unsafe tarball capture command: ${tarballCommands.join(' | ')}`,
    );
  },
);

test(
  'PKG-02: prepack never claims that an npm payload was verified when payload verification was skipped',
  { todo: 'either verify the ignore-scripts packlist during prepack or emit a narrower status message' },
  () => {
    const source = read('scripts/release/verify-release.js');
    const skipsPayloadVerification =
      /if\s*\(\s*!argv\.includes\(['"]--prepack['"]\)\s*\)\s*\{\s*verifyPack\(version\)/s
        .test(source);
    const claimsPayloadVerified =
      /console\.log\([^\n]*metadata and npm payload verified/.test(source);

    assert.equal(
      skipsPayloadVerification && claimsPayloadVerified,
      false,
      'a skipped payload check and a payload-verified success message cannot coexist',
    );
  },
);

test(
  'PLATFORM-01: Linux setup rejects unsupported distributions before invoking apt-get',
  { todo: 'validate ID/ID_LIKE from os-release before running Debian-specific setup' },
  () => {
    const source = read('scripts/install_linux.sh');
    const distroProbe = source.search(/(?:\/etc\/os-release|lsb_release)/);
    const distroIdentity = source.search(/\b(?:ID|ID_LIKE)\b/);
    const unsupportedExit = source.search(/unsupported[^\n]*(?:distribution|distro)|(?:distribution|distro)[^\n]*unsupported/i);
    const apt = source.search(/\bapt-get\b/);

    assert.ok(distroProbe >= 0 && distroProbe < apt, 'distribution must be probed before apt-get');
    assert.ok(distroIdentity >= 0 && distroIdentity < apt, 'ID/ID_LIKE must be checked before apt-get');
    assert.ok(unsupportedExit >= 0 && unsupportedExit < apt, 'unsupported distributions must fail first');
  },
);

test(
  'LICENSE-01: canonical MIT license text is present in npm and Python release contracts',
  { todo: 'add LICENSE and include it in every built artifact and release allowlist' },
  () => {
    const licensePath = path.join(root, 'LICENSE');
    const licenseText = fs.existsSync(licensePath) ? fs.readFileSync(licensePath, 'utf8') : '';
    const pkg = json('package.json');
    const pyproject = read('pyproject.toml');
    const releaseSource = read('scripts/release/verify-release.js');
    const results = {
      canonicalFile: fs.existsSync(licensePath) && fs.statSync(licensePath).isFile(),
      canonicalMitText:
        /Permission is hereby granted, free of charge/.test(licenseText) &&
        /THE SOFTWARE IS PROVIDED [“"]AS IS[”"]/.test(licenseText),
      npmArtifactContract: pkg.files.includes('LICENSE'),
      pythonArtifactContract:
        /license(?:-files|_files)?\s*=\s*(?:\{[^\n]*file\s*=\s*)?["']?LICENSE/i.test(pyproject),
      reviewedReleasePayload: /['"]LICENSE['"]/.test(releaseSource),
    };

    assert.deepEqual(results, {
      canonicalFile: true,
      canonicalMitText: true,
      npmArtifactContract: true,
      pythonArtifactContract: true,
      reviewedReleasePayload: true,
    });
  },
);

test(
  'A2F-01: the Audio2Face extra names the published service protocol package used by the driver',
  { todo: 'replace the impossible nvidia-ace constraint and update protocol imports/release allowlists' },
  () => {
    const pyproject = read('pyproject.toml');
    const driver = read('src/custback/avatar/audio2face.py');
    const releaseSource = read('scripts/release/verify-release.js');
    const extra = pyproject.match(/\baudio2face\s*=\s*\[([\s\S]*?)\]/);

    assert.ok(extra, 'pyproject must define the audio2face optional dependency group');
    assert.match(extra[1], /nvidia-audio2face-3d/i);
    assert.doesNotMatch(extra[1], /nvidia-ace\s*>=\s*1\.2/i);
    assert.match(driver, /nvidia_audio2face_3d/);
    assert.match(releaseSource, /nvidia-audio2face-3d/i);
  },
);

test(
  'DEPLOY-01: the remote deployment guide separates renderer and control credentials over WSS/HTTPS',
  { todo: 'document both remote trust planes, their distinct tokens, and TLS transports' },
  () => {
    const readme = read('README.md');
    const start = readme.indexOf('### Running the avatar service on another host');
    const next = readme.indexOf('\n### ', start + 4);
    const guide = start >= 0 ? readme.slice(start, next >= 0 ? next : undefined) : '';
    const requirements = {
      sectionPresent: start >= 0,
      rendererToken:
        /renderer(?:-scoped)?[- ]+(?:credential|token)|renderer[_ -]token/i.test(guide),
      controlToken:
        /avatar[- ]control[- ]+(?:credential|token)|control[_ -]token/i.test(guide),
      explicitlySeparate: /\b(?:separate|distinct)\b/i.test(guide),
      rendererWss: /\bwss:\/\//i.test(guide),
      controlHttps: /\bhttps:\/\//i.test(guide),
    };

    assert.deepEqual(requirements, {
      sectionPresent: true,
      rendererToken: true,
      controlToken: true,
      explicitlySeparate: true,
      rendererWss: true,
      controlHttps: true,
    });
  },
);

test(
  'HYGIENE-01: generated ONNX Runtime profiles are ignored and rejected as stale release artifacts',
  { todo: 'ignore onnxruntime_profile__*.json and make staleArtifacts reject them' },
  (t) => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-onnx-profile-'));
    t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
    const profile = 'onnxruntime_profile__2026-07-16_12-34-56.json';
    fs.writeFileSync(path.join(fixture, profile), '{}\n');

    const ignoreRules = read('.gitignore').split('\n')
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith('#'));
    const results = {
      ignored: ignoreRules.includes('onnxruntime_profile__*.json'),
      rejectedByReleaseGate: release.staleArtifacts(fixture).includes(profile),
    };

    assert.deepEqual(results, { ignored: true, rejectedByReleaseGate: true });
  },
);
