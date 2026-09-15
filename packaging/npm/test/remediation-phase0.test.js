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
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const installer = require('../install');
const launcher = require('../custback');
const managed = require('../managed-venv');
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
  (t) => {
    const sources = [read('packaging/npm/install.js'), read('packaging/npm/custback.js')];
    for (const source of sources) {
      assert.equal(
        /DEFAULT_VENV\s*=\s*path\.join\(PKG_ROOT,\s*['"]\.venv['"]\s*\)/.test(source),
        false,
        'the default venv must not be stored below the replaceable package root',
      );
    }
    const prefix = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-phase0-prefix-'));
    t.after(() => fs.rmSync(prefix, { recursive: true, force: true }));
    const packageRoot = path.join(prefix, 'lib', 'node_modules', 'custback');
    fs.mkdirSync(packageRoot, { recursive: true });
    const target = managed.defaultTargetForPackage(packageRoot);
    const generationRoot = managed.ensureGenerationsRoot(target);
    const makeGeneration = () => {
      const generation = managed.createGeneration(generationRoot);
      fs.mkdirSync(path.join(generation, 'bin'));
      fs.writeFileSync(path.join(generation, 'pyvenv.cfg'), 'home = /python\n');
      fs.writeFileSync(path.join(generation, 'bin', 'python'), '#!/bin/sh\n');
      fs.writeFileSync(path.join(generation, 'bin', 'custback'), '#!/bin/sh\n');
      managed.markGeneration(generation, target);
      return generation;
    };
    const rollback = makeGeneration();
    managed.promoteGeneration({
      target,
      generation: rollback,
      inspection: managed.inspectTarget(target),
      validateActive() {},
    });
    const active = makeGeneration();
    managed.promoteGeneration({
      target,
      generation: active,
      inspection: managed.inspectTarget(target),
      validateActive() {},
    });
    installer.writeInstallIntent(target, ['gpu']);

    fs.rmSync(packageRoot, { recursive: true });
    fs.mkdirSync(packageRoot, { recursive: true });
    assert.equal(managed.defaultTargetForPackage(packageRoot), target);
    assert.equal(fs.realpathSync(target), fs.realpathSync(active));
    managed.validateGeneration(rollback, target, generationRoot);
    assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['gpu']);
  },
);

test(
  'NPM-01: an absent CUSTBACK_EXTRAS preserves persisted intent while an explicit empty value clears it',
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
    const intent = { requestedExtras: ['gpu'] };
    assert.deepEqual(installer.resolveRequestedExtras({}, intent), ['gpu']);
    assert.deepEqual(installer.resolveRequestedExtras({ CUSTBACK_EXTRAS: '' }, intent), []);
  },
);

test(
  'NPM-01: config/avatar.yaml participates in the managed-environment source digest',
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
  () => {
    const pkg = json('package.json');
    const lock = json('package-lock.json');
    assert.match(read('README.md'), /\(docs\/user-guide\.md\)/);
    const readme = read('docs/user-guide.md');
    const expectedBins = {
      custback: 'packaging/npm/custback.js',
      'custback-avatar': 'packaging/npm/custback.js',
      'custback-npm-migrate': 'packaging/npm/migrate-legacy.js',
    };

    assert.deepEqual(pkg.bin, expectedBins);
    assert.deepEqual(lock.packages[''].bin, expectedBins);
    if (process.platform !== 'win32') {
      assert.notEqual(
        fs.statSync(path.join(root, 'packaging', 'npm', 'custback.js')).mode & 0o111,
        0,
        'npm bin target must be executable',
      );
    }
    assert.match(readme, /\bcustback avatar\b/);
    assert.match(readme, /\bcustback-avatar\b/);
    assert.match(readme, /custback avatar config export/);
    assert.equal(launcher.invokedAsAvatar('/npm-prefix/bin/custback-avatar'), true);
    let exported = Buffer.alloc(0);
    launcher.exportAvatarConfig([], {
      write(chunk) { exported = Buffer.concat([exported, chunk]); },
    });
    assert.deepEqual(exported, fs.readFileSync(path.join(root, 'config', 'avatar.yaml')));
  },
);

test(
  'PKG-02: the documented npm pack command is safe for shell command substitution',
  () => {
    const tarballCommands = read('docs/user-guide.md').split('\n')
      .filter((line) => line.includes('TARBALL='));
    assert.ok(tarballCommands.length > 0, 'User guide must document local tarball installation');
    assert.equal(
      tarballCommands.every((line) => line.includes('TARBALL=$(npm pack --silent)')),
      true,
      `unsafe tarball capture command: ${tarballCommands.join(' | ')}`,
    );
  },
);

test(
  'PKG-02: prepack never claims that an npm payload was verified when payload verification was skipped',
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
  'PKG-02: npm pack --silent stdout is one installable filename',
  (t) => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-pack-contract-'));
    const checkout = path.join(fixture, 'checkout');
    const destination = path.join(fixture, 'pack');
    const cache = path.join(fixture, 'cache');
    const prefix = path.join(fixture, 'prefix');
    const version = json('package.json').version;
    t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
    fs.cpSync(root, checkout, {
      recursive: true,
      filter(source) {
        const relative = path.relative(root, source);
        const parts = relative.split(path.sep);
        return !parts.some((part) => [
          '.agents', '.codex', '.git', '.pytest_cache', '.venv', '__pycache__',
          'build', 'dist', 'node_modules',
        ].includes(part) || part.endsWith('.egg-info') ||
          part.endsWith('.custback-generations')) &&
          !relative.endsWith('.tgz') && !relative.endsWith('.whl') &&
          !relative.endsWith('.tar.gz') && !relative.endsWith('.pyc') &&
          !/(^|[\\/])onnxruntime_profile__.*\.json$/.test(relative);
      },
    });
    fs.mkdirSync(destination);

    // Normalize this copy to a closed registry so the fixture remains robust if
    // a future blocker is added. This only exercises npm's post-prepack stdout
    // contract; it is not release evidence.
    const registryPath = path.join(
      checkout, 'scripts', 'release', 'remediation-blockers.json',
    );
    const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
    registry.release_blocked = false;
    registry.blockers = registry.blockers.map((blocker) => {
      const regression = { ...blocker.regression };
      delete regression.guard;
      return { ...blocker, status: 'resolved', regression };
    });
    fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
    const fixtureReleasePath = path.join(
      checkout, 'scripts', 'release', 'verify-release.js',
    );
    const fixtureRelease = require(fixtureReleasePath);
    const closedDigest = fixtureRelease.remediationContractDigest(
      fixtureRelease.remediationRegistry(checkout),
    );
    const fixtureReleaseSource = fs.readFileSync(fixtureReleasePath, 'utf8');
    fs.writeFileSync(
      fixtureReleasePath,
      fixtureReleaseSource.replace(
        /const REVIEWED_REMEDIATION_CONTRACT_SHA256\s*=\s*\n?\s*'[^']+';/,
        `const REVIEWED_REMEDIATION_CONTRACT_SHA256 = '${closedDigest}';`,
      ),
    );

    const packed = spawnSync('npm', [
      'pack', '--silent', '--pack-destination', destination, '--cache', cache,
    ], {
      cwd: checkout,
      encoding: 'utf8',
      timeout: 120_000,
      env: { ...process.env, CUSTBACK_SKIP_INSTALL: '1' },
    });
    assert.equal(packed.error, undefined, packed.error && packed.error.message);
    assert.equal(packed.status, 0, packed.stderr);
    const filename = `custback-${version}.tgz`;
    assert.equal(packed.stdout, `${filename}\n`);

    const tarball = path.join(destination, filename);
    assert.equal(fs.statSync(tarball).isFile(), true);
    const installed = spawnSync('npm', [
      'install', '--prefix', prefix, '--cache', cache, tarball,
    ], {
      cwd: fixture,
      encoding: 'utf8',
      timeout: 120_000,
      env: { ...process.env, CUSTBACK_SKIP_INSTALL: '1' },
    });
    assert.equal(installed.error, undefined, installed.error && installed.error.message);
    assert.equal(installed.status, 0, installed.stderr);
    assert.equal(
      fs.statSync(path.join(prefix, 'node_modules', 'custback', 'package.json')).isFile(),
      true,
    );
  },
);

test(
  'PLATFORM-01: Linux setup rejects unsupported distributions before invoking apt-get',
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
        /^license\s*=\s*["']MIT["']\s*$/m.test(pyproject) &&
        /^license-files\s*=\s*\[\s*["']LICENSE["']\s*\]\s*$/m.test(pyproject),
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
  () => {
    assert.match(read('README.md'), /\(docs\/user-guide\.md\)/);
    const readme = read('docs/user-guide.md');
    const start = readme.indexOf('### Running the avatar service on another host');
    const next = readme.indexOf('\n### ', start + 4);
    const summary = start >= 0 ? readme.slice(start, next >= 0 ? next : undefined) : '';
    const guide = read('docs/remote-deployment.md');
    const requirements = {
      sectionPresent:
        start >= 0 && /\(remote-deployment\.md\)/.test(summary),
      rendererToken:
        /renderer(?:-scoped)?[- ]+(?:credential|token)|renderer[_ -]token/i.test(guide),
      controlToken:
        /avatar[- ]control[- ]+(?:credential|token)|control[_ -]token/i.test(guide),
      explicitlySeparate: /\b(?:separate|distinct)\b/i.test(guide),
      rendererWss: /\bwss:\/\//i.test(guide),
      controlHttps: /\bhttps:\/\//i.test(guide),
      trustAndRotation:
        /source\.tls_ca_file/.test(guide) && /avatar\.tls_ca_file/.test(guide) &&
        /\brotation\b/i.test(guide),
      outageBehavior:
        /privacy slate/i.test(guide) && /avatar_auth_failed/.test(guide) &&
        /avatar_unreachable/.test(guide),
    };

    assert.deepEqual(requirements, {
      sectionPresent: true,
      rendererToken: true,
      controlToken: true,
      explicitlySeparate: true,
      rendererWss: true,
      controlHttps: true,
      trustAndRotation: true,
      outageBehavior: true,
    });
  },
);

test(
  'HYGIENE-01: generated ONNX Runtime profiles are ignored and rejected as stale release artifacts',
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
