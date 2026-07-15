'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const release = require('../../../scripts/release/verify-release');

const root = path.resolve(__dirname, '..', '..', '..');

test('release metadata versions and required compatibility bounds agree', () => {
  assert.equal(release.verifyVersions(root), '0.4.0');
  assert.doesNotThrow(() => release.verifyNpmMetadata(root));
  assert.doesNotThrow(() => release.verifyDependencies(root));
  assert.doesNotThrow(() => release.verifyDocs(root));
});

test('remediation registry tracks resolved work and keeps the release frozen', () => {
  const blockers = release.remediationBlockers(root);
  assert.equal(blockers.length, 26);
  assert.deepEqual(
    blockers.filter((entry) => entry.status === 'resolved').map((entry) => entry.id),
    ['SEC-01', 'TOKEN-01', 'TRANS-01', 'PRIV-01', 'SEG-03'],
  );
  assert.equal(blockers.filter((entry) => entry.status === 'open').length, 21);
  assert.doesNotThrow(() => release.verifyBlockerRegressionCoverage(root));
  assert.throws(
    () => release.verifyNoReleaseBlockers(root),
    /release blocked by 21 open remediation blocker.*A2F-01.*HYGIENE-01/,
  );
});

test('remediation registry fails closed on missing, malformed, or inconsistent state', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-blocker-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const directory = path.join(fixture, 'scripts', 'release');
  fs.mkdirSync(directory, { recursive: true });

  assert.throws(
    () => release.remediationBlockers(fixture),
    /registry is missing/,
  );

  const registryPath = path.join(directory, 'remediation-blockers.json');
  fs.writeFileSync(registryPath, '{bad');
  assert.throws(
    () => release.remediationBlockers(fixture),
    /invalid JSON/,
  );

  fs.writeFileSync(registryPath, JSON.stringify({
    schema_version: 1,
    release_blocked: false,
    blockers: [
      {
        id: 'SEC-01', phase: 1, status: 'open', title: 'still open',
        regression: 'tests/security.test.js',
      },
    ],
  }));
  assert.throws(
    () => release.remediationBlockers(fixture),
    /release_blocked must be true/,
  );

  fs.writeFileSync(registryPath, JSON.stringify({
    schema_version: 1,
    release_blocked: true,
    blockers: [
      {
        id: 'SEC-01', phase: 1, status: 'open', title: 'first',
        regression: 'tests/security.test.js',
      },
      {
        id: 'SEC-01', phase: 1, status: 'resolved', title: 'duplicate',
        regression: 'tests/security.test.js',
      },
    ],
  }));
  assert.throws(
    () => release.remediationBlockers(fixture),
    /duplicate id SEC-01/,
  );
});

test('release version parity rejects a mismatched lock root', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-version-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'src', 'custback'), { recursive: true });
  fs.writeFileSync(path.join(fixture, 'package.json'), JSON.stringify({ version: '0.3.0' }));
  fs.writeFileSync(path.join(fixture, 'package-lock.json'), JSON.stringify({
    version: '0.3.0',
    packages: { '': { version: '0.2.0' } },
  }));
  fs.writeFileSync(path.join(fixture, 'pyproject.toml'), '[project]\nversion = "0.3.0"\n');
  fs.writeFileSync(path.join(fixture, 'src', 'custback', '__init__.py'), '__version__ = "0.3.0"\n');
  assert.throws(() => release.verifyVersions(fixture), /release version mismatch/);
});

test('source fallback verification ignores a matching comment decoy', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-fallback-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'src', 'custback'), { recursive: true });
  fs.copyFileSync(path.join(root, 'package.json'), path.join(fixture, 'package.json'));
  fs.copyFileSync(path.join(root, 'package-lock.json'), path.join(fixture, 'package-lock.json'));
  // Match the real package version so only the fallback line disagrees.
  const packageVersion = JSON.parse(
    fs.readFileSync(path.join(root, 'package.json'), 'utf8'),
  ).version;
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    `[project]\nversion = "${packageVersion}"\n`,
  );
  fs.writeFileSync(path.join(fixture, 'src', 'custback', '__init__.py'), `
from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("custback")
except PackageNotFoundError:
    # __version__ = "${packageVersion}"
    __version__ = "9.9.9"
`);
  assert.throws(
    () => release.verifyVersions(fixture),
    /source-tree __version__ fallback does not match package version/,
  );
});

test('npm metadata and lock root are exact reviewed allowlists', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-npm-metadata-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const packagePath = path.join(fixture, 'package.json');
  const lockPath = path.join(fixture, 'package-lock.json');
  const originalPackage = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const originalLock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));

  fs.writeFileSync(packagePath, JSON.stringify({
    ...originalPackage,
    scripts: { ...originalPackage.scripts, preinstall: 'node unexpected.js' },
  }));
  fs.writeFileSync(lockPath, JSON.stringify(originalLock));
  assert.throws(
    () => release.verifyNpmMetadata(fixture),
    /package.json metadata must exactly match/,
  );

  fs.writeFileSync(packagePath, JSON.stringify(originalPackage));
  fs.writeFileSync(lockPath, JSON.stringify({
    ...originalLock,
    packages: {
      '': { ...originalLock.packages[''], hasInstallScript: false },
    },
  }));
  assert.throws(
    () => release.verifyNpmMetadata(fixture),
    /package-lock.json must exactly mirror/,
  );
});

test('release and install probes remain active with Python optimization', () => {
  const files = [
    path.join(root, 'scripts', 'release', 'verify-release.js'),
    path.join(root, 'packaging', 'npm', 'install.js'),
    path.join(root, 'packaging', 'npm', 'custback.js'),
  ];
  for (const file of files) {
    assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /\bassert\s+/);
  }
  const previous = process.env.PYTHONOPTIMIZE;
  process.env.PYTHONOPTIMIZE = '2';
  try {
    assert.equal(release.verifyVersions(root), '0.4.0');
  } finally {
    if (previous === undefined) delete process.env.PYTHONOPTIMIZE;
    else process.env.PYTHONOPTIMIZE = previous;
  }
});

test('npm pack payload parsing rejects ambiguous or unsafe manifests', () => {
  const version = '0.3.0';
  const artifact = {
    name: 'custback',
    version,
    filename: `custback-${version}.tgz`,
    files: [{ path: 'package.json' }, { path: 'packaging/npm/custback.js' }],
  };
  assert.deepEqual(
    release.parseNpmPackPayload(JSON.stringify([artifact]), version).names,
    ['package.json', 'packaging/npm/custback.js'],
  );
  assert.throws(() => release.parseNpmPackPayload('{bad', version), /invalid JSON/);
  assert.throws(() => release.parseNpmPackPayload(JSON.stringify([]), version), /unexpected payload/);
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{ ...artifact, version: '9.9.9' }]), version),
    /identity is inconsistent/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{ ...artifact, files: [] }]), version),
    /no file manifest/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{
      ...artifact, files: [{ path: '../outside' }],
    }]), version),
    /invalid file entry/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{
      ...artifact, files: [{ path: 'package.json' }, { path: 'package.json' }],
    }]), version),
    /duplicate file entries/,
  );
});

test('dependency verification reads runtime tables instead of matching stray text', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-dependency-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const spec = 'pyvirtualcam>=0.11,<1';
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    `${original.replace(`    "${spec}",\n`, '')}\n# "${spec}"\n`,
  );
  assert.throws(() => release.verifyDependencies(fixture), /missing bounded core dependency/);
});

test('dependency verification enforces the complete bounded dev test set', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-dev-dependency-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace('    "httpx2>=2,<3",\n', ''),
  );
  assert.throws(() => release.verifyDependencies(fixture), /invalid bounded dev dependencies/);
});

test('dependency verification rejects unreviewed extras and console scripts', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-extra-gate-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace(
      '[project.scripts]',
      'unreviewed = ["totally-unbounded"]\n\n[project.scripts]',
    ),
  );
  assert.throws(
    () => release.verifyDependencies(fixture),
    /optional dependency groups must be exactly/,
  );

  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace(
      'custback = "custback.__main__:main"',
      'custback = "custback.__main__:main"\nunreviewed = "evil:main"',
    ),
  );
  assert.throws(
    () => release.verifyDependencies(fixture),
    /console scripts must be exactly/,
  );
});

test('stale artifact reporting is deterministic and non-mutating', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-release-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  for (const name of ['custback-0.1.0.tgz', 'debug.txt', 'uninstall.log', 'keep.txt']) {
    fs.writeFileSync(path.join(fixture, name), name);
  }
  fs.mkdirSync(path.join(fixture, 'build'));
  fs.mkdirSync(path.join(fixture, 'src'));
  fs.mkdirSync(path.join(fixture, 'src', 'custback.egg-info'));
  assert.deepEqual(
    release.staleArtifacts(fixture).sort(),
    ['build', 'custback-0.1.0.tgz', 'debug.txt', 'src/custback.egg-info', 'uninstall.log'],
  );
  assert.equal(fs.readFileSync(path.join(fixture, 'keep.txt'), 'utf8'), 'keep.txt');
});
