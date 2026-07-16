'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const zlib = require('node:zlib');

const release = require('../../../scripts/release/verify-release');

const root = path.resolve(__dirname, '..', '..', '..');

function tarballWithFile(name, contents) {
  const data = Buffer.from(contents);
  const header = Buffer.alloc(512);
  header.write(name, 0, 100, 'utf8');
  header.write('0000644\0', 100, 8, 'ascii');
  header.write('0000000\0', 108, 8, 'ascii');
  header.write('0000000\0', 116, 8, 'ascii');
  header.write(`${data.length.toString(8).padStart(11, '0')}\0`, 124, 12, 'ascii');
  header[156] = '0'.charCodeAt(0);
  const padding = Buffer.alloc(Math.ceil(data.length / 512) * 512 - data.length);
  return zlib.gzipSync(Buffer.concat([header, data, padding, Buffer.alloc(1024)]));
}

test('release metadata versions and required compatibility bounds agree', () => {
  assert.equal(release.verifyVersions(root), '0.4.0');
  assert.doesNotThrow(() => release.verifyNpmMetadata(root));
  assert.doesNotThrow(() => release.verifyDependencies(root));
  assert.doesNotThrow(() => release.verifyLicenseMetadata(root));
  assert.doesNotThrow(() => release.verifyDocs(root));
  assert.doesNotThrow(() => release.verifyCiWorkflow(root));
  assert.doesNotThrow(() => release.verifyPlatformScope(root));
});

test('prepack verifies the non-recursive packlist and reserves artifact installs', () => {
  assert.deepEqual(release.releasePlan(['--prepack']), {
    builtArtifacts: false,
    success: 'metadata and non-recursive npm packlist verified',
  });
  assert.equal(release.releasePlan([]).builtArtifacts, true);
  assert.equal(release.releasePlan(['--quick']).builtArtifacts, false);
  assert.throws(() => release.releasePlan(['--prepack', '--quick']), /invalid release/);
  assert.throws(() => release.releasePlan(['--unknown']), /invalid release/);

  const source = fs.readFileSync(
    path.join(root, 'scripts', 'release', 'verify-release.js'), 'utf8',
  );
  assert.match(
    source,
    /\['pack', '--dry-run', '--json', '--ignore-scripts'/,
    'packlist verification must suppress lifecycle recursion',
  );
});

test('full release temporary roots reject Linux memory-backed filesystems', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-release-base-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const baseEnv = { CUSTBACK_RELEASE_TMPDIR: fixture };
  const tmpfs = () => ({ type: 0x01021994 });
  const ramfs = () => ({ type: 0x858458f6 });
  const disk = () => ({ type: 0xef53 });

  assert.equal(release.filesystemIsMemoryBacked(fixture, tmpfs, 'linux'), true);
  assert.equal(release.filesystemIsMemoryBacked(fixture, ramfs, 'linux'), true);
  assert.equal(release.filesystemIsMemoryBacked(fixture, tmpfs, 'darwin'), false);
  assert.throws(
    () => release.filesystemIsMemoryBacked(fixture, null, 'linux'),
    /cannot determine whether release temporary storage is memory-backed/,
  );
  assert.throws(
    () => release.createReleaseTemporaryRoot({ env: baseEnv, statfs: tmpfs, platform: 'linux' }),
    /memory-backed filesystem/,
  );
  assert.throws(
    () => release.createReleaseTemporaryRoot({ env: baseEnv, statfs: ramfs, platform: 'linux' }),
    /memory-backed filesystem/,
  );

  const allowed = release.createReleaseTemporaryRoot({
    env: { ...baseEnv, CUSTBACK_RELEASE_ALLOW_TMPFS: '1' },
    statfs: tmpfs,
    platform: 'linux',
  });
  assert.equal(path.dirname(allowed), fs.realpathSync(fixture));
  assert.equal(fs.statSync(allowed).isDirectory(), true);
  fs.rmSync(allowed, { recursive: true, force: true });

  const diskRoot = release.createReleaseTemporaryRoot({
    env: baseEnv,
    statfs: disk,
    platform: 'linux',
  });
  assert.equal(fs.statSync(diskRoot).isDirectory(), true);
  fs.rmSync(diskRoot, { recursive: true, force: true });

  const checkout = fs.mkdtempSync(path.join(fixture, 'checkout-'));
  const nestedBase = path.join(checkout, 'release-scratch');
  fs.mkdirSync(nestedBase);
  assert.throws(
    () => release.createReleaseTemporaryRoot({
      env: { CUSTBACK_RELEASE_TMPDIR: nestedBase },
      root: checkout,
      statfs: disk,
      platform: 'linux',
    }),
    /outside the source checkout/,
  );
});

test('release profile directories are disposed after success and failure', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-disposable-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));

  const successful = path.join(fixture, 'successful');
  assert.equal(release.withDisposableDirectory(successful, (directory) => {
    fs.mkdirSync(directory);
    fs.writeFileSync(path.join(directory, 'payload'), 'large environment');
    return 42;
  }), 42);
  assert.equal(fs.existsSync(successful), false);

  const failed = path.join(fixture, 'failed');
  assert.throws(() => release.withDisposableDirectory(failed, (directory) => {
    fs.mkdirSync(directory);
    fs.writeFileSync(path.join(directory, 'payload'), 'partial environment');
    throw new Error('profile failed');
  }), /profile failed/);
  assert.equal(fs.existsSync(failed), false);

  const existing = path.join(fixture, 'existing');
  fs.mkdirSync(existing);
  assert.throws(
    () => release.withDisposableDirectory(existing, () => undefined),
    /refusing to reuse/,
  );
  assert.equal(fs.existsSync(existing), true);
});

test('release scratch environment is relocated and restored after failures', () => {
  const env = { TMPDIR: '/old/tmpdir', TEMP: '/old/temp', KEEP: 'unchanged' };
  const before = { ...env };
  assert.throws(() => release.withTemporaryEnvironment({
    TMPDIR: '/disk/release',
    TMP: '/disk/release',
    TEMP: '/disk/release',
  }, () => {
    assert.equal(env.TMPDIR, '/disk/release');
    assert.equal(env.TMP, '/disk/release');
    assert.equal(env.TEMP, '/disk/release');
    throw new Error('artifact failure');
  }, env), /artifact failure/);
  assert.deepEqual(env, before);
});

test('release native build parallelism is memory-bounded by default', () => {
  assert.equal(release.releaseBuildJobs({}), '2');
  assert.equal(release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: '1' }), '1');
  assert.equal(release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: '32' }), '32');
  for (const invalid of ['', '0', '-1', '2.5', '33', 'many']) {
    assert.throws(
      () => release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: invalid }),
      /integer from 1 through 32/,
    );
  }
});

test('advertised built-wheel extra profiles cover every platform-feasible group', () => {
  assert.deepEqual(
    release.extraArtifactProfiles('linux', 'x64').map((profile) => profile.name),
    ['mediapipe', 'rvm', 'gpu', 'audio2face', 'dev'],
  );
  assert.deepEqual(
    release.extraArtifactProfiles('darwin', 'x64').map((profile) => profile.name),
    ['mediapipe', 'rvm', 'audio2face', 'dev'],
  );
  const audio2face = release.extraArtifactProfiles('linux', 'x64')
    .find((profile) => profile.name === 'audio2face');
  assert.deepEqual(audio2face.extras, ['audio2face', 'dev']);
  assert.equal(audio2face.test, 'tests/test_audio2face_protocol.py');
});

test('npm artifact LICENSE must be byte-identical to the canonical source', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-license-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const canonical = release.canonicalLicense(root);
  const matching = path.join(fixture, 'matching.tgz');
  const changed = path.join(fixture, 'changed.tgz');
  fs.writeFileSync(matching, tarballWithFile('package/LICENSE', canonical));
  fs.writeFileSync(changed, tarballWithFile('package/LICENSE', `${canonical}changed\n`));
  assert.doesNotThrow(() => release.verifyNpmArtifactLicense(matching, root));
  assert.throws(
    () => release.verifyNpmArtifactLicense(changed, root),
    /not byte-identical/,
  );
});

test('remediation registry records all 26 blockers as resolved', () => {
  const blockers = release.remediationBlockers(root);
  assert.equal(blockers.length, 26);
  assert.deepEqual(
    blockers.filter((entry) => entry.status === 'resolved').map((entry) => entry.id),
    [
      'SEC-01', 'TOKEN-01', 'TRANS-01', 'PRIV-01',
      'A2F-01',
      'CFG-01', 'CFG-02', 'LIFE-01', 'LIFE-02',
      'STOR-01', 'STOR-02', 'SEG-01', 'SEG-02', 'SEG-03',
      'RENDER-01', 'RENDER-02', 'API-01',
      'NPM-01', 'PKG-01', 'PKG-02', 'DEPLOY-01',
      'MISC-01', 'MISC-02',
      'LICENSE-01', 'PLATFORM-01', 'HYGIENE-01',
    ],
  );
  assert.deepEqual(blockers.filter((entry) => entry.status === 'open'), []);
  assert.doesNotThrow(() => release.verifyBlockerRegressionCoverage(root));
  assert.doesNotThrow(() => release.verifyNoReleaseBlockers(root));
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
  for (const name of [
    'custback-0.1.0.tgz',
    'debug.txt',
    'onnxruntime_profile__2026-07-16_12-34-56.json',
    'uninstall.log',
    'keep.txt',
  ]) {
    fs.writeFileSync(path.join(fixture, name), name);
  }
  fs.mkdirSync(path.join(fixture, 'build'));
  fs.mkdirSync(path.join(fixture, 'src'));
  fs.mkdirSync(path.join(fixture, 'src', 'custback.egg-info'));
  assert.deepEqual(
    release.staleArtifacts(fixture).sort(),
    [
      'build',
      'custback-0.1.0.tgz',
      'debug.txt',
      'onnxruntime_profile__2026-07-16_12-34-56.json',
      'src/custback.egg-info',
      'uninstall.log',
    ],
  );
  assert.equal(fs.readFileSync(path.join(fixture, 'keep.txt'), 'utf8'), 'keep.txt');
});
