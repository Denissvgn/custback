'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const installer = require('../install');

test('interpreter identity probe executes valid Python and reports support', () => {
  const identity = installer.pythonIdentity(
    process.env.CUSTBACK_TEST_PYTHON || 'python3'
  );
  assert.ok(identity, 'python3 must be a supported CPython 3.10-3.14 interpreter');
  assert.equal(identity.major, 3);
  assert.ok(identity.minor >= 10 && identity.minor < 15);
  assert.match(identity.version, /^3\.(?:10|11|12|13|14)\.\d+$/);
  assert.ok(path.isAbsolute(identity.executable));
});

test('installer uses a tested pip range instead of an unbounded upgrade', () => {
  assert.equal(installer.PIP_SPEC, 'pip>=23,<27');
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'install.js'), 'utf8');
  assert.doesNotMatch(source, /'--upgrade',\s*'pip'/);
});

test('installer reaches interrupted-promotion recovery under the install lock', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'install.js'), 'utf8');
  const prepare = source.indexOf('managed.prepareInstallTarget(target)');
  const lock = source.indexOf('managed.withInstallLock(generationRoot');
  const recover = source.indexOf('managed.recoverInterruptedPromotion(target, generationRoot)');
  assert.ok(prepare >= 0, 'installer must use recovery-aware target preparation');
  assert.ok(lock > prepare, 'installer must lock the prepared generations root');
  assert.ok(recover > lock, 'interrupted promotion must recover inside the install lock');
});

function completeStamp(expected, selectedExtras = expected.requestedExtras) {
  return {
    schema: 2,
    ...expected,
    selectedExtras,
    capabilities: {
      mediapipe: selectedExtras.includes('mediapipe'),
      rvm: selectedExtras.includes('rvm') || selectedExtras.includes('gpu'),
      cuda_provider: selectedExtras.includes('gpu'),
    },
    createdAt: '2026-07-14T00:00:00.000Z',
  };
}

test('extras are normalized and invalid/conflicting requests fail', () => {
  assert.deepEqual(installer.parseExtras(' gpu,mediapipe,gpu '), ['gpu', 'mediapipe']);
  assert.throws(() => installer.parseExtras('dev'), /unsupported CUSTBACK_EXTRAS/);
  assert.throws(() => installer.parseExtras('gpu,rvm'), /cannot combine rvm and gpu/);
});

test('only implicit MediaPipe can be removed from fallback attempts', () => {
  assert.deepEqual(installer.installAttempts([]), [['mediapipe'], []]);
  assert.deepEqual(installer.installAttempts(['gpu']), [['gpu', 'mediapipe'], ['gpu']]);
  assert.deepEqual(installer.installAttempts(['rvm']), [['mediapipe', 'rvm'], ['rvm']]);
  assert.deepEqual(installer.installAttempts(['mediapipe']), [['mediapipe']]);
  assert.deepEqual(
    installer.installAttempts(['gpu', 'mediapipe']),
    [['gpu', 'mediapipe']],
  );
});

test('truthful JSON stamp comparison includes source, interpreter, and requests', () => {
  const expected = {
    packageVersion: '0.3.0',
    sourceDigest: `sha256:${'a'.repeat(64)}`,
    python: { executable: '/python', version: '3.12.1', cacheTag: 'cpython-312' },
    requestedExtras: ['gpu'],
  };
  const stamp = completeStamp(expected, ['gpu']);
  assert.equal(installer.stampMatches(stamp, expected), true);
  assert.equal(installer.stampMatches({ ...stamp, selectedExtras: [] }, expected), false);
  assert.equal(installer.stampMatches({ ...stamp, requestedExtras: [] }, expected), false);
  assert.equal(installer.stampMatches({ ...stamp, sourceDigest: 'sha256:other' }, expected), false);
});

test('install stamp extras are canonical, supported, and include every request', () => {
  assert.equal(installer.validExtraSelection(['gpu'], ['gpu']), true);
  assert.equal(installer.validExtraSelection([], ['mediapipe']), true);
  assert.equal(installer.validExtraSelection(['gpu'], ['rvm']), false);
  assert.equal(installer.validExtraSelection([], ['unknown']), false);
  assert.equal(installer.validExtraSelection([], ['gpu', 'rvm']), false);
  assert.equal(installer.validExtraSelection([], ['mediapipe', 'gpu']), false);
  const expected = {
    packageVersion: '0.3.0',
    sourceDigest: `sha256:${'c'.repeat(64)}`,
    python: { executable: '/python', version: '3.12.1', cacheTag: 'cpython-312' },
    requestedExtras: ['gpu'],
  };
  const stamp = completeStamp(expected, ['gpu']);
  assert.equal(installer.validInstallStamp(stamp), true);
  assert.equal(installer.validInstallStamp({
    ...stamp,
    capabilities: { ...stamp.capabilities, cuda_provider: false },
  }), false);
});

test('stamp reuse requires a healthy environment and force always rebuilds', () => {
  const expected = {
    packageVersion: '0.3.0',
    sourceDigest: `sha256:${'b'.repeat(64)}`,
    python: { executable: '/python', version: '3.12.1', cacheTag: 'cpython-312' },
    requestedExtras: ['rvm'],
  };
  const stamp = completeStamp(expected, ['rvm']);
  const base = {
    force: false,
    inspectionKind: 'symlink',
    appExists: true,
    stamp,
    expected,
  };
  assert.equal(installer.reusableEnvironment({ ...base, validate() {} }), true);
  assert.equal(installer.reusableEnvironment({
    ...base,
    validate() { throw new Error('pip check failed'); },
  }), false);
  assert.equal(installer.reusableEnvironment({ ...base, force: true, validate() {} }), false);
});

test('corrupt active install metadata can be treated as stale without hiding strict reads', (t) => {
  const target = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-stamp-test-'));
  t.after(() => fs.rmSync(target, { recursive: true, force: true }));
  fs.writeFileSync(path.join(target, '.custback-install.json'), '{invalid');
  let observed = null;
  assert.equal(installer.readActiveStamp(target, (err) => { observed = err; }), null);
  assert.match(observed.message, /invalid custback metadata/);
  assert.throws(() => installer.readActiveStamp(target), /invalid custback metadata/);
});

test('launcher rebuild delegates to installer and never recursively deletes VENV_DIR', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'custback.js'), 'utf8');
  assert.match(source, /bootstrap\(true\)/);
  assert.doesNotMatch(source, /rmSync\s*\(\s*VENV_DIR/);
  assert.doesNotMatch(source, /rmSync\s*\(\s*targetPath/);
});

test('doctor keeps optional capabilities and machine setup non-fatal', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'custback.js'), 'utf8');
  assert.match(source, /note\('MediaPipe segmentation not installed'/);
  assert.match(source, /warn\('custback virtual camera device not found'/);
  assert.doesNotMatch(source, /report\('MediaPipe segmentation not installed'/);
  assert.match(source, /requested\.includes\('mediapipe'\)/);
  assert.match(source, /requested\.includes\('gpu'\)/);
  assert.match(source, /installer\.validInstallStamp\(stamp\)/);
  assert.match(source, /stamp\.sourceDigest === installer\.sourceDigest\(\)/);
  assert.match(source, /timeout: PROBE_TIMEOUT_MS/);
});

test('release smoke reuses one managed target, rebuilds, and bounds subprocesses', () => {
  const source = fs.readFileSync(
    path.resolve(__dirname, '..', '..', '..', 'scripts', 'release', 'verify-release.js'),
    'utf8',
  );
  assert.match(source, /CUSTBACK_VENV: managedVenv/);
  assert.match(source, /runChecked\(launcher, \['rebuild'\], \{ env: smokeEnv \}\)/);
  assert.match(source, /timeout: COMMAND_TIMEOUT_MS/);
  const installerSource = fs.readFileSync(path.resolve(__dirname, '..', 'install.js'), 'utf8');
  assert.match(installerSource, /timeout: INSTALL_TIMEOUT_MS/);
});
