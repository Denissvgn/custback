'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const installer = require('../install');
const launcher = require('../custback');
const managed = require('../managed-venv');

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
    schema: installer.INSTALL_STAMP_SCHEMA,
    ...expected,
    selectedExtras,
    capabilities: {
      mediapipe: selectedExtras.includes('mediapipe'),
      audio2face: selectedExtras.includes('audio2face'),
      rvm: selectedExtras.includes('rvm') || selectedExtras.includes('gpu'),
      cuda_provider: selectedExtras.includes('gpu'),
      cuda_inference: selectedExtras.includes('gpu'),
    },
    createdAt: '2026-07-14T00:00:00.000Z',
  };
}

test('extras are normalized and invalid/conflicting requests fail', () => {
  assert.deepEqual(installer.parseExtras(' audio2face,gpu,audio2face '), ['audio2face', 'gpu']);
  assert.throws(() => installer.parseExtras('dev'), /unsupported CUSTBACK_EXTRAS/);
  assert.throws(() => installer.parseExtras('gpu,rvm'), /cannot combine rvm and gpu/);
  assert.throws(
    () => installer.parseExtras('audio2face,mediapipe'),
    /conflicting protobuf requirements/,
  );
});

test('only implicit MediaPipe can be removed from fallback attempts', () => {
  assert.deepEqual(installer.installAttempts([]), [['mediapipe'], []]);
  assert.deepEqual(installer.installAttempts(['gpu']), [['gpu', 'mediapipe'], ['gpu']]);
  assert.deepEqual(installer.installAttempts(['rvm']), [['mediapipe', 'rvm'], ['rvm']]);
  assert.deepEqual(installer.installAttempts(['mediapipe']), [['mediapipe']]);
  assert.deepEqual(installer.installAttempts(['audio2face']), [['audio2face']]);
  assert.deepEqual(
    installer.installAttempts(['audio2face', 'gpu']),
    [['audio2face', 'gpu']],
  );
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
  assert.equal(installer.validExtraSelection(['audio2face'], ['audio2face']), true);
  assert.equal(installer.validExtraSelection([], ['audio2face', 'mediapipe']), false);
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
  assert.equal(installer.validInstallStamp({
    ...stamp,
    capabilities: { ...stamp.capabilities, cuda_inference: false },
  }), false);
  assert.equal(installer.validInstallStamp({ ...stamp, schema: 3 }), false);
});

test('CUDA capability parser fails closed on malformed or CPU-fallback evidence', () => {
  const verified = {
    schema: 1,
    onnxruntime: true,
    cuda_provider: true,
    cuda_inference: true,
    active_providers: ['CUDAExecutionProvider'],
    output_verified: true,
    profile_verified: true,
    error: '',
  };
  assert.deepEqual(installer.parseCudaProbeOutput(JSON.stringify(verified)), verified);
  assert.equal(installer.parseCudaProbeOutput('{bad').cuda_inference, false);
  assert.equal(installer.parseCudaProbeOutput(JSON.stringify({
    ...verified,
    active_providers: [123],
  })).cuda_inference, false);
  assert.equal(installer.parseCudaProbeOutput(JSON.stringify({
    ...verified,
    active_providers: ['CPUExecutionProvider'],
  })).cuda_inference, false);
  const cpuFallback = installer.parseCudaProbeOutput(JSON.stringify({
    ...verified,
    cuda_inference: false,
    active_providers: ['CPUExecutionProvider'],
    profile_verified: false,
    error: 'probe did not execute on CUDAExecutionProvider',
  }));
  assert.equal(cpuFallback.cuda_provider, true);
  assert.equal(cpuFallback.cuda_inference, false);
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

test('default target, generations, and extras intent survive package-directory replacement', (t) => {
  const prefix = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-prefix-test-'));
  t.after(() => fs.rmSync(prefix, { recursive: true, force: true }));
  const packageRoot = path.join(prefix, 'lib', 'node_modules', 'custback');
  fs.mkdirSync(packageRoot, { recursive: true });
  const target = managed.defaultTargetForPackage(packageRoot);
  assert.equal(target, path.join(prefix, managed.DEFAULT_TARGET_NAME));
  assert.equal(managed.isWithin(packageRoot, target), false);

  const generations = managed.ensureGenerationsRoot(target);
  const fakeGeneration = () => {
    const generation = managed.createGeneration(generations);
    fs.mkdirSync(path.join(generation, 'bin'));
    fs.writeFileSync(path.join(generation, 'pyvenv.cfg'), 'home = /python\n');
    fs.writeFileSync(path.join(generation, 'bin', 'python'), '#!/bin/sh\n');
    fs.writeFileSync(path.join(generation, 'bin', 'custback'), '#!/bin/sh\n');
    managed.markGeneration(generation, target);
    return generation;
  };
  const rollback = fakeGeneration();
  managed.promoteGeneration({
    target,
    generation: rollback,
    inspection: managed.inspectTarget(target),
    validateActive() {},
  });
  const active = fakeGeneration();
  const promotion = managed.promoteGeneration({
    target,
    generation: active,
    inspection: managed.inspectTarget(target),
    validateActive() {},
  });
  assert.equal(path.resolve(promotion.previousGeneration), path.resolve(rollback));
  const activeIdentity = fs.statSync(active);
  const rollbackIdentity = fs.statSync(rollback);
  installer.writeInstallIntent(target, ['audio2face', 'gpu']);
  fs.rmSync(packageRoot, { recursive: true });
  fs.mkdirSync(packageRoot, { recursive: true });

  assert.equal(managed.defaultTargetForPackage(packageRoot), target);
  assert.equal(fs.existsSync(generations), true);
  assert.equal(fs.realpathSync(target), fs.realpathSync(active));
  assert.equal(fs.statSync(active).ino, activeIdentity.ino);
  assert.equal(fs.statSync(active).dev, activeIdentity.dev);
  assert.equal(fs.statSync(rollback).ino, rollbackIdentity.ino);
  assert.equal(fs.statSync(rollback).dev, rollbackIdentity.dev);
  managed.validateGeneration(rollback, target, generations);
  assert.deepEqual(
    installer.readInstallIntent(target).requestedExtras,
    ['audio2face', 'gpu'],
  );
});

test('npm prefix inference handles local, global, scoped, and checkout layouts', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-prefix-layout-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  assert.equal(
    managed.npmPrefixForPackage(path.join(root, 'node_modules', 'custback')),
    root,
  );
  assert.equal(
    managed.npmPrefixForPackage(path.join(root, 'lib', 'node_modules', 'custback')),
    root,
  );
  assert.equal(
    managed.npmPrefixForPackage(path.join(root, 'node_modules', '@scope', 'custback')),
    root,
  );
  assert.equal(
    managed.npmPrefixForPackage(path.join(root, 'checkout')),
    root,
  );
});

test('extras intent distinguishes absent and empty input and is private metadata', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-intent-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const target = path.join(root, 'venv');
  installer.writeInstallIntent(target, ['gpu']);
  const intent = installer.readInstallIntent(target);
  assert.deepEqual(installer.resolveRequestedExtras({}, intent), ['gpu']);
  assert.deepEqual(installer.resolveRequestedExtras({ CUSTBACK_EXTRAS: '' }, intent), []);
  assert.deepEqual(
    installer.resolveRequestedExtras({ CUSTBACK_EXTRAS: ' audio2face,gpu ' }, intent),
    ['audio2face', 'gpu'],
  );
  if (process.platform !== 'win32') {
    assert.equal(fs.statSync(installer.installIntentPath(target)).mode & 0o777, 0o600);
  }
});

test('a newer promoted stamp wins over stale intent after intent publication failure', () => {
  const expected = {
    packageVersion: '0.4.0',
    sourceDigest: `sha256:${'d'.repeat(64)}`,
    python: { executable: '/python', version: '3.12.1', cacheTag: 'cpython-312' },
    requestedExtras: [],
  };
  const active = {
    ...completeStamp(expected, []),
    createdAt: '2026-07-16T12:00:01.000Z',
  };
  const staleIntent = {
    owner: managed.OWNER,
    schema: installer.INSTALL_INTENT_SCHEMA,
    logicalTarget: '/managed/venv',
    requestedExtras: ['gpu'],
    updatedAt: '2026-07-16T12:00:00.000Z',
  };
  assert.deepEqual(installer.resolveRequestedExtras({}, staleIntent, active), []);
  assert.deepEqual(
    installer.resolveRequestedExtras({}, { ...staleIntent, updatedAt: '2026-07-16T12:00:02.000Z' }, active),
    ['gpu'],
  );
});

test('legacy package-local environments are read for intent but never relocated', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-legacy-read-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const legacy = path.join(root, '.venv');
  fs.mkdirSync(path.join(legacy, 'bin'), { recursive: true });
  fs.writeFileSync(path.join(legacy, 'pyvenv.cfg'), 'home = /python\n');
  fs.writeFileSync(path.join(legacy, 'bin', 'python'), '#!/bin/sh\n');
  fs.writeFileSync(path.join(legacy, 'bin', 'custback'), '#!/bin/sh\n');
  fs.writeFileSync(path.join(legacy, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 [gpu]');
  assert.deepEqual(installer.readLegacyRequestedExtras(legacy), ['gpu']);
  assert.equal(fs.lstatSync(legacy).isDirectory(), true);
  assert.equal(fs.existsSync(path.join(legacy, 'bin', 'custback')), true);
});

test('launcher exposes durable extras flags, avatar alias dispatch, and config export', (t) => {
  assert.deepEqual(launcher.parseRebuildArgs([]), { extras: undefined });
  assert.deepEqual(launcher.parseRebuildArgs(['--extras', 'gpu,audio2face']), {
    extras: 'audio2face,gpu',
  });
  assert.deepEqual(launcher.parseRebuildArgs(['--extras=']), { extras: '' });
  assert.throws(() => launcher.parseRebuildArgs(['--extras']), /usage/);
  assert.equal(launcher.invokedAsAvatar('/prefix/bin/custback-avatar'), true);
  assert.equal(launcher.invokedAsAvatar('/prefix/bin/custback'), false);
  assert.deepEqual(launcher.avatarConfigExportArgs(['config', 'export', 'out.yaml']), ['out.yaml']);

  let stdout = Buffer.alloc(0);
  launcher.exportAvatarConfig([], { write(chunk) { stdout = Buffer.concat([stdout, chunk]); } });
  assert.deepEqual(stdout, fs.readFileSync(launcher.AVATAR_CONFIG));
  const destination = path.join(
    fs.mkdtempSync(path.join(os.tmpdir(), 'custback-config-export-')),
    'avatar.yaml',
  );
  t.after(() => fs.rmSync(path.dirname(destination), { recursive: true, force: true }));
  assert.equal(launcher.exportAvatarConfig([destination]), 0);
  assert.deepEqual(fs.readFileSync(destination), fs.readFileSync(launcher.AVATAR_CONFIG));
  assert.equal(fs.statSync(destination).mode & 0o777, 0o600);
  assert.throws(() => launcher.exportAvatarConfig([destination]), /EEXIST/);
});

test('launcher rebuild delegates to installer and never recursively deletes VENV_DIR', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'custback.js'), 'utf8');
  assert.match(source, /bootstrap\(true,\s*parsed\.extras\)/);
  assert.doesNotMatch(source, /rmSync\s*\(\s*VENV_DIR/);
  assert.doesNotMatch(source, /rmSync\s*\(\s*targetPath/);
});

test('doctor explains installed backend quality tiers with actionable rebuild commands', () => {
  assert.deepEqual(launcher.backendQualityProfile({ mediapipe: true, rvm: true }), {
    level: 'note',
    label: 'backend quality tier: matting',
    hint: 'RVM true-alpha matting is installed; confirm the active provider in runtime status',
  });

  const defaultProfile = launcher.backendQualityProfile({ mediapipe: true, rvm: false });
  assert.equal(defaultProfile.level, 'note');
  assert.equal(defaultProfile.label, 'backend quality tier: segmentation');
  assert.match(defaultProfile.hint, /MediaPipe confidence-mask segmentation is installed/);
  assert.match(defaultProfile.hint, /custback rebuild --extras rvm \(CPU\)/);
  assert.match(defaultProfile.hint, /custback rebuild --extras gpu \(NVIDIA\/CUDA\)/);

  const coreProfile = launcher.backendQualityProfile({ mediapipe: false, rvm: false });
  assert.equal(coreProfile.level, 'warn');
  assert.equal(coreProfile.label, 'backend quality tier: heuristic');
  assert.match(coreProfile.hint, /custback rebuild --extras mediapipe/);
  assert.match(coreProfile.hint, /custback rebuild --extras rvm \(CPU\)/);
  assert.match(coreProfile.hint, /custback rebuild --extras gpu \(NVIDIA\/CUDA\)/);
});

test('doctor keeps optional capabilities and machine setup non-fatal', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '..', 'custback.js'), 'utf8');
  assert.match(source, /note\('MediaPipe confidence-mask segmentation tier not installed'/);
  assert.match(source, /note\('RVM true-alpha matting tier not installed'/);
  assert.match(source, /warn\('custback virtual camera device not found'/);
  assert.doesNotMatch(
    source,
    /report\('MediaPipe confidence-mask segmentation tier not installed'/,
  );
  assert.match(source, /requested\.includes\('mediapipe'\)/);
  assert.match(source, /requested\.includes\('gpu'\)/);
  assert.match(source, /installer\.cudaProbe\(python\)/);
  assert.match(source, /verified CUDA inference/);
  assert.match(source, /installer\.validInstallStamp\(stamp\)/);
  assert.match(source, /stamp\.sourceDigest === installer\.sourceDigest\(\)/);
  assert.match(source, /timeout: PROBE_TIMEOUT_MS/);
  const installerSource = fs.readFileSync(path.resolve(__dirname, '..', 'install.js'), 'utf8');
  assert.match(installerSource, /custback\.gpu_probe/);
  assert.match(installerSource, /cuda_inference/);
  assert.match(installerSource, /timeout: CUDA_PROBE_TIMEOUT_MS/);
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
