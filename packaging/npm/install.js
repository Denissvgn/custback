#!/usr/bin/env node
/** Build and atomically promote custback's private Python environment. */

'use strict';

const { spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const managed = require('./managed-venv');

const PKG_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_VENV = managed.defaultTargetForPackage(PKG_ROOT);
const LEGACY_DEFAULT_VENV = path.resolve(PKG_ROOT, '.venv');
const PYTHON_CANDIDATES = [
  'python3.12', 'python3.11', 'python3.10',
  'python3.13', 'python3.14', 'python3', 'python',
];
const ALLOWED_EXTRAS = new Set(['audio2face', 'mediapipe', 'rvm', 'gpu']);
const INSTALL_STAMP_SCHEMA = 4;
const INSTALL_INTENT_SCHEMA = 1;
const PIP_SPEC = 'pip>=23,<27';
const configuredTimeout = Number(process.env.CUSTBACK_INSTALL_TIMEOUT_MS);
const INSTALL_TIMEOUT_MS = Number.isSafeInteger(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 15 * 60 * 1000;
const configuredCudaProbeTimeout = Number(process.env.CUSTBACK_CUDA_PROBE_TIMEOUT_MS);
const CUDA_PROBE_TIMEOUT_MS = Number.isSafeInteger(configuredCudaProbeTimeout) &&
  configuredCudaProbeTimeout > 0 ? configuredCudaProbeTimeout : 60 * 1000;

function spawn(command, args, options = {}) {
  return spawnSync(command, args, { timeout: INSTALL_TIMEOUT_MS, ...options });
}

function log(message) {
  console.log(`[custback install] ${message}`);
}

function run(command, args, options = {}) {
  return spawn(command, args, { stdio: 'inherit', ...options });
}

function pythonIdentity(binary) {
  const code = [
    'import json, os, ssl, sys, sysconfig, venv',
    'print(json.dumps({',
    '  "executable": os.path.realpath(sys.executable),',
    '  "version": ".".join(map(str, sys.version_info[:3])),',
    '  "major": sys.version_info[0], "minor": sys.version_info[1],',
    '  "cache_tag": sys.implementation.cache_tag,',
    '}))',
  ].join('\n');
  const result = spawn(binary, ['-c', code], { encoding: 'utf8' });
  if (result.status !== 0 || !result.stdout) return null;
  try {
    const identity = JSON.parse(result.stdout.trim());
    if (identity.major !== 3 || identity.minor < 10 || identity.minor >= 15) return null;
    return identity;
  } catch {
    return null;
  }
}

function ensurepipWorks(binary) {
  const probe = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-py-probe-'));
  try {
    return spawn(binary, ['-m', 'venv', probe], { stdio: 'ignore' }).status === 0;
  } finally {
    fs.rmSync(probe, { recursive: true, force: true });
  }
}

function findPython(candidates = PYTHON_CANDIDATES) {
  const rejected = [];
  for (const binary of candidates) {
    const identity = pythonIdentity(binary);
    if (!identity) continue;
    if (!ensurepipWorks(binary)) {
      rejected.push(binary);
      continue;
    }
    return { binary, identity, rejected };
  }
  return { binary: null, identity: null, rejected };
}

function parseExtras(raw = '') {
  const extras = [...new Set(raw.split(',').map((item) => item.trim()).filter(Boolean))].sort();
  const unknown = extras.filter((extra) => !ALLOWED_EXTRAS.has(extra));
  if (unknown.length) {
    throw new Error(
      `unsupported CUSTBACK_EXTRAS: ${unknown.join(', ')}; ` +
      'choose audio2face, mediapipe, rvm, or gpu'
    );
  }
  if (extras.includes('rvm') && extras.includes('gpu')) {
    throw new Error('CUSTBACK_EXTRAS cannot combine rvm and gpu (conflicting ONNX runtimes)');
  }
  if (extras.includes('audio2face') && extras.includes('mediapipe')) {
    throw new Error(
      'CUSTBACK_EXTRAS cannot combine audio2face and mediapipe ' +
      '(separate supported driver profiles)'
    );
  }
  return extras;
}

function installAttempts(requested) {
  if (requested.includes('mediapipe') || requested.includes('audio2face')) return [requested];
  const preferred = [...requested, 'mediapipe'].sort();
  return requested.length ? [preferred, requested] : [preferred, []];
}

function walkFiles(root, include = () => true) {
  if (!fs.existsSync(root)) return [];
  const result = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === '__pycache__' || entry.name.endsWith('.egg-info')) continue;
      result.push(...walkFiles(full, include));
    } else if (entry.isFile() && include(full)) {
      result.push(full);
    }
  }
  return result;
}

function sourceDigest(pkgRoot = PKG_ROOT) {
  const roots = [
    path.join(pkgRoot, 'package.json'),
    path.join(pkgRoot, 'pyproject.toml'),
    ...walkFiles(path.join(pkgRoot, 'config'), (file) => /\.ya?ml$/i.test(file)),
    ...walkFiles(
      path.join(pkgRoot, 'src', 'custback'),
      (file) => file.endsWith('.py') || /\.ya?ml$/i.test(file),
    ),
    ...walkFiles(path.join(pkgRoot, 'scripts'), (file) => file.endsWith('.sh')),
    ...walkFiles(path.join(pkgRoot, 'packaging', 'npm'), (file) =>
      file.endsWith('.js') && !file.includes(`${path.sep}test${path.sep}`)),
  ].filter((file) => fs.existsSync(file)).sort();
  const hash = crypto.createHash('sha256');
  for (const file of roots) {
    hash.update(path.relative(pkgRoot, file));
    hash.update('\0');
    hash.update(fs.readFileSync(file));
    hash.update('\0');
  }
  return `sha256:${hash.digest('hex')}`;
}

function sameArray(left, right) {
  return Array.isArray(left) && Array.isArray(right) &&
    left.length === right.length && left.every((value, index) => value === right[index]);
}

function validRequestedExtras(extras) {
  return Array.isArray(extras) &&
    extras.every((extra) => typeof extra === 'string' && ALLOWED_EXTRAS.has(extra)) &&
    sameArray(extras, [...new Set(extras)].sort()) &&
    !(extras.includes('rvm') && extras.includes('gpu')) &&
    !(extras.includes('audio2face') && extras.includes('mediapipe'));
}

function validExtraSelection(requested, selected) {
  return validRequestedExtras(requested) && validRequestedExtras(selected) &&
    requested.every((extra) => selected.includes(extra));
}

function validInstallStamp(stamp) {
  const valid = Boolean(stamp) && stamp.schema === INSTALL_STAMP_SCHEMA &&
    typeof stamp.packageVersion === 'string' && stamp.packageVersion.length > 0 &&
    typeof stamp.sourceDigest === 'string' && /^sha256:[0-9a-f]{64}$/.test(stamp.sourceDigest) &&
    stamp.python && typeof stamp.python.executable === 'string' &&
    path.isAbsolute(stamp.python.executable) &&
    typeof stamp.python.version === 'string' && /^3\.(?:10|11|12|13|14)\.\d+$/.test(stamp.python.version) &&
    typeof stamp.python.cacheTag === 'string' && stamp.python.cacheTag.length > 0 &&
    validExtraSelection(stamp.requestedExtras, stamp.selectedExtras) &&
    stamp.capabilities && typeof stamp.capabilities.mediapipe === 'boolean' &&
    typeof stamp.capabilities.audio2face === 'boolean' &&
    typeof stamp.capabilities.rvm === 'boolean' &&
    typeof stamp.capabilities.cuda_provider === 'boolean' &&
    typeof stamp.capabilities.cuda_inference === 'boolean' &&
    typeof stamp.createdAt === 'string' && Number.isFinite(Date.parse(stamp.createdAt));
  return valid &&
    (!stamp.selectedExtras.includes('mediapipe') || stamp.capabilities.mediapipe) &&
    (!stamp.selectedExtras.includes('audio2face') || stamp.capabilities.audio2face) &&
    (!(stamp.selectedExtras.includes('rvm') || stamp.selectedExtras.includes('gpu')) ||
      stamp.capabilities.rvm) &&
    (!stamp.selectedExtras.includes('gpu') ||
      (stamp.capabilities.cuda_provider && stamp.capabilities.cuda_inference));
}

function installIntentPath(target) {
  return `${path.resolve(target)}.custback-install-intent.json`;
}

function validInstallIntent(intent, target) {
  return Boolean(intent) && intent.owner === managed.OWNER &&
    intent.schema === INSTALL_INTENT_SCHEMA &&
    typeof intent.logicalTarget === 'string' && path.isAbsolute(intent.logicalTarget) &&
    path.resolve(intent.logicalTarget) === path.resolve(target) &&
    validRequestedExtras(intent.requestedExtras) &&
    typeof intent.updatedAt === 'string' && Number.isFinite(Date.parse(intent.updatedAt));
}

function readInstallIntent(target) {
  const file = installIntentPath(target);
  let stat;
  try {
    stat = fs.lstatSync(file);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
  if (!stat.isFile() || stat.isSymbolicLink()) {
    throw new Error(`custback install intent is not a regular owned file: ${file}`);
  }
  const intent = managed.readJson(file);
  if (!validInstallIntent(intent, target)) {
    throw new Error(`custback install intent does not match ${path.resolve(target)}`);
  }
  return intent;
}

function writeInstallIntent(target, requestedExtras) {
  if (!validRequestedExtras(requestedExtras)) {
    throw new Error('refusing to persist invalid custback extras intent');
  }
  const file = installIntentPath(target);
  const existing = fs.existsSync(file) ? fs.lstatSync(file) : null;
  if (existing && (!existing.isFile() || existing.isSymbolicLink())) {
    throw new Error(`custback install intent is not a regular owned file: ${file}`);
  }
  if (existing) readInstallIntent(target);
  managed.writeJson(file, {
    owner: managed.OWNER,
    schema: INSTALL_INTENT_SCHEMA,
    logicalTarget: path.resolve(target),
    requestedExtras: [...requestedExtras],
    updatedAt: new Date().toISOString(),
  });
}

function resolveRequestedExtras(env, persistedIntent, activeStamp = null) {
  if (Object.prototype.hasOwnProperty.call(env, 'CUSTBACK_EXTRAS')) {
    return parseExtras(env.CUSTBACK_EXTRAS);
  }
  if (persistedIntent) {
    // Promotion is authoritative before intent publication. If publication
    // failed after an otherwise successful promotion, the newer valid active
    // stamp prevents the stale intent from undoing that environment on the
    // next absent-env run; reuse will repair the intent file.
    if (validInstallStamp(activeStamp) &&
        !sameArray(persistedIntent.requestedExtras, activeStamp.requestedExtras) &&
        Date.parse(activeStamp.createdAt) > Date.parse(persistedIntent.updatedAt)) {
      return [...activeStamp.requestedExtras];
    }
    return [...persistedIntent.requestedExtras];
  }
  if (activeStamp && Object.prototype.hasOwnProperty.call(activeStamp, 'requestedExtras')) {
    if (!validRequestedExtras(activeStamp.requestedExtras)) {
      throw new Error('active managed venv contains invalid persisted extras metadata');
    }
    return [...activeStamp.requestedExtras];
  }
  return [];
}

function stampMatches(stamp, expected) {
  return validInstallStamp(stamp) && stamp.packageVersion === expected.packageVersion &&
    stamp.sourceDigest === expected.sourceDigest &&
    stamp.python && stamp.python.executable === expected.python.executable &&
    stamp.python.version === expected.python.version &&
    stamp.python.cacheTag === expected.python.cacheTag &&
    sameArray(stamp.requestedExtras, expected.requestedExtras);
}

function venvPython(directory) {
  return path.join(directory, 'bin', 'python');
}

function appBinary(directory) {
  return path.join(directory, 'bin', 'custback');
}

function emptyCudaProbe(error = '') {
  return {
    schema: 1,
    onnxruntime: false,
    cuda_provider: false,
    cuda_inference: false,
    active_providers: [],
    output_verified: false,
    profile_verified: false,
    error,
  };
}

function parseCudaProbeOutput(stdout) {
  try {
    const parsed = JSON.parse((stdout || '').trim());
    const valid = parsed && parsed.schema === 1 &&
      typeof parsed.onnxruntime === 'boolean' &&
      typeof parsed.cuda_provider === 'boolean' &&
      typeof parsed.cuda_inference === 'boolean' &&
      Array.isArray(parsed.active_providers) &&
      parsed.active_providers.every((provider) => typeof provider === 'string') &&
      typeof parsed.output_verified === 'boolean' &&
      typeof parsed.profile_verified === 'boolean' &&
      typeof parsed.error === 'string' &&
      (!parsed.cuda_inference || (
        parsed.onnxruntime && parsed.cuda_provider && parsed.output_verified &&
        parsed.profile_verified && parsed.active_providers.includes('CUDAExecutionProvider')
      ));
    return valid ? parsed : emptyCudaProbe('malformed CUDA probe result');
  } catch {
    return emptyCudaProbe('malformed CUDA probe JSON');
  }
}

function cudaProbe(python) {
  const result = spawn(python, ['-m', 'custback.gpu_probe', '--json'], {
    encoding: 'utf8',
    timeout: CUDA_PROBE_TIMEOUT_MS,
  });
  if (result.status !== 0) {
    const detail = (result.stderr || result.error?.message || 'probe process failed')
      .trim().split('\n').pop();
    return emptyCudaProbe(detail);
  }
  return parseCudaProbeOutput(result.stdout);
}

function capabilities(python) {
  const mediaPipe = spawn(python, ['-c', 'import mediapipe'], {
    encoding: 'utf8',
    timeout: CUDA_PROBE_TIMEOUT_MS,
  });
  const audio2face = spawn(python, ['-c', [
    'import grpc',
    'from nvidia_audio2face_3d import audio2face_pb2_grpc, messages_pb2',
    'from nvidia_ace import animation_pb2, audio_pb2',
  ].join('\n')], {
    encoding: 'utf8',
    timeout: CUDA_PROBE_TIMEOUT_MS,
  });
  const cuda = cudaProbe(python);
  return {
    mediapipe: mediaPipe.status === 0,
    audio2face: audio2face.status === 0,
    rvm: cuda.onnxruntime,
    cuda_provider: cuda.cuda_provider,
    cuda_inference: cuda.cuda_inference,
  };
}

function validateEnvironment(directory, packageVersion, selectedExtras, quiet = false) {
  const python = venvPython(directory);
  const stdio = quiet ? 'ignore' : 'inherit';
  const coreProbe = [
    'import importlib.metadata as m',
    'import custback, custback.api.server, cv2, fastapi, multipart, numpy, pydantic, PIL, pyvirtualcam, uvicorn, websockets, yaml',
    `expected = ${JSON.stringify(packageVersion)}`,
    'metadata_version = m.version("custback")',
    'if metadata_version != expected:\n    raise RuntimeError(f"metadata version {metadata_version!r} != {expected!r}")',
    'if custback.__version__ != expected:\n    raise RuntimeError(f"source version {custback.__version__!r} != {expected!r}")',
  ].join('\n');
  if (spawn(python, ['-c', coreProbe], { stdio }).status !== 0) {
    throw new Error('installed core imports/version probe failed');
  }
  if (spawn(python, ['-m', 'pip', 'check'], { stdio }).status !== 0) {
    throw new Error('pip dependency check failed');
  }
  if (spawn(python, ['-m', 'custback', '--help'], { stdio }).status !== 0) {
    throw new Error('custback CLI smoke probe failed');
  }
  const observed = capabilities(python);
  if (selectedExtras.includes('mediapipe') && !observed.mediapipe) {
    throw new Error('selected mediapipe extra cannot be imported');
  }
  if (selectedExtras.includes('audio2face') && !observed.audio2face) {
    throw new Error('selected Audio2Face protocol modules cannot be imported');
  }
  if ((selectedExtras.includes('rvm') || selectedExtras.includes('gpu')) && !observed.rvm) {
    throw new Error('selected ONNX Runtime extra cannot be imported');
  }
  if (selectedExtras.includes('gpu') && !observed.cuda_inference) {
    throw new Error('selected gpu extra cannot complete verified CUDA inference');
  }
  return observed;
}

function reusableEnvironment({ force, inspectionKind, appExists, stamp, expected, validate }) {
  if (force || inspectionKind === 'absent' || !appExists || !stampMatches(stamp, expected)) {
    return false;
  }
  try {
    validate(stamp.selectedExtras);
    return true;
  } catch {
    return false;
  }
}

function writeGenerationMetadata(generation, target, stamp) {
  managed.markGeneration(generation, target);
  managed.writeJson(path.join(generation, managed.INSTALL_STAMP), stamp);
}

function readActiveStamp(target, onInvalid = null) {
  const file = path.join(target, managed.INSTALL_STAMP);
  if (!fs.existsSync(file)) return null;
  try {
    return managed.readJson(file);
  } catch (err) {
    if (!onInvalid) throw err;
    onInvalid(err);
    return null;
  }
}

function buildGeneration({ generationRoot, python, attempts, packageVersion, target, stampBase }) {
  for (const extras of attempts) {
    const generation = managed.createGeneration(generationRoot);
    let keep = false;
    try {
      log(`creating candidate venv ${generation}`);
      if (run(python.binary, ['-m', 'venv', generation]).status !== 0) {
        throw new Error('venv creation failed');
      }
      const candidatePython = venvPython(generation);
      if (run(candidatePython, [
        '-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', PIP_SPEC,
      ]).status !== 0) {
        throw new Error('pip bootstrap upgrade failed');
      }
      const spec = extras.length ? `${PKG_ROOT}[${extras.join(',')}]` : PKG_ROOT;
      log(`installing custback${extras.length ? ` (extras: ${extras.join(', ')})` : ' (core)'}`);
      const result = run(candidatePython, [
        '-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', spec,
      ]);
      if (result.status !== 0) {
        if (extras.includes('mediapipe') && !stampBase.requestedExtras.includes('mediapipe')) {
          log('implicit MediaPipe install failed; retrying without that optional capability');
          continue;
        }
        throw new Error(`pip install failed for requested extras [${extras.join(',')}]`);
      }
      let observed;
      try {
        observed = validateEnvironment(generation, packageVersion, extras);
      } catch (err) {
        if (extras.includes('mediapipe') && !stampBase.requestedExtras.includes('mediapipe')) {
          log(`implicit MediaPipe candidate failed validation (${err.message}); retrying without it`);
          continue;
        }
        throw err;
      }
      const stamp = {
        schema: INSTALL_STAMP_SCHEMA,
        packageVersion,
        sourceDigest: stampBase.sourceDigest,
        python: stampBase.python,
        requestedExtras: stampBase.requestedExtras,
        selectedExtras: extras,
        capabilities: observed,
        createdAt: new Date().toISOString(),
      };
      writeGenerationMetadata(generation, target, stamp);
      keep = true;
      return { generation, stamp };
    } finally {
      if (!keep) managed.removeCreatedGeneration(generation, generationRoot);
    }
  }
  throw new Error('no permitted custback dependency set could be installed');
}

function readLegacyRequestedExtras(target = LEGACY_DEFAULT_VENV) {
  if (!fs.existsSync(target)) return null;
  const inspection = managed.inspectTarget(target);
  if (inspection.kind === 'absent') return null;
  const stamp = readActiveStamp(target);
  if (stamp && Object.prototype.hasOwnProperty.call(stamp, 'requestedExtras')) {
    if (!validRequestedExtras(stamp.requestedExtras)) {
      throw new Error('legacy managed venv contains invalid extras metadata');
    }
    return [...stamp.requestedExtras];
  }
  if (inspection.legacy) {
    const legacy = fs.readFileSync(path.join(target, managed.LEGACY_STAMP), 'utf8').trim();
    const match = legacy.match(/\[([^\]]*)\]$/);
    if (match) return parseExtras(match[1].replace(/\s+/g, ','));
  }
  return null;
}

function main() {
  if (process.env.CUSTBACK_SKIP_INSTALL === '1') {
    log('CUSTBACK_SKIP_INSTALL=1, skipping Python bootstrap');
    return 0;
  }
  if (process.platform === 'win32') {
    console.error('custback supports Linux and macOS only');
    return 1;
  }
  try {
    if (typeof process.getuid === 'function' && process.getuid() === 0) {
      log('warning: installing the private environment as root is unnecessary; prefer an unprivileged npm prefix');
    }
    const target = managed.assertSafeTarget(process.env.CUSTBACK_VENV || DEFAULT_VENV, {
      pkgRoot: PKG_ROOT,
    });
    const explicitExtras = Object.prototype.hasOwnProperty.call(process.env, 'CUSTBACK_EXTRAS');
    // Validate explicit intent before doing interpreter probes or filesystem
    // mutation. An empty value deliberately means core-only.
    if (explicitExtras) parseExtras(process.env.CUSTBACK_EXTRAS);
    const packageVersion = require(path.join(PKG_ROOT, 'package.json')).version;
    const python = findPython();
    if (!python.binary) {
      if (python.rejected.length) {
        throw new Error(
          `found ${python.rejected.join(', ')} but venv/ensurepip is unavailable; ` +
          'install the matching python3-venv package and run custback rebuild'
        );
      }
      throw new Error('no usable Python >=3.10,<3.15 found on PATH');
    }
    log(`using ${python.binary} (Python ${python.identity.version})`);

    // Reject unsafe/unowned targets before creating sibling metadata, while
    // allowing a journal-proven dangling promotion to reach locked recovery.
    const generationRoot = managed.prepareInstallTarget(target);
    return managed.withInstallLock(generationRoot, () => {
      managed.recoverInterruptedPromotion(target, generationRoot);
      const inspection = managed.inspectTarget(target);
      const activeStamp = inspection.kind === 'absent' ? null : readActiveStamp(target, (err) => {
        log(`managed venv install metadata is corrupt (${err.message}); rebuilding`);
      });
      // Malformed or unowned intent fails closed instead of silently dropping
      // extras or replacing a colliding file.
      const persistedIntent = readInstallIntent(target);
      let legacyStamp = null;
      if (!explicitExtras && !persistedIntent && !activeStamp &&
          !Object.prototype.hasOwnProperty.call(process.env, 'CUSTBACK_VENV')) {
        try {
          const requestedExtras = readLegacyRequestedExtras();
          if (requestedExtras) {
            legacyStamp = { requestedExtras };
            log('rebuilding the legacy package-local venv at the durable npm-prefix target');
          }
        } catch (err) {
          log(`warning: legacy package-local venv was not adopted (${err.message})`);
        }
      }
      const requestedExtras = resolveRequestedExtras(
        process.env,
        persistedIntent,
        activeStamp || legacyStamp,
      );
      const expected = {
        packageVersion,
        sourceDigest: sourceDigest(),
        python: {
          executable: python.identity.executable,
          version: python.identity.version,
          cacheTag: python.identity.cache_tag,
        },
        requestedExtras,
      };
      const reuse = reusableEnvironment({
        force: process.env.CUSTBACK_FORCE_REBUILD === '1',
        inspectionKind: inspection.kind,
        appExists: fs.existsSync(appBinary(target)),
        stamp: activeStamp,
        expected,
        validate: (selectedExtras) =>
          validateEnvironment(target, packageVersion, selectedExtras, true),
      });
      if (reuse) {
        if (!persistedIntent ||
            !sameArray(persistedIntent.requestedExtras, requestedExtras)) {
          writeInstallIntent(target, requestedExtras);
        }
        log('managed venv is already up to date and healthy');
        return 0;
      }
      if (activeStamp && stampMatches(activeStamp, expected) &&
          process.env.CUSTBACK_FORCE_REBUILD !== '1') {
        log('managed venv metadata matched but health validation failed; rebuilding');
      }

      const built = buildGeneration({
        generationRoot,
        python,
        attempts: installAttempts(requestedExtras),
        packageVersion,
        target,
        stampBase: expected,
      });
      let promoted = false;
      try {
        const promotion = managed.promoteGeneration({
          target,
          generation: built.generation,
          inspection,
          validateActive: (logicalTarget) => {
            const stamp = readActiveStamp(logicalTarget);
            if (!stampMatches(stamp, expected)) throw new Error('promoted install stamp mismatch');
            validateEnvironment(logicalTarget, packageVersion, stamp.selectedExtras, true);
          },
        });
        promoted = true;
        // Persist only after the new environment has been promoted and its
        // active-path validation succeeded. Failed candidates retain the last
        // successful extras choice.
        writeInstallIntent(target, requestedExtras);
        const keep = new Set([path.resolve(built.generation)]);
        if (promotion.previousGeneration) keep.add(path.resolve(promotion.previousGeneration));
        try {
          managed.cleanupGenerations(generationRoot, target, keep, 0);
        } catch (err) {
          log(`warning: could not clean an older owned generation: ${err.message}`);
        }
      } finally {
        if (!promoted && fs.existsSync(built.generation)) {
          if (managed.generationIsReferenced(target, built.generation, generationRoot)) {
            log('warning: retaining candidate referenced by incomplete promotion state');
          } else {
            managed.removeCreatedGeneration(built.generation, generationRoot);
          }
        }
      }
      log(`installed extras: ${built.stamp.selectedExtras.join(', ') || 'core only'}`);
      log('done; run "custback setup" once, then "custback doctor"');
      return 0;
    });
  } catch (err) {
    console.error(`[custback install] ${err.message}`);
    return 1;
  }
}

module.exports = {
  allowedExtras: () => [...ALLOWED_EXTRAS].sort(),
  buildGeneration,
  capabilities,
  cudaProbe,
  CUDA_PROBE_TIMEOUT_MS,
  DEFAULT_VENV,
  findPython,
  INSTALL_INTENT_SCHEMA,
  INSTALL_STAMP_SCHEMA,
  installIntentPath,
  installAttempts,
  LEGACY_DEFAULT_VENV,
  main,
  parseCudaProbeOutput,
  parseExtras,
  PIP_SPEC,
  pythonIdentity,
  readActiveStamp,
  readInstallIntent,
  readLegacyRequestedExtras,
  resolveRequestedExtras,
  reusableEnvironment,
  sourceDigest,
  stampMatches,
  validExtraSelection,
  validInstallIntent,
  validInstallStamp,
  validRequestedExtras,
  validateEnvironment,
  writeInstallIntent,
};

if (require.main === module) process.exit(main());
