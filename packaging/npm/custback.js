#!/usr/bin/env node
/** npm launcher and machine-facing setup/doctor commands for custback. */

'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const managed = require('./managed-venv');
const installer = require('./install');

const PKG_ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_VENV = path.join(PKG_ROOT, '.venv');
const configuredProbeTimeout = Number(process.env.CUSTBACK_DOCTOR_TIMEOUT_MS);
const PROBE_TIMEOUT_MS = Number.isSafeInteger(configuredProbeTimeout) && configuredProbeTimeout > 0
  ? configuredProbeTimeout
  : 60 * 1000;

function runProbe(command, args, options = {}) {
  return spawnSync(command, args, { timeout: PROBE_TIMEOUT_MS, ...options });
}

function targetPath() {
  return managed.assertSafeTarget(process.env.CUSTBACK_VENV || DEFAULT_VENV, { pkgRoot: PKG_ROOT });
}

function appPath(target = targetPath()) {
  return path.join(target, 'bin', 'custback');
}

function pythonPath(target = targetPath()) {
  return path.join(target, 'bin', 'python');
}

function bootstrap(force = false) {
  const env = { ...process.env };
  if (force) env.CUSTBACK_FORCE_REBUILD = '1';
  const result = spawnSync(process.execPath, [path.join(__dirname, 'install.js')], {
    stdio: 'inherit',
    env,
  });
  return result.status === 0;
}

function hasCustbackCamera() {
  const base = '/sys/class/video4linux';
  if (!fs.existsSync(base)) return false;
  return fs.readdirSync(base).some((device) => {
    try {
      return fs.readFileSync(path.join(base, device, 'name'), 'utf8').includes('custback Camera');
    } catch {
      return false;
    }
  });
}

function setup() {
  if (!['linux', 'darwin'].includes(process.platform)) {
    console.error(`custback setup is unsupported on ${process.platform}`);
    return 1;
  }
  const script = process.platform === 'darwin'
    ? path.join(PKG_ROOT, 'scripts', 'install_macos.sh')
    : path.join(PKG_ROOT, 'scripts', 'install_linux.sh');
  console.log(`running ${script}`);
  return spawnSync('bash', [script], { stdio: 'inherit' }).status ?? 1;
}

function probeImport(python, moduleName) {
  return runProbe(python, ['-c', `import ${moduleName}`], { encoding: 'utf8' });
}

function doctor() {
  let failures = 0;
  const report = (label, good, hint = '') => {
    console.log(`${good ? '  ok ' : 'FAIL '} ${label}${good || !hint ? '' : ` — ${hint}`}`);
    if (!good) failures += 1;
  };
  const warn = (label, hint = '') => {
    console.log(`WARN  ${label}${hint ? ` — ${hint}` : ''}`);
  };
  const note = (label, hint = '') => {
    console.log(`note  ${label}${hint ? ` — ${hint}` : ''}`);
  };

  report(`platform ${process.platform}`, ['linux', 'darwin'].includes(process.platform),
    'custback supports Linux and macOS');

  let target;
  let inspection;
  try {
    target = targetPath();
    inspection = managed.inspectTarget(target);
    report('managed Python venv', inspection.kind !== 'absent', 'run: custback rebuild');
  } catch (err) {
    report('managed Python venv', false, err.message);
    return 1;
  }
  if (inspection.kind === 'absent') return 1;

  const python = pythonPath(target);
  const app = appPath(target);
  report('custback launcher', fs.existsSync(app), 'run: custback rebuild');
  if (!fs.existsSync(app) || !fs.existsSync(python)) return 1;

  const packageVersion = require(path.join(PKG_ROOT, 'package.json')).version;
  const coreCode = [
    'import importlib.metadata as m',
    'import custback, custback.api.server, cv2, fastapi, multipart, numpy, pydantic, PIL, pyvirtualcam, uvicorn, websockets, yaml',
    `expected = ${JSON.stringify(packageVersion)}`,
    'metadata_version = m.version("custback")',
    'if metadata_version != expected:\n    raise RuntimeError(f"metadata version {metadata_version!r} != {expected!r}")',
    'if custback.__version__ != expected:\n    raise RuntimeError(f"source version {custback.__version__!r} != {expected!r}")',
  ].join('\n');
  const core = runProbe(python, ['-c', coreCode], { encoding: 'utf8' });
  report('core Python imports and version', core.status === 0,
    (core.stderr || '').trim().split('\n').pop());
  const pipCheck = runProbe(python, ['-m', 'pip', 'check'], { encoding: 'utf8' });
  report('Python dependency consistency', pipCheck.status === 0,
    (pipCheck.stdout || pipCheck.stderr || '').trim().split('\n').pop());

  let stamp = null;
  try {
    stamp = managed.readJson(path.join(target, managed.INSTALL_STAMP));
    report('install metadata', installer.validInstallStamp(stamp) &&
      stamp.packageVersion === packageVersion && stamp.sourceDigest === installer.sourceDigest(),
      'metadata is stale; run: custback rebuild');
  } catch (err) {
    report('install metadata', false, `${err.message}; run: custback rebuild`);
  }

  const selected = stamp && Array.isArray(stamp.selectedExtras) ? stamp.selectedExtras : [];
  const requested = stamp && Array.isArray(stamp.requestedExtras) ? stamp.requestedExtras : [];
  if (selected.includes('mediapipe')) {
    const result = probeImport(python, 'mediapipe');
    if (result.status === 0) {
      report('MediaPipe segmentation', true);
    } else if (requested.includes('mediapipe')) {
      report('MediaPipe segmentation', false, (result.stderr || '').trim().split('\n').pop());
    } else {
      warn('implicit MediaPipe segmentation is unavailable',
        'heuristic fallback remains available; run custback rebuild to repair it');
    }
  } else if (requested.includes('mediapipe')) {
    report('requested MediaPipe segmentation', false, 'requested extra is absent; run: custback rebuild');
  } else {
    note('MediaPipe segmentation not installed',
      'optional; heuristic fallback is available (CUSTBACK_EXTRAS=mediapipe custback rebuild)');
  }

  if (selected.includes('rvm') || selected.includes('gpu')) {
    const cuda = installer.cudaProbe(python);
    if (cuda.onnxruntime) {
      report('RVM ONNX Runtime', true);
    } else if (requested.includes('rvm') || requested.includes('gpu')) {
      report('RVM ONNX Runtime', false, cuda.error || 'ONNX Runtime import failed');
    } else {
      warn('implicit RVM ONNX Runtime is unavailable');
    }
    if (cuda.onnxruntime && selected.includes('gpu')) {
      if (requested.includes('gpu')) {
        report('verified CUDA inference', cuda.cuda_inference,
          cuda.error || 'GPU extra was requested but execution fell back from CUDA');
      } else if (!cuda.cuda_inference) {
        warn('implicit CUDA inference is unavailable', cuda.error);
      }
    } else if (cuda.onnxruntime) {
      note(`CUDA provider ${cuda.cuda_provider ? 'registered' : 'not registered'}; CPU RVM is healthy`);
    }
  } else if (requested.includes('rvm') || requested.includes('gpu')) {
    report('requested RVM backend', false, 'requested extra is absent; run: custback rebuild');
  } else {
    note('RVM matting not installed',
      'optional; use CUSTBACK_EXTRAS=rvm (CPU) or gpu (NVIDIA) with custback rebuild');
  }

  if (process.platform === 'linux') {
    if (!hasCustbackCamera()) {
      warn('custback virtual camera device not found', 'run: custback setup');
    } else {
      note('custback virtual camera device is present');
    }
  } else if (process.platform === 'darwin') {
    if (!fs.existsSync('/Applications/OBS.app')) {
      warn('OBS virtual camera is not installed', 'run: custback setup');
    } else {
      note('OBS installation is present');
    }
  }
  return failures ? 1 : 0;
}

function managedAppExists(target) {
  const inspection = managed.inspectTarget(target);
  return inspection.kind !== 'absent' && fs.existsSync(appPath(target));
}

function main(argv = process.argv.slice(2)) {
  const command = argv[0];
  if (command === 'setup') return setup();
  if (command === 'doctor') return doctor();
  if (command === 'rebuild') return bootstrap(true) ? 0 : 1;

  try {
    const target = targetPath();
    if (!managedAppExists(target) && !bootstrap(false)) {
      console.error('custback: managed Python environment is missing and bootstrap failed');
      return 1;
    }
    if (!managedAppExists(target)) {
      console.error('custback: bootstrap completed without a valid managed launcher');
      return 1;
    }
    const result = spawnSync(appPath(target), argv, { stdio: 'inherit' });
    if (result.error) {
      console.error(`custback: failed to launch ${appPath(target)}: ${result.error.message}`);
      return 1;
    }
    return result.status ?? 1;
  } catch (err) {
    console.error(`custback: ${err.message}`);
    return 1;
  }
}

module.exports = {
  appPath,
  bootstrap,
  doctor,
  hasCustbackCamera,
  main,
  managedAppExists,
  pythonPath,
  targetPath,
};

if (require.main === module) process.exit(main());
