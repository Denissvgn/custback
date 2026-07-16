#!/usr/bin/env node
/** Non-mutating release metadata and npm payload gate. */

'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { isDeepStrictEqual } = require('util');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..', '..');
const managed = require(path.join(ROOT, 'packaging', 'npm', 'managed-venv'));
const installer = require(path.join(ROOT, 'packaging', 'npm', 'install'));
const configuredTimeout = Number(process.env.CUSTBACK_RELEASE_TIMEOUT_MS);
const COMMAND_TIMEOUT_MS = Number.isSafeInteger(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 15 * 60 * 1000;
const LINUX_MEMORY_BACKED_FILESYSTEM_MAGICS = new Set([
  0x01021994, // TMPFS_MAGIC
  0x858458f6, // RAMFS_MAGIC
]);
const REVIEWED_PYTHON_MODULES = [
  'custback/__init__.py',
  'custback/__main__.py',
  'custback/api/__init__.py',
  'custback/api/avatar_proxy.py',
  'custback/api/security.py',
  'custback/api/server.py',
  'custback/api/streaming.py',
  'custback/api/webui.py',
  'custback/avatar/__init__.py',
  'custback/avatar/__main__.py',
  'custback/avatar/api.py',
  'custback/avatar/audio2face.py',
  'custback/avatar/config.py',
  'custback/avatar/drivers.py',
  'custback/avatar/renderer.py',
  'custback/avatar/rig.py',
  'custback/avatar/service.py',
  'custback/avatar/state.py',
  'custback/avatar/store.py',
  'custback/backgrounds.py',
  'custback/capture.py',
  'custback/compositor.py',
  'custback/config.py',
  'custback/config_merge.py',
  'custback/diagnostics.py',
  'custback/gpu_probe.py',
  'custback/hub.py',
  'custback/pipeline.py',
  'custback/preview.py',
  'custback/segmentation.py',
  'custback/vcam.py',
];
const REVIEWED_PYTHON_TESTS = [
  'tests/test_api.py',
  'tests/test_api_lifecycle.py',
  'tests/test_api_security.py',
  'tests/test_audio2face_protocol.py',
  'tests/test_avatar_api.py',
  'tests/test_avatar_config.py',
  'tests/test_avatar_drivers.py',
  'tests/test_avatar_proxy.py',
  'tests/test_avatar_rig.py',
  'tests/test_avatar_service.py',
  'tests/test_avatar_store.py',
  'tests/test_capture.py',
  'tests/test_config.py',
  'tests/test_config_merge.py',
  'tests/test_diagnostics.py',
  'tests/test_gpu_probe.py',
  'tests/test_model_acquisition.py',
  'tests/test_pipeline.py',
  'tests/test_preview.py',
  'tests/test_processing.py',
  'tests/test_remediation_runtime.py',
  'tests/test_remediation_security.py',
  'tests/test_segmentation_rvm.py',
  'tests/test_streaming.py',
  'tests/test_webui.py',
];
const REVIEWED_NPM_PAYLOAD = [
  'LICENSE',
  'README.md',
  'config/avatar.yaml',
  'config/default.yaml',
  'docs/remote-deployment.md',
  'examples/avatar_client.py',
  'package.json',
  'packaging/npm/custback.js',
  'packaging/npm/install.js',
  'packaging/npm/managed-venv.js',
  'pyproject.toml',
  'scripts/install_linux.sh',
  'scripts/install_macos.sh',
  'scripts/release/verify-release.js',
  ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
];
const REVIEWED_BUILD_REQUIREMENTS = ['setuptools>=77,<84'];
const REVIEWED_CORE_DEPENDENCIES = [
  'numpy>=1.24,<3',
  'opencv-contrib-python>=4.8,<6',
  'pillow>=10,<13',
  'pydantic>=2.7,<3',
  'pyvirtualcam>=0.11,<1',
  'fastapi>=0.110,<1',
  'uvicorn>=0.29,<1',
  'pyyaml>=6.0,<7',
  'websockets>=12.0,<17',
  'python-multipart>=0.0.9,<1',
  'httpx>=0.27,<0.29',
];
const REVIEWED_OPTIONAL_DEPENDENCIES = {
  mediapipe: ['mediapipe>=0.10.14,<0.11'],
  rvm: ['onnxruntime>=1.17,<2'],
  gpu: ['onnxruntime-gpu>=1.17,<1.27'],
  audio2face: [
    'grpcio>=1.67,<1.67.2',
    'nvidia-ace==1.0.0',
    'nvidia-audio2face-3d==1.3.0',
    'protobuf>=5.29.3,<6',
    'sounddevice>=0.4,<0.6',
  ],
  dev: [
    'build>=1.2,<2',
    'pytest>=8.0,<10',
    'pytest-timeout>=2.3,<3',
    'httpx>=0.27,<0.29',
    'httpx2>=2,<3',
  ],
};
const REVIEWED_CONSOLE_SCRIPTS = {
  custback: 'custback.__main__:main',
  'custback-avatar': 'custback.avatar.__main__:main',
};
const REVIEWED_ACTIONS = new Set([
  'actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955',
  'actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020',
  'actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065',
]);
const REVIEWED_LICENSE_COPYRIGHT = 'Copyright (c) 2026 Bramen';
const REVIEWED_NPM_METADATA = {
  name: 'custback',
  description: 'Virtual camera with background replacement for meeting apps (Ubuntu / Debian / macOS)',
  license: 'MIT',
  bin: {
    custback: 'packaging/npm/custback.js',
    'custback-avatar': 'packaging/npm/custback.js',
  },
  scripts: {
    postinstall: 'node packaging/npm/install.js',
    test: 'node --test packaging/npm/test/*.test.js',
    doctor: 'node packaging/npm/custback.js doctor',
    'release:check': 'node scripts/release/verify-release.js',
    prepack: 'node scripts/release/verify-release.js --prepack',
  },
  files: [
    'LICENSE',
    'packaging/npm/*.js',
    'src/**/*.py',
    'config/*.yaml',
    'docs/*.md',
    'scripts/*.sh',
    'scripts/release/*.js',
    'examples/*.py',
    'pyproject.toml',
  ],
  os: ['linux', 'darwin'],
  engines: { node: '>=18' },
  keywords: [
    'virtual-camera',
    'background-replacement',
    'webcam',
    'v4l2loopback',
    'obs',
  ],
};

function fail(message) {
  throw new Error(message);
}

function filesystemIsMemoryBacked(
  directory,
  statfs = fs.statfsSync,
  platform = process.platform,
) {
  if (platform !== 'linux') return false;
  if (typeof statfs !== 'function') {
    fail('cannot determine whether release temporary storage is memory-backed');
  }
  return LINUX_MEMORY_BACKED_FILESYSTEM_MAGICS.has(Number(statfs(directory).type));
}

function releaseBuildJobs(env = process.env) {
  const configured = env.CUSTBACK_RELEASE_BUILD_JOBS;
  if (configured === undefined) return '2';
  const value = String(configured);
  if (!/^[1-9][0-9]*$/.test(value) || Number(value) > 32) {
    fail('CUSTBACK_RELEASE_BUILD_JOBS must be an integer from 1 through 32');
  }
  return value;
}

function createReleaseTemporaryRoot(options = {}) {
  const env = options.env || process.env;
  const configured = env.CUSTBACK_RELEASE_TMPDIR;
  const candidate = configured || options.defaultBase || os.tmpdir();
  let base;
  try {
    base = fs.realpathSync(path.resolve(candidate));
    if (!fs.statSync(base).isDirectory()) {
      fail('release temporary base must be an existing directory');
    }
  } catch (err) {
    if (err.message === 'release temporary base must be an existing directory') throw err;
    fail(`release temporary base is unavailable: ${err.message}`);
  }
  const sourceRoot = fs.realpathSync(options.root || ROOT);
  const sourceRelative = path.relative(sourceRoot, base);
  if (!sourceRelative ||
      (!sourceRelative.startsWith(`..${path.sep}`) && sourceRelative !== '..' &&
       !path.isAbsolute(sourceRelative))) {
    fail('release temporary base must be outside the source checkout');
  }
  const allowTmpfs = env.CUSTBACK_RELEASE_ALLOW_TMPFS === '1';
  if (filesystemIsMemoryBacked(base, options.statfs, options.platform) && !allowTmpfs) {
    fail(
      `${base} is a memory-backed filesystem; set CUSTBACK_RELEASE_TMPDIR to an ` +
      'existing disk-backed directory (or explicitly set CUSTBACK_RELEASE_ALLOW_TMPFS=1)'
    );
  }
  return fs.mkdtempSync(path.join(base, 'custback-release-'));
}

function withDisposableDirectory(directory, callback) {
  if (fs.existsSync(directory)) {
    fail(`refusing to reuse disposable release directory: ${directory}`);
  }
  try {
    return callback(directory);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function withTemporaryEnvironment(updates, callback, env = process.env) {
  const previous = new Map();
  for (const [name, value] of Object.entries(updates)) {
    previous.set(name, Object.prototype.hasOwnProperty.call(env, name) ? env[name] : undefined);
    env[name] = value;
  }
  try {
    return callback();
  } finally {
    for (const [name, value] of previous) {
      if (value === undefined) delete env[name];
      else env[name] = value;
    }
  }
}

function canonicalLicense(root = ROOT) {
  const licensePath = path.join(root, 'LICENSE');
  let text;
  try {
    const stat = fs.lstatSync(licensePath);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      fail('LICENSE must be a regular, non-symlink file');
    }
    text = fs.readFileSync(licensePath, 'utf8');
  } catch (err) {
    if (err.message.startsWith('LICENSE must')) throw err;
    fail(`LICENSE is missing or unreadable: ${err.message}`);
  }
  const required = [
    'MIT License',
    REVIEWED_LICENSE_COPYRIGHT,
    'Permission is hereby granted, free of charge, to any person obtaining a copy',
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
  ];
  if (!text.endsWith('\n') || required.some((line) => !text.includes(line))) {
    fail(`LICENSE must contain the reviewed MIT grant and ${REVIEWED_LICENSE_COPYRIGHT}`);
  }
  return Buffer.from(text, 'utf8');
}

function verifyLicenseMetadata(root = ROOT) {
  canonicalLicense(root);
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const project = tableBody(pyproject, 'project');
  if (stringValue(project, 'license') !== 'MIT' ||
      !isDeepStrictEqual(arrayValue(project, 'license-files'), ['LICENSE'])) {
    fail('pyproject.toml must declare MIT and ship only the canonical LICENSE file');
  }
  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  if (pkg.license !== 'MIT' || !Array.isArray(pkg.files) || !pkg.files.includes('LICENSE')) {
    fail('package.json must declare MIT and explicitly ship LICENSE');
  }
}

function remediationBlockers(root = ROOT) {
  const registryPath = path.join(root, 'scripts', 'release', 'remediation-blockers.json');
  if (!fs.existsSync(registryPath)) {
    fail('release remediation blocker registry is missing');
  }
  let registry;
  try {
    registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
  } catch (err) {
    fail(`release remediation blocker registry is invalid JSON: ${err.message}`);
  }
  if (!registry || registry.schema_version !== 1 || !Array.isArray(registry.blockers)) {
    fail('release remediation blocker registry must use schema_version 1 with a blockers array');
  }
  const seen = new Set();
  const blockers = registry.blockers.map((entry) => {
    if (!entry || typeof entry !== 'object' ||
        typeof entry.id !== 'string' || !/^[A-Z][A-Z0-9]+-\d{2}$/.test(entry.id) ||
        !Number.isSafeInteger(entry.phase) || entry.phase < 1 ||
        !['open', 'resolved'].includes(entry.status) ||
        typeof entry.title !== 'string' || entry.title.trim() === '' ||
        typeof entry.regression !== 'string' || entry.regression.trim() === '' ||
        path.isAbsolute(entry.regression) ||
        entry.regression.split(/[\\/]/).includes('..')) {
      fail('release remediation blocker registry contains an invalid entry');
    }
    if (seen.has(entry.id)) {
      fail(`release remediation blocker registry contains duplicate id ${entry.id}`);
    }
    seen.add(entry.id);
    return { ...entry };
  });
  const open = blockers.filter((entry) => entry.status === 'open');
  if (registry.release_blocked !== (open.length > 0)) {
    fail('release_blocked must be true exactly while remediation blockers remain open');
  }
  return blockers;
}

function verifyBlockerRegressionCoverage(root = ROOT) {
  for (const blocker of remediationBlockers(root)) {
    const regressionPath = path.join(root, blocker.regression);
    if (!fs.existsSync(regressionPath) || !fs.statSync(regressionPath).isFile()) {
      fail(`${blocker.id} regression file is missing: ${blocker.regression}`);
    }
    const source = fs.readFileSync(regressionPath, 'utf8');
    if (!source.includes(blocker.id)) {
      fail(`${blocker.id} is not referenced by ${blocker.regression}`);
    }
  }
}

function verifyNoReleaseBlockers(root = ROOT) {
  verifyBlockerRegressionCoverage(root);
  const open = remediationBlockers(root).filter((entry) => entry.status === 'open');
  if (open.length) {
    fail(
      `release blocked by ${open.length} open remediation blocker(s): ` +
      `${open.map((entry) => entry.id).join(', ')}; see REMEDIATION_PLAN.md`,
    );
  }
}

function projectVersion(pyproject) {
  const project = pyproject.match(/\[project\]([\s\S]*?)(?:\n\[|$)/);
  if (!project) fail('pyproject.toml has no [project] table');
  const version = project[1].match(/^version\s*=\s*"([^"]+)"/m);
  if (!version) fail('pyproject.toml [project] has no static version');
  return version[1];
}

function tableBody(toml, name) {
  const header = `[${name}]`;
  const start = toml.indexOf(header);
  if (start < 0) fail(`pyproject.toml has no ${header} table`);
  const contentStart = toml.indexOf('\n', start + header.length);
  if (contentStart < 0) return '';
  const rest = toml.slice(contentStart + 1);
  const nextTable = rest.search(/^\[/m);
  return nextTable < 0 ? rest : rest.slice(0, nextTable);
}

function arrayValue(table, key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = table.match(new RegExp(`^${escaped}\\s*=\\s*\\[([\\s\\S]*?)\\]`, 'm'));
  if (!match) fail(`pyproject.toml has no ${key} array in the expected table`);
  const uncommented = match[1].split('\n').map((line) => line.split('#', 1)[0]).join('\n');
  return [...uncommented.matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
}

function assignmentKeys(table) {
  return [...table.matchAll(/^([A-Za-z0-9_-]+)\s*=/gm)].map((entry) => entry[1]);
}

function stringValue(table, key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = table.match(new RegExp(`^${escaped}\\s*=\\s*"([^"]+)"\\s*$`, 'm'));
  if (!match) fail(`pyproject.toml has no exact ${key} string in the expected table`);
  return match[1];
}

function sourceFallbackVersion(filename) {
  const python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
  const parseCode = `
import ast, json, pathlib, sys
source_path = pathlib.Path(sys.argv[1])
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
values = []
for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
    if not isinstance(handler.type, ast.Name) or handler.type.id != "PackageNotFoundError":
        continue
    for node in handler.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise RuntimeError("__version__ fallback must be a literal string")
        values.append(value)
if len(values) != 1:
    raise RuntimeError(f"expected exactly one PackageNotFoundError __version__ fallback, found {len(values)}")
print(json.dumps(values[0]))
`;
  const result = spawnSync(python, ['-c', parseCode, filename], {
    encoding: 'utf8',
    timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
  });
  if (result.error && result.status === null) {
    fail(`could not inspect source-tree __version__: ${result.error.message}`);
  }
  if (result.status !== 0) {
    fail(`invalid source-tree __version__ fallback: ${(result.stderr || '').trim()}`);
  }
  try {
    return JSON.parse(result.stdout);
  } catch (err) {
    fail(`source-tree __version__ probe returned invalid JSON: ${err.message}`);
  }
}

function verifyNpmMetadata(root = ROOT) {
  const packagePath = path.join(root, 'package.json');
  const lockPath = path.join(root, 'package-lock.json');
  const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
  const { version, ...metadata } = pkg;
  if (typeof version !== 'string' || version === '' ||
      !isDeepStrictEqual(metadata, REVIEWED_NPM_METADATA)) {
    fail('package.json metadata must exactly match the reviewed release allowlist');
  }
  const expectedLock = {
    name: REVIEWED_NPM_METADATA.name,
    version,
    lockfileVersion: 3,
    requires: true,
    packages: {
      '': {
        name: REVIEWED_NPM_METADATA.name,
        version,
        hasInstallScript: true,
        license: REVIEWED_NPM_METADATA.license,
        os: REVIEWED_NPM_METADATA.os,
        bin: REVIEWED_NPM_METADATA.bin,
        engines: REVIEWED_NPM_METADATA.engines,
      },
    },
  };
  if (!isDeepStrictEqual(lock, expectedLock)) {
    fail('package-lock.json must exactly mirror reviewed package metadata');
  }
}

function verifyVersions(root = ROOT) {
  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const initPath = path.join(root, 'src', 'custback', '__init__.py');
  const versions = {
    package: pkg.version,
    lock: lock.version,
    lockRoot: lock.packages && lock.packages[''] && lock.packages[''].version,
    python: projectVersion(pyproject),
  };
  const unique = new Set(Object.values(versions));
  if (unique.size !== 1) fail(`release version mismatch: ${JSON.stringify(versions)}`);
  if (sourceFallbackVersion(initPath) !== pkg.version) {
    fail('source-tree __version__ fallback does not match package version');
  }
  verifyNpmMetadata(root);
  return pkg.version;
}

function verifyDependencies(root = ROOT) {
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const buildSystem = arrayValue(tableBody(pyproject, 'build-system'), 'requires');
  if (JSON.stringify(buildSystem) !== JSON.stringify(REVIEWED_BUILD_REQUIREMENTS)) {
    fail('build-system requirements must be exactly setuptools>=77,<84');
  }
  const core = arrayValue(tableBody(pyproject, 'project'), 'dependencies');
  for (const spec of REVIEWED_CORE_DEPENDENCIES) {
    if (!core.includes(spec)) fail(`missing bounded core dependency: ${spec}`);
  }
  const unexpectedCore = core.filter((spec) => !REVIEWED_CORE_DEPENDENCIES.includes(spec));
  if (unexpectedCore.length || core.length !== REVIEWED_CORE_DEPENDENCIES.length) {
    fail(`unexpected core dependencies: ${unexpectedCore.join(', ') || 'duplicate entries'}`);
  }
  const optional = tableBody(pyproject, 'project.optional-dependencies');
  const optionalKeys = assignmentKeys(optional);
  const reviewedOptionalKeys = Object.keys(REVIEWED_OPTIONAL_DEPENDENCIES);
  if (new Set(optionalKeys).size !== optionalKeys.length ||
      optionalKeys.length !== reviewedOptionalKeys.length ||
      optionalKeys.some((key) => !reviewedOptionalKeys.includes(key))) {
    fail(`optional dependency groups must be exactly ${reviewedOptionalKeys.join(', ')}`);
  }
  for (const [extra, expected] of Object.entries(REVIEWED_OPTIONAL_DEPENDENCIES)) {
    const dependencies = arrayValue(optional, extra);
    const unexpected = dependencies.filter((spec) => !expected.includes(spec));
    const missing = expected.filter((spec) => !dependencies.includes(spec));
    if (unexpected.length || missing.length || dependencies.length !== expected.length) {
      fail(
        `invalid bounded ${extra} dependencies; missing: ${missing.join(', ') || 'none'}; ` +
        `unexpected: ${unexpected.join(', ') || 'none'}`
      );
    }
  }
  const scripts = tableBody(pyproject, 'project.scripts');
  const scriptKeys = assignmentKeys(scripts);
  const expectedScripts = Object.keys(REVIEWED_CONSOLE_SCRIPTS);
  if (scriptKeys.length !== expectedScripts.length ||
      scriptKeys.some((key) => !expectedScripts.includes(key))) {
    fail(`console scripts must be exactly ${expectedScripts.join(', ')}`);
  }
  for (const [name, entrypoint] of Object.entries(REVIEWED_CONSOLE_SCRIPTS)) {
    if (stringValue(scripts, name) !== entrypoint) {
      fail(`console script ${name} must map exactly to ${entrypoint}`);
    }
  }
  if (!pyproject.includes('requires-python = ">=3.10,<3.15"')) {
    fail('Python support range must be >=3.10,<3.15');
  }
  if (/"opencv-python[<>=]/.test(pyproject)) {
    fail('opencv-python conflicts with the selected opencv-contrib-python distribution');
  }
}

function verifyReviewedSourceFiles(root, names) {
  for (const name of names) {
    let current = root;
    for (const part of name.split('/')) {
      current = path.join(current, part);
      let stat;
      try {
        stat = fs.lstatSync(current);
      } catch (err) {
        fail(`reviewed release source is missing: ${name} (${err.message})`);
      }
      if (stat.isSymbolicLink()) fail(`release payload source must not be a symlink: ${name}`);
    }
    if (!fs.lstatSync(current).isFile()) fail(`reviewed release source is not a file: ${name}`);
  }
}

function expectedNpmPayload(root) {
  verifyReviewedSourceFiles(root, REVIEWED_NPM_PAYLOAD);
  return new Set(REVIEWED_NPM_PAYLOAD);
}

function verifyNpmPayload(names, root) {
  const forbidden = names.filter((name) =>
    /(^|\/)\.venv(?:\/|$)/.test(name) || name.includes('custback-generations') ||
    name.startsWith('packaging/npm/test/') || name.endsWith('.tgz') || name.endsWith('.whl') ||
    name.endsWith('.tar.gz') || name.includes('__pycache__') || name.endsWith('.pyc') ||
    /(^|\/)onnxruntime_profile__.*\.json$/.test(name) ||
    name === 'debug.txt' || name === 'uninstall.log');
  if (forbidden.length) fail(`npm artifact contains forbidden files: ${forbidden.join(', ')}`);
  const expected = expectedNpmPayload(root);
  const actual = new Set(names);
  const unexpected = names.filter((name) => !expected.has(name));
  const missing = [...expected].filter((name) => !actual.has(name));
  if (unexpected.length || missing.length) {
    fail(
      `npm artifact manifest mismatch; missing: ${missing.join(', ') || 'none'}; ` +
      `unexpected: ${unexpected.join(', ') || 'none'}`
    );
  }
}

function parseNpmPackPayload(stdout, version) {
  let payload;
  try {
    payload = JSON.parse(stdout);
  } catch (err) {
    fail(`npm pack returned invalid JSON: ${err.message}`);
  }
  if (!Array.isArray(payload) || payload.length !== 1 || !payload[0] ||
      typeof payload[0] !== 'object' || Array.isArray(payload[0])) {
    fail('npm pack returned an unexpected payload');
  }
  const artifact = payload[0];
  const expectedFilename = `custback-${version}.tgz`;
  if (artifact.name !== 'custback' || artifact.version !== version ||
      artifact.filename !== expectedFilename) {
    fail('npm pack artifact identity is inconsistent');
  }
  if (!Array.isArray(artifact.files) || artifact.files.length === 0) {
    fail('npm pack artifact has no file manifest');
  }
  const names = artifact.files.map((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry) ||
        typeof entry.path !== 'string' || entry.path === '' || entry.path === '.' ||
        entry.path.endsWith('/') || entry.path.includes('\\') ||
        path.posix.isAbsolute(entry.path) || path.posix.normalize(entry.path) !== entry.path ||
        entry.path.split('/').includes('..')) {
      fail('npm pack artifact contains an invalid file entry');
    }
    return entry.path;
  });
  if (new Set(names).size !== names.length) {
    fail('npm pack artifact contains duplicate file entries');
  }
  return { artifact, names };
}

function tarString(block, start, length) {
  const end = block.indexOf(0, start);
  const limit = end >= start && end < start + length ? end : start + length;
  return block.subarray(start, limit).toString('utf8');
}

function npmTarballFile(tarball, wanted) {
  let archive;
  try {
    archive = zlib.gunzipSync(fs.readFileSync(tarball));
  } catch (err) {
    fail(`npm artifact is not a readable gzip stream: ${err.message}`);
  }
  const seen = new Set();
  for (let offset = 0; offset + 512 <= archive.length;) {
    const header = archive.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;
    const name = tarString(header, 0, 100);
    const prefix = tarString(header, 345, 155);
    const pathname = prefix ? `${prefix}/${name}` : name;
    const rawSize = tarString(header, 124, 12).trim();
    if (!/^[0-7]+$/.test(rawSize)) {
      fail(`npm artifact has an invalid tar size for ${pathname || '<unnamed>'}`);
    }
    const size = Number.parseInt(rawSize, 8);
    if (!Number.isSafeInteger(size)) fail(`npm artifact tar entry is too large: ${pathname}`);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;
    if (dataEnd > archive.length) fail(`npm artifact tar entry is truncated: ${pathname}`);
    const type = String.fromCharCode(header[156] || 48);
    if (type === '0') {
      if (!pathname || path.posix.isAbsolute(pathname) ||
          path.posix.normalize(pathname) !== pathname ||
          pathname.split('/').includes('..') || seen.has(pathname)) {
        fail(`npm artifact has an unsafe or duplicate tar entry: ${pathname}`);
      }
      seen.add(pathname);
      if (pathname === wanted) return Buffer.from(archive.subarray(dataStart, dataEnd));
    }
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  fail(`npm artifact is missing ${wanted}`);
}

function verifyNpmArtifactLicense(tarball, root = ROOT) {
  const expected = canonicalLicense(root);
  const actual = npmTarballFile(tarball, 'package/LICENSE');
  if (!actual.equals(expected)) {
    fail('npm artifact LICENSE is not byte-identical to the canonical LICENSE');
  }
}

function staleArtifacts(root = ROOT) {
  const staleNames = new Set(['debug.txt', 'uninstall.log']);
  const stale = [];
  for (const name of fs.readdirSync(root)) {
    const full = path.join(root, name);
    if (staleNames.has(name) || /^onnxruntime_profile__.*\.json$/.test(name) ||
        name.endsWith('.tgz') || name.endsWith('.whl') ||
        name.endsWith('.tar.gz') || (['build', 'dist'].includes(name) && fs.statSync(full).isDirectory())) {
      stale.push(name);
    }
  }
  const sourceRoot = path.join(root, 'src');
  if (fs.existsSync(sourceRoot)) {
    for (const name of fs.readdirSync(sourceRoot)) {
      if (name.endsWith('.egg-info')) stale.push(path.join('src', name));
    }
  }
  return stale.sort();
}

function verifyDocs(root = ROOT) {
  const readme = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
  if (/custback-\d+\.\d+\.\d+\.tgz/.test(readme)) {
    fail('README hard-codes a versioned npm tarball');
  }
  if (!readme.includes('TARBALL=$(npm pack --silent)')) {
    fail('README local npm tarball capture must use npm pack --silent');
  }
  if (!readme.includes('(docs/remote-deployment.md)')) {
    fail('README must link the two-host remote deployment guide');
  }
  const deployment = fs.readFileSync(
    path.join(root, 'docs', 'remote-deployment.md'), 'utf8',
  );
  const requirements = [
    [/renderer-scoped token/i, 'renderer-scoped token'],
    [/avatar-control token/i, 'avatar-control token'],
    [/\bwss:\/\//i, 'WSS renderer endpoint'],
    [/\bhttps:\/\//i, 'HTTPS avatar-control endpoint'],
    [/source\.tls_ca_file/, 'renderer CA configuration'],
    [/avatar\.tls_ca_file/, 'avatar-control CA configuration'],
    [/firewall rule/i, 'firewall direction'],
    [/\brotation\b/i, 'credential rotation'],
    [/privacy slate/i, 'renderer-outage privacy behavior'],
    [/avatar_auth_failed/, 'control-token failure behavior'],
    [/avatar_unreachable/, 'control-plane outage behavior'],
  ];
  for (const [pattern, description] of requirements) {
    if (!pattern.test(deployment)) {
      fail(`remote deployment guide is missing ${description}`);
    }
  }
}

function verifyCiWorkflow(root = ROOT) {
  const workflow = fs.readFileSync(path.join(root, '.github', 'workflows', 'ci.yml'), 'utf8');
  const uses = [...workflow.matchAll(/^\s*-\s+uses:\s+([^\s#]+)/gm)]
    .map((match) => match[1]);
  if (!uses.length || uses.some((action) => !REVIEWED_ACTIONS.has(action))) {
    fail(`CI actions must use the reviewed commit SHA pins: ${uses.join(', ') || 'none'}`);
  }
  if (/\bnpm\s+install\b/.test(workflow) || !/\bnpm\s+ci\b/.test(workflow)) {
    fail('CI must use npm ci and must not use npm install');
  }
}

function verifyPlatformScope(root = ROOT) {
  const scriptPath = path.join(root, 'scripts', 'install_linux.sh');
  const source = fs.readFileSync(scriptPath, 'utf8');
  const guard = source.search(/unsupported Linux distribution/i);
  const identity = source.search(/DISTRO_ID=/);
  const identityLike = source.search(/DISTRO_ID_LIKE=/);
  const apt = source.search(/\bapt-get\b/);
  if (guard < 0 || identity < 0 || identityLike < 0 || apt < 0 ||
      guard > apt || identity > apt || identityLike > apt ||
      !/ubuntu\|debian/.test(source)) {
    fail('Linux installer must allow only Ubuntu/Debian before apt-get');
  }

  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-platform-check-'));
  try {
    const releasePath = path.join(fixture, 'os-release');
    const markerPath = path.join(fixture, 'mutation-attempted');
    const fakeBin = path.join(fixture, 'bin');
    fs.mkdirSync(fakeBin);
    fs.writeFileSync(releasePath, 'ID=fedora\nID_LIKE="rhel centos"\n');
    const fakeSudo = path.join(fakeBin, 'sudo');
    fs.writeFileSync(fakeSudo, '#!/bin/sh\n: > "$CUSTBACK_MUTATION_MARKER"\nexit 99\n');
    fs.chmodSync(fakeSudo, 0o700);
    const result = spawnSync('/bin/bash', [scriptPath], {
      encoding: 'utf8',
      timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
      env: {
        ...process.env,
        CUSTBACK_OS_RELEASE_FILE: releasePath,
        CUSTBACK_MUTATION_MARKER: markerPath,
        PATH: `${fakeBin}${path.delimiter}${process.env.PATH || ''}`,
      },
    });
    if ((result.error && result.status === null) || result.status === 0 ||
        !/unsupported Linux distribution/i.test(result.stderr || '') ||
        fs.existsSync(markerPath)) {
      fail('Linux installer did not reject an unsupported distro before mutation');
    }
  } finally {
    fs.rmSync(fixture, { recursive: true, force: true });
  }
}

function verifyPack(version, root = ROOT) {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-npm-cache-'));
  try {
    const result = spawnSync(
      'npm',
      ['pack', '--dry-run', '--json', '--ignore-scripts', '--cache', cache],
      { cwd: root, encoding: 'utf8', timeout: Math.min(COMMAND_TIMEOUT_MS, 2 * 60 * 1000) },
    );
    if (result.error) fail(`npm pack dry-run could not start: ${result.error.message}`);
    if (result.status !== 0) fail(`npm pack dry-run failed: ${(result.stderr || '').trim()}`);
    const { names } = parseNpmPackPayload(result.stdout, version);
    for (const required of [
      'LICENSE',
      'README.md',
      'docs/remote-deployment.md',
      'package.json',
      'packaging/npm/custback.js',
      'packaging/npm/install.js',
      'packaging/npm/managed-venv.js',
      'config/default.yaml',
      'pyproject.toml',
      'scripts/install_linux.sh',
      'scripts/install_macos.sh',
      'scripts/release/verify-release.js',
      'src/custback/__init__.py',
      'src/custback/__main__.py',
    ]) {
      if (!names.includes(required)) fail(`npm artifact is missing ${required}`);
    }
    verifyNpmPayload(names, root);
  } finally {
    fs.rmSync(cache, { recursive: true, force: true });
  }
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
    timeout: COMMAND_TIMEOUT_MS,
    ...options,
  });
  if (result.error) fail(`${command} could not start: ${result.error.message}`);
  if (result.status !== 0) {
    fail(
      `${command} ${args.join(' ')} failed:\n${(result.stdout || '').trim()}\n${(result.stderr || '').trim()}`
    );
  }
  return result;
}

function extraArtifactProfiles(platform = process.platform, arch = process.arch) {
  const profiles = [
    {
      name: 'mediapipe',
      extras: ['mediapipe'],
      probe: 'import importlib.metadata as m; import mediapipe; m.version("mediapipe")',
    },
    {
      name: 'rvm',
      extras: ['rvm'],
      probe: 'import importlib.metadata as m; import onnxruntime; m.version("onnxruntime")',
    },
    {
      name: 'audio2face',
      extras: ['audio2face', 'dev'],
      test: 'tests/test_audio2face_protocol.py',
    },
    {
      name: 'dev',
      extras: ['dev'],
      probe: 'import importlib.metadata as m; import build, pytest; m.version("httpx2")',
    },
  ];
  // onnxruntime-gpu publishes Linux x86-64 wheels. Generic CI runners can
  // validate resolution and metadata without claiming CUDA execution.
  if (platform === 'linux' && arch === 'x64') {
    profiles.splice(2, 0, {
      name: 'gpu',
      extras: ['gpu'],
      probe: 'import importlib.metadata as m; m.version("onnxruntime-gpu")',
    });
  }
  return profiles;
}

function installAndProbeExtra(python, temporaryRoot, artifact, profile, source) {
  const venv = path.join(temporaryRoot, `${profile.name}-artifact-venv`);
  return withDisposableDirectory(venv, () => {
    runChecked(python, ['-m', 'venv', venv]);
    const venvPython = path.join(venv, 'bin', 'python');
    const spec = `${artifact}[${profile.extras.join(',')}]`;
    runChecked(venvPython, [
      '-m', 'pip', 'install', '--disable-pip-version-check', spec,
    ]);
    runChecked(venvPython, ['-m', 'pip', 'check']);
    if (profile.probe) runChecked(venvPython, ['-c', profile.probe]);
    if (profile.test) {
      runChecked(venvPython, [
        '-m', 'pytest', '-q', path.join(source, profile.test),
      ], { cwd: source });
    }
  });
}

function stageCleanSource(root, destination) {
  const excludedNames = new Set([
    '.agents', '.codex', '.git', '.venv', '.pytest_cache', 'build', 'dist', '__pycache__',
    'debug.txt', 'uninstall.log',
  ]);
  fs.cpSync(root, destination, {
    recursive: true,
    filter(source) {
      const relative = path.relative(root, source);
      if (relative === '') return true;
      const parts = relative.split(path.sep);
      if (parts.some((part) => excludedNames.has(part) || part.endsWith('.egg-info') ||
          part.endsWith('.custback-generations'))) return false;
      if (relative.endsWith('.tgz') || relative.endsWith('.whl') ||
          relative.endsWith('.tar.gz') || relative.endsWith('.pyc') ||
          /(^|[\\/])onnxruntime_profile__.*\.json$/.test(relative)) return false;
      return true;
    },
  });
}

function verifyPythonArtifacts(version, temporaryRoot, root = ROOT) {
  const python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
  const source = path.join(temporaryRoot, 'source');
  const output = path.join(temporaryRoot, 'python-dist');
  const buildTools = path.join(temporaryRoot, 'build-tools-venv');
  verifyReviewedSourceFiles(root, [
    'LICENSE',
    'README.md',
    'pyproject.toml',
    ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
    ...REVIEWED_PYTHON_TESTS,
  ]);
  stageCleanSource(root, source);
  fs.mkdirSync(output);
  withDisposableDirectory(buildTools, () => {
    runChecked(python, ['-m', 'venv', buildTools]);
    const buildPython = path.join(buildTools, 'bin', 'python');
    runChecked(buildPython, [
      '-m', 'pip', 'install', '--disable-pip-version-check',
      'pip>=23,<27', 'build>=1.2,<2',
    ]);
    runChecked(buildPython, [
      '-m', 'build', '--sdist', '--wheel', '--outdir', output, source,
    ]);
  });
  const files = fs.readdirSync(output);
  const wheels = files.filter((name) => name.endsWith('.whl'));
  const sdists = files.filter((name) => name.endsWith('.tar.gz'));
  if (files.length !== 2 || wheels.length !== 1 || sdists.length !== 1) {
    fail(`Python build produced an unexpected artifact set: ${files.join(', ')}`);
  }
  const [wheel] = wheels;
  const [sdist] = sdists;
  const normalized = version.replace(/-/g, '_');
  const escaped = normalized.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const wheelPattern = new RegExp(`^custback-${escaped}-(?:\\d+-)?[^-]+-[^-]+-[^-]+\\.whl$`);
  if (!wheelPattern.test(wheel) || sdist !== `custback-${version}.tar.gz`) {
    fail(`Python artifact filenames do not match ${version}: ${wheel}, ${sdist}`);
  }

  const inspectCode = `
import configparser, email.parser, io, json, pathlib, re, sys, tarfile, zipfile
(
    version, wheel_path, sdist_path, module_json, test_json,
    core_json, optional_json, scripts_json, license_path,
) = sys.argv[1:]
bad = ("/.venv/", "__pycache__", ".pyc", ".tgz", "/debug.txt", "/uninstall.log", "/build/")
source_modules = set(json.loads(module_json))
source_tests = set(json.loads(test_json))
core_dependencies = json.loads(core_json)
optional_dependencies = json.loads(optional_json)
console_scripts = json.loads(scripts_json)
license_bytes = pathlib.Path(license_path).read_bytes()

def require(condition, detail):
    if not condition:
        raise RuntimeError(repr(detail))

def normalize_requirement(value):
    requirement, separator, marker = value.partition(";")
    match = re.fullmatch(r"\\s*([A-Za-z0-9_.-]+)\\s*(.*?)\\s*", requirement)
    require(match, value)
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    specifier = match.group(2).strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1]
    specs = tuple(sorted(part.replace(" ", "") for part in specifier.split(",") if part.strip()))
    extra = ""
    if separator:
        marker_match = re.fullmatch(
            r"\\s*extra\\s*==\\s*['\\\"]([A-Za-z0-9_.-]+)['\\\"]\\s*",
            marker,
        )
        require(marker_match, value)
        extra = marker_match.group(1)
    return name, specs, extra

expected_requires = {normalize_requirement(item) for item in core_dependencies}
for extra, dependencies in optional_dependencies.items():
    expected_requires |= {
        normalize_requirement(f"{item}; extra == '{extra}'") for item in dependencies
    }

def verify_metadata(text):
    metadata = email.parser.Parser().parsestr(text)
    require(metadata["Name"].lower() == "custback", metadata["Name"])
    require(metadata["Version"] == version, metadata["Version"])
    require(metadata.get_all("License-File", []) == ["LICENSE"], metadata.items())
    requires_python = {
        item.strip() for item in metadata["Requires-Python"].split(",")
    }
    require(requires_python == {">=3.10", "<3.15"}, metadata["Requires-Python"])
    provides = metadata.get_all("Provides-Extra", [])
    require(len(provides) == len(set(provides)), provides)
    require(set(provides) == set(optional_dependencies), provides)
    requires = metadata.get_all("Requires-Dist", [])
    normalized = [normalize_requirement(item) for item in requires]
    require(len(normalized) == len(set(normalized)), requires)
    require(set(normalized) == expected_requires, {
        "missing": sorted(expected_requires - set(normalized)),
        "unexpected": sorted(set(normalized) - expected_requires),
    })

def verify_entry_points(text):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_file(io.StringIO(text))
    require(parser.sections() == ["console_scripts"], parser.sections())
    require(dict(parser.items("console_scripts")) == console_scripts, dict(
        parser.items("console_scripts")
    ))

with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    require(len(names) == len(set(names)), names)
    require(not [n for n in names if any(x in "/" + n for x in bad)], names)
    require(not [n for n in names if n.endswith("/")], names)
    dist_info = f"custback-{version}.dist-info"
    expected = source_modules | {
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/top_level.txt",
        f"{dist_info}/RECORD",
    }
    require(set(names) == expected, {
        "missing": sorted(expected - set(names)),
        "unexpected": sorted(set(names) - expected),
    })
    metadata = f"{dist_info}/METADATA"
    require(archive.read(f"{dist_info}/licenses/LICENSE") == license_bytes, "wheel LICENSE")
    text = archive.read(metadata).decode()
    require(f"Version: {version}\\n" in text, text[:500])
    verify_metadata(text)
    verify_entry_points(archive.read(f"{dist_info}/entry_points.txt").decode())
with tarfile.open(sdist_path, "r:gz") as archive:
    members = archive.getmembers()
    member_names = [m.name.rstrip("/") for m in members]
    require(len(member_names) == len(set(member_names)), member_names)
    require(not [m.name for m in members if not (m.isfile() or m.isdir())], [m.name for m in members])
    names = {m.name for m in members if m.isfile()}
    require(not [n for n in names if any(x in "/" + n for x in bad)], sorted(names))
    prefix = f"custback-{version}"
    expected = {
        f"{prefix}/LICENSE",
        f"{prefix}/PKG-INFO",
        f"{prefix}/README.md",
        f"{prefix}/pyproject.toml",
        f"{prefix}/setup.cfg",
        f"{prefix}/src/custback.egg-info/PKG-INFO",
        f"{prefix}/src/custback.egg-info/SOURCES.txt",
        f"{prefix}/src/custback.egg-info/dependency_links.txt",
        f"{prefix}/src/custback.egg-info/entry_points.txt",
        f"{prefix}/src/custback.egg-info/requires.txt",
        f"{prefix}/src/custback.egg-info/top_level.txt",
    }
    expected |= {f"{prefix}/src/{name}" for name in source_modules}
    expected |= {f"{prefix}/{name}" for name in source_tests}
    require(names == expected, {
        "missing": sorted(expected - names),
        "unexpected": sorted(names - expected),
    })
    allowed_directories = {prefix}
    for name in expected:
        parent = pathlib.PurePosixPath(name).parent
        while str(parent) != ".":
            allowed_directories.add(str(parent))
            parent = parent.parent
    actual_directories = {m.name.rstrip("/") for m in members if m.isdir()}
    require(not (actual_directories - allowed_directories), sorted(
        actual_directories - allowed_directories
    ))
    pkg_info = f"{prefix}/PKG-INFO"
    require(archive.extractfile(f"{prefix}/LICENSE").read() == license_bytes, "sdist LICENSE")
    text = archive.extractfile(pkg_info).read().decode()
    require(f"Version: {version}\\n" in text, text[:500])
    verify_metadata(text)
    egg_info = f"{prefix}/src/custback.egg-info"
    verify_metadata(archive.extractfile(f"{egg_info}/PKG-INFO").read().decode())
    verify_entry_points(archive.extractfile(f"{egg_info}/entry_points.txt").read().decode())
`;
  runChecked(python, [
    '-c', inspectCode, version, path.join(output, wheel), path.join(output, sdist),
    JSON.stringify(REVIEWED_PYTHON_MODULES), JSON.stringify(REVIEWED_PYTHON_TESTS),
    JSON.stringify(REVIEWED_CORE_DEPENDENCIES),
    JSON.stringify(REVIEWED_OPTIONAL_DEPENDENCIES),
    JSON.stringify(REVIEWED_CONSOLE_SCRIPTS),
    path.join(root, 'LICENSE'),
  ]);

  const importProbe = [
    'import importlib.metadata as m',
    'import custback, custback.api.server, cv2, fastapi, numpy, pydantic, PIL, pyvirtualcam, uvicorn, websockets, yaml',
    `expected = ${JSON.stringify(version)}`,
    'metadata_version = m.version("custback")',
    'if metadata_version != expected:\n    raise RuntimeError(f"metadata version {metadata_version!r} != {expected!r}")',
    'if custback.__version__ != expected:\n    raise RuntimeError(f"source version {custback.__version__!r} != {expected!r}")',
  ].join('\n');
  for (const [kind, artifact] of [
    ['wheel', path.join(output, wheel)],
    ['sdist', path.join(output, sdist)],
  ]) {
    const venv = path.join(temporaryRoot, `${kind}-smoke-venv`);
    withDisposableDirectory(venv, () => {
      runChecked(python, ['-m', 'venv', venv]);
      const venvPython = path.join(venv, 'bin', 'python');
      runChecked(venvPython, [
        '-m', 'pip', 'install', '--disable-pip-version-check', artifact,
      ]);
      runChecked(venvPython, ['-m', 'pip', 'check']);
      runChecked(venvPython, ['-c', importProbe]);
      runChecked(venvPython, ['-m', 'custback', '--help']);
    });
  }

  const audio2faceProfile = extraArtifactProfiles()
    .find((profile) => profile.name === 'audio2face');
  if (!audio2faceProfile) fail('audio2face artifact profile is missing');
  const sourceProfile = { ...audio2faceProfile, name: 'audio2face-source' };
  installAndProbeExtra(python, temporaryRoot, source, sourceProfile, source);
  for (const profile of extraArtifactProfiles()) {
    installAndProbeExtra(
      python, temporaryRoot, path.join(output, wheel), profile, source,
    );
  }
  return { wheel: path.join(output, wheel), sdist: path.join(output, sdist) };
}

function verifyNpmArtifactInstall(version, temporaryRoot, root = ROOT) {
  const packDirectory = path.join(temporaryRoot, 'npm-pack');
  const cache = path.join(temporaryRoot, 'npm-cache');
  const prefix = path.join(temporaryRoot, 'npm-prefix');
  const managedVenv = path.join(temporaryRoot, 'npm-managed-venv');
  fs.mkdirSync(packDirectory);
  const packed = runChecked('npm', [
    'pack', '--json', '--ignore-scripts', '--pack-destination', packDirectory, '--cache', cache,
  ], { cwd: root });
  const { artifact, names } = parseNpmPackPayload(packed.stdout, version);
  verifyNpmPayload(names, root);
  const tarball = path.join(packDirectory, artifact.filename);
  verifyNpmArtifactLicense(tarball, root);
  const smokeEnv = {
    ...process.env,
    CUSTBACK_VENV: managedVenv,
    CUSTBACK_EXTRAS: '',
    CUSTBACK_SKIP_INSTALL: '0',
    CUSTBACK_FORCE_REBUILD: '0',
  };
  runChecked('npm', [
    'install', '--global', tarball, '--prefix', prefix, '--cache', cache,
  ], {
    cwd: temporaryRoot,
    env: smokeEnv,
  });
  const launcher = path.join(prefix, 'bin', 'custback');
  runChecked(launcher, ['--help'], { env: smokeEnv });
  runChecked(launcher, ['rebuild'], { env: smokeEnv });
  runChecked(launcher, ['doctor'], { env: smokeEnv });
  const stamp = JSON.parse(fs.readFileSync(path.join(managedVenv, managed.INSTALL_STAMP), 'utf8'));
  if (!installer.validInstallStamp(stamp) || stamp.packageVersion !== version ||
      stamp.sourceDigest !== installer.sourceDigest(root) || stamp.requestedExtras.length !== 0) {
    fail('npm smoke install stamp is inconsistent with the artifact');
  }
}

function verifyBuiltArtifacts(version, root = ROOT) {
  const buildJobs = releaseBuildJobs();
  const temporaryRoot = createReleaseTemporaryRoot({ root });
  try {
    return withTemporaryEnvironment({
      TMPDIR: temporaryRoot,
      TMP: temporaryRoot,
      TEMP: temporaryRoot,
      CMAKE_BUILD_PARALLEL_LEVEL: buildJobs,
      GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS: buildJobs,
      MAKEFLAGS: `-j${buildJobs}`,
      MAX_JOBS: buildJobs,
    }, () => {
      verifyPythonArtifacts(version, temporaryRoot, root);
      verifyNpmArtifactInstall(version, temporaryRoot, root);
    });
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
}

function releasePlan(argv) {
  const allowed = new Set(['--prepack', '--quick']);
  const unknown = argv.filter((arg) => !allowed.has(arg));
  if (unknown.length || new Set(argv).size !== argv.length ||
      (argv.includes('--prepack') && argv.includes('--quick'))) {
    fail(`invalid release verification arguments: ${argv.join(' ') || '<none>'}`);
  }
  const prepack = argv.includes('--prepack');
  const quick = argv.includes('--quick');
  return {
    builtArtifacts: !prepack && !quick,
    success: prepack
      ? 'metadata and non-recursive npm packlist verified'
      : quick
        ? 'metadata and npm packlist verified (quick mode)'
        : 'metadata, npm packlist, and built artifacts verified',
  };
}

function main(argv = process.argv.slice(2)) {
  try {
    const plan = releasePlan(argv);
    const version = verifyVersions();
    verifyDependencies();
    verifyLicenseMetadata();
    verifyDocs();
    verifyCiWorkflow();
    verifyPlatformScope();
    verifyNoReleaseBlockers();
    const stale = staleArtifacts();
    if (stale.length) {
      fail(`stale release artifacts must be removed before release: ${stale.join(', ')}`);
    }
    // --ignore-scripts makes this safe to call from prepack without invoking
    // the prepack lifecycle recursively. Artifact builds/installs are reserved
    // for the explicit, full release:check path.
    verifyPack(version);
    if (plan.builtArtifacts) verifyBuiltArtifacts(version);
    console.log(`[custback release] ${plan.success} for ${version}`);
    return 0;
  } catch (err) {
    console.error(`[custback release] ${err.message}`);
    return 1;
  }
}

module.exports = {
  canonicalLicense,
  createReleaseTemporaryRoot,
  extraArtifactProfiles,
  filesystemIsMemoryBacked,
  main,
  parseNpmPackPayload,
  projectVersion,
  releaseBuildJobs,
  releasePlan,
  remediationBlockers,
  staleArtifacts,
  verifyDependencies,
  verifyDocs,
  verifyBuiltArtifacts,
  verifyBlockerRegressionCoverage,
  verifyCiWorkflow,
  verifyLicenseMetadata,
  verifyNpmArtifactLicense,
  verifyNpmArtifactInstall,
  verifyNpmMetadata,
  verifyNoReleaseBlockers,
  verifyPack,
  verifyPlatformScope,
  verifyPythonArtifacts,
  verifyVersions,
  withDisposableDirectory,
  withTemporaryEnvironment,
};

if (require.main === module) process.exit(main());
