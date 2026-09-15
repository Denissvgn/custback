#!/usr/bin/env node
/** Artifact-only, non-authorizing Phase 6 migration qualification. */

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { isDeepStrictEqual } = require('node:util');

const phase6 = require('./phase6-evidence');

const ROOT = path.resolve(__dirname, '..', '..');
const CANDIDATE_VERSION = JSON.parse(
  fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'),
).version;
const REFERENCE_COMMIT = '099001e9f25a3ea1d0c820b066a6318e4172d549';
const REFERENCE_VERSION = '0.3.0';
const REPORT_ID = 'custback-phase6-migration-qualification-v1';
const COMMAND_TIMEOUT_MS = 30 * 60 * 1000;
const SHA256_RE = /^[0-9a-f]{64}$/;

const INPUT_DEFINITIONS = Object.freeze([
  {
    option: 'wheel',
    id: 'python-wheel',
    role: 'candidate',
    filename: `custback-${CANDIDATE_VERSION}-py3-none-any.whl`,
  },
  {
    option: 'sdist',
    id: 'python-sdist',
    role: 'candidate',
    filename: `custback-${CANDIDATE_VERSION}.tar.gz`,
  },
  {
    option: 'npm',
    id: 'npm-tarball',
    role: 'candidate',
    filename: `custback-${CANDIDATE_VERSION}.tgz`,
  },
  {
    option: 'reference-wheel',
    id: 'reference-0.3.0-wheel',
    role: 'reference',
    filename: 'custback-0.3.0-py3-none-any.whl',
  },
  {
    option: 'reference-sdist',
    id: 'reference-0.3.0-sdist',
    role: 'reference',
    filename: 'custback-0.3.0.tar.gz',
  },
  {
    option: 'reference-npm',
    id: 'reference-0.3.0-npm',
    role: 'reference',
    filename: 'custback-0.3.0.tgz',
  },
]);

const SCENARIOS = Object.freeze([
  {
    id: 'candidate-wheel-clean-install',
    fixtures: [],
    checks: ['isolated-venv', 'installed-version', 'pip-check', 'no-source-import-path'],
  },
  {
    id: 'candidate-sdist-clean-install',
    fixtures: [],
    checks: ['isolated-venv', 'installed-version', 'pip-check', 'no-source-import-path'],
  },
  {
    id: 'reference-wheel-upgrade',
    fixtures: ['reference-0.3.0-wheel'],
    checks: ['reference-version', 'candidate-upgrade', 'pip-check'],
  },
  {
    id: 'reference-sdist-upgrade',
    fixtures: ['reference-0.3.0-sdist'],
    checks: ['reference-version', 'candidate-upgrade', 'pip-check'],
  },
  {
    id: 'config-local-in-place-migration',
    fixtures: ['config-camera-device-local'],
    checks: ['atomic-config', 'private-backup', 'immutable-target', 'idempotent-retry'],
  },
  {
    id: 'config-remote-explicit-refusal',
    fixtures: ['config-camera-device-remote'],
    checks: ['operator-action-required', 'zero-writes', 'secret-redaction'],
  },
  {
    id: 'core-avatar-store-repair',
    fixtures: ['core-store-pre-ownership-ledger', 'avatar-store-pre-ownership-ledger'],
    checks: ['assets-preserved', 'private-modes', 'quota-ownership-preserved'],
  },
  {
    id: 'python-reinstall-uninstall-reinstall',
    fixtures: [],
    checks: ['force-reinstall', 'uninstall', 'reinstall', 'external-state-preserved'],
  },
  {
    id: 'candidate-npm-clean-install',
    fixtures: [],
    checks: ['isolated-prefix', 'managed-generation', 'doctor', 'no-source-import-path'],
  },
  {
    id: 'npm-artifact-migration-contracts',
    fixtures: [
      'npm-legacy-text-stamp-package-local',
      'npm-install-stamp-v2-package-local',
      'npm-install-stamp-v3-package-local',
      'npm-promotion-journal-v1-prepared',
      'npm-promotion-journal-v1-committed',
    ],
    checks: ['artifact-tests', 'durable-boundaries', 'unsafe-links-refused'],
  },
  {
    id: 'reference-npm-preupgrade-bridge',
    fixtures: ['reference-0.3.0-npm'],
    checks: ['reference-layout', 'preupgrade-bridge', 'extras-intent'],
  },
  {
    id: 'npm-package-directory-replacement',
    fixtures: ['npm-prefix-generations-v1'],
    checks: ['active-generation', 'rollback-generation', 'journal-recovery', 'extras-preserved'],
  },
  {
    id: 'npm-retry-rollback',
    fixtures: [],
    checks: ['failed-rebuild', 'active-rollback', 'zero-surviving-journals'],
  },
  {
    id: 'npm-reinstall-uninstall-reinstall',
    fixtures: [],
    checks: ['uninstall', 'durable-state', 'reinstall', 'extras-preserved'],
  },
]);

function fail(message) {
  throw new Error(message);
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value) &&
    Object.getPrototypeOf(value) === Object.prototype;
}

function exactKeys(value, expected, label) {
  if (!isPlainObject(value)) fail(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${label} must contain exactly: ${wanted.join(', ')}`);
  }
}

function exactIds(actual, expected, label) {
  if (!Array.isArray(actual) || actual.some((id) => typeof id !== 'string') ||
      new Set(actual).size !== actual.length ||
      JSON.stringify([...actual].sort()) !== JSON.stringify([...expected].sort())) {
    fail(`${label} does not match the reviewed contract`);
  }
}

function lstatOrNull(filename) {
  try {
    return fs.lstatSync(filename);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

function sha256File(filename) {
  const before = fs.lstatSync(filename);
  if (!before.isFile() || before.isSymbolicLink()) {
    fail(`artifact must be a regular non-symlink file: ${filename}`);
  }
  const descriptor = fs.openSync(
    filename,
    fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0),
  );
  try {
    const opened = fs.fstatSync(descriptor);
    if (opened.dev !== before.dev || opened.ino !== before.ino) {
      fail(`artifact changed while opening: ${filename}`);
    }
    const hash = crypto.createHash('sha256');
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let position = 0;
    while (true) {
      const count = fs.readSync(descriptor, buffer, 0, buffer.length, position);
      if (count === 0) break;
      hash.update(buffer.subarray(0, count));
      position += count;
    }
    const after = fs.lstatSync(filename);
    if (after.dev !== opened.dev || after.ino !== opened.ino ||
        after.size !== opened.size) {
      fail(`artifact changed while hashing: ${filename}`);
    }
    return {
      sha256: hash.digest('hex'),
      size: opened.size,
      identity: `${opened.dev}:${opened.ino}`,
    };
  } finally {
    fs.closeSync(descriptor);
  }
}

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (!argument.startsWith('--') || index + 1 >= argv.length) {
      fail('migration qualification requires named artifact and report paths');
    }
    const name = argument.slice(2);
    if (![...INPUT_DEFINITIONS.map((entry) => entry.option), 'report', 'matrix-id'].includes(name) ||
        Object.hasOwn(options, name)) {
      fail(`unknown or duplicate migration qualification option: ${argument}`);
    }
    const value = argv[index += 1];
    if (name === 'matrix-id') {
      if (!/^[a-z0-9]+(?:[._-][a-z0-9]+)*$/.test(value)) {
        fail('--matrix-id is invalid');
      }
      options[name] = value;
    } else {
      if (!path.isAbsolute(value)) fail(`${argument} must be an absolute path`);
      options[name] = path.resolve(value);
    }
  }
  for (const name of [
    ...INPUT_DEFINITIONS.map((entry) => entry.option), 'report', 'matrix-id',
  ]) {
    if (!options[name]) fail(`missing required migration qualification option: --${name}`);
  }
  return options;
}

function detectRuntime(python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3') {
  const result = runChecked(python, [
    '-I', '-c', 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")',
  ]);
  return {
    platform: process.platform,
    python: result.stdout.trim(),
    node: process.versions.node.split('.')[0],
  };
}

function validateMigrationRuntime(options, manifest) {
  const definition = manifest.matrices.migration.find(
    (entry) => entry.id === options['matrix-id'],
  );
  if (!definition) fail('migration matrix id is not present in the reviewed manifest');
  const runtime = options.runtime || detectRuntime(options.python);
  const expectedPlatform = definition.os === 'macos' ? 'darwin' : 'linux';
  if (runtime.platform !== expectedPlatform || runtime.python !== definition.python ||
      runtime.node !== definition.node) {
    fail(
      `migration runtime does not match ${definition.id}: ` +
      `${runtime.platform}/python-${runtime.python}/node-${runtime.node}`,
    );
  }
  return { ...definition, platform: runtime.platform };
}

function validateArtifactInputs(options, manifest = phase6.loadManifest()) {
  phase6.validateManifest(manifest);
  const candidateDefinitions = new Map(manifest.artifacts.map((entry) => [entry.id, entry]));
  const fixtureIds = new Set(manifest.legacy_fixtures.map((entry) => entry.id));
  const records = [];
  for (const definition of INPUT_DEFINITIONS) {
    const filename = path.resolve(options[definition.option] || '');
    if (!path.isAbsolute(options[definition.option] || '')) {
      fail(`--${definition.option} must be an absolute path`);
    }
    const basename = path.basename(filename);
    const manifestPattern = candidateDefinitions.get(definition.id)?.filename_pattern;
    if (definition.role === 'candidate' &&
        (!manifestPattern || !new RegExp(manifestPattern).test(basename))) {
      fail(`artifact filename does not match ${definition.id}: ${basename}`);
    }
    if (basename !== definition.filename) {
      fail(`artifact filename must be exactly ${definition.filename}: ${basename}`);
    }
    if (definition.role === 'reference' && !fixtureIds.has(definition.id)) {
      fail(`reference artifact id is absent from the migration fixture manifest: ${definition.id}`);
    }
    const digest = sha256File(filename);
    if (digest.size <= 0) fail(`artifact is empty: ${filename}`);
    records.push({
      id: definition.id,
      role: definition.role,
      filename: basename,
      sha256: digest.sha256,
      size: digest.size,
      _path: filename,
      _identity: digest.identity,
    });
  }
  if (new Set(records.map((entry) => entry._identity)).size !== records.length) {
    fail('migration qualification artifacts must be distinct files');
  }
  if (new Set(records.map((entry) => entry.sha256)).size !== records.length) {
    fail('migration qualification artifacts must have distinct SHA-256 digests');
  }
  return records;
}

function sanitizedEnvironment(temporaryRoot, base = process.env) {
  const environment = { ...base };
  for (const name of [
    'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'NODE_PATH', 'NODE_OPTIONS',
    'PYTHONINSPECT', 'PYTHONSTARTUP', 'NPM_CONFIG_PREFIX', 'npm_config_prefix',
    'NPM_CONFIG_USERCONFIG', 'npm_config_userconfig', 'CUSTBACK_VENV',
    'CUSTBACK_EXTRAS', 'CUSTBACK_SKIP_INSTALL', 'CUSTBACK_FORCE_REBUILD',
    'CUSTBACK_INSTALL_TIMEOUT_MS', 'CUSTBACK_DOCTOR_TIMEOUT_MS',
    'CUSTBACK_CUDA_PROBE_TIMEOUT_MS',
  ]) delete environment[name];
  const home = path.join(temporaryRoot, 'home');
  const cache = path.join(temporaryRoot, 'cache');
  const tmp = path.join(temporaryRoot, 'tmp');
  for (const directory of [home, cache, tmp]) {
    fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  }
  Object.assign(environment, {
    HOME: home,
    XDG_CONFIG_HOME: path.join(home, '.config'),
    XDG_CACHE_HOME: cache,
    PIP_CACHE_DIR: path.join(cache, 'pip'),
    PIP_CONFIG_FILE: os.devNull,
    PYTHONNOUSERSITE: '1',
    PYTHONSAFEPATH: '1',
    PIP_DISABLE_PIP_VERSION_CHECK: '1',
    PIP_NO_INPUT: '1',
    TMPDIR: tmp,
    TMP: tmp,
    TEMP: tmp,
    PWD: temporaryRoot,
    NPM_CONFIG_USERCONFIG: path.join(home, '.npmrc'),
    CUSTBACK_RELEASE_TMPDIR: temporaryRoot,
  });
  return environment;
}

function createTemporaryRoot(base = process.env.CUSTBACK_RELEASE_TMPDIR || os.tmpdir()) {
  const resolvedBase = fs.realpathSync(path.resolve(base));
  const metadata = fs.lstatSync(resolvedBase);
  if (!metadata.isDirectory()) fail('migration qualification temporary base is not a directory');
  const relative = path.relative(fs.realpathSync(ROOT), resolvedBase);
  if (!relative || (!relative.startsWith(`..${path.sep}`) && relative !== '..' &&
      !path.isAbsolute(relative))) {
    fail('migration qualification temporary base must be outside the source checkout');
  }
  const temporaryRoot = fs.mkdtempSync(path.join(resolvedBase, 'custback-migrations-'));
  fs.chmodSync(temporaryRoot, 0o700);
  return temporaryRoot;
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    env: options.env,
    encoding: 'utf8',
    maxBuffer: 32 * 1024 * 1024,
    timeout: options.timeout || COMMAND_TIMEOUT_MS,
  });
  // Some hardened hosts report a post-spawn EPERM alongside a completed child
  // status. A real launch failure has no numeric exit status; never discard a
  // completed child's result merely because that advisory error is present.
  if (result.error && !Number.isInteger(result.status)) {
    fail(`${command} could not start: ${result.error.message}`);
  }
  const expected = options.expectedStatus ?? 0;
  if (result.status !== expected) {
    fail(
      `${command} ${args.join(' ')} exited ${result.status}; expected ${expected}:\n` +
      `${(result.stdout || '').trim()}\n${(result.stderr || '').trim()}`,
    );
  }
  return result;
}

function assertMode(filename, mode) {
  if ((fs.lstatSync(filename).mode & 0o777) !== mode) {
    fail(`${filename} does not have mode ${mode.toString(8)}`);
  }
}

class ArtifactMigrationExecutor {
  constructor(context) {
    this.context = context;
    this.artifacts = Object.fromEntries(context.artifacts.map((entry) => [entry.id, entry._path]));
    this.env = context.environment;
    this.root = context.temporaryRoot;
    this.python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
    this.venvs = {};
    this.localConfig = null;
    this.storeSnapshot = null;
    this.extractedNpm = null;
    this.npmState = null;
  }

  execute(definition) {
    const method = this[`scenario_${definition.id.replaceAll('-', '_')}`];
    if (typeof method !== 'function') fail(`migration scenario has no executor: ${definition.id}`);
    method.call(this);
    return [...definition.checks];
  }

  createVenv(name, initialArtifact, expectedVersion) {
    const directory = path.join(this.root, name);
    runChecked(this.python, ['-m', 'venv', directory], {
      cwd: this.root, env: this.env,
    });
    const python = path.join(directory, 'bin', 'python');
    runChecked(python, [
      '-m', 'pip', 'install', '--disable-pip-version-check', initialArtifact,
    ], { cwd: this.root, env: this.env });
    this.probePython(python, expectedVersion);
    this.venvs[name] = { directory, python };
    return this.venvs[name];
  }

  probePython(python, expectedVersion) {
    const code = [
      'import importlib.metadata as m, json, os, sys',
      'import custback',
      `expected = ${JSON.stringify(expectedVersion)}`,
      'assert m.version("custback") == expected',
      'assert custback.__version__ == expected',
      'assert not os.environ.get("PYTHONPATH")',
      'print(json.dumps({"version": expected, "prefix": sys.prefix}))',
    ].join('\n');
    runChecked(python, ['-I', '-c', code], { cwd: this.root, env: this.env });
    runChecked(python, ['-m', 'pip', 'check'], { cwd: this.root, env: this.env });
  }

  upgradePython(venv, artifact) {
    runChecked(venv.python, [
      '-m', 'pip', 'install', '--disable-pip-version-check', '--upgrade', artifact,
    ], { cwd: this.root, env: this.env });
    this.probePython(venv.python, CANDIDATE_VERSION);
  }

  scenario_candidate_wheel_clean_install() {
    this.createVenv(
      'candidate-wheel', this.artifacts['python-wheel'], CANDIDATE_VERSION,
    );
  }

  scenario_candidate_sdist_clean_install() {
    this.createVenv(
      'candidate-sdist', this.artifacts['python-sdist'], CANDIDATE_VERSION,
    );
  }

  scenario_reference_wheel_upgrade() {
    const venv = this.createVenv(
      'reference-wheel', this.artifacts['reference-0.3.0-wheel'], REFERENCE_VERSION,
    );
    this.upgradePython(venv, this.artifacts['python-wheel']);
  }

  scenario_reference_sdist_upgrade() {
    const venv = this.createVenv(
      'reference-sdist', this.artifacts['reference-0.3.0-sdist'], REFERENCE_VERSION,
    );
    this.upgradePython(venv, this.artifacts['python-sdist']);
  }

  scenario_config_local_in_place_migration() {
    const venv = this.venvs['candidate-wheel'];
    const directory = path.join(this.root, 'config-local');
    fs.mkdirSync(directory, { mode: 0o700 });
    const config = path.join(directory, 'config.yaml');
    const original = Buffer.from(
      'background:\n  mode: camera\n  camera_device: /dev/video2\n',
    );
    fs.writeFileSync(config, original, { mode: 0o644 });
    const launcher = path.join(venv.directory, 'bin', 'custback');
    runChecked(launcher, [
      'migrate', '--config', config, '--target-id', 'legacy-camera',
    ], { cwd: directory, env: this.env });
    const migrated = fs.readFileSync(config);
    const backup = path.join(directory, '.config.yaml.custback-migration.backup');
    if (!fs.readFileSync(backup).equals(original) || migrated.equals(original)) {
      fail('local config migration did not retain the exact original backup');
    }
    assertMode(config, 0o600);
    assertMode(backup, 0o600);
    const probe = [
      'import json, sys',
      'from custback.config import AppConfig',
      'cfg = AppConfig.load(sys.argv[1])',
      'assert cfg.background.camera_device == ""',
      'assert cfg.background.camera_target == "legacy-camera"',
      'assert cfg.backdrop_targets["legacy-camera"].source == "/dev/video2"',
      'print(json.dumps(cfg.to_dict(), default=list))',
    ].join('\n');
    runChecked(venv.python, ['-I', '-c', probe, config], {
      cwd: directory, env: this.env,
    });
    runChecked(launcher, [
      'migrate', '--config', config, '--target-id', 'legacy-camera',
    ], { cwd: directory, env: this.env });
    if (!fs.readFileSync(config).equals(migrated)) fail('idempotent config retry changed bytes');
    this.localConfig = { backup, config, snapshot: migrated };
  }

  scenario_config_remote_explicit_refusal() {
    const venv = this.venvs['candidate-wheel'];
    const directory = path.join(this.root, 'config-remote');
    fs.mkdirSync(directory, { mode: 0o700 });
    const config = path.join(directory, 'config.yaml');
    const secret = 'qualification-secret-canary';
    const original = Buffer.from(
      `background:\n  mode: camera\n  camera_device: https://user:${secret}@camera.invalid/live\n`,
    );
    fs.writeFileSync(config, original, { mode: 0o600 });
    const launcher = path.join(venv.directory, 'bin', 'custback');
    const result = runChecked(launcher, [
      'migrate', '--config', config, '--target-id', 'legacy-camera',
    ], { cwd: directory, env: this.env, expectedStatus: 4 });
    if ((result.stdout || '').includes(secret) || (result.stderr || '').includes(secret)) {
      fail('remote migration refusal leaked the legacy source');
    }
    if (!fs.readFileSync(config).equals(original) || fs.readdirSync(directory).length !== 1) {
      fail('remote migration refusal changed on-disk state');
    }
  }

  scenario_core_avatar_store_repair() {
    const venv = this.venvs['candidate-wheel'];
    const base = path.join(this.root, 'stores');
    const core = path.join(base, 'core');
    const rigs = path.join(base, 'rigs');
    const avatar = path.join(base, 'avatar');
    const rig = path.join(rigs, 'office');
    for (const directory of [base, core, rigs, avatar, rig]) {
      fs.mkdirSync(directory, { recursive: true, mode: 0o755 });
      fs.chmodSync(directory, 0o755);
    }
    const assets = [
      [path.join(core, 'scene.png'), Buffer.from('core-asset')],
      [path.join(rig, 'body.png'), Buffer.from('rig-asset')],
      [path.join(avatar, 'room.mp4'), Buffer.from('avatar-asset')],
    ];
    for (const [filename, contents] of assets) {
      fs.writeFileSync(filename, contents, { mode: 0o644 });
      fs.chmodSync(filename, 0o644);
    }
    const pending = path.join(core, '.upload-owned.part');
    fs.writeFileSync(pending, 'reserved-upload', { mode: 0o640 });
    const ledgerCode = [
      'import pathlib, sys',
      'from custback.storage_tx import OwnershipLedger',
      'root, pending = map(pathlib.Path, sys.argv[1:])',
      'ledger = OwnershipLedger(root)',
      'record = ledger.begin(pending, kind="file", reserved_bytes=15, reserved_slots=1)',
      'ledger.bind(record, pending)',
      'ledger.close()',
    ].join('\n');
    runChecked(venv.python, ['-I', '-c', ledgerCode, core, pending], {
      cwd: base, env: this.env,
    });
    const launcher = path.join(venv.directory, 'bin', 'custback');
    const storeArgs = [
      'migrate', '--repair-storage', '--core-store', core,
      '--avatar-rigs-store', rigs, '--avatar-backgrounds-store', avatar,
    ];
    runChecked(launcher, storeArgs, { cwd: base, env: this.env });
    for (const directory of [core, rigs, avatar, rig]) assertMode(directory, 0o700);
    for (const [filename, contents] of [...assets, [pending, Buffer.from('reserved-upload')]]) {
      assertMode(filename, 0o600);
      if (!fs.readFileSync(filename).equals(contents)) fail(`store asset changed: ${filename}`);
    }
    const quotaProbe = [
      'import pathlib, sys',
      'from custback.storage_tx import OwnershipLedger',
      'ledger = OwnershipLedger(pathlib.Path(sys.argv[1]))',
      'assert ledger.reserved_bytes == 15 and ledger.reserved_slots == 1',
      'assert len(ledger.records) == 1',
      'ledger.close()',
    ].join('\n');
    runChecked(venv.python, ['-I', '-c', quotaProbe, core], {
      cwd: base, env: this.env,
    });
    this.storeSnapshot = Object.fromEntries(
      [...assets, [pending, Buffer.from('reserved-upload')]].map(([name]) => [
        name, crypto.createHash('sha256').update(fs.readFileSync(name)).digest('hex'),
      ]),
    );
  }

  scenario_python_reinstall_uninstall_reinstall() {
    const venv = this.venvs['candidate-wheel'];
    runChecked(venv.python, [
      '-m', 'pip', 'install', '--force-reinstall', '--no-deps', this.artifacts['python-wheel'],
    ], { cwd: this.root, env: this.env });
    this.probePython(venv.python, CANDIDATE_VERSION);
    runChecked(venv.python, ['-m', 'pip', 'uninstall', '-y', 'custback'], {
      cwd: this.root, env: this.env,
    });
    runChecked(venv.python, [
      '-I', '-c', 'import importlib.util; assert importlib.util.find_spec("custback") is None',
    ], { cwd: this.root, env: this.env });
    runChecked(venv.python, [
      '-m', 'pip', 'install', '--no-deps', this.artifacts['python-wheel'],
    ], { cwd: this.root, env: this.env });
    this.probePython(venv.python, CANDIDATE_VERSION);
    if (!fs.readFileSync(this.localConfig.config).equals(this.localConfig.snapshot)) {
      fail('Python reinstall changed migrated external configuration');
    }
    for (const [filename, digest] of Object.entries(this.storeSnapshot)) {
      const current = crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
      if (current !== digest) fail(`Python reinstall changed external asset: ${filename}`);
    }
  }

  npmInstall(prefix, artifact, environment, { ignoreScripts = false } = {}) {
    const cache = path.join(this.root, 'npm-cache');
    fs.mkdirSync(cache, { recursive: true });
    const args = [
      'install', '--global', artifact, '--prefix', prefix, '--cache', cache,
      '--no-audit', '--no-fund',
    ];
    if (ignoreScripts) args.push('--ignore-scripts');
    return runChecked('npm', args, { cwd: this.root, env: environment });
  }

  npmTarget(prefix) {
    return path.join(prefix, '.custback-venv');
  }

  npmGenerationRoot(prefix) {
    return path.join(prefix, '.custback-venv.custback-generations');
  }

  npmPackageRoot(prefix) {
    return path.join(prefix, 'lib', 'node_modules', 'custback');
  }

  inspectNpmState(prefix, requestedExtras) {
    const target = this.npmTarget(prefix);
    const root = this.npmGenerationRoot(prefix);
    if (!fs.lstatSync(target).isSymbolicLink() ||
        !fs.lstatSync(root).isDirectory() || fs.lstatSync(root).isSymbolicLink()) {
      fail('npm managed target is not an owned generational symlink');
    }
    const intent = JSON.parse(fs.readFileSync(`${target}.custback-install-intent.json`, 'utf8'));
    const stamp = JSON.parse(fs.readFileSync(path.join(target, '.custback-install.json'), 'utf8'));
    if (JSON.stringify(intent.requestedExtras) !== JSON.stringify(requestedExtras) ||
        JSON.stringify(stamp.requestedExtras) !== JSON.stringify(requestedExtras)) {
      fail('npm explicit extras intent was not preserved');
    }
    return {
      active: fs.realpathSync(target),
      entries: fs.readdirSync(root).filter((name) => /^(?:gen|legacy)-/.test(name)).sort(),
      intent: fs.readFileSync(`${target}.custback-install-intent.json`),
      root,
      target,
    };
  }

  scenario_candidate_npm_clean_install() {
    const prefix = path.join(this.root, 'npm-clean-prefix');
    this.npmInstall(prefix, this.artifacts['npm-tarball'], {
      ...this.env, CUSTBACK_EXTRAS: '',
    });
    const launcher = path.join(prefix, 'bin', 'custback');
    runChecked(launcher, ['doctor'], { cwd: this.root, env: this.env });
    this.inspectNpmState(prefix, []);
  }

  extractCandidateNpm() {
    if (this.extractedNpm) return this.extractedNpm;
    const directory = path.join(this.root, 'candidate-npm-extracted');
    fs.mkdirSync(directory, { mode: 0o700 });
    runChecked('tar', ['-xzf', this.artifacts['npm-tarball'], '-C', directory], {
      cwd: this.root, env: this.env,
    });
    const packageRoot = path.join(directory, 'package');
    const migration = path.join(packageRoot, 'packaging', 'npm', 'migrate-legacy.js');
    const metadata = fs.lstatSync(migration);
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      fail('candidate npm artifact is missing its migration bridge');
    }
    this.extractedNpm = packageRoot;
    return packageRoot;
  }

  scenario_npm_artifact_migration_contracts() {
    const packageRoot = this.extractCandidateNpm();
    const testFile = path.join(
      packageRoot, 'packaging', 'npm', 'test', 'phase6-migration.test.js',
    );
    runChecked(process.execPath, ['--test', testFile], { cwd: this.root, env: this.env });
  }

  scenario_reference_npm_preupgrade_bridge() {
    const prefix = path.join(this.root, 'npm-upgrade-prefix');
    this.npmInstall(prefix, this.artifacts['reference-0.3.0-npm'], {
      ...this.env, CUSTBACK_EXTRAS: 'rvm',
    });
    const packageRoot = this.npmPackageRoot(prefix);
    const legacy = path.join(packageRoot, '.venv');
    if (!fs.lstatSync(legacy).isSymbolicLink()) {
      fail('reference npm artifact did not create the reviewed package-local generations');
    }
    const bridge = path.join(this.extractCandidateNpm(), 'packaging', 'npm', 'migrate-legacy.js');
    runChecked(process.execPath, [bridge, '--prefix', prefix], {
      cwd: this.root, env: { ...this.env, CUSTBACK_SKIP_INSTALL: '1' },
    });
    const before = this.inspectNpmState(prefix, ['rvm']);
    if (!fs.lstatSync(legacy).isSymbolicLink() ||
        fs.realpathSync(legacy) !== fs.realpathSync(before.target)) {
      fail('preupgrade bridge did not publish its compatibility link');
    }
    this.npmState = { before, prefix };
  }

  scenario_npm_package_directory_replacement() {
    const { prefix } = this.npmState;
    this.npmInstall(prefix, this.artifacts['npm-tarball'], this.env);
    const after = this.inspectNpmState(prefix, ['rvm']);
    const activeName = path.basename(after.active);
    if (after.entries.length < 2 || !after.entries.includes(activeName) ||
        !after.entries.some((name) => name !== activeName)) {
      fail('npm package replacement did not preserve active and rollback generations');
    }
    if (lstatOrNull(`${after.target}.custback-preupgrade.json`) ||
        lstatOrNull(path.join(after.root, '.migration-pending.json'))) {
      fail('successful npm replacement left a migration journal');
    }
    runChecked(path.join(prefix, 'bin', 'custback'), ['doctor'], {
      cwd: this.root, env: this.env,
    });
    this.npmState.after = after;
  }

  scenario_npm_retry_rollback() {
    const { after, prefix } = this.npmState;
    const launcher = path.join(prefix, 'bin', 'custback');
    runChecked(launcher, ['rebuild', '--extras', 'rvm'], {
      cwd: this.root,
      env: { ...this.env, PIP_NO_INDEX: '1' },
      expectedStatus: 1,
    });
    const retried = this.inspectNpmState(prefix, ['rvm']);
    if (retried.active !== after.active ||
        JSON.stringify(retried.entries) !== JSON.stringify(after.entries)) {
      fail('failed npm rebuild changed active or rollback generations');
    }
    if (lstatOrNull(path.join(retried.root, '.migration-pending.json'))) {
      fail('failed npm rebuild left a promotion journal');
    }
  }

  scenario_npm_reinstall_uninstall_reinstall() {
    const { prefix } = this.npmState;
    const before = this.inspectNpmState(prefix, ['rvm']);
    runChecked('npm', [
      'uninstall', '--global', 'custback', '--prefix', prefix,
      '--cache', path.join(this.root, 'npm-cache'), '--no-audit', '--no-fund',
    ], { cwd: this.root, env: this.env });
    if (!fs.existsSync(before.target) || !fs.existsSync(before.root)) {
      fail('npm uninstall removed durable managed state');
    }
    this.npmInstall(prefix, this.artifacts['npm-tarball'], this.env);
    const after = this.inspectNpmState(prefix, ['rvm']);
    if (after.active !== before.active ||
        !after.intent.equals(before.intent)) {
      fail('npm uninstall/reinstall did not reuse exact healthy state and intent');
    }
  }
}

function fixtureCoverage(manifest) {
  const expected = manifest.legacy_fixtures.map((entry) => entry.id);
  const covered = new Set(SCENARIOS.flatMap((entry) => entry.fixtures));
  exactIds([...covered], expected, 'migration scenario fixture ids');
  return expected.map((id) => ({
    id,
    scenario_ids: SCENARIOS.filter((entry) => entry.fixtures.includes(id)).map((entry) => entry.id),
  }));
}

function publicArtifactRecords(records) {
  return records.map(({ _path, _identity, ...entry }) => entry);
}

function buildReport(context, scenarioResults, temporaryRemoved) {
  const bindings = context.artifacts.map((entry) => entry.sha256).sort();
  const fixtures = fixtureCoverage(context.manifest).map((entry) => ({
    ...entry,
    conclusion: 'success',
    artifact_sha256s: [...bindings],
  }));
  return {
    schema_version: 1,
    report_id: REPORT_ID,
    authorization: 'diagnostic-only',
    conclusion: 'success',
    manifest_id: context.manifest.manifest_id,
    manifest_sha256: phase6.manifestDigest(context.manifest),
    generated_at: new Date().toISOString(),
    reference_source: {
      commit: REFERENCE_COMMIT,
      version: REFERENCE_VERSION,
      publication_status: 'unpublished-source-reference',
      description: 'never-published reference artifacts rebuilt from the reviewed reference commit',
    },
    matrix: context.matrix,
    isolation: {
      platform: process.platform,
      architecture: process.arch,
      source_import_path: false,
      temporary_root_outside_source: true,
      temporary_root_removed: temporaryRemoved,
    },
    artifacts: publicArtifactRecords(context.artifacts),
    scenarios: scenarioResults.map((entry) => ({
      ...entry,
      artifact_sha256s: [...bindings],
    })),
    fixtures,
  };
}

function validateReport(report, manifest = phase6.loadManifest()) {
  phase6.validateManifest(manifest);
  exactKeys(report, [
    'schema_version', 'report_id', 'authorization', 'conclusion', 'manifest_id',
    'manifest_sha256', 'generated_at', 'reference_source', 'matrix', 'isolation', 'artifacts',
    'scenarios', 'fixtures',
  ], 'migration qualification report');
  if (report.schema_version !== 1 || report.report_id !== REPORT_ID ||
      report.authorization !== 'diagnostic-only' || report.conclusion !== 'success' ||
      report.manifest_id !== manifest.manifest_id ||
      report.manifest_sha256 !== phase6.manifestDigest(manifest) ||
      !Number.isFinite(Date.parse(report.generated_at))) {
    fail('migration qualification report header is invalid');
  }
  exactKeys(report.reference_source, [
    'commit', 'version', 'publication_status', 'description',
  ], 'migration qualification reference_source');
  if (report.reference_source.commit !== REFERENCE_COMMIT ||
      report.reference_source.version !== REFERENCE_VERSION ||
      report.reference_source.publication_status !== 'unpublished-source-reference' ||
      !/rebuilt from the reviewed reference commit/.test(report.reference_source.description)) {
    fail('migration qualification reference artifacts are misrepresented');
  }
  exactKeys(
    report.matrix,
    ['id', 'os', 'python', 'node', 'artifact_ids', 'platform'],
    'migration qualification matrix',
  );
  const matrix = manifest.matrices.migration.find((entry) => entry.id === report.matrix.id);
  const expectedPlatform = matrix && matrix.os === 'macos' ? 'darwin' : 'linux';
  if (!matrix || !isDeepStrictEqual(
    report.matrix,
    { ...matrix, platform: expectedPlatform },
  )) {
    fail('migration qualification matrix is not the reviewed runtime');
  }
  exactKeys(report.isolation, [
    'platform', 'architecture', 'source_import_path',
    'temporary_root_outside_source', 'temporary_root_removed',
  ], 'migration qualification isolation');
  if (report.isolation.source_import_path !== false ||
      report.isolation.temporary_root_outside_source !== true ||
      report.isolation.temporary_root_removed !== true) {
    fail('migration qualification did not prove isolated cleanup');
  }
  const expectedArtifactIds = INPUT_DEFINITIONS.map((entry) => entry.id);
  exactIds(report.artifacts.map((entry) => entry.id), expectedArtifactIds, 'report artifact ids');
  for (const [index, entry] of report.artifacts.entries()) {
    exactKeys(entry, ['id', 'role', 'filename', 'sha256', 'size'], `report artifact ${index}`);
    const definition = INPUT_DEFINITIONS.find((candidate) => candidate.id === entry.id);
    if (entry.role !== definition.role || entry.filename !== definition.filename ||
        !SHA256_RE.test(entry.sha256) || !Number.isSafeInteger(entry.size) || entry.size <= 0) {
      fail(`report artifact ${entry.id} is invalid`);
    }
  }
  if (new Set(report.artifacts.map((entry) => entry.sha256)).size !== report.artifacts.length) {
    fail('report artifact digests must be distinct');
  }
  const bindings = report.artifacts.map((entry) => entry.sha256).sort();
  exactIds(report.scenarios.map((entry) => entry.id), SCENARIOS.map((entry) => entry.id),
    'report scenario ids');
  for (const entry of report.scenarios) {
    exactKeys(entry, [
      'id', 'conclusion', 'fixture_ids', 'checks', 'artifact_sha256s',
    ], `report scenario ${entry.id}`);
    const definition = SCENARIOS.find((candidate) => candidate.id === entry.id);
    if (entry.conclusion !== 'success') fail(`report scenario ${entry.id} did not succeed`);
    exactIds(entry.fixture_ids, definition.fixtures, `report scenario ${entry.id} fixtures`);
    exactIds(entry.checks, definition.checks, `report scenario ${entry.id} checks`);
    exactIds(entry.artifact_sha256s, bindings, `report scenario ${entry.id} bindings`);
  }
  const fixtureIds = manifest.legacy_fixtures.map((entry) => entry.id);
  exactIds(report.fixtures.map((entry) => entry.id), fixtureIds, 'report fixture ids');
  for (const entry of report.fixtures) {
    exactKeys(entry, [
      'id', 'scenario_ids', 'conclusion', 'artifact_sha256s',
    ], `report fixture ${entry.id}`);
    const expectedScenarios = SCENARIOS
      .filter((scenario) => scenario.fixtures.includes(entry.id))
      .map((scenario) => scenario.id);
    if (entry.conclusion !== 'success' || expectedScenarios.length === 0) {
      fail(`report fixture ${entry.id} did not succeed`);
    }
    exactIds(entry.scenario_ids, expectedScenarios, `report fixture ${entry.id} scenarios`);
    exactIds(entry.artifact_sha256s, bindings, `report fixture ${entry.id} bindings`);
  }
  return report;
}

function qualifyMigrations(options) {
  const manifest = options.manifest || phase6.loadManifest();
  const matrix = validateMigrationRuntime(options, manifest);
  const artifacts = validateArtifactInputs(options, manifest);
  const temporaryRoot = createTemporaryRoot(options.temporaryBase);
  const context = {
    artifacts,
    environment: sanitizedEnvironment(temporaryRoot, options.env || process.env),
    manifest,
    matrix,
    temporaryRoot,
  };
  const executor = options.executor || new ArtifactMigrationExecutor(context);
  const scenarioResults = [];
  let removed = false;
  try {
    for (const definition of SCENARIOS) {
      const checks = executor.execute(definition, context);
      exactIds(checks, definition.checks, `scenario ${definition.id} executed checks`);
      scenarioResults.push({
        id: definition.id,
        conclusion: 'success',
        fixture_ids: [...definition.fixtures],
        checks: [...checks],
      });
    }
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
    removed = !fs.existsSync(temporaryRoot);
  }
  if (!removed) fail('migration qualification temporary root survived cleanup');
  return validateReport(buildReport(context, scenarioResults, removed), manifest);
}

function writeReport(filename, report) {
  const resolved = path.resolve(filename);
  const existing = lstatOrNull(resolved);
  if (existing) fail(`migration qualification report already exists: ${resolved}`);
  fs.writeFileSync(resolved, `${JSON.stringify(report, null, 2)}\n`, {
    encoding: 'utf8', mode: 0o600, flag: 'wx',
  });
  assertMode(resolved, 0o600);
}

function main(argv = process.argv.slice(2)) {
  try {
    const options = parseArgs(argv);
    const report = qualifyMigrations(options);
    writeReport(options.report, report);
    process.stderr.write(
      `[custback migrations] artifact qualification passed (diagnostic only): ${options.report}\n`,
    );
    return 0;
  } catch (err) {
    process.stderr.write(`[custback migrations] ${err.message}\n`);
    return 1;
  }
}

module.exports = {
  ArtifactMigrationExecutor,
  CANDIDATE_VERSION,
  INPUT_DEFINITIONS,
  REFERENCE_COMMIT,
  REPORT_ID,
  ROOT,
  SCENARIOS,
  buildReport,
  createTemporaryRoot,
  detectRuntime,
  fixtureCoverage,
  main,
  parseArgs,
  qualifyMigrations,
  sanitizedEnvironment,
  sha256File,
  validateArtifactInputs,
  validateMigrationRuntime,
  validateReport,
  writeReport,
};

if (require.main === module) process.exit(main());
