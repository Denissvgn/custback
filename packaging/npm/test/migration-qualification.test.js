'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const migration = require('../../../scripts/release/qualify-migrations');
const phase6 = require('../../../scripts/release/phase6-evidence');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function artifactFixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-migration-runner-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const options = {};
  for (const [index, definition] of migration.INPUT_DEFINITIONS.entries()) {
    const filename = path.join(directory, definition.filename);
    fs.writeFileSync(filename, `artifact ${definition.id} ${index}\n`, { mode: 0o600 });
    options[definition.option] = filename;
  }
  options['matrix-id'] = 'ubuntu-migration-python-3.12-node-22';
  options.report = path.join(directory, 'migration-report.json');
  return { directory, manifest: phase6.loadManifest(), options };
}

function argvFor(options) {
  return [
    ...migration.INPUT_DEFINITIONS.flatMap((definition) => [
      `--${definition.option}`, options[definition.option],
    ]),
    '--matrix-id', options['matrix-id'],
    '--report', options.report,
  ];
}

const REVIEWED_RUNTIME = { platform: 'linux', python: '3.12', node: '22' };

test('migration qualification requires six exact absolute artifact paths', (t) => {
  const { options } = artifactFixture(t);
  assert.deepEqual(migration.parseArgs(argvFor(options)), options);

  assert.throws(
    () => migration.parseArgs(argvFor(options).slice(2)),
    /missing required.*--wheel/,
  );
  const relative = argvFor(options);
  relative[1] = path.basename(relative[1]);
  assert.throws(() => migration.parseArgs(relative), /--wheel must be an absolute path/);
  assert.throws(
    () => migration.parseArgs([...argvFor(options), '--wheel', options.wheel]),
    /unknown or duplicate/,
  );
});

test('artifact inputs are regular, uniquely hashed, and named for exact versions', (t) => {
  const { directory, manifest, options } = artifactFixture(t);
  const records = migration.validateArtifactInputs(options, manifest);
  assert.deepEqual(
    records.map((entry) => entry.id),
    migration.INPUT_DEFINITIONS.map((entry) => entry.id),
  );
  assert.equal(new Set(records.map((entry) => entry.sha256)).size, records.length);
  assert.ok(records.every((entry) => path.isAbsolute(entry._path)));

  const wrongName = { ...options };
  wrongName.wheel = path.join(directory, 'custback-9.9.9-py3-none-any.whl');
  fs.writeFileSync(wrongName.wheel, 'wrong release\n');
  assert.throws(
    () => migration.validateArtifactInputs(wrongName, manifest),
    /must be exactly custback-/,
  );

  const referenceNpm = options['reference-npm'];
  fs.rmSync(referenceNpm);
  fs.symlinkSync(options.npm, referenceNpm);
  assert.throws(
    () => migration.validateArtifactInputs(options, manifest),
    /regular non-symlink/,
  );
  fs.rmSync(referenceNpm);
  fs.writeFileSync(referenceNpm, 'restored reference npm\n');

  fs.rmSync(options['reference-sdist']);
  fs.linkSync(options.sdist, options['reference-sdist']);
  assert.throws(
    () => migration.validateArtifactInputs(options, manifest),
    /distinct files/,
  );
});

test('the scenario set covers exactly every manifest migration fixture', () => {
  const manifest = phase6.loadManifest();
  const coverage = migration.fixtureCoverage(manifest);
  assert.deepEqual(
    coverage.map((entry) => entry.id),
    manifest.legacy_fixtures.map((entry) => entry.id),
  );
  assert.ok(coverage.every((entry) => entry.scenario_ids.length > 0));
  assert.deepEqual(
    [...new Set(migration.SCENARIOS.flatMap((entry) => entry.fixtures))].sort(),
    manifest.legacy_fixtures.map((entry) => entry.id).sort(),
  );
});

test('the runner removes source-path injection from isolated child environments', (t) => {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-migration-env-'));
  t.after(() => fs.rmSync(temporaryRoot, { recursive: true, force: true }));
  const env = migration.sanitizedEnvironment(temporaryRoot, {
    PATH: process.env.PATH,
    PYTHONPATH: '/checkout/src',
    PYTHONHOME: '/untrusted/python',
    VIRTUAL_ENV: '/untrusted/venv',
    NODE_PATH: '/checkout/node_modules',
    NODE_OPTIONS: '--require=/tmp/inject.js',
    NPM_CONFIG_PREFIX: '/untrusted/prefix',
    NPM_CONFIG_USERCONFIG: '/checkout/.npmrc',
    CUSTBACK_VENV: '/untrusted/custback',
    CUSTBACK_EXTRAS: 'gpu',
    CUSTBACK_SKIP_INSTALL: '1',
    CUSTBACK_FORCE_REBUILD: '1',
  });
  for (const name of [
    'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'NODE_PATH', 'NODE_OPTIONS',
    'NPM_CONFIG_PREFIX', 'CUSTBACK_VENV', 'CUSTBACK_EXTRAS',
    'CUSTBACK_SKIP_INSTALL', 'CUSTBACK_FORCE_REBUILD',
  ]) assert.equal(Object.hasOwn(env, name), false, name);
  assert.equal(env.PYTHONNOUSERSITE, '1');
  assert.equal(env.PIP_CONFIG_FILE, os.devNull);
  assert.ok(env.NPM_CONFIG_USERCONFIG.startsWith(temporaryRoot));
  assert.equal(env.CUSTBACK_RELEASE_TMPDIR, temporaryRoot);
  assert.ok(env.HOME.startsWith(temporaryRoot));
});

test('a fake executor produces a strict diagnostic-only artifact-bound report', (t) => {
  const { manifest, options } = artifactFixture(t);
  let temporaryRoot;
  const seen = [];
  const executor = {
    execute(definition, context) {
      temporaryRoot = context.temporaryRoot;
      assert.equal(fs.existsSync(temporaryRoot), true);
      assert.equal(path.relative(migration.ROOT, temporaryRoot).startsWith('..'), true);
      seen.push(definition.id);
      return [...definition.checks];
    },
  };
  const report = migration.qualifyMigrations({
    ...options,
    env: { PATH: process.env.PATH, PYTHONPATH: path.join(migration.ROOT, 'src') },
    executor,
    manifest,
    runtime: REVIEWED_RUNTIME,
    temporaryBase: os.tmpdir(),
  });

  assert.equal(fs.existsSync(temporaryRoot), false);
  assert.deepEqual(seen, migration.SCENARIOS.map((entry) => entry.id));
  assert.equal(report.authorization, 'diagnostic-only');
  assert.equal(report.reference_source.publication_status, 'unpublished-source-reference');
  assert.match(
    report.reference_source.description,
    /never-published reference artifacts rebuilt from the reviewed reference commit/,
  );
  assert.deepEqual(
    report.fixtures.map((entry) => entry.id),
    manifest.legacy_fixtures.map((entry) => entry.id),
  );
  const bindings = report.artifacts.map((entry) => entry.sha256).sort();
  assert.ok(report.scenarios.every(
    (entry) => assert.deepEqual(entry.artifact_sha256s, bindings) === undefined,
  ));
  assert.ok(report.fixtures.every(
    (entry) => assert.deepEqual(entry.artifact_sha256s, bindings) === undefined,
  ));

  migration.writeReport(options.report, report);
  assert.deepEqual(JSON.parse(fs.readFileSync(options.report)), report);
  assert.equal(fs.lstatSync(options.report).mode & 0o777, 0o600);
  assert.throws(() => migration.writeReport(options.report, report), /already exists/);
});

test('report validation rejects fixture drift, false publication claims, and substitution', (t) => {
  const { manifest, options } = artifactFixture(t);
  const executor = { execute: (definition) => [...definition.checks] };
  const report = migration.qualifyMigrations({
    ...options, executor, manifest, runtime: REVIEWED_RUNTIME, temporaryBase: os.tmpdir(),
  });

  const missing = clone(report);
  missing.fixtures.pop();
  assert.throws(() => migration.validateReport(missing, manifest), /fixture ids/);

  const added = clone(report);
  added.fixtures.push({
    id: 'unreviewed-fixture',
    scenario_ids: [added.scenarios[0].id],
    conclusion: 'success',
    artifact_sha256s: added.artifacts.map((entry) => entry.sha256),
  });
  assert.throws(() => migration.validateReport(added, manifest), /fixture ids/);

  const failed = clone(report);
  failed.scenarios[0].conclusion = 'failure';
  assert.throws(() => migration.validateReport(failed, manifest), /did not succeed/);

  const published = clone(report);
  published.reference_source.publication_status = 'published';
  assert.throws(() => migration.validateReport(published, manifest), /misrepresented/);

  const substituted = clone(report);
  substituted.fixtures[0].artifact_sha256s[0] = 'f'.repeat(64);
  assert.throws(() => migration.validateReport(substituted, manifest), /bindings/);

  const wrongRole = clone(report);
  wrongRole.artifacts[0].role = 'reference';
  assert.throws(() => migration.validateReport(wrongRole, manifest), /artifact.*invalid/);

  const wrongMatrix = clone(report);
  wrongMatrix.matrix.node = '20';
  assert.throws(() => migration.validateReport(wrongMatrix, manifest), /reviewed runtime/);

  const extraField = clone(report);
  extraField.locally_approved = true;
  assert.throws(() => migration.validateReport(extraField, manifest), /must contain exactly/);
});
