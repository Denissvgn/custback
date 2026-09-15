'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const candidate = require('../../../scripts/release/build-candidate');
const phase6 = require('../../../scripts/release/phase6-evidence');

const root = path.resolve(__dirname, '..', '..', '..');

test('candidate artifact records require exactly one immutable file per reviewed id', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-candidate-records-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const files = [
    'custback-0.4.0-py3-none-any.whl',
    'custback-0.4.0.tar.gz',
    'custback-0.4.0.tgz',
  ];
  files.forEach((name, index) => fs.writeFileSync(path.join(directory, name), `artifact ${index}`));
  const records = candidate.artifactRecords(directory, phase6.loadManifest());
  assert.deepEqual(records.map((entry) => entry.id), [
    'python-wheel', 'python-sdist', 'npm-tarball',
  ]);
  assert.ok(records.every((entry) => /^[0-9a-f]{64}$/.test(entry.sha256)));
  assert.equal(new Set(records.map((entry) => entry.sha256)).size, 3);

  fs.writeFileSync(path.join(directory, 'unexpected.txt'), 'not publishable');
  assert.throws(() => candidate.artifactRecords(directory), /unexpected files/);
});

test('candidate hashing rejects symlinks and detects content substitution', (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-candidate-hash-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const file = path.join(directory, 'artifact');
  const alias = path.join(directory, 'alias');
  fs.writeFileSync(file, 'first');
  fs.symlinkSync(file, alias);
  const first = candidate.sha256File(file);
  fs.writeFileSync(file, 'second');
  const second = candidate.sha256File(file);
  assert.notEqual(first.sha256, second.sha256);
  assert.throws(() => candidate.sha256File(alias), /non-symlink/);
});

test('release provenance is exact-run bound and local provenance is diagnostic only', () => {
  const commit = 'a'.repeat(40);
  const local = candidate.githubProvenance(commit, {});
  assert.equal(local.provider, 'local-diagnostic');
  const env = {
    GITHUB_ACTIONS: 'true',
    GITHUB_REPOSITORY: 'example/custback',
    GITHUB_SHA: commit,
    GITHUB_RUN_ID: '42',
    GITHUB_RUN_ATTEMPT: '2',
    GITHUB_WORKFLOW_REF: 'example/custback/.github/workflows/release.yml@refs/tags/v0.4.0',
  };
  assert.equal(candidate.githubProvenance(commit, env).provider, 'github-actions');
  assert.throws(
    () => candidate.githubProvenance(commit, { ...env, GITHUB_SHA: 'b'.repeat(40) }),
    /SHA does not match/,
  );
});

test('candidate prepack overwrites its trusted source bridge metadata', () => {
  const builderSource = fs.readFileSync(
    path.join(root, 'scripts', 'release', 'build-candidate.js'),
    'utf8',
  );
  assert.match(builderSource, /release\.verifyVisualPolicyRollout\(root,/);
  assert.match(builderSource, /if \(diagnostic\) npmArguments\.push\('--ignore-scripts'\)/);

  const source = {
    commit: 'a'.repeat(40),
    tree: 'b'.repeat(40),
  };
  const hostile = {
    KEEP: 'yes',
    CUSTBACK_SKIP_INSTALL: '0',
    CUSTBACK_RELEASE_GIT_ROOT: '/untrusted',
    CUSTBACK_RELEASE_SOURCE_COMMIT: 'c'.repeat(40),
    CUSTBACK_RELEASE_SOURCE_TREE: 'd'.repeat(40),
  };
  const sanitized = candidate.withoutReleaseSourceBridge(hostile);
  assert.deepEqual(sanitized, {
    KEEP: 'yes',
    CUSTBACK_SKIP_INSTALL: '0',
  });

  const env = candidate.trustedPrepackEnvironment(root, source, hostile);
  assert.equal(env.KEEP, 'yes');
  assert.equal(env.CUSTBACK_SKIP_INSTALL, '1');
  assert.equal(env.CUSTBACK_RELEASE_GIT_ROOT, fs.realpathSync(root));
  assert.equal(env.CUSTBACK_RELEASE_SOURCE_COMMIT, source.commit);
  assert.equal(env.CUSTBACK_RELEASE_SOURCE_TREE, source.tree);
  assert.throws(
    () => candidate.trustedPrepackEnvironment(root, { commit: 'bad', tree: source.tree }),
    /exact source commit metadata/,
  );
});

test('candidate output must stay outside the checkout and start empty', (t) => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-candidate-output-'));
  t.after(() => fs.rmSync(base, { recursive: true, force: true }));
  const output = path.join(base, 'out');
  assert.equal(candidate.prepareOutput(output), output);
  fs.writeFileSync(path.join(output, 'existing'), 'collision');
  assert.throws(() => candidate.prepareOutput(output), /empty real directory/);
});

test('Python build products cannot contaminate the npm source snapshot', (t) => {
  const { execFileSync } = require('node:child_process');
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-build-snapshots-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const repository = path.join(directory, 'repository');
  const scratch = path.join(directory, 'scratch');
  fs.mkdirSync(repository);
  fs.mkdirSync(scratch);
  const git = (...args) => execFileSync('git', ['-C', repository, ...args], {
    encoding: 'utf8',
    env: { ...process.env, GIT_CONFIG_NOSYSTEM: '1' },
  }).trim();
  git('init', '--quiet');
  fs.writeFileSync(path.join(repository, 'source.py'), 'committed source\n');
  git('add', 'source.py');
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
    '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture');
  const commit = git('rev-parse', 'HEAD');
  fs.writeFileSync(path.join(repository, 'source.py'), 'uncommitted source\n');
  const sources = candidate.stageBuildSources(repository, commit, scratch);
  fs.mkdirSync(path.join(sources.python, 'build'));
  fs.mkdirSync(path.join(sources.python, 'src', 'custback.egg-info'), { recursive: true });
  fs.writeFileSync(path.join(sources.python, 'source.py'), 'modified by a build\n');
  assert.deepEqual(fs.readdirSync(sources.npm), ['source.py']);
  assert.equal(fs.readFileSync(path.join(sources.npm, 'source.py'), 'utf8'), 'committed source\n');
  assert.equal(fs.readFileSync(path.join(repository, 'source.py'), 'utf8'), 'uncommitted source\n');
});
