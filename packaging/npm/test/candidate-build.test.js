'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const candidate = require('../../../scripts/release/build-candidate');
const phase6 = require('../../../scripts/release/phase6-evidence');

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

test('candidate output must stay outside the checkout and start empty', (t) => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-candidate-output-'));
  t.after(() => fs.rmSync(base, { recursive: true, force: true }));
  const output = path.join(base, 'out');
  assert.equal(candidate.prepareOutput(output), output);
  fs.writeFileSync(path.join(output, 'existing'), 'collision');
  assert.throws(() => candidate.prepareOutput(output), /empty real directory/);
});
