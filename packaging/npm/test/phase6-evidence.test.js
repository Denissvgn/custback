'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const phase6 = require('../../../scripts/release/phase6-evidence');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function resultSet(ids, digests) {
  return ids.map((id) => ({
    id,
    conclusion: 'success',
    artifact_sha256s: [...digests],
  }));
}

function reportSet(ids, digests, category) {
  return ids.map((id, index) => ({
    id,
    sha256: phase6.sha256Buffer(Buffer.from(`${category}:${id}:${index}`)),
    conclusion: 'success',
    artifact_sha256s: [...digests],
  }));
}

function evidenceFixture(t) {
  const manifest = phase6.loadManifest();
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-phase6-evidence-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const filenames = {
    'python-wheel': 'custback-0.4.0-py3-none-any.whl',
    'python-sdist': 'custback-0.4.0.tar.gz',
    'npm-tarball': 'custback-0.4.0.tgz',
  };
  const artifacts = manifest.artifacts.map((definition, index) => {
    const filename = filenames[definition.id];
    const contents = Buffer.from(`phase6 artifact ${definition.id} ${index}\n`, 'utf8');
    fs.writeFileSync(path.join(directory, filename), contents);
    return {
      id: definition.id,
      filename,
      sha256: phase6.sha256Buffer(contents),
      size: contents.length,
    };
  });
  const digests = artifacts.map((entry) => entry.sha256);
  const provenance = {
    provider: 'github-actions',
    repository: 'example/custback',
    commit: 'a'.repeat(40),
    run_id: '987654321',
    run_attempt: 2,
    workflow_ref: 'example/custback/.github/workflows/release.yml@refs/tags/v0.4.0',
  };
  const evidence = {
    schema_version: 1,
    manifest_id: manifest.manifest_id,
    manifest_sha256: phase6.manifestDigest(manifest),
    generated_at: '2026-07-16T12:34:56.000Z',
    provenance,
    artifacts,
    gates: resultSet(manifest.required_gates, digests),
    jobs: resultSet(manifest.workflow.required_job_ids, digests),
    fixtures: resultSet(manifest.legacy_fixtures.map((entry) => entry.id), digests),
    matrices: Object.fromEntries(
      Object.entries(manifest.matrices).map(([name, entries]) => [
        name,
        resultSet(entries.map((entry) => entry.id), digests),
      ]),
    ),
    stress: manifest.stress_families.map((entry, index) => ({
      id: entry.id,
      conclusion: 'success',
      iterations: entry.minimum_iterations,
      seed: `phase6:${index + 1}`,
      artifact_sha256s: [...digests],
    })),
    tls: resultSet(manifest.tls_scenarios, digests),
    reports: {
      migrations: reportSet(
        manifest.matrices.migration.map((entry) => entry.id),
        digests,
        'migrations',
      ),
      two_host: reportSet(manifest.tls_scenarios, digests, 'two-host'),
      clean_hosts: reportSet(
        ['clean-host-repeat-a', 'clean-host-repeat-b'],
        digests,
        'clean-hosts',
      ),
    },
  };
  const artifactPaths = Object.fromEntries(
    artifacts.map((entry) => [entry.id, path.join(directory, entry.filename)]),
  );
  const env = {
    GITHUB_ACTIONS: 'true',
    GITHUB_REPOSITORY: provenance.repository,
    GITHUB_SHA: provenance.commit,
    GITHUB_RUN_ID: provenance.run_id,
    GITHUB_RUN_ATTEMPT: String(provenance.run_attempt),
    GITHUB_WORKFLOW_REF: provenance.workflow_ref,
  };
  const evidencePath = path.join(directory, 'phase6-evidence.json');
  fs.writeFileSync(evidencePath, `${JSON.stringify(evidence)}\n`);
  return { artifactPaths, directory, env, evidence, evidencePath, manifest };
}

test('the Phase 6 manifest is finite and enumerates every required contract family', () => {
  const manifest = phase6.loadManifest();
  assert.equal(manifest.schema_version, 1);
  assert.deepEqual(
    manifest.artifacts.map((entry) => entry.id),
    ['python-wheel', 'python-sdist', 'npm-tarball'],
  );
  assert.deepEqual(
    Object.keys(manifest.matrices),
    ['python', 'node', 'migration', 'platform', 'optional_backends'],
  );
  assert.deepEqual(
    manifest.matrices.python.map((entry) => entry.id),
    [
      'ubuntu-python-3.10', 'ubuntu-python-3.11', 'ubuntu-python-3.12',
      'ubuntu-python-3.13', 'ubuntu-python-3.14', 'macos-python-3.12',
    ],
  );
  assert.deepEqual(
    manifest.matrices.node.map((entry) => entry.id),
    ['ubuntu-node-18', 'ubuntu-node-20', 'ubuntu-node-22', 'macos-node-20'],
  );
  assert.deepEqual(
    manifest.matrices.migration.map((entry) => entry.id),
    [
      'ubuntu-migration-python-3.10-node-18',
      'ubuntu-migration-python-3.11-node-20',
      'ubuntu-migration-python-3.12-node-22',
      'ubuntu-migration-python-3.13-node-20',
      'ubuntu-migration-python-3.14-node-20',
      'macos-migration-python-3.12-node-20',
    ],
  );
  assert.equal(manifest.stress_families.length, 7);
  assert.ok(manifest.stress_families.every((entry) => entry.minimum_iterations >= 100));
  assert.ok(manifest.tls_scenarios.includes('failed-tls-handshake-no-payload'));
  assert.ok(manifest.workflow.required_job_ids.includes('phase6-evidence'));
  assert.equal(manifest.workflow.aggregate_job_id, 'release-gate');
  assert.equal(manifest.workflow.publish_job_id, 'publish');
  assert.match(phase6.manifestDigest(manifest), /^[0-9a-f]{64}$/);
});

test('the manifest schema rejects additions, duplicates, and unknown artifact references', () => {
  const manifest = phase6.loadManifest();
  const added = clone(manifest);
  added.local_override = true;
  assert.throws(() => phase6.validateManifest(added), /invalid schema.*local_override/);

  const duplicate = clone(manifest);
  duplicate.workflow.required_job_ids.push(duplicate.workflow.required_job_ids[0]);
  assert.throws(() => phase6.validateManifest(duplicate), /duplicate entries/);

  const unknownArtifact = clone(manifest);
  unknownArtifact.matrices.python[0].artifact_ids = ['unreviewed-artifact'];
  assert.throws(() => phase6.validateManifest(unknownArtifact), /unknown artifact ids/);
});

test('complete diagnostic evidence validates but is not promoted to trusted evidence', (t) => {
  const { evidence, manifest } = evidenceFixture(t);
  assert.equal(phase6.validateEvidence(evidence, manifest), evidence);
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true,
      env: {},
      artifactPaths: {},
    }),
    /requires a GitHub Actions run context/,
  );
});

test('every evidence collection requires the exact manifest-defined ID set', (t) => {
  const { evidence, manifest } = evidenceFixture(t);
  const mutations = [
    ['artifact omission', (candidate) => candidate.artifacts.pop()],
    ['gate omission', (candidate) => candidate.gates.pop()],
    ['job addition', (candidate) => candidate.jobs.push({
      ...candidate.jobs[0], id: 'unreviewed-job',
    })],
    ['fixture omission', (candidate) => candidate.fixtures.shift()],
    ['Python matrix omission', (candidate) => candidate.matrices.python.pop()],
    ['Node matrix omission', (candidate) => candidate.matrices.node.pop()],
    ['migration matrix omission', (candidate) => candidate.matrices.migration.pop()],
    ['platform matrix omission', (candidate) => candidate.matrices.platform.pop()],
    ['optional matrix omission', (candidate) => candidate.matrices.optional_backends.pop()],
    ['stress omission', (candidate) => candidate.stress.pop()],
    ['TLS scenario omission', (candidate) => candidate.tls.pop()],
    ['migration report omission', (candidate) => candidate.reports.migrations.pop()],
  ];
  for (const [label, mutate] of mutations) {
    const candidate = clone(evidence);
    mutate(candidate);
    assert.throws(
      () => phase6.validateEvidence(candidate, manifest),
      /required set|required artifact|duplicate ids/,
      label,
    );
  }
});

test('evidence and result objects reject unknown schema fields', (t) => {
  const { evidence, manifest } = evidenceFixture(t);
  const topLevel = clone(evidence);
  topLevel.locally_approved = true;
  assert.throws(
    () => phase6.validateEvidence(topLevel, manifest),
    /invalid schema.*locally_approved/,
  );

  const result = clone(evidence);
  result.gates[0].note = 'trust me';
  assert.throws(
    () => phase6.validateEvidence(result, manifest),
    /invalid schema.*note/,
  );
});

test('all results are bound to the complete exact artifact digest set', (t) => {
  const { evidence, manifest } = evidenceFixture(t);
  const missing = clone(evidence);
  missing.stress[0].artifact_sha256s.pop();
  assert.throws(
    () => phase6.validateEvidence(missing, manifest),
    /does not match the required set/,
  );

  const substituted = clone(evidence);
  substituted.tls[0].artifact_sha256s[0] = 'f'.repeat(64);
  assert.throws(
    () => phase6.validateEvidence(substituted, manifest),
    /does not match the required set/,
  );

  const failed = clone(evidence);
  failed.jobs[0].conclusion = 'failure';
  assert.throws(
    () => phase6.validateEvidence(failed, manifest),
    /conclusion must be success/,
  );
});

test('manifest changes invalidate evidence from the previous reviewed set', (t) => {
  const { evidence, manifest } = evidenceFixture(t);
  const changed = clone(manifest);
  changed.tls_scenarios.push('unreviewed-shortcut');
  assert.throws(
    () => phase6.validateEvidence(evidence, changed),
    /manifest SHA-256 does not match/,
  );
});

test('trusted evidence requires the exact GitHub commit, run, attempt, and workflow', (t) => {
  const { artifactPaths, env, evidence, manifest } = evidenceFixture(t);
  assert.equal(
    phase6.validateEvidence(evidence, manifest, { trusted: true, env, artifactPaths }),
    evidence,
  );
  for (const [name, value, pattern] of [
    ['GITHUB_SHA', 'b'.repeat(40), /commit does not match/],
    ['GITHUB_RUN_ID', '987654322', /run_id does not match/],
    ['GITHUB_RUN_ATTEMPT', '3', /run_attempt does not match/],
    [
      'GITHUB_WORKFLOW_REF',
      'example/custback/.github/workflows/other.yml@refs/tags/v0.4.0',
      /workflow_ref does not match/,
    ],
  ]) {
    assert.throws(
      () => phase6.validateEvidence(evidence, manifest, {
        trusted: true,
        env: { ...env, [name]: value },
        artifactPaths,
      }),
      pattern,
      name,
    );
  }
});

test('trusted validation recomputes every artifact SHA-256 and rejects substitution', (t) => {
  const { artifactPaths, env, evidence, manifest } = evidenceFixture(t);
  const wheel = artifactPaths['python-wheel'];
  fs.appendFileSync(wheel, 'substituted\n');
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true,
      env,
      artifactPaths,
    }),
    /size or SHA-256 does not match/,
  );
});

test('trusted validation rejects missing, extra, relative, and aliased artifact paths', (t) => {
  const { artifactPaths, directory, env, evidence, manifest } = evidenceFixture(t);
  const missing = { ...artifactPaths };
  delete missing['npm-tarball'];
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true, env, artifactPaths: missing,
    }),
    /does not match the required set/,
  );

  const extra = { ...artifactPaths, unexpected: artifactPaths['npm-tarball'] };
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true, env, artifactPaths: extra,
    }),
    /does not match the required set/,
  );

  const relative = { ...artifactPaths, 'npm-tarball': 'custback-0.4.0.tgz' };
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true, env, artifactPaths: relative,
    }),
    /must be absolute/,
  );

  const aliasDirectory = path.join(directory, 'aliases');
  fs.mkdirSync(aliasDirectory);
  const alias = path.join(aliasDirectory, evidence.artifacts[0].filename);
  fs.symlinkSync(artifactPaths['python-wheel'], alias);
  assert.throws(
    () => phase6.validateEvidence(evidence, manifest, {
      trusted: true,
      env,
      artifactPaths: { ...artifactPaths, 'python-wheel': alias },
    }),
    /regular, non-symlink file/,
  );
});

test('cryptographic provenance verifies every digest and evidence with exact identity', (t) => {
  const { artifactPaths, env, evidence, evidencePath } = evidenceFixture(t);
  const calls = [];
  const authenticated = { ...env, GH_TOKEN: 'test-token' };
  assert.equal(phase6.verifyGithubAttestations(evidence, artifactPaths, {
    env: authenticated,
    evidencePath,
    spawn(command, args, options) {
      calls.push({ command, args, options });
      return { status: 0, stdout: '[{"verificationResult":{}}]', stderr: '' };
    },
  }), true);
  assert.equal(calls.length, 4);
  assert.ok(calls.some((call) => call.args.includes(evidencePath)));
  for (const call of calls) {
    assert.equal(call.command, 'gh');
    assert.ok(call.args.includes('--deny-self-hosted-runners'));
    assert.deepEqual(
      call.args.slice(call.args.indexOf('--repo'), call.args.indexOf('--repo') + 2),
      ['--repo', evidence.provenance.repository],
    );
    assert.deepEqual(
      call.args.slice(
        call.args.indexOf('--source-digest'),
        call.args.indexOf('--source-digest') + 2,
      ),
      ['--source-digest', evidence.provenance.commit],
    );
    assert.deepEqual(
      call.args.slice(call.args.indexOf('--source-ref'), call.args.indexOf('--source-ref') + 2),
      ['--source-ref', 'refs/tags/v0.4.0'],
    );
  }
});

test('cryptographic provenance rejects missing authentication and failed verification', (t) => {
  const { artifactPaths, env, evidence, evidencePath } = evidenceFixture(t);
  assert.throws(
    () => phase6.verifyGithubAttestations(evidence, artifactPaths, { env }),
    /absolute evidence path/,
  );
  fs.writeFileSync(evidencePath, JSON.stringify({ ...evidence, manifest_id: 'substituted' }));
  assert.throws(
    () => phase6.verifyGithubAttestations(evidence, artifactPaths, { env, evidencePath }),
    /differs from the validated evidence/,
  );
  fs.writeFileSync(evidencePath, `${JSON.stringify(evidence)}\n`);
  assert.throws(
    () => phase6.verifyGithubAttestations(evidence, artifactPaths, { env, evidencePath }),
    /requires GH_TOKEN/,
  );
  assert.throws(
    () => phase6.verifyGithubAttestations(evidence, artifactPaths, {
      env: { ...env, GH_TOKEN: 'test-token' },
      evidencePath,
      spawn() { return { status: 1, stdout: '', stderr: 'no signed bundle' }; },
    }),
    /verification failed.*no signed bundle/,
  );
});
