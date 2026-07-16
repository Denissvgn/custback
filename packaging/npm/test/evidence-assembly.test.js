'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const assembly = require('../../../scripts/release/assemble-evidence');
const migrations = require('../../../scripts/release/qualify-migrations');
const phase6 = require('../../../scripts/release/phase6-evidence');

function reportBindings(manifest) {
  const records = (ids, category) => ids.map((id, index) => ({
    id,
    sha256: phase6.sha256Buffer(Buffer.from(`${category}:${id}:${index}`)),
  }));
  return {
    migrations: records(
      manifest.matrices.migration.map((entry) => entry.id),
      'migrations',
    ),
    two_host: records(manifest.tls_scenarios, 'two-host'),
    clean_hosts: records(['clean-host-repeat-a', 'clean-host-repeat-b'], 'clean-hosts'),
  };
}

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-assembly-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const manifest = phase6.loadManifest();
  const provenance = {
    provider: 'github-actions',
    repository: 'example/custback',
    commit: 'a'.repeat(40),
    run_id: '99',
    run_attempt: 1,
    workflow_ref: 'example/custback/.github/workflows/release.yml@refs/tags/v0.4.0',
  };
  const env = {
    GITHUB_ACTIONS: 'true',
    GITHUB_REPOSITORY: provenance.repository,
    GITHUB_SHA: provenance.commit,
    GITHUB_RUN_ID: provenance.run_id,
    GITHUB_RUN_ATTEMPT: '1',
    GITHUB_WORKFLOW_REF: provenance.workflow_ref,
  };
  const names = {
    'python-wheel': 'custback-0.4.0-py3-none-any.whl',
    'python-sdist': 'custback-0.4.0.tar.gz',
    'npm-tarball': 'custback-0.4.0.tgz',
  };
  const artifacts = manifest.artifacts.map((definition, index) => {
    const filename = names[definition.id];
    const contents = Buffer.from(`candidate ${index}\n`);
    fs.writeFileSync(path.join(directory, filename), contents);
    return {
      id: definition.id,
      filename,
      sha256: phase6.sha256Buffer(contents),
      size: contents.length,
    };
  });
  const candidate = {
    schema_version: 1,
    authorization: 'release',
    manifest_id: manifest.manifest_id,
    manifest_sha256: phase6.manifestDigest(manifest),
    generated_at: new Date().toISOString(),
    source: { commit: provenance.commit, tree: 'b'.repeat(40), version: '0.4.0' },
    provenance,
    artifacts,
  };
  const candidatePath = path.join(directory, 'candidate-manifest.json');
  fs.writeFileSync(candidatePath, `${JSON.stringify(candidate)}\n`);
  const bytes = fs.readFileSync(candidatePath);
  const report = assembly.qualificationTemplate(bytes, manifest, {
    reports: reportBindings(manifest),
  });
  const reportPath = path.join(directory, 'report.json');
  fs.writeFileSync(reportPath, `${JSON.stringify(report)}\n`);
  const needs = Object.fromEntries(
    manifest.workflow.required_job_ids
      .filter((id) => id !== assembly.SELF_JOB_ID)
      .map((id) => [id, { result: 'success', outputs: {} }]),
  );
  return { candidate, candidatePath, directory, env, manifest, needs, reportPath };
}

function dynamicReportFixture(t) {
  const value = fixture(t);
  const referenceDirectory = path.join(value.directory, 'references');
  const migrationDirectory = path.join(value.directory, 'migration-reports');
  const twoHostDirectory = path.join(value.directory, 'two-host-reports');
  const cleanHostDirectory = path.join(value.directory, 'clean-host-reports');
  for (const directory of [
    referenceDirectory, migrationDirectory, twoHostDirectory, cleanHostDirectory,
  ]) fs.mkdirSync(directory);

  const candidateById = new Map(value.candidate.artifacts.map((entry) => [entry.id, entry]));
  const inputs = { report: path.join(value.directory, 'unused.json') };
  for (const [index, definition] of migrations.INPUT_DEFINITIONS.entries()) {
    if (definition.role === 'candidate') {
      inputs[definition.option] = path.join(
        value.directory,
        candidateById.get(definition.id).filename,
      );
    } else {
      const filename = path.join(referenceDirectory, definition.filename);
      fs.writeFileSync(filename, `reference ${definition.id} ${index}\n`);
      inputs[definition.option] = filename;
    }
  }
  const artifacts = migrations.validateArtifactInputs(inputs, value.manifest);
  const scenarioResults = migrations.SCENARIOS.map((definition) => ({
    id: definition.id,
    conclusion: 'success',
    fixture_ids: [...definition.fixtures],
    checks: [...definition.checks],
  }));
  for (const definition of value.manifest.matrices.migration) {
    const platform = definition.os === 'macos' ? 'darwin' : 'linux';
    const report = migrations.buildReport({
      artifacts,
      manifest: value.manifest,
      matrix: { ...definition, platform },
    }, scenarioResults, true);
    report.isolation.platform = platform;
    fs.writeFileSync(
      path.join(migrationDirectory, `migration-${definition.id}.json`),
      `${JSON.stringify(report)}\n`,
    );
  }

  const writeTwoHost = (directory, prefix, id, clean, index) => {
    fs.writeFileSync(path.join(directory, `${prefix}${id}.json`), `${JSON.stringify({
      schema_version: 1,
      wheel_sha256: candidateById.get('python-wheel').sha256,
      clean_host_evidence: clean,
      scenario_ids: [id],
      scenarios: [{
        id,
        duration_s: index + 0.25,
        meeting_address: `172.30.${index}.2`,
        renderer_address: `172.30.${index}.3`,
      }],
    })}\n`);
  };
  value.manifest.tls_scenarios.forEach((id, index) => {
    writeTwoHost(twoHostDirectory, 'two-host-', id, false, index);
  });
  ['clean-host-repeat-a', 'clean-host-repeat-b'].forEach((id, index) => {
    writeTwoHost(cleanHostDirectory, 'clean-host-', id, true, index + 40);
  });
  return {
    ...value,
    cleanHostDirectory,
    migrationDirectory,
    referenceDirectory,
    twoHostDirectory,
  };
}

test('same-run successful dependencies assemble evidence bound to all artifacts', (t) => {
  const value = fixture(t);
  const assembled = assembly.assembleEvidence({
    candidate: value.candidatePath,
    report: value.reportPath,
    needs: value.needs,
    env: value.env,
  });
  assert.doesNotThrow(() => phase6.validateEvidence(assembled, value.manifest, {
    trusted: true,
    env: value.env,
    artifactPaths: phase6.artifactPathsFromDirectory(assembled, value.directory),
  }));
  assert.equal(assembled.jobs.find((entry) => entry.id === 'phase6-evidence').conclusion, 'success');
});

test('assembly rejects a diagnostic candidate, failed job, and stale report', (t) => {
  const value = fixture(t);
  const candidate = JSON.parse(fs.readFileSync(value.candidatePath));
  candidate.authorization = 'diagnostic';
  fs.writeFileSync(value.candidatePath, JSON.stringify(candidate));
  assert.throws(() => assembly.assembleEvidence({
    candidate: value.candidatePath, report: value.reportPath, needs: value.needs, env: value.env,
  }), /authorizing release candidate/);

  const next = fixture(t);
  next.needs.stress.result = 'failure';
  assert.throws(() => assembly.assembleEvidence({
    candidate: next.candidatePath, report: next.reportPath, needs: next.needs, env: next.env,
  }), /stress did not succeed/);

  const stale = fixture(t);
  const report = JSON.parse(fs.readFileSync(stale.reportPath));
  report.candidate_manifest_sha256 = 'f'.repeat(64);
  fs.writeFileSync(stale.reportPath, JSON.stringify(report));
  assert.throws(() => assembly.assembleEvidence({
    candidate: stale.candidatePath, report: stale.reportPath, needs: stale.needs, env: stale.env,
  }), /not bound to the exact candidate/);
});

test('assembly rejects omitted, added, and skipped required dependencies', (t) => {
  const value = fixture(t);
  delete value.needs.migrations;
  assert.throws(() => assembly.validateJobNeeds(value.needs, value.manifest), /exactly/);
  value.needs.migrations = { result: 'skipped' };
  assert.throws(() => assembly.validateJobNeeds(value.needs, value.manifest), /did not succeed/);
  value.needs.unreviewed = { result: 'success' };
  assert.throws(() => assembly.validateJobNeeds(value.needs, value.manifest), /exactly/);
});

test('evidence assembly consumes exact artifact-bound migration and host reports', (t) => {
  const value = dynamicReportFixture(t);
  const options = {
    candidate: value.candidate,
    candidateDirectory: value.directory,
    directory: value.migrationDirectory,
    referenceDirectory: value.referenceDirectory,
    twoHostDirectory: value.twoHostDirectory,
    cleanHostDirectory: value.cleanHostDirectory,
    manifest: value.manifest,
  };
  const reports = assembly.collectReportBindings(options);
  assert.deepEqual(
    reports.migrations.map((entry) => entry.id),
    value.manifest.matrices.migration.map((entry) => entry.id),
  );
  assert.deepEqual(
    reports.two_host.map((entry) => entry.id),
    value.manifest.tls_scenarios,
  );
  assert.deepEqual(
    reports.clean_hosts.map((entry) => entry.id),
    ['clean-host-repeat-a', 'clean-host-repeat-b'],
  );

  const missing = path.join(
    value.twoHostDirectory,
    `two-host-${value.manifest.tls_scenarios[0]}.json`,
  );
  fs.rmSync(missing);
  assert.throws(
    () => assembly.collectReportBindings(options),
    /exact reviewed report set/,
  );
});
