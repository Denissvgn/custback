#!/usr/bin/env node
/** Assemble exact same-run Phase 6 evidence after every reviewed job succeeds. */

'use strict';

const fs = require('fs');
const path = require('path');
const { isDeepStrictEqual } = require('util');

const phase6 = require('./phase6-evidence');
const migrations = require('./qualify-migrations');

const SELF_JOB_ID = 'phase6-evidence';
const SHA256_RE = /^[0-9a-f]{64}$/;

function fail(message) {
  throw new Error(message);
}

function exactKeys(value, expected, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail(`${label} must be an object`);
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    fail(`${label} must contain exactly: ${wanted.join(', ')}`);
  }
}

function readRegularJson(filename, label) {
  const resolved = path.resolve(filename);
  const metadata = fs.lstatSync(resolved);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`${label} must be a regular, non-symlink file`);
  }
  try {
    return { bytes: fs.readFileSync(resolved), value: JSON.parse(fs.readFileSync(resolved, 'utf8')) };
  } catch (err) {
    fail(`${label} is invalid JSON: ${err.message}`);
  }
}

function exactIds(actual, expected, label) {
  if (!Array.isArray(actual) || actual.some((id) => typeof id !== 'string')) {
    fail(`${label} must be an array of ids`);
  }
  const left = [...actual].sort();
  const right = [...expected].sort();
  if (new Set(actual).size !== actual.length || JSON.stringify(left) !== JSON.stringify(right)) {
    fail(`${label} does not match the reviewed manifest`);
  }
}

function readExactReports(directory, prefix, ids, label) {
  const resolved = path.resolve(directory);
  const metadata = fs.lstatSync(resolved);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail(`${label} must be a regular, non-symlink directory`);
  }
  const expected = ids.map((id) => `${prefix}${id}.json`);
  const actual = fs.readdirSync(resolved).sort();
  if (JSON.stringify(actual) !== JSON.stringify([...expected].sort())) {
    fail(`${label} does not contain the exact reviewed report set`);
  }
  return ids.map((id) => {
    const report = readRegularJson(path.join(resolved, `${prefix}${id}.json`), `${label} ${id}`);
    return {
      id,
      sha256: phase6.sha256Buffer(report.bytes),
      value: report.value,
    };
  });
}

function publicMigrationArtifacts(records) {
  return records.map(({ _path, _identity, ...entry }) => entry);
}

function validateMigrationReportSet(options) {
  const manifest = options.manifest;
  const ids = manifest.matrices.migration.map((entry) => entry.id);
  const reports = readExactReports(options.directory, 'migration-', ids, 'migration reports');
  const candidateById = new Map(options.candidate.artifacts.map((entry) => [entry.id, entry]));
  const inputs = { report: path.resolve(options.directory, '.unused-report.json') };
  for (const definition of migrations.INPUT_DEFINITIONS) {
    if (definition.role === 'candidate') {
      const artifact = candidateById.get(definition.id);
      if (!artifact) fail(`candidate is missing ${definition.id}`);
      inputs[definition.option] = path.join(options.candidateDirectory, artifact.filename);
    } else {
      inputs[definition.option] = path.join(options.referenceDirectory, definition.filename);
    }
  }
  const artifacts = publicMigrationArtifacts(
    migrations.validateArtifactInputs(inputs, manifest),
  );
  const expectedCandidate = artifacts
    .filter((entry) => entry.role === 'candidate')
    .map(({ role, ...entry }) => entry);
  if (!isDeepStrictEqual(expectedCandidate, options.candidate.artifacts)) {
    fail('migration reports do not use the exact release candidate artifacts');
  }
  const definitions = new Map(manifest.matrices.migration.map((entry) => [entry.id, entry]));
  for (const report of reports) {
    migrations.validateReport(report.value, manifest);
    if (!isDeepStrictEqual(report.value.artifacts, artifacts)) {
      fail(`migration report ${report.id} is not bound to the downloaded artifacts`);
    }
    const definition = definitions.get(report.id);
    const platform = definition.os === 'macos' ? 'darwin' : 'linux';
    if (report.value.isolation.platform !== platform ||
        !isDeepStrictEqual(report.value.matrix, { ...definition, platform })) {
      fail(`migration report ${report.id} ran on the wrong reviewed runtime`);
    }
  }
  if (new Set(reports.map((entry) => entry.sha256)).size !== reports.length) {
    fail('migration runtime reports must be distinct');
  }
  return reports.map(({ id, sha256 }) => ({ id, sha256 }));
}

function validateTwoHostReportSet(options) {
  const cleanIds = ['clean-host-repeat-a', 'clean-host-repeat-b'];
  const clean = options.clean === true;
  const ids = clean ? cleanIds : options.manifest.tls_scenarios;
  const prefix = clean ? 'clean-host-' : 'two-host-';
  const reports = readExactReports(options.directory, prefix, ids, `${prefix}reports`);
  const wheel = options.candidate.artifacts.find((entry) => entry.id === 'python-wheel');
  if (!wheel) fail('candidate is missing python-wheel');
  for (const report of reports) {
    exactKeys(report.value, [
      'schema_version', 'wheel_sha256', 'clean_host_evidence', 'scenario_ids', 'scenarios',
    ], `${prefix}report ${report.id}`);
    if (report.value.schema_version !== 1 || report.value.wheel_sha256 !== wheel.sha256 ||
        report.value.clean_host_evidence !== clean) {
      fail(`${prefix}report ${report.id} is not bound to the candidate and host class`);
    }
    exactIds(report.value.scenario_ids, [report.id], `${prefix}report scenario ids`);
    if (!Array.isArray(report.value.scenarios) || report.value.scenarios.length !== 1) {
      fail(`${prefix}report ${report.id} must contain one scenario result`);
    }
    const result = report.value.scenarios[0];
    exactKeys(
      result,
      ['id', 'duration_s', 'meeting_address', 'renderer_address'],
      `${prefix}scenario ${report.id}`,
    );
    if (result.id !== report.id || typeof result.duration_s !== 'number' ||
        !Number.isFinite(result.duration_s) || result.duration_s < 0 ||
        typeof result.meeting_address !== 'string' || result.meeting_address === '' ||
        typeof result.renderer_address !== 'string' || result.renderer_address === '' ||
        result.meeting_address === result.renderer_address) {
      fail(`${prefix}scenario ${report.id} has an invalid result`);
    }
  }
  return reports.map(({ id, sha256 }) => ({ id, sha256 }));
}

function validateReportBindings(value, manifest) {
  exactKeys(value, ['migrations', 'two_host', 'clean_hosts'], 'qualification reports');
  for (const [name, ids] of [
    ['migrations', manifest.matrices.migration.map((entry) => entry.id)],
    ['two_host', manifest.tls_scenarios],
    ['clean_hosts', ['clean-host-repeat-a', 'clean-host-repeat-b']],
  ]) {
    if (!Array.isArray(value[name])) fail(`qualification reports.${name} must be an array`);
    exactIds(value[name].map((entry) => entry.id), ids, `qualification reports.${name}`);
    for (const entry of value[name]) {
      exactKeys(entry, ['id', 'sha256'], `qualification reports.${name}.${entry.id}`);
      if (!SHA256_RE.test(entry.sha256)) {
        fail(`qualification reports.${name}.${entry.id} has an invalid SHA-256`);
      }
    }
    if (new Set(value[name].map((entry) => entry.sha256)).size !== value[name].length) {
      fail(`qualification reports.${name} contains duplicate report digests`);
    }
  }
  return value;
}

function collectReportBindings(options) {
  return {
    migrations: validateMigrationReportSet(options),
    two_host: validateTwoHostReportSet({ ...options, directory: options.twoHostDirectory }),
    clean_hosts: validateTwoHostReportSet({
      ...options,
      directory: options.cleanHostDirectory,
      clean: true,
    }),
  };
}

function validateCandidate(candidate, manifest, env) {
  exactKeys(candidate, [
    'schema_version', 'authorization', 'manifest_id', 'manifest_sha256', 'generated_at',
    'source', 'provenance', 'artifacts',
  ], 'candidate manifest');
  if (candidate.schema_version !== 1 || candidate.authorization !== 'release') {
    fail('only an authorizing release candidate can produce trusted evidence');
  }
  exactKeys(candidate.source, ['commit', 'tree', 'version'], 'candidate source');
  if (candidate.manifest_id !== manifest.manifest_id ||
      candidate.manifest_sha256 !== phase6.manifestDigest(manifest)) {
    fail('candidate does not use the reviewed Phase 6 manifest');
  }
  if (candidate.source.commit !== candidate.provenance.commit) {
    fail('candidate source commit and provenance differ');
  }
  phase6.validateGithubContext(candidate.provenance, env);
  const definitions = new Map(manifest.artifacts.map((entry) => [entry.id, entry]));
  if (!Array.isArray(candidate.artifacts)) fail('candidate artifacts must be an array');
  exactIds(candidate.artifacts.map((entry) => entry.id), [...definitions.keys()], 'candidate artifacts');
  for (const entry of candidate.artifacts) {
    exactKeys(entry, ['id', 'filename', 'sha256', 'size'], `candidate artifact ${entry.id}`);
    const definition = definitions.get(entry.id);
    if (!definition || !new RegExp(definition.filename_pattern).test(entry.filename) ||
        !/^[0-9a-f]{64}$/.test(entry.sha256) ||
        !Number.isSafeInteger(entry.size) || entry.size <= 0) {
      fail(`candidate artifact ${entry.id} is invalid`);
    }
  }
}

function qualificationTemplate(candidateBytes, manifest, options = {}) {
  const iterations = options.iterations || Object.fromEntries(
    manifest.stress_families.map((entry) => [entry.id, entry.minimum_iterations]),
  );
  const seed = options.seed || 'phase6:required';
  const reports = validateReportBindings(options.reports, manifest);
  return {
    schema_version: 1,
    candidate_manifest_sha256: phase6.sha256Buffer(candidateBytes),
    gates: manifest.required_gates.map((id) => id),
    fixtures: manifest.legacy_fixtures.map((entry) => entry.id),
    matrices: Object.fromEntries(
      Object.entries(manifest.matrices).map(([name, entries]) => [
        name, entries.map((entry) => entry.id),
      ]),
    ),
    stress: manifest.stress_families.map((entry) => ({
      id: entry.id,
      iterations: iterations[entry.id],
      seed,
    })),
    tls: [...manifest.tls_scenarios],
    reports,
  };
}

function validateQualification(report, candidateBytes, manifest) {
  exactKeys(report, [
    'schema_version', 'candidate_manifest_sha256', 'gates', 'fixtures', 'matrices',
    'stress', 'tls', 'reports',
  ], 'qualification report');
  if (report.schema_version !== 1 ||
      report.candidate_manifest_sha256 !== phase6.sha256Buffer(candidateBytes)) {
    fail('qualification report is not bound to the exact candidate manifest');
  }
  exactIds(report.gates, manifest.required_gates, 'qualification gates');
  exactIds(
    report.fixtures,
    manifest.legacy_fixtures.map((entry) => entry.id),
    'qualification fixtures',
  );
  exactKeys(report.matrices, Object.keys(manifest.matrices), 'qualification matrices');
  for (const [name, entries] of Object.entries(manifest.matrices)) {
    exactIds(report.matrices[name], entries.map((entry) => entry.id), `${name} matrix`);
  }
  if (!Array.isArray(report.stress)) fail('qualification stress must be an array');
  exactIds(
    report.stress.map((entry) => entry.id),
    manifest.stress_families.map((entry) => entry.id),
    'qualification stress',
  );
  const definitions = new Map(manifest.stress_families.map((entry) => [entry.id, entry]));
  for (const entry of report.stress) {
    exactKeys(entry, ['id', 'iterations', 'seed'], `stress ${entry.id}`);
    if (!Number.isSafeInteger(entry.iterations) ||
        entry.iterations < definitions.get(entry.id).minimum_iterations ||
        typeof entry.seed !== 'string' || !/^[A-Za-z0-9_.:-]+$/.test(entry.seed)) {
      fail(`stress ${entry.id} does not meet its reviewed run contract`);
    }
  }
  exactIds(report.tls, manifest.tls_scenarios, 'qualification TLS scenarios');
  validateReportBindings(report.reports, manifest);
}

function validateJobNeeds(needs, manifest) {
  const expected = manifest.workflow.required_job_ids.filter((id) => id !== SELF_JOB_ID);
  exactKeys(needs, expected, 'release job dependencies');
  for (const id of expected) {
    const value = needs[id];
    if (!value || typeof value !== 'object' || value.result !== 'success') {
      fail(`required release job ${id} did not succeed`);
    }
  }
  return [...expected, SELF_JOB_ID];
}

function resultSet(ids, digests) {
  return ids.map((id) => ({
    id,
    conclusion: 'success',
    artifact_sha256s: [...digests],
  }));
}

function assembleEvidence(options) {
  const manifest = phase6.loadManifest(options.manifest);
  const candidateFile = readRegularJson(options.candidate, 'candidate manifest');
  validateCandidate(candidateFile.value, manifest, options.env || process.env);
  const reportFile = readRegularJson(options.report, 'qualification report');
  validateQualification(reportFile.value, candidateFile.bytes, manifest);
  const jobs = validateJobNeeds(options.needs, manifest);
  const artifacts = candidateFile.value.artifacts;
  const digests = artifacts.map((entry) => entry.sha256);
  const report = reportFile.value;
  const assembled = {
    schema_version: 1,
    manifest_id: manifest.manifest_id,
    manifest_sha256: phase6.manifestDigest(manifest),
    generated_at: new Date().toISOString(),
    provenance: candidateFile.value.provenance,
    artifacts,
    gates: resultSet(report.gates, digests),
    jobs: resultSet(jobs, digests),
    fixtures: resultSet(report.fixtures, digests),
    matrices: Object.fromEntries(
      Object.entries(report.matrices).map(([name, ids]) => [name, resultSet(ids, digests)]),
    ),
    stress: report.stress.map((entry) => ({
      ...entry,
      conclusion: 'success',
      artifact_sha256s: [...digests],
    })),
    tls: resultSet(report.tls, digests),
    reports: Object.fromEntries(
      Object.entries(report.reports).map(([name, entries]) => [
        name,
        entries.map((entry) => ({
          ...entry,
          conclusion: 'success',
          artifact_sha256s: [...digests],
        })),
      ]),
    ),
  };
  phase6.validateEvidence(assembled, manifest);
  return assembled;
}

function writeExclusive(filename, value) {
  fs.writeFileSync(path.resolve(filename), `${JSON.stringify(value, null, 2)}\n`, {
    encoding: 'utf8', mode: 0o600, flag: 'wx',
  });
}

function main(argv = process.argv.slice(2)) {
  try {
    const [command, ...args] = argv;
    if (command === 'template' && args.length === 6) {
      const candidate = readRegularJson(args[0], 'candidate manifest');
      const manifest = phase6.loadManifest();
      validateCandidate(candidate.value, manifest, process.env);
      const reports = collectReportBindings({
        candidate: candidate.value,
        candidateDirectory: path.dirname(path.resolve(args[0])),
        migrationDirectory: path.resolve(args[1]),
        directory: path.resolve(args[1]),
        referenceDirectory: path.resolve(args[2]),
        twoHostDirectory: path.resolve(args[3]),
        cleanHostDirectory: path.resolve(args[4]),
        manifest,
      });
      writeExclusive(args[5], qualificationTemplate(candidate.bytes, manifest, {
        seed: process.env.CUSTBACK_STRESS_SEED || 'phase6:required',
        reports,
      }));
      return 0;
    }
    if (command === 'assemble' && args.length === 3) {
      let needs;
      try { needs = JSON.parse(process.env.CUSTBACK_JOB_RESULTS || ''); } catch {
        fail('CUSTBACK_JOB_RESULTS must contain the exact GitHub Actions needs JSON');
      }
      const assembled = assembleEvidence({
        candidate: args[0], report: args[1], needs,
      });
      writeExclusive(args[2], assembled);
      return 0;
    }
    fail(
      'usage: assemble-evidence.js template CANDIDATE MIGRATION_REPORTS ' +
      'REFERENCES TWO_HOST_REPORTS ' +
      'CLEAN_HOST_REPORTS OUT | assemble CANDIDATE REPORT OUT',
    );
  } catch (err) {
    process.stderr.write(`[custback evidence assembly] ${err.message}\n`);
    return 1;
  }
}

module.exports = {
  SELF_JOB_ID,
  assembleEvidence,
  main,
  qualificationTemplate,
  collectReportBindings,
  readExactReports,
  validateMigrationReportSet,
  validateReportBindings,
  validateTwoHostReportSet,
  validateCandidate,
  validateJobNeeds,
  validateQualification,
};

if (require.main === module) process.exit(main());
