#!/usr/bin/env node
/** Strict Phase 6 release evidence and artifact binding validation. */

'use strict';

const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..', '..');
const DEFAULT_MANIFEST = path.join(__dirname, 'required-gates.json');
const SHA256_RE = /^[0-9a-f]{64}$/;
const ID_RE = /^[a-z0-9]+(?:[._-][a-z0-9]+)*$/;

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
    const missing = wanted.filter((key) => !actual.includes(key));
    const unexpected = actual.filter((key) => !wanted.includes(key));
    fail(
      `${label} has an invalid schema; missing: ${missing.join(', ') || 'none'}; ` +
      `unexpected: ${unexpected.join(', ') || 'none'}`,
    );
  }
}

function requiredString(value, label, pattern = null) {
  if (typeof value !== 'string' || value.trim() !== value || value === '' ||
      value.length > 512 || (pattern && !pattern.test(value))) {
    fail(`${label} is invalid`);
  }
  return value;
}

function identifier(value, label) {
  return requiredString(value, label, ID_RE);
}

function nonEmptyArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) fail(`${label} must be a non-empty array`);
  return value;
}

function uniqueStrings(value, label, validator = identifier) {
  const items = nonEmptyArray(value, label).map((item, index) =>
    validator(item, `${label}[${index}]`));
  if (new Set(items).size !== items.length) fail(`${label} contains duplicate entries`);
  return items;
}

function exactStringSet(actual, expected, label, validator = identifier) {
  const values = uniqueStrings(actual, label, validator);
  const actualSet = new Set(values);
  const expectedSet = new Set(expected);
  const missing = expected.filter((item) => !actualSet.has(item));
  const unexpected = values.filter((item) => !expectedSet.has(item));
  if (missing.length || unexpected.length || values.length !== expected.length) {
    fail(
      `${label} does not match the required set; missing: ${missing.join(', ') || 'none'}; ` +
      `unexpected: ${unexpected.join(', ') || 'none'}`,
    );
  }
  return values;
}

function validateIdObjects(value, label, validateEntry) {
  const entries = nonEmptyArray(value, label);
  const ids = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entryLabel = `${label}[${index}]`;
    validateEntry(entries[index], entryLabel);
    ids.push(identifier(entries[index].id, `${entryLabel}.id`));
  }
  if (new Set(ids).size !== ids.length) fail(`${label} contains duplicate ids`);
  return ids;
}

function validateArtifactReferences(value, artifactIds, label) {
  const references = uniqueStrings(value, label);
  const known = new Set(artifactIds);
  const unknown = references.filter((item) => !known.has(item));
  if (unknown.length) fail(`${label} contains unknown artifact ids: ${unknown.join(', ')}`);
}

function validateManifest(manifest) {
  exactKeys(manifest, [
    'schema_version', 'manifest_id', 'artifacts', 'legacy_fixtures', 'matrices',
    'stress_families', 'tls_scenarios', 'required_gates', 'workflow',
  ], 'Phase 6 manifest');
  if (manifest.schema_version !== 1) fail('Phase 6 manifest schema_version must be 1');
  identifier(manifest.manifest_id, 'Phase 6 manifest manifest_id');

  const artifactIds = validateIdObjects(manifest.artifacts, 'manifest.artifacts', (entry, label) => {
    exactKeys(entry, ['id', 'filename_pattern'], label);
    identifier(entry.id, `${label}.id`);
    const pattern = requiredString(entry.filename_pattern, `${label}.filename_pattern`);
    if (!pattern.startsWith('^') || !pattern.endsWith('$')) {
      fail(`${label}.filename_pattern must be fully anchored`);
    }
    try {
      new RegExp(pattern);
    } catch (err) {
      fail(`${label}.filename_pattern is invalid: ${err.message}`);
    }
  });

  validateIdObjects(manifest.legacy_fixtures, 'manifest.legacy_fixtures', (entry, label) => {
    exactKeys(entry, ['id', 'kind', 'format_version'], label);
    identifier(entry.id, `${label}.id`);
    identifier(entry.kind, `${label}.kind`);
    identifier(entry.format_version, `${label}.format_version`);
  });

  exactKeys(
    manifest.matrices,
    ['python', 'node', 'migration', 'platform', 'optional_backends'],
    'manifest.matrices',
  );
  validateIdObjects(manifest.matrices.python, 'manifest.matrices.python', (entry, label) => {
    exactKeys(entry, ['id', 'os', 'python', 'artifact_ids'], label);
    identifier(entry.id, `${label}.id`);
    if (!['ubuntu', 'macos'].includes(entry.os)) fail(`${label}.os is invalid`);
    requiredString(entry.python, `${label}.python`, /^3\.(?:10|11|12|13|14)$/);
    validateArtifactReferences(entry.artifact_ids, artifactIds, `${label}.artifact_ids`);
  });
  validateIdObjects(manifest.matrices.node, 'manifest.matrices.node', (entry, label) => {
    exactKeys(entry, ['id', 'os', 'node', 'artifact_ids'], label);
    identifier(entry.id, `${label}.id`);
    if (!['ubuntu', 'macos'].includes(entry.os)) fail(`${label}.os is invalid`);
    requiredString(entry.node, `${label}.node`, /^(?:18|20|22)$/);
    validateArtifactReferences(entry.artifact_ids, artifactIds, `${label}.artifact_ids`);
  });
  validateIdObjects(
    manifest.matrices.migration,
    'manifest.matrices.migration',
    (entry, label) => {
      exactKeys(entry, ['id', 'os', 'python', 'node', 'artifact_ids'], label);
      identifier(entry.id, `${label}.id`);
      if (!['ubuntu', 'macos'].includes(entry.os)) fail(`${label}.os is invalid`);
      requiredString(entry.python, `${label}.python`, /^3\.(?:10|11|12|13|14)$/);
      requiredString(entry.node, `${label}.node`, /^(?:18|20|22)$/);
      validateArtifactReferences(entry.artifact_ids, artifactIds, `${label}.artifact_ids`);
      exactStringSet(
        entry.artifact_ids,
        artifactIds,
        `${label}.artifact_ids`,
      );
    },
  );
  validateIdObjects(manifest.matrices.platform, 'manifest.matrices.platform', (entry, label) => {
    exactKeys(entry, ['id', 'os', 'arch'], label);
    identifier(entry.id, `${label}.id`);
    if (!['ubuntu', 'macos'].includes(entry.os)) fail(`${label}.os is invalid`);
    if (!['x86_64', 'arm64', 'native'].includes(entry.arch)) fail(`${label}.arch is invalid`);
  });
  validateIdObjects(
    manifest.matrices.optional_backends,
    'manifest.matrices.optional_backends',
    (entry, label) => {
      exactKeys(entry, ['id', 'os', 'python', 'contract'], label);
      identifier(entry.id, `${label}.id`);
      if (!['ubuntu', 'macos'].includes(entry.os)) fail(`${label}.os is invalid`);
      requiredString(entry.python, `${label}.python`, /^3\.(?:10|11|12|13|14)$/);
      identifier(entry.contract, `${label}.contract`);
    },
  );

  validateIdObjects(manifest.stress_families, 'manifest.stress_families', (entry, label) => {
    exactKeys(entry, ['id', 'minimum_iterations'], label);
    identifier(entry.id, `${label}.id`);
    if (!Number.isSafeInteger(entry.minimum_iterations) || entry.minimum_iterations < 100) {
      fail(`${label}.minimum_iterations must be an integer of at least 100`);
    }
  });
  uniqueStrings(manifest.tls_scenarios, 'manifest.tls_scenarios');
  uniqueStrings(manifest.required_gates, 'manifest.required_gates');

  exactKeys(
    manifest.workflow,
    ['required_job_ids', 'aggregate_job_id', 'publish_job_id'],
    'manifest.workflow',
  );
  const requiredJobs = uniqueStrings(
    manifest.workflow.required_job_ids,
    'manifest.workflow.required_job_ids',
  );
  const aggregate = identifier(
    manifest.workflow.aggregate_job_id,
    'manifest.workflow.aggregate_job_id',
  );
  const publish = identifier(manifest.workflow.publish_job_id, 'manifest.workflow.publish_job_id');
  if (aggregate === publish || requiredJobs.includes(aggregate) || requiredJobs.includes(publish)) {
    fail('aggregate and publish jobs must be distinct from required dependency jobs');
  }
  return manifest;
}

function canonicalJson(value) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') {
    return JSON.stringify(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) fail('cannot canonicalize a non-finite number');
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (isPlainObject(value)) {
    return `{${Object.keys(value).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  fail('cannot canonicalize an unsupported JSON value');
}

function sha256Buffer(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function manifestDigest(manifest) {
  validateManifest(manifest);
  return sha256Buffer(Buffer.from(canonicalJson(manifest), 'utf8'));
}

function readJsonFile(filename, label) {
  const resolved = path.resolve(filename);
  let metadata;
  try {
    metadata = fs.lstatSync(resolved);
  } catch (err) {
    fail(`${label} is unavailable: ${err.message}`);
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`${label} must be a regular, non-symlink file`);
  }
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(resolved, 'utf8'));
  } catch (err) {
    fail(`${label} is invalid JSON: ${err.message}`);
  }
  return parsed;
}

function loadManifest(filename = DEFAULT_MANIFEST) {
  return validateManifest(readJsonFile(filename, 'Phase 6 manifest'));
}

function expectedIds(entries) {
  return entries.map((entry) => entry.id);
}

function validateProvenance(value) {
  exactKeys(value, [
    'provider', 'repository', 'commit', 'run_id', 'run_attempt', 'workflow_ref',
  ], 'evidence.provenance');
  if (value.provider !== 'github-actions') {
    fail('evidence.provenance.provider must be github-actions');
  }
  requiredString(
    value.repository,
    'evidence.provenance.repository',
    /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/,
  );
  requiredString(value.commit, 'evidence.provenance.commit', /^[0-9a-f]{40}$/);
  requiredString(value.run_id, 'evidence.provenance.run_id', /^[1-9][0-9]*$/);
  if (!Number.isSafeInteger(value.run_attempt) || value.run_attempt < 1) {
    fail('evidence.provenance.run_attempt must be a positive integer');
  }
  requiredString(value.workflow_ref, 'evidence.provenance.workflow_ref');
  if (!value.workflow_ref.startsWith(`${value.repository}/.github/workflows/`) ||
      !value.workflow_ref.includes('@')) {
    fail('evidence.provenance.workflow_ref is not bound to the repository workflow');
  }
}

function validateGeneratedAt(value) {
  requiredString(value, 'evidence.generated_at');
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString() !== value) {
    fail('evidence.generated_at must be a canonical ISO-8601 timestamp');
  }
}

function validateArtifactEvidence(value, manifest) {
  const definitions = new Map(manifest.artifacts.map((entry) => [entry.id, entry]));
  const entries = nonEmptyArray(value, 'evidence.artifacts');
  const ids = [];
  const filenames = [];
  const digests = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const label = `evidence.artifacts[${index}]`;
    exactKeys(entry, ['id', 'filename', 'sha256', 'size'], label);
    const id = identifier(entry.id, `${label}.id`);
    const definition = definitions.get(id);
    if (!definition) fail(`${label}.id is not a required artifact`);
    const filename = requiredString(
      entry.filename,
      `${label}.filename`,
      /^[A-Za-z0-9][A-Za-z0-9_.+-]*$/,
    );
    if (!new RegExp(definition.filename_pattern).test(filename)) {
      fail(`${label}.filename does not match the reviewed artifact pattern`);
    }
    requiredString(entry.sha256, `${label}.sha256`, SHA256_RE);
    if (!Number.isSafeInteger(entry.size) || entry.size <= 0) {
      fail(`${label}.size must be a positive integer`);
    }
    ids.push(id);
    filenames.push(filename);
    digests.push(entry.sha256);
  }
  exactStringSet(ids, expectedIds(manifest.artifacts), 'evidence artifact ids');
  if (new Set(filenames).size !== filenames.length) {
    fail('evidence.artifacts contains duplicate filenames');
  }
  if (new Set(digests).size !== digests.length) {
    fail('evidence.artifacts contains duplicate SHA-256 digests');
  }
  return digests;
}

function validateBindings(value, digests, label) {
  exactStringSet(value, digests, label, (item, itemLabel) =>
    requiredString(item, itemLabel, SHA256_RE));
}

function validateResultSet(value, expected, digests, label) {
  const entries = nonEmptyArray(value, label);
  const ids = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const entryLabel = `${label}[${index}]`;
    exactKeys(entry, ['id', 'conclusion', 'artifact_sha256s'], entryLabel);
    ids.push(identifier(entry.id, `${entryLabel}.id`));
    if (entry.conclusion !== 'success') fail(`${entryLabel}.conclusion must be success`);
    validateBindings(entry.artifact_sha256s, digests, `${entryLabel}.artifact_sha256s`);
  }
  exactStringSet(ids, expected, `${label} ids`);
}

function validateStress(value, manifest, digests) {
  const definitions = new Map(manifest.stress_families.map((entry) => [entry.id, entry]));
  const entries = nonEmptyArray(value, 'evidence.stress');
  const ids = [];
  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    const label = `evidence.stress[${index}]`;
    exactKeys(
      entry,
      ['id', 'conclusion', 'iterations', 'seed', 'artifact_sha256s'],
      label,
    );
    const id = identifier(entry.id, `${label}.id`);
    const definition = definitions.get(id);
    if (!definition) fail(`${label}.id is not a required stress family`);
    if (entry.conclusion !== 'success') fail(`${label}.conclusion must be success`);
    if (!Number.isSafeInteger(entry.iterations) ||
        entry.iterations < definition.minimum_iterations) {
      fail(`${label}.iterations is below the reviewed minimum`);
    }
    requiredString(entry.seed, `${label}.seed`, /^[A-Za-z0-9_.:-]+$/);
    validateBindings(entry.artifact_sha256s, digests, `${label}.artifact_sha256s`);
    ids.push(id);
  }
  exactStringSet(ids, expectedIds(manifest.stress_families), 'evidence.stress ids');
}

function validateReportEvidence(value, manifest, digests) {
  exactKeys(value, ['migrations', 'two_host', 'clean_hosts'], 'evidence.reports');
  for (const [name, expected] of [
    ['migrations', expectedIds(manifest.matrices.migration)],
    ['two_host', manifest.tls_scenarios],
    ['clean_hosts', ['clean-host-repeat-a', 'clean-host-repeat-b']],
  ]) {
    const entries = nonEmptyArray(value[name], `evidence.reports.${name}`);
    const ids = [];
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const label = `evidence.reports.${name}[${index}]`;
      exactKeys(
        entry,
        ['id', 'sha256', 'conclusion', 'artifact_sha256s'],
        label,
      );
      ids.push(identifier(entry.id, `${label}.id`));
      requiredString(entry.sha256, `${label}.sha256`, SHA256_RE);
      if (entry.conclusion !== 'success') fail(`${label}.conclusion must be success`);
      validateBindings(entry.artifact_sha256s, digests, `${label}.artifact_sha256s`);
    }
    exactStringSet(ids, expected, `${labelForReportSet(name)} ids`);
  }
}

function labelForReportSet(name) {
  return `evidence.reports.${name}`;
}

function validateEvidence(evidence, manifest = loadManifest(), options = {}) {
  validateManifest(manifest);
  exactKeys(evidence, [
    'schema_version', 'manifest_id', 'manifest_sha256', 'generated_at', 'provenance',
    'artifacts', 'gates', 'jobs', 'fixtures', 'matrices', 'stress', 'tls', 'reports',
  ], 'Phase 6 evidence');
  if (evidence.schema_version !== 1) fail('Phase 6 evidence schema_version must be 1');
  if (evidence.manifest_id !== manifest.manifest_id) {
    fail('Phase 6 evidence manifest_id does not match the reviewed manifest');
  }
  requiredString(evidence.manifest_sha256, 'evidence.manifest_sha256', SHA256_RE);
  if (evidence.manifest_sha256 !== manifestDigest(manifest)) {
    fail('Phase 6 evidence manifest SHA-256 does not match the reviewed manifest');
  }
  validateGeneratedAt(evidence.generated_at);
  validateProvenance(evidence.provenance);
  const digests = validateArtifactEvidence(evidence.artifacts, manifest);
  validateResultSet(
    evidence.gates,
    manifest.required_gates,
    digests,
    'evidence.gates',
  );
  validateResultSet(
    evidence.jobs,
    manifest.workflow.required_job_ids,
    digests,
    'evidence.jobs',
  );
  validateResultSet(
    evidence.fixtures,
    expectedIds(manifest.legacy_fixtures),
    digests,
    'evidence.fixtures',
  );
  exactKeys(
    evidence.matrices,
    ['python', 'node', 'migration', 'platform', 'optional_backends'],
    'evidence.matrices',
  );
  for (const name of ['python', 'node', 'migration', 'platform', 'optional_backends']) {
    validateResultSet(
      evidence.matrices[name],
      expectedIds(manifest.matrices[name]),
      digests,
      `evidence.matrices.${name}`,
    );
  }
  validateStress(evidence.stress, manifest, digests);
  validateResultSet(evidence.tls, manifest.tls_scenarios, digests, 'evidence.tls');
  validateReportEvidence(evidence.reports, manifest, digests);

  if (options.trusted === true) {
    validateGithubContext(evidence.provenance, options.env || process.env);
    verifyArtifactFiles(evidence.artifacts, options.artifactPaths);
  }
  return evidence;
}

function validateGithubContext(provenance, env = process.env) {
  if (!env || env.GITHUB_ACTIONS !== 'true') {
    fail('trusted Phase 6 evidence requires a GitHub Actions run context');
  }
  const requiredEnvironment = [
    'GITHUB_REPOSITORY', 'GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT',
    'GITHUB_WORKFLOW_REF',
  ];
  for (const name of requiredEnvironment) {
    if (typeof env[name] !== 'string' || env[name] === '') {
      fail(`trusted Phase 6 evidence requires ${name}`);
    }
  }
  const comparisons = {
    repository: env.GITHUB_REPOSITORY,
    commit: env.GITHUB_SHA,
    run_id: env.GITHUB_RUN_ID,
    workflow_ref: env.GITHUB_WORKFLOW_REF,
  };
  for (const [field, expected] of Object.entries(comparisons)) {
    if (provenance[field] !== expected) {
      fail(`Phase 6 evidence ${field} does not match the GitHub Actions run`);
    }
  }
  if (String(provenance.run_attempt) !== env.GITHUB_RUN_ATTEMPT) {
    fail('Phase 6 evidence run_attempt does not match the GitHub Actions run');
  }
}

function sha256Descriptor(descriptor) {
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let position = 0;
  while (true) {
    const count = fs.readSync(descriptor, buffer, 0, buffer.length, position);
    if (count === 0) break;
    hash.update(buffer.subarray(0, count));
    position += count;
  }
  return hash.digest('hex');
}

function verifyArtifactFile(entry, filename) {
  if (typeof filename !== 'string' || !path.isAbsolute(filename)) {
    fail(`trusted artifact path for ${entry.id} must be absolute`);
  }
  let before;
  try {
    before = fs.lstatSync(filename);
  } catch (err) {
    fail(`trusted artifact ${entry.id} is unavailable: ${err.message}`);
  }
  if (!before.isFile() || before.isSymbolicLink()) {
    fail(`trusted artifact ${entry.id} must be a regular, non-symlink file`);
  }
  if (path.basename(filename) !== entry.filename) {
    fail(`trusted artifact ${entry.id} filename does not match evidence`);
  }
  const noFollow = fs.constants.O_NOFOLLOW || 0;
  let descriptor;
  try {
    descriptor = fs.openSync(filename, fs.constants.O_RDONLY | noFollow);
    const opened = fs.fstatSync(descriptor);
    if (!opened.isFile() || opened.dev !== before.dev || opened.ino !== before.ino) {
      fail(`trusted artifact ${entry.id} changed while opening`);
    }
    const digest = sha256Descriptor(descriptor);
    const after = fs.lstatSync(filename);
    if (!after.isFile() || after.isSymbolicLink() || after.dev !== opened.dev ||
        after.ino !== opened.ino || after.size !== opened.size) {
      fail(`trusted artifact ${entry.id} changed while hashing`);
    }
    if (opened.size !== entry.size || digest !== entry.sha256) {
      fail(`trusted artifact ${entry.id} size or SHA-256 does not match evidence`);
    }
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
  }
}

function verifyArtifactFiles(artifacts, artifactPaths) {
  if (!isPlainObject(artifactPaths)) {
    fail('trusted Phase 6 evidence requires an artifact path map');
  }
  const expected = artifacts.map((entry) => entry.id);
  const actual = Object.keys(artifactPaths);
  exactStringSet(actual, expected, 'trusted artifact path ids');
  const resolved = new Set();
  for (const entry of artifacts) {
    verifyArtifactFile(entry, artifactPaths[entry.id]);
    const real = fs.realpathSync(artifactPaths[entry.id]);
    if (resolved.has(real)) fail('trusted artifacts must use distinct files');
    resolved.add(real);
  }
}

function verifyGithubAttestations(evidence, artifactPaths, options = {}) {
  const env = options.env || process.env;
  validateGithubContext(evidence.provenance, env);
  verifyArtifactFiles(evidence.artifacts, artifactPaths);
  const evidencePath = options.evidencePath;
  if (typeof evidencePath !== 'string' || !path.isAbsolute(evidencePath)) {
    fail('cryptographic evidence verification requires an absolute evidence path');
  }
  const evidenceMetadata = fs.lstatSync(evidencePath);
  if (!evidenceMetadata.isFile() || evidenceMetadata.isSymbolicLink()) {
    fail('cryptographic evidence subject must be a regular, non-symlink file');
  }
  const evidenceOnDisk = readJsonFile(evidencePath, 'cryptographic evidence subject');
  if (canonicalJson(evidenceOnDisk) !== canonicalJson(evidence)) {
    fail('cryptographic evidence subject differs from the validated evidence');
  }
  if (!env.GH_TOKEN && !env.GITHUB_TOKEN) {
    fail('cryptographic attestation verification requires GH_TOKEN');
  }
  const workflowSeparator = evidence.provenance.workflow_ref.lastIndexOf('@');
  const signerWorkflow = evidence.provenance.workflow_ref.slice(0, workflowSeparator);
  const sourceRef = evidence.provenance.workflow_ref.slice(workflowSeparator + 1);
  if (!signerWorkflow || !sourceRef) {
    fail('evidence workflow_ref cannot identify its signer and source ref');
  }
  const spawn = options.spawn || spawnSync;
  const subjects = [
    ...evidence.artifacts.map((artifact) => ({
      label: artifact.id,
      filename: artifactPaths[artifact.id],
    })),
    { label: 'phase6-evidence', filename: evidencePath },
  ];
  for (const subject of subjects) {
    const filename = subject.filename;
    const result = spawn('gh', [
      'attestation', 'verify', filename,
      '--repo', evidence.provenance.repository,
      '--signer-workflow', signerWorkflow,
      '--signer-digest', evidence.provenance.commit,
      '--source-digest', evidence.provenance.commit,
      '--source-ref', sourceRef,
      '--deny-self-hosted-runners',
      '--format', 'json',
    ], {
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
      timeout: 2 * 60 * 1000,
      env,
    });
    if (result.error) {
      fail(
        `GitHub attestation verifier could not start for ${subject.label}: ` +
        result.error.message,
      );
    }
    if (result.status !== 0) {
      fail(
        `GitHub attestation verification failed for ${subject.label}: ` +
        `${(result.stderr || result.stdout || '').trim()}`,
      );
    }
    let verified;
    try { verified = JSON.parse(result.stdout); } catch (err) {
      fail(
        `GitHub attestation result is invalid JSON for ${subject.label}: ${err.message}`,
      );
    }
    if (!Array.isArray(verified) || verified.length === 0) {
      fail(`GitHub returned no verified attestation for ${subject.label}`);
    }
  }
  const evidenceAfterVerification = readJsonFile(
    evidencePath,
    'cryptographic evidence subject',
  );
  if (canonicalJson(evidenceAfterVerification) !== canonicalJson(evidence)) {
    fail('cryptographic evidence subject changed during attestation verification');
  }
  return true;
}

function loadEvidence(filename) {
  return readJsonFile(filename, 'Phase 6 evidence');
}

function artifactPathsFromDirectory(evidence, directory) {
  const resolved = path.resolve(directory);
  const metadata = fs.lstatSync(resolved);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail('artifact directory must be a regular, non-symlink directory');
  }
  return Object.fromEntries(
    evidence.artifacts.map((entry) => [entry.id, path.join(resolved, entry.filename)]),
  );
}

function main(argv = process.argv.slice(2)) {
  try {
    const [command, ...args] = argv;
    if (command === 'manifest-digest' && args.length <= 1) {
      const manifest = loadManifest(args[0] || DEFAULT_MANIFEST);
      process.stdout.write(`${manifestDigest(manifest)}\n`);
      return 0;
    }
    if (command === 'validate' && (args.length === 1 || args.length === 2)) {
      const evidence = loadEvidence(args[0]);
      const manifest = loadManifest(args[1] || DEFAULT_MANIFEST);
      validateEvidence(evidence, manifest);
      process.stdout.write('Phase 6 diagnostic evidence schema verified (not trusted)\n');
      return 0;
    }
    if (command === 'validate-trusted' && (args.length === 2 || args.length === 3)) {
      const evidence = loadEvidence(args[0]);
      const manifest = loadManifest(args[2] || DEFAULT_MANIFEST);
      const artifactPaths = artifactPathsFromDirectory(evidence, args[1]);
      validateEvidence(evidence, manifest, {
        trusted: true,
        artifactPaths,
      });
      verifyGithubAttestations(evidence, artifactPaths, {
        evidencePath: path.resolve(args[0]),
      });
      process.stdout.write(
        'Phase 6 signed evidence document, artifacts, and provenance verified\n',
      );
      return 0;
    }
    fail(
      'usage: phase6-evidence.js manifest-digest [MANIFEST] | ' +
      'validate EVIDENCE [MANIFEST] | ' +
      'validate-trusted EVIDENCE ARTIFACT_DIR [MANIFEST]',
    );
  } catch (err) {
    process.stderr.write(`[custback phase6 evidence] ${err.message}\n`);
    return 1;
  }
}

module.exports = {
  DEFAULT_MANIFEST,
  ROOT,
  artifactPathsFromDirectory,
  canonicalJson,
  loadEvidence,
  loadManifest,
  main,
  manifestDigest,
  sha256Buffer,
  validateEvidence,
  validateGithubContext,
  validateManifest,
  verifyArtifactFiles,
  verifyGithubAttestations,
};

if (require.main === module) process.exit(main());
