'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.resolve(__dirname, '..', '..', '..');
const workflowPath = path.join(root, '.github', 'workflows', 'release.yml');
const manifest = JSON.parse(fs.readFileSync(
  path.join(root, 'scripts', 'release', 'required-gates.json'),
  'utf8',
));
const source = fs.readFileSync(workflowPath, 'utf8');

const CHECKOUT = 'actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955';
const SETUP_PYTHON = 'actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065';
const SETUP_NODE = 'actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020';
const UPLOAD = 'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02';
const DOWNLOAD = 'actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093';
const ATTEST = 'actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6';
const ALLOWED_ACTIONS = new Set([
  CHECKOUT, SETUP_PYTHON, SETUP_NODE, UPLOAD, DOWNLOAD, ATTEST,
]);

function sorted(values) {
  return [...values].sort();
}

function jobBlocks(text) {
  const lines = text.split(/\r?\n/);
  const jobsLine = lines.findIndex((line) => line === 'jobs:');
  assert.notEqual(jobsLine, -1, 'workflow has no jobs mapping');
  const starts = [];
  for (let index = jobsLine + 1; index < lines.length; index += 1) {
    const match = lines[index].match(/^  ([a-z0-9]+(?:-[a-z0-9]+)*):$/);
    if (match) starts.push({ id: match[1], index });
  }
  const blocks = new Map();
  starts.forEach((entry, index) => {
    const end = index + 1 < starts.length ? starts[index + 1].index : lines.length;
    blocks.set(entry.id, lines.slice(entry.index, end).join('\n'));
  });
  return blocks;
}

function needsFrom(block) {
  const inline = block.match(/^    needs:\s*\[([^\]]*)\]\s*$/m);
  if (inline) {
    return inline[1].split(',').map((item) => item.trim()).filter(Boolean);
  }
  const lines = block.split(/\r?\n/);
  const start = lines.findIndex((line) => line === '    needs:');
  if (start < 0) return [];
  const needs = [];
  for (let index = start + 1; index < lines.length; index += 1) {
    const match = lines[index].match(/^      - ([a-z0-9]+(?:-[a-z0-9]+)*)$/);
    if (!match) break;
    needs.push(match[1]);
  }
  return needs;
}

function matrixIds(block) {
  return [...block.matchAll(/^          - id: ([a-z0-9]+(?:[._-][a-z0-9]+)*)$/gm)]
    .map((match) => match[1]);
}

function matrixEntryBlock(block, id) {
  const marker = `          - id: ${id}`;
  const start = block.indexOf(marker);
  assert.notEqual(start, -1, `missing matrix entry ${id}`);
  const next = block.indexOf('\n          - id:', start + marker.length);
  return block.slice(start, next < 0 ? block.length : next);
}

const jobs = jobBlocks(source);

test('release qualification triggers only manually or from version tags', () => {
  const trigger = source.slice(source.indexOf('on:'), source.indexOf('\npermissions:'));
  assert.match(trigger, /^on:\n  workflow_dispatch:\n  push:\n    tags:\n      - "v\*\.\*\.\*"\n$/);
  assert.doesNotMatch(trigger, /pull_request|schedule|branches|release:/);
});

test('manual qualification cannot publish and tagged publication checks its main ancestry', () => {
  const block = jobs.get('publish');
  assert.match(block, /github\.event_name == 'push'/);
  assert.match(block, /github\.ref_type == 'tag'/);
  assert.match(block, /github\.ref_protected/);
  assert.match(block, /vars\.CUSTBACK_PUBLISH_ENABLED == 'true'/);
  assert.match(block, /fetch-depth: 0/);
  const guard = block.indexOf('phase6-evidence.js authorize-publication');
  const upload = block.indexOf('twine upload');
  assert.ok(guard >= 0 && guard < upload);
  assert.match(block, /git merge-base --is-ancestor "\$GITHUB_SHA" origin\/main/);
  assert.match(jobs.get('artifact-build'), /github\.repository == 'Denissvgn\/custback'/);
  assert.match(jobs.get('artifact-build'), /refs\/heads\/main/);
});

test('workflow job IDs exactly match the reviewed manifest plus aggregate and publish', () => {
  const expected = [
    ...manifest.workflow.required_job_ids,
    manifest.workflow.aggregate_job_id,
    manifest.workflow.publish_job_id,
  ];
  assert.deepEqual(sorted(jobs.keys()), sorted(expected));
});

test('all external actions use the exact reviewed commit pins', () => {
  const uses = [...source.matchAll(/^\s*(?:-\s+)?uses: ([^\s#]+)/gm)]
    .map((match) => match[1]);
  assert.ok(uses.length > 0);
  assert.deepEqual(new Set(uses), ALLOWED_ACTIONS);
  assert.ok(uses.every((action) => /@[0-9a-f]{40}$/.test(action)));
  assert.ok(uses.includes(UPLOAD));
  assert.ok(uses.includes(DOWNLOAD));
  assert.equal(uses.filter((action) => action === ATTEST).length, 2);
});

test('every job provisions reviewed Node instead of relying on an ambient runner binary', () => {
  const setup = new RegExp(SETUP_NODE.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  for (const [id, block] of jobs) assert.match(block, setup, id);
});

test('artifact build uses the authorizing path once and attests all exact files', () => {
  const block = jobs.get('artifact-build');
  assert.match(block, /node scripts\/release\/build-candidate\.js --output/);
  assert.doesNotMatch(block, /build-candidate\.js[^\n]*--diagnostic/);
  assert.equal((source.match(/build-candidate\.js/g) || []).length, 1);
  assert.ok(block.indexOf('build-candidate.js') < block.indexOf(ATTEST));
  assert.match(block, /subject-path:\s*\|[\s\S]*candidate\/\*\.whl/);
  assert.match(block, /candidate\/\*\.tar\.gz/);
  assert.match(block, /candidate\/\*\.tgz/);
  assert.match(block, /id-token: write/);
  assert.match(block, /attestations: write/);
  assert.match(block, new RegExp(UPLOAD.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(block, /f01baadfa3b1e2a1ef19eceda315eedf06fbe883/);
  assert.match(block, /custback-unpublished-reference-0\.3\.0/);
  assert.match(block, /npm pack[^\n]*--ignore-scripts/);

  const registry = JSON.parse(fs.readFileSync(
    path.join(root, 'scripts', 'release', 'remediation-blockers.json'),
    'utf8',
  ));
  const rel = registry.blockers.find((entry) => entry.id === 'REL-01');
  assert.deepEqual({ status: rel.status, release_blocked: registry.release_blocked }, {
    status: 'resolved', release_blocked: false,
  });
});

test('OIDC and attestation write authority are scoped only to jobs that need them', () => {
  assert.equal((source.match(/^      attestations: write$/gm) || []).length, 2);
  assert.equal((source.match(/^      id-token: write$/gm) || []).length, 3);
  assert.doesNotMatch(source.slice(0, source.indexOf('\njobs:')), /id-token|attestations/);
  assert.doesNotMatch(jobs.get('release-gate'), /id-token: write|attestations: write/);
  assert.match(jobs.get('phase6-evidence'), /^      attestations: write$/m);
  assert.match(jobs.get('phase6-evidence'), /^      id-token: write$/m);
  for (const id of ['release-gate', 'publish']) {
    assert.match(jobs.get(id), /^      attestations: read$/m, id);
  }
  for (const id of ['phase6-evidence', 'release-gate', 'publish']) {
    assert.match(jobs.get(id), /^      GH_TOKEN: \$\{\{ github\.token \}\}$/m, id);
  }
});

test('every reviewed downstream job consumes the uploaded candidate and never rebuilds it', () => {
  for (const id of manifest.workflow.required_job_ids) {
    if (id === 'artifact-build') continue;
    const block = jobs.get(id);
    assert.ok(needsFrom(block).includes('artifact-build'), `${id} does not need artifact-build`);
    assert.match(block, new RegExp(DOWNLOAD.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), id);
    assert.match(block, /name: custback-release-candidate/, id);
    assert.doesNotMatch(block, /build-candidate\.js|python -m build|npm pack/, id);
  }
});

test('runtime and scenario matrices exactly match required-gates.json', () => {
  const expected = {
    'python-core': manifest.matrices.python.map((entry) => entry.id),
    node: manifest.matrices.node.map((entry) => entry.id),
    'optional-backends': manifest.matrices.optional_backends.map((entry) => entry.id),
    'artifact-validation': manifest.matrices.platform.map((entry) => entry.id),
    migrations: manifest.matrices.migration.map((entry) => entry.id),
    stress: manifest.stress_families.map((entry) => entry.id),
    'two-host-tls': manifest.tls_scenarios,
  };
  for (const [job, ids] of Object.entries(expected)) {
    assert.deepEqual(matrixIds(jobs.get(job)), ids, job);
  }
  assert.deepEqual(
    matrixIds(jobs.get('two-clean-hosts')),
    ['clean-host-repeat-a', 'clean-host-repeat-b'],
  );
});

test('vision optional profiles exercise every visual-consistency contract', () => {
  const block = jobs.get('optional-backends');
  const requiredTests = [
    'tests/test_geometry.py',
    'tests/test_background_geometry.py',
    'tests/test_capture.py',
    'tests/test_capture_geometry.py',
    'tests/test_color.py',
    'tests/test_video_color.py',
    'tests/test_processing.py',
    'tests/test_pipeline.py',
    'tests/test_canonical_canvas.py',
    'tests/test_output_geometry.py',
    'tests/test_visual_consistency_e2e.py',
    'tests/test_visual_consistency_qualification.py',
  ];
  for (const id of [
    'mediapipe-python-3.11-wheel',
    'mediapipe-python-3.12-wheel',
    'rvm-python-3.11-wheel',
    'rvm-python-3.12-wheel',
  ]) {
    const entry = matrixEntryBlock(block, id);
    for (const filename of requiredTests) assert.match(entry, new RegExp(filename), id);
  }
});

test('OpenCV compatibility profiles exercise every visual-consistency contract', () => {
  const block = jobs.get('opencv-compatibility');
  for (const filename of [
    'tests/test_geometry.py',
    'tests/test_background_geometry.py',
    'tests/test_capture.py',
    'tests/test_capture_geometry.py',
    'tests/test_color.py',
    'tests/test_video_color.py',
    'tests/test_processing.py',
    'tests/test_pipeline.py',
    'tests/test_canonical_canvas.py',
    'tests/test_output_geometry.py',
    'tests/test_visual_consistency_e2e.py',
    'tests/test_visual_consistency_qualification.py',
  ]) {
    assert.match(block, new RegExp(filename), filename);
  }
});

test('migration qualification consumes exact candidate and unpublished reference artifacts', () => {
  const block = jobs.get('migrations');
  assert.match(block, /name: custback-unpublished-reference-0\.3\.0/);
  assert.match(block, /node scripts\/release\/qualify-migrations\.js/);
  for (const option of [
    'wheel', 'sdist', 'npm', 'reference-wheel', 'reference-sdist', 'reference-npm', 'report',
  ]) {
    assert.match(block, new RegExp(`--${option}(?: | \\\\)`), option);
  }
  assert.doesNotMatch(block, /CUSTBACK_MIGRATION_FIXTURE_ID|pytest/);
  assert.match(block, /custback-migration-\$\{\{ matrix\.id \}\}/);
});

test('dynamic migration and two-host reports are retained and consumed by evidence assembly', () => {
  for (const [job, reportPrefix, artifactPrefix] of [
    ['two-host-tls', 'two-host-', 'custback-two-host-'],
    ['two-clean-hosts', 'clean-host-', 'custback-clean-host-'],
  ]) {
    const block = jobs.get(job);
    assert.match(block, /--result/);
    assert.match(block, new RegExp(`${reportPrefix}\\$\\{\\{ matrix\\.id \\}\\}\\.json`));
    assert.match(block, new RegExp(`name: ${artifactPrefix}\\$\\{\\{ matrix\\.id \\}\\}`));
    assert.match(block, new RegExp(UPLOAD.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  }
  const evidence = jobs.get('phase6-evidence');
  for (const pattern of [
    'custback-migration-*', 'custback-two-host-*', 'custback-clean-host-*',
  ]) {
    assert.match(evidence, new RegExp(`pattern: ${pattern.replace('*', '\\*')}`));
  }
  assert.equal((evidence.match(/merge-multiple: true/g) || []).length, 3);
  assert.match(evidence, /migration-reports/);
  assert.match(evidence, /two-host-reports/);
  assert.match(evidence, /clean-host-reports/);
});

test('same-run evidence is itself attested before trusted validation and upload', () => {
  const block = jobs.get('phase6-evidence');
  const assemble = block.indexOf('assemble-evidence.js assemble');
  const attest = block.indexOf(ATTEST);
  const validate = block.indexOf('phase6-evidence.js validate-trusted');
  const upload = block.lastIndexOf(UPLOAD);
  assert.ok(assemble >= 0 && assemble < attest);
  assert.ok(attest < validate && validate < upload);
  assert.match(block, /subject-path: \$\{\{ runner\.temp \}\}\/phase6-evidence\.json/);
});

test('same-run evidence and aggregate gates always evaluate every exact required job', () => {
  const evidence = jobs.get('phase6-evidence');
  const aggregate = jobs.get('release-gate');
  assert.match(evidence, /^    if: \$\{\{ always\(\) \}\}$/m);
  assert.match(aggregate, /^    if: \$\{\{ always\(\) \}\}$/m);
  assert.deepEqual(
    sorted(needsFrom(evidence)),
    sorted(manifest.workflow.required_job_ids.filter((id) => id !== 'phase6-evidence')),
  );
  assert.deepEqual(sorted(needsFrom(aggregate)), sorted(manifest.workflow.required_job_ids));
  assert.match(evidence, /assemble-evidence\.js assemble/);
  assert.match(evidence, /assemble-evidence\.js template/);
  assert.match(evidence, /phase6-evidence\.js validate-trusted/);
  assert.match(aggregate, /phase6-evidence\.js validate-trusted/);
  assert.match(aggregate, /toJSON\(needs\)/);
});

test('release gate and publish reverify exact-repository GitHub attestations', () => {
  for (const id of ['release-gate', 'publish']) {
    const block = jobs.get(id);
    assert.match(block, /gh attestation verify/);
    assert.match(block, /--repo "\$GITHUB_REPOSITORY"/);
    assert.match(
      block,
      /--signer-workflow "\$GITHUB_REPOSITORY\/\.github\/workflows\/release\.yml"/,
    );
    assert.match(block, /--signer-digest "\$GITHUB_SHA"/);
    assert.match(block, /--source-digest "\$GITHUB_SHA"/);
    assert.match(block, /--source-ref "\$GITHUB_REF"/);
    assert.match(block, /--deny-self-hosted-runners/);
    for (const suffix of ['\\.whl', '\\.tar\\.gz', '\\.tgz']) {
      assert.match(block, new RegExp(`candidate/\\*${suffix}`));
    }
  }
});

test('publish depends only on a successful release-gate and uploads exact files', () => {
  const block = jobs.get('publish');
  assert.deepEqual(needsFrom(block), ['release-gate']);
  assert.match(block, /if: \$\{\{ needs\.release-gate\.result == 'success' &&/);
  assert.match(block, /phase6-evidence\.js validate-trusted/);
  assert.match(block, /python -m twine upload --non-interactive "\$wheel" "\$sdist"/);
  assert.match(block, /npm publish "\$tarball" --access public --provenance/);
  assert.doesNotMatch(block, /build-candidate\.js|python -m build|npm pack/);
  assert.deepEqual(needsFrom(block), [manifest.workflow.aggregate_job_id]);
});
