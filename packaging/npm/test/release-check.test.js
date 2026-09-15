'use strict';

const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const zlib = require('node:zlib');

const release = require('../../../scripts/release/verify-release');

const root = path.resolve(__dirname, '..', '..', '..');

function tarballWithFile(name, contents) {
  const data = Buffer.from(contents);
  const header = Buffer.alloc(512);
  header.write(name, 0, 100, 'utf8');
  header.write('0000644\0', 100, 8, 'ascii');
  header.write('0000000\0', 108, 8, 'ascii');
  header.write('0000000\0', 116, 8, 'ascii');
  header.write(`${data.length.toString(8).padStart(11, '0')}\0`, 124, 12, 'ascii');
  header[156] = '0'.charCodeAt(0);
  const padding = Buffer.alloc(Math.ceil(data.length / 512) * 512 - data.length);
  return zlib.gzipSync(Buffer.concat([header, data, padding, Buffer.alloc(1024)]));
}

test('release metadata versions and required compatibility bounds agree', () => {
  assert.equal(release.verifyVersions(root), '0.4.0');
  assert.doesNotThrow(() => release.verifyNpmMetadata(root));
  assert.doesNotThrow(() => release.verifyDependencies(root));
  assert.doesNotThrow(() => release.verifyLicenseMetadata(root));
  assert.doesNotThrow(() => release.verifyCoreConfigTemplate(root));
  assert.doesNotThrow(() => release.verifyDocs(root));
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(root));
  assert.doesNotThrow(() => release.verifyMattePolicyRollout(root));
  assert.doesNotThrow(() => release.verifyCiWorkflow(root));
  assert.doesNotThrow(() => release.verifyPlatformScope(root));
});

test('CI vision compatibility profiles retain integrated qualification coverage', () => {
  const workflow = fs.readFileSync(
    path.join(root, '.github', 'workflows', 'full-ci.yml'),
    'utf8',
  );
  const jobBlock = (id) => {
    const marker = `  ${id}:\n`;
    const start = workflow.indexOf(marker);
    assert.notEqual(start, -1, `missing CI job ${id}`);
    const remainder = workflow.slice(start + marker.length);
    const next = remainder.search(/\n  [a-z0-9]+(?:-[a-z0-9]+)*:\n/);
    return workflow.slice(
      start,
      next < 0 ? workflow.length : start + marker.length + next,
    );
  };
  const opencv = jobBlock('opencv-compatibility');
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
    assert.match(opencv, new RegExp(filename), filename);
  }
  const optional = jobBlock('optional-backends');
  assert.match(optional, /timeout 180s python -m pytest -q/);
  assert.doesNotMatch(optional, /--ignore|--deselect/);
});

test('matte defaults remain held until every exact release authority changes', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-matte-rollout-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  for (const relative of [
    'config/default.yaml',
    'scripts/release/matte-policy-rollout.json',
    'src/custback/api/webui.py',
    'src/custback/default.yaml',
    'src/custback/system-profile-catalog.json',
  ]) {
    const target = path.join(fixture, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.copyFileSync(path.join(root, relative), target);
  }
  const source = JSON.parse(fs.readFileSync(
    path.join(root, 'scripts', 'release', 'matte-policy-rollout.json'),
    'utf8',
  ));
  const manifestPath = path.join(
    fixture, 'scripts', 'release', 'matte-policy-rollout.json',
  );
  const runtimeContract = {
    patch: structuredClone(source.compatibility_policy.patch),
    status: {
      schema: 'custback.matte-rollout-status',
      version: 1,
      stage: 'compatibility_hold',
      decision: 'held_pending_physical_qualification',
      configured_schema_version: 1,
      config_version: 0,
      qualified_default_active: false,
      preset_catalog_version: 1,
      preset_evidence_status: 'not_qualified',
      legacy_policy_available: true,
      legacy_policy_active: true,
      rollback_patch_id: 'matte-legacy-v1',
      patch_attempts: 0,
      patch_in_flight: 0,
      patch_successes: 0,
      patch_failures: 0,
      legacy_rollbacks: 0,
      last_outcome: 'none',
    },
  };
  const verify = (manifest, contract = runtimeContract) => {
    fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
    return release.verifyMattePolicyRollout(fixture, {
      runtimeContract: contract,
    });
  };

  assert.doesNotThrow(() => verify(source));

  const promoted = structuredClone(source);
  promoted.active_stage = 'qualified_default';
  assert.throws(() => verify(promoted), /manifest header is invalid/);

  const privateField = structuredClone(source);
  privateField.private_report_path = '/private/qualification/report.json';
  assert.throws(() => verify(privateField), /manifest header is invalid/);

  const preset = structuredClone(source);
  preset.preset_catalog.profiles[0].quality_claim = true;
  assert.throws(() => verify(preset), /catalog or patch digest disagrees/);

  const catalogPath = path.join(
    fixture, 'src', 'custback', 'system-profile-catalog.json',
  );
  const catalogBytes = fs.readFileSync(catalogPath);
  const locallyScreenedCatalog = JSON.parse(catalogBytes.toString('utf8'));
  locallyScreenedCatalog.axes.quality.profiles.balanced.evidence_state =
    'locally_screened';
  const locallyScreenedBytes = Buffer.from(
    `${JSON.stringify(locallyScreenedCatalog, null, 2)}\n`,
  );
  fs.writeFileSync(catalogPath, locallyScreenedBytes);
  const locallyScreened = structuredClone(source);
  locallyScreened.preset_catalog.catalog_sha256 = crypto.createHash('sha256')
    .update(locallyScreenedBytes).digest('hex');
  locallyScreened.preset_catalog.profiles.find(
    (profile) => profile.axis === 'quality' && profile.id === 'balanced',
  ).evidence_status = 'locally_screened';
  assert.doesNotThrow(() => verify(locallyScreened));
  fs.writeFileSync(catalogPath, catalogBytes);

  const fabricated = structuredClone(source);
  fabricated.promotion.status = 'qualified';
  fabricated.promotion.authority = 'operator-asserted';
  assert.throws(() => verify(fabricated), /pending physical evidence/);

  const driftedPatch = structuredClone(source);
  driftedPatch.compatibility_policy.patch.segmentation.threshold = 0.6;
  assert.throws(() => verify(driftedPatch), /compatibility patch or digest/);

  const omittedEvidence = structuredClone(source);
  omittedEvidence.promotion.evidence.pop();
  assert.throws(() => verify(omittedEvidence), /pending physical evidence/);

  const destructive = structuredClone(source);
  destructive.rollback.delete_model_cache = true;
  assert.throws(() => verify(destructive), /rollback or reaction separation/);

  const reactions = structuredClone(source);
  reactions.reactions.included = true;
  assert.throws(() => verify(reactions), /rollback or reaction separation/);

  const helperDrift = structuredClone(runtimeContract);
  helperDrift.patch.segmentation.threshold = 0.6;
  assert.throws(
    () => verify(source, helperDrift),
    /Python rollback\/status contract disagrees/,
  );

  const statusDrift = structuredClone(runtimeContract);
  statusDrift.status.qualified_default_active = true;
  assert.throws(
    () => verify(source, statusDrift),
    /Python rollback\/status contract disagrees/,
  );
  const privateStatus = structuredClone(runtimeContract);
  privateStatus.status.raw_error = '/private/provider/error';
  assert.throws(
    () => verify(source, privateStatus),
    /Python rollback\/status contract disagrees/,
  );
});

test('visual defaults cannot advance without a distinct commit and approval record', (t) => {
  const verifier = fs.readFileSync(
    path.join(root, 'scripts', 'release', 'verify-release.js'),
    'utf8',
  );
  assert.match(verifier, /visual_consistency_qualification\.py/);
  assert.match(verifier, /'--claim', 'release'/);
  assert.match(verifier, /'--expected-commit', expectedCommit/);
  assert.match(verifier, /'--evidence-root', path\.dirname\(reportPath\)/);

  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-rollout-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'config'), { recursive: true });
  fs.mkdirSync(path.join(fixture, 'docs'), { recursive: true });
  fs.mkdirSync(path.join(fixture, 'scripts', 'release'), { recursive: true });
  const sourceManifest = JSON.parse(fs.readFileSync(
    path.join(root, 'scripts', 'release', 'visual-policy-rollout.json'),
    'utf8',
  ));
  const manifestPath = path.join(
    fixture, 'scripts', 'release', 'visual-policy-rollout.json',
  );
  const configPath = path.join(fixture, 'config', 'default.yaml');
  const writeManifest = (manifest) => {
    fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  };
  fs.writeFileSync(
    configPath,
    'schema_version: 1\ncamera:\n  fit_mode: stretch\n' +
      'compositing:\n  blend_space: srgb_legacy\n' +
      '  color_correction:\n    mode: "off"\n',
  );
  writeManifest(sourceManifest);
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(fixture));

  const unapproved = structuredClone(sourceManifest);
  unapproved.active_stage = 'camera-cover';
  unapproved.stages[0].status = 'complete';
  unapproved.stages[1].status = 'active';
  fs.writeFileSync(
    configPath,
    'schema_version: 2\ncamera:\n  fit_mode: cover\n' +
      'compositing:\n  blend_space: srgb_legacy\n' +
      '  color_correction:\n    mode: "off"\n',
  );
  writeManifest(unapproved);
  assert.throws(
    () => release.verifyVisualPolicyRollout(fixture),
    /distinct commit-bound evidence/,
  );

  const commit = 'a'.repeat(40);
  const evidence = 'docs/camera-cover-approval.json';
  unapproved.stages[1].change_commit = commit;
  unapproved.stages[1].evidence = [evidence];
  fs.writeFileSync(
    path.join(fixture, evidence),
    `${JSON.stringify({
      release_qualified: true,
      source: { commit, clean: true },
    })}\n`,
  );
  writeManifest(unapproved);
  let strictValidationCalls = 0;
  const qualificationValidator = (reportPath, validation) => {
    strictValidationCalls += 1;
    assert.equal(validation.expectedCommit, commit);
    assert.equal(validation.evidenceRoot, path.dirname(reportPath));
    return JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  };
  fs.writeFileSync(
    path.join(fixture, evidence),
    `${JSON.stringify({
      release_qualified: true,
      source: { commit: 'b'.repeat(40), clean: true },
    })}\n`,
  );
  assert.throws(
    () => release.verifyVisualPolicyRollout(
      fixture,
      { qualificationValidator },
    ),
    /not bound to its clean change commit/,
  );
  fs.writeFileSync(
    path.join(fixture, evidence),
    `${JSON.stringify({
      release_qualified: true,
      source: { commit, clean: true },
    })}\n`,
  );
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(
    fixture,
    { qualificationValidator },
  ));
  assert.equal(strictValidationCalls, 2);

  const linearCommit = 'c'.repeat(40);
  const linearEvidence = 'docs/linear-compositing-approval.json';
  const linear = structuredClone(unapproved);
  linear.active_stage = 'linear-compositing';
  linear.stages[1].status = 'complete';
  linear.stages[2].status = 'active';
  linear.stages[2].change_commit = linearCommit;
  linear.stages[2].evidence = [linearEvidence];
  fs.writeFileSync(
    path.join(fixture, linearEvidence),
    `${JSON.stringify({
      release_qualified: true,
      source: { commit: linearCommit, clean: true },
    })}\n`,
  );
  fs.writeFileSync(
    configPath,
    'schema_version: 3\ncamera:\n  fit_mode: cover\n' +
      'compositing:\n  blend_space: linear_srgb\n' +
      '  color_correction:\n    mode: "off"\n',
  );
  writeManifest(linear);
  const validatedCommits = [];
  const historicalValidator = (reportPath, validation) => {
    const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    validatedCommits.push(validation.expectedCommit);
    assert.equal(validation.expectedCommit, report.source.commit);
    assert.equal(validation.evidenceRoot, path.dirname(reportPath));
    return report;
  };
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(
    fixture,
    { qualificationValidator: historicalValidator },
  ));
  assert.deepEqual(validatedCommits, [commit, linearCommit]);

  const automaticCommit = 'd'.repeat(40);
  const automaticEvidence = 'docs/automatic-correction-approval.json';
  const automatic = structuredClone(linear);
  automatic.active_stage = 'automatic-correction';
  automatic.stages[2].status = 'complete';
  automatic.stages[3].status = 'active';
  automatic.stages[3].change_commit = automaticCommit;
  automatic.stages[3].evidence = [automaticEvidence];
  fs.writeFileSync(
    path.join(fixture, automaticEvidence),
    `${JSON.stringify({
      release_qualified: true,
      source: { commit: automaticCommit, clean: true },
    })}\n`,
  );
  fs.writeFileSync(
    configPath,
    'schema_version: 4\ncamera:\n  fit_mode: cover\n' +
      'compositing:\n  blend_space: linear_srgb\n' +
      '  color_correction:\n    mode: auto\n',
  );
  writeManifest(automatic);
  validatedCommits.length = 0;
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(
    fixture,
    { qualificationValidator: historicalValidator },
  ));
  assert.deepEqual(
    validatedCommits,
    [commit, linearCommit, automaticCommit],
  );

  fs.writeFileSync(
    configPath,
    'schema_version: 2\ncamera:\n  fit_mode: stretch\n' +
      'compositing:\n  blend_space: srgb_legacy\n' +
      '  color_correction:\n    mode: "off"\n',
  );
  writeManifest(unapproved);
  assert.throws(
    () => release.verifyVisualPolicyRollout(fixture, { qualificationValidator }),
    /does not match the active rollout stage/,
  );
});

test('staged prepack binds approval files to a clean trusted Git source', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-rollout-bridge-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const trusted = path.join(fixture, 'trusted');
  const staged = path.join(fixture, 'staged');
  fs.mkdirSync(trusted);
  fs.mkdirSync(staged);

  const git = (args) => {
    const result = spawnSync('git', ['-C', trusted, ...args], {
      encoding: 'utf8',
    });
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  git(['init', '--quiet']);
  git(['config', 'user.name', 'Custback Release Test']);
  git(['config', 'user.email', 'release-test@invalid.example']);

  const write = (relative, contents) => {
    const target = path.join(trusted, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, contents);
  };
  const compatibilityConfig =
    'schema_version: 1\ncamera:\n  fit_mode: stretch\n' +
    'compositing:\n  blend_space: srgb_legacy\n' +
    '  color_correction:\n    mode: "off"\n';
  const coverConfig =
    'schema_version: 2\ncamera:\n  fit_mode: cover\n' +
    'compositing:\n  blend_space: srgb_legacy\n' +
    '  color_correction:\n    mode: "off"\n';
  const rollout = JSON.parse(fs.readFileSync(
    path.join(root, 'scripts', 'release', 'visual-policy-rollout.json'),
    'utf8',
  ));
  write('config/default.yaml', compatibilityConfig);
  write('src/custback/default.yaml', compatibilityConfig);
  write(
    'scripts/release/visual-policy-rollout.json',
    `${JSON.stringify(rollout, null, 2)}\n`,
  );
  write('scripts/release/visual-qualification-manifest.json', '{}\n');
  write('scripts/release/visual_consistency_qualification.py', '# test fixture\n');
  git(['add', '.']);
  git(['commit', '--quiet', '-m', 'compatibility policy']);

  // The isolated visual-default commit exists before its evidence, avoiding a
  // self-referential commit hash in the report or ledger.
  write('config/default.yaml', coverConfig);
  write('src/custback/default.yaml', coverConfig);
  git(['add', 'config/default.yaml', 'src/custback/default.yaml']);
  git(['commit', '--quiet', '-m', 'enable camera cover default']);
  const changeCommit = git(['rev-parse', '--verify', 'HEAD']);

  const evidence = 'docs/camera-cover-approval.json';
  const approved = structuredClone(rollout);
  approved.active_stage = 'camera-cover';
  approved.stages[0].status = 'complete';
  approved.stages[1].status = 'active';
  approved.stages[1].change_commit = changeCommit;
  approved.stages[1].evidence = [evidence];
  write(
    'scripts/release/visual-policy-rollout.json',
    `${JSON.stringify(approved, null, 2)}\n`,
  );
  write(
    evidence,
    `${JSON.stringify({
      release_qualified: true,
      source: { commit: changeCommit, clean: true },
    })}\n`,
  );
  git(['add', 'scripts/release/visual-policy-rollout.json', evidence]);
  git(['commit', '--quiet', '-m', 'attach camera cover approval']);
  const approvalCommit = git(['rev-parse', '--verify', 'HEAD']);
  const approvalTree = git(['rev-parse', '--verify', 'HEAD^{tree}']);
  assert.notEqual(approvalCommit, changeCommit);

  const stagedPaths = [
    'config/default.yaml',
    evidence,
    'scripts/release/visual-policy-rollout.json',
    'scripts/release/visual-qualification-manifest.json',
    'scripts/release/visual_consistency_qualification.py',
    'src/custback/default.yaml',
  ];
  const syncStage = () => {
    for (const relative of stagedPaths) {
      const target = path.join(staged, relative);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.copyFileSync(path.join(trusted, relative), target);
    }
  };
  syncStage();
  const bridgeEnv = {
    CUSTBACK_RELEASE_GIT_ROOT: fs.realpathSync(trusted),
    CUSTBACK_RELEASE_SOURCE_COMMIT: approvalCommit,
    CUSTBACK_RELEASE_SOURCE_TREE: approvalTree,
  };
  const committedFiles = Object.fromEntries(
    stagedPaths.map((relative) => [
      relative,
      fs.readFileSync(path.join(trusted, relative)),
    ]),
  );
  const gitRunner = (_command, args, options) => {
    const commandArgs = args.slice(2);
    let stdout;
    if (commandArgs[0] === 'rev-parse' &&
        commandArgs[1] === '--show-toplevel') {
      stdout = `${fs.realpathSync(trusted)}\n`;
    } else if (commandArgs.join(' ') === 'rev-parse --verify HEAD') {
      stdout = `${approvalCommit}\n`;
    } else if (commandArgs.join(' ') === 'rev-parse --verify HEAD^{tree}') {
      stdout = `${approvalTree}\n`;
    } else if (commandArgs[0] === 'status') {
      stdout = fs.existsSync(path.join(trusted, 'untracked'))
        ? '?? untracked\n'
        : '';
    } else if (commandArgs[0] === 'show') {
      const separator = commandArgs[1].indexOf(':');
      const commit = commandArgs[1].slice(0, separator);
      const relative = commandArgs[1].slice(separator + 1);
      if (commit !== approvalCommit || !committedFiles[relative]) {
        return { status: 1, stdout: '', stderr: 'unknown object' };
      }
      stdout = Buffer.from(committedFiles[relative]);
    } else {
      return { status: 1, stdout: '', stderr: 'unexpected git command' };
    }
    if (options.encoding === null) {
      stdout = Buffer.isBuffer(stdout) ? stdout : Buffer.from(stdout);
    }
    return { status: 0, stdout, stderr: options.encoding === null ? Buffer.alloc(0) : '' };
  };
  let validationCalls = 0;
  const qualificationValidator = (reportPath, validation) => {
    validationCalls += 1;
    assert.equal(validation.expectedCommit, changeCommit);
    return JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  };
  assert.doesNotThrow(() => release.verifyVisualPolicyRollout(staged, {
    env: bridgeEnv,
    gitRunner,
    qualificationValidator,
  }));
  assert.equal(validationCalls, 1);

  fs.appendFileSync(path.join(staged, evidence), ' \n');
  assert.throws(
    () => release.verifyVisualPolicyRollout(staged, {
      env: bridgeEnv,
      gitRunner,
      qualificationValidator,
    }),
    /differs from trusted commit/,
  );
  syncStage();

  fs.writeFileSync(path.join(trusted, 'untracked'), 'dirty');
  assert.throws(
    () => release.verifyVisualPolicyRollout(staged, {
      env: bridgeEnv,
      gitRunner,
      qualificationValidator,
    }),
    /not the exact clean source commit/,
  );
  fs.rmSync(path.join(trusted, 'untracked'));
  assert.throws(
    () => release.verifyVisualPolicyRollout(staged, {
      env: {
        ...bridgeEnv,
        CUSTBACK_RELEASE_SOURCE_TREE: 'f'.repeat(40),
      },
      gitRunner,
      qualificationValidator,
    }),
    /not the exact clean source commit/,
  );
});

test('prepack verifies the non-recursive packlist and reserves artifact installs', () => {
  assert.deepEqual(release.releasePlan(['--prepack']), {
    builtArtifacts: false,
    success: 'metadata and non-recursive npm packlist verified',
  });
  assert.equal(release.releasePlan([]).builtArtifacts, true);
  assert.equal(release.releasePlan(['--quick']).builtArtifacts, false);
  assert.throws(() => release.releasePlan(['--prepack', '--quick']), /invalid release/);
  assert.throws(() => release.releasePlan(['--package-smoke']), /invalid release/);
  assert.throws(() => release.releasePlan(['--unknown']), /invalid release/);

  const source = fs.readFileSync(
    path.join(root, 'scripts', 'release', 'verify-release.js'), 'utf8',
  );
  assert.match(
    source,
    /\['pack', '--dry-run', '--json', '--ignore-scripts'/,
    'packlist verification must suppress lifecycle recursion',
  );
});

test('full release temporary roots reject Linux memory-backed filesystems', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-release-base-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const baseEnv = { CUSTBACK_RELEASE_TMPDIR: fixture };
  const tmpfs = () => ({ type: 0x01021994 });
  const ramfs = () => ({ type: 0x858458f6 });
  const disk = () => ({ type: 0xef53 });

  assert.equal(release.filesystemIsMemoryBacked(fixture, tmpfs, 'linux'), true);
  assert.equal(release.filesystemIsMemoryBacked(fixture, ramfs, 'linux'), true);
  assert.equal(release.filesystemIsMemoryBacked(fixture, tmpfs, 'darwin'), false);
  assert.throws(
    () => release.filesystemIsMemoryBacked(fixture, null, 'linux'),
    /cannot determine whether release temporary storage is memory-backed/,
  );
  assert.throws(
    () => release.createReleaseTemporaryRoot({ env: baseEnv, statfs: tmpfs, platform: 'linux' }),
    /memory-backed filesystem/,
  );
  assert.throws(
    () => release.createReleaseTemporaryRoot({ env: baseEnv, statfs: ramfs, platform: 'linux' }),
    /memory-backed filesystem/,
  );

  const allowed = release.createReleaseTemporaryRoot({
    env: { ...baseEnv, CUSTBACK_RELEASE_ALLOW_TMPFS: '1' },
    statfs: tmpfs,
    platform: 'linux',
  });
  assert.equal(path.dirname(allowed), fs.realpathSync(fixture));
  assert.equal(fs.statSync(allowed).isDirectory(), true);
  fs.rmSync(allowed, { recursive: true, force: true });

  const diskRoot = release.createReleaseTemporaryRoot({
    env: baseEnv,
    statfs: disk,
    platform: 'linux',
  });
  assert.equal(fs.statSync(diskRoot).isDirectory(), true);
  fs.rmSync(diskRoot, { recursive: true, force: true });

  const checkout = fs.mkdtempSync(path.join(fixture, 'checkout-'));
  const nestedBase = path.join(checkout, 'release-scratch');
  fs.mkdirSync(nestedBase);
  assert.throws(
    () => release.createReleaseTemporaryRoot({
      env: { CUSTBACK_RELEASE_TMPDIR: nestedBase },
      root: checkout,
      statfs: disk,
      platform: 'linux',
    }),
    /outside the source checkout/,
  );
});

test('release profile directories are disposed after success and failure', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-disposable-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));

  const successful = path.join(fixture, 'successful');
  assert.equal(release.withDisposableDirectory(successful, (directory) => {
    fs.mkdirSync(directory);
    fs.writeFileSync(path.join(directory, 'payload'), 'large environment');
    return 42;
  }), 42);
  assert.equal(fs.existsSync(successful), false);

  const failed = path.join(fixture, 'failed');
  assert.throws(() => release.withDisposableDirectory(failed, (directory) => {
    fs.mkdirSync(directory);
    fs.writeFileSync(path.join(directory, 'payload'), 'partial environment');
    throw new Error('profile failed');
  }), /profile failed/);
  assert.equal(fs.existsSync(failed), false);

  const existing = path.join(fixture, 'existing');
  fs.mkdirSync(existing);
  assert.throws(
    () => release.withDisposableDirectory(existing, () => undefined),
    /refusing to reuse/,
  );
  assert.equal(fs.existsSync(existing), true);
});

test('release scratch environment is relocated and restored after failures', () => {
  const env = { TMPDIR: '/old/tmpdir', TEMP: '/old/temp', KEEP: 'unchanged' };
  const before = { ...env };
  assert.throws(() => release.withTemporaryEnvironment({
    TMPDIR: '/disk/release',
    TMP: '/disk/release',
    TEMP: '/disk/release',
  }, () => {
    assert.equal(env.TMPDIR, '/disk/release');
    assert.equal(env.TMP, '/disk/release');
    assert.equal(env.TEMP, '/disk/release');
    throw new Error('artifact failure');
  }, env), /artifact failure/);
  assert.deepEqual(env, before);
});

test('release native build parallelism is memory-bounded by default', () => {
  assert.equal(release.releaseBuildJobs({}), '2');
  assert.equal(release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: '1' }), '1');
  assert.equal(release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: '32' }), '32');
  for (const invalid of ['', '0', '-1', '2.5', '33', 'many']) {
    assert.throws(
      () => release.releaseBuildJobs({ CUSTBACK_RELEASE_BUILD_JOBS: invalid }),
      /integer from 1 through 32/,
    );
  }
});

test('advertised built-wheel extra profiles cover every platform-feasible group', () => {
  assert.deepEqual(
    release.extraArtifactProfiles('linux', 'x64').map((profile) => profile.name),
    ['mediapipe', 'rvm', 'gpu', 'audio2face', 'dev'],
  );
  assert.deepEqual(
    release.extraArtifactProfiles('darwin', 'x64').map((profile) => profile.name),
    ['mediapipe', 'rvm', 'audio2face', 'dev'],
  );
  const audio2face = release.extraArtifactProfiles('linux', 'x64')
    .find((profile) => profile.name === 'audio2face');
  assert.deepEqual(audio2face.extras, ['audio2face', 'dev']);
  assert.equal(audio2face.test, 'tests/test_audio2face_protocol.py');
});

test('npm artifact LICENSE must be byte-identical to the canonical source', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-license-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const canonical = release.canonicalLicense(root);
  const matching = path.join(fixture, 'matching.tgz');
  const changed = path.join(fixture, 'changed.tgz');
  fs.writeFileSync(matching, tarballWithFile('package/LICENSE', canonical));
  fs.writeFileSync(changed, tarballWithFile('package/LICENSE', `${canonical}changed\n`));
  assert.doesNotThrow(() => release.verifyNpmArtifactLicense(matching, root));
  assert.throws(
    () => release.verifyNpmArtifactLicense(changed, root),
    /not byte-identical/,
  );
});

test('remediation registry records every registered blocker resolved', () => {
  const registry = release.remediationRegistry(root);
  const blockers = release.remediationBlockers(root);
  assert.equal(registry.phase, 6);
  assert.equal(registry.release_blocked, false);
  assert.equal(blockers.length, 31);
  assert.deepEqual(blockers.map((entry) => entry.id), [
    'SEC-01', 'TOKEN-01', 'TRANS-01', 'PRIV-01', 'A2F-01',
    'CFG-01', 'CFG-02', 'LIFE-01', 'LIFE-02', 'STOR-01', 'STOR-02',
    'SEG-01', 'SEG-02', 'SEG-03', 'RENDER-01', 'RENDER-02', 'API-01',
    'NPM-01', 'PKG-01', 'PKG-02', 'DEPLOY-01', 'MISC-01', 'MISC-02',
    'LICENSE-01', 'PLATFORM-01', 'HYGIENE-01', 'SEC-02', 'REL-01', 'DEP-01',
    'STATUS-01', 'LOCK-01',
  ]);
  assert.equal(blockers.filter((entry) => entry.status === 'resolved').length, 31);
  assert.deepEqual(
    blockers.filter((entry) => entry.status === 'open').map((entry) => entry.id),
    [],
  );
  assert.equal(blockers.find((entry) => entry.id === 'REL-01').phase, 6);
  assert.doesNotThrow(() => release.verifyBlockerRegressionCoverage(root));
  assert.doesNotThrow(() => release.verifyNoReleaseBlockers(root));
});

function assertPhase6ReleaseIntegrity() {
  const workflow = fs.readFileSync(
    path.join(root, '.github', 'workflows', 'release.yml'),
    'utf8',
  );
  assert.match(workflow, /^\s{2}publish:\s*$/m);
  assert.equal(
    fs.existsSync(path.join(root, 'scripts', 'release', 'required-gates.json')),
    true,
    'the versioned required-gate allow-list is missing',
  );
  assert.match(workflow, /needs:\s*\[[^\]]*release-gate[^\]]*\]/s);
}

test(
  'REL-01: production publish requires exact Phase 6 gate evidence',
  () => {
    assertPhase6ReleaseIntegrity();
    const rel = release.remediationBlockers(root)
      .find((entry) => entry.id === 'REL-01');
    assert.equal(rel.status, 'resolved');
  },
);

test(
  'REL-01: resolved registry no longer blocks the installed Phase 6 publish machinery',
  () => {
    assert.doesNotThrow(assertPhase6ReleaseIntegrity);
    assert.doesNotThrow(() => release.verifyNoReleaseBlockers(root));
  },
);

test('remediation registry fails closed on missing, malformed, or inconsistent state', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-blocker-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const directory = path.join(fixture, 'scripts', 'release');
  fs.mkdirSync(directory, { recursive: true });

  assert.throws(
    () => release.remediationBlockers(fixture),
    /registry is missing/,
  );

  const registryPath = path.join(directory, 'remediation-blockers.json');
  fs.writeFileSync(registryPath, '{bad');
  assert.throws(
    () => release.remediationBlockers(fixture),
    /invalid JSON/,
  );

  fs.writeFileSync(registryPath, JSON.stringify({
    schema_version: 2,
    phase: 5,
    release_blocked: false,
    blockers: [
      {
        id: 'SEC-01', phase: 1, status: 'open', title: 'still open',
        regression: {
          runner: 'node', file: 'tests/security.test.js', test: 'SEC-01 regression',
        },
      },
    ],
  }));
  assert.throws(
    () => release.remediationBlockers(fixture),
    /release_blocked must be true/,
  );

  fs.writeFileSync(registryPath, JSON.stringify({
    schema_version: 2,
    phase: 5,
    release_blocked: true,
    blockers: [
      {
        id: 'SEC-01', phase: 1, status: 'open', title: 'first',
        regression: {
          runner: 'node', file: 'tests/security.test.js', test: 'SEC-01 first',
        },
      },
      {
        id: 'SEC-01', phase: 1, status: 'resolved', title: 'duplicate',
        regression: {
          runner: 'node', file: 'tests/security.test.js', test: 'SEC-01 duplicate',
        },
      },
    ],
  }));
  assert.throws(
    () => release.remediationBlockers(fixture),
    /duplicate id SEC-01/,
  );
});

test('remediation registry requires an exact executable test identifier', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-regression-id-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'scripts', 'release'), { recursive: true });
  fs.mkdirSync(path.join(fixture, 'tests'));
  fs.writeFileSync(
    path.join(fixture, 'tests', 'regression.py'),
    '# test_exact_node is mentioned but was never collected\n',
  );
  fs.writeFileSync(
    path.join(fixture, 'scripts', 'release', 'remediation-blockers.json'),
    JSON.stringify({
      schema_version: 2,
      phase: 5,
      release_blocked: false,
      blockers: [{
        id: 'SEC-01', phase: 1, status: 'resolved', title: 'exact identifier',
        regression: {
          runner: 'pytest', test: 'tests/regression.py::test_exact_node',
        },
      }],
    }),
  );
  assert.throws(
    () => release.verifyBlockerRegressionCoverage(
      fixture, { enforceReviewedContract: false },
    ),
    /pytest node is not defined/,
  );
});

test('reviewed remediation contract rejects blocker or node substitution', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-blocker-contract-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const directory = path.join(fixture, 'scripts', 'release');
  fs.mkdirSync(directory, { recursive: true });
  const registry = JSON.parse(
    fs.readFileSync(
      path.join(root, 'scripts', 'release', 'remediation-blockers.json'),
      'utf8',
    ),
  );
  registry.blockers[0] = {
    ...registry.blockers[0],
    id: 'FAKE-99',
    regression: registry.blockers[1].regression,
  };
  fs.writeFileSync(
    path.join(directory, 'remediation-blockers.json'),
    JSON.stringify(registry),
  );
  assert.throws(
    () => release.verifyBlockerRegressionCoverage(fixture),
    /reviewed Phase 5 blocker contract/,
  );
});

test('exact Node regression outcome rejects missing and duplicate TAP names', () => {
  const name = 'REL-01: exact TODO';
  assert.equal(
    release.exactNodeTapOutcome('REL-01', name, `ok 1 - ${name} # TODO open\n`),
    `ok 1 - ${name} # TODO open`,
  );
  assert.throws(
    () => release.exactNodeTapOutcome('REL-01', name, '1..0\n'),
    /0 exact TAP outcomes/,
  );
  assert.throws(
    () => release.exactNodeTapOutcome(
      'REL-01', name, `ok 1 - ${name} # TODO open\nok 2 - ${name}\n`,
    ),
    /2 exact TAP outcomes/,
  );
});

test('release version parity rejects a mismatched lock root', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-version-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'src', 'custback'), { recursive: true });
  fs.writeFileSync(path.join(fixture, 'package.json'), JSON.stringify({ version: '0.3.0' }));
  fs.writeFileSync(path.join(fixture, 'package-lock.json'), JSON.stringify({
    version: '0.3.0',
    packages: { '': { version: '0.2.0' } },
  }));
  fs.writeFileSync(path.join(fixture, 'pyproject.toml'), '[project]\nversion = "0.3.0"\n');
  fs.writeFileSync(path.join(fixture, 'src', 'custback', '__init__.py'), '__version__ = "0.3.0"\n');
  assert.throws(() => release.verifyVersions(fixture), /release version mismatch/);
});

test('source fallback verification ignores a matching comment decoy', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-fallback-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  fs.mkdirSync(path.join(fixture, 'src', 'custback'), { recursive: true });
  fs.copyFileSync(path.join(root, 'package.json'), path.join(fixture, 'package.json'));
  fs.copyFileSync(path.join(root, 'package-lock.json'), path.join(fixture, 'package-lock.json'));
  // Match the real package version so only the fallback line disagrees.
  const packageVersion = JSON.parse(
    fs.readFileSync(path.join(root, 'package.json'), 'utf8'),
  ).version;
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    `[project]\nversion = "${packageVersion}"\n`,
  );
  fs.writeFileSync(path.join(fixture, 'src', 'custback', '__init__.py'), `
from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("custback")
except PackageNotFoundError:
    # __version__ = "${packageVersion}"
    __version__ = "9.9.9"
`);
  assert.throws(
    () => release.verifyVersions(fixture),
    /source-tree __version__ fallback does not match package version/,
  );
});

test('npm metadata and lock root are exact reviewed allowlists', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-npm-metadata-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const packagePath = path.join(fixture, 'package.json');
  const lockPath = path.join(fixture, 'package-lock.json');
  const originalPackage = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const originalLock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));

  fs.writeFileSync(packagePath, JSON.stringify({
    ...originalPackage,
    scripts: { ...originalPackage.scripts, preinstall: 'node unexpected.js' },
  }));
  fs.writeFileSync(lockPath, JSON.stringify(originalLock));
  assert.throws(
    () => release.verifyNpmMetadata(fixture),
    /package.json metadata must exactly match/,
  );

  fs.writeFileSync(packagePath, JSON.stringify(originalPackage));
  fs.writeFileSync(lockPath, JSON.stringify({
    ...originalLock,
    packages: {
      '': { ...originalLock.packages[''], hasInstallScript: false },
    },
  }));
  assert.throws(
    () => release.verifyNpmMetadata(fixture),
    /package-lock.json must exactly mirror/,
  );
});

test('release and install probes remain active with Python optimization', () => {
  const files = [
    path.join(root, 'scripts', 'release', 'verify-release.js'),
    path.join(root, 'packaging', 'npm', 'install.js'),
    path.join(root, 'packaging', 'npm', 'custback.js'),
  ];
  for (const file of files) {
    assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /\bassert\s+/);
  }
  const previous = process.env.PYTHONOPTIMIZE;
  process.env.PYTHONOPTIMIZE = '2';
  try {
    assert.equal(release.verifyVersions(root), '0.4.0');
  } finally {
    if (previous === undefined) delete process.env.PYTHONOPTIMIZE;
    else process.env.PYTHONOPTIMIZE = previous;
  }
});

test('npm pack payload parsing rejects ambiguous or unsafe manifests', () => {
  const version = '0.3.0';
  const artifact = {
    name: 'custback',
    version,
    filename: `custback-${version}.tgz`,
    files: [{ path: 'package.json' }, { path: 'packaging/npm/custback.js' }],
  };
  assert.deepEqual(
    release.parseNpmPackPayload(JSON.stringify([artifact]), version).names,
    ['package.json', 'packaging/npm/custback.js'],
  );
  assert.throws(() => release.parseNpmPackPayload('{bad', version), /invalid JSON/);
  assert.throws(() => release.parseNpmPackPayload(JSON.stringify([]), version), /unexpected payload/);
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{ ...artifact, version: '9.9.9' }]), version),
    /identity is inconsistent/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{ ...artifact, files: [] }]), version),
    /no file manifest/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{
      ...artifact, files: [{ path: '../outside' }],
    }]), version),
    /invalid file entry/,
  );
  assert.throws(
    () => release.parseNpmPackPayload(JSON.stringify([{
      ...artifact, files: [{ path: 'package.json' }, { path: 'package.json' }],
    }]), version),
    /duplicate file entries/,
  );
});

test('dependency verification reads runtime tables instead of matching stray text', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-dependency-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const spec = "pyvirtualcam>=0.11,<1; sys_platform != 'win32' or (platform_machine != 'ARM64' and platform_machine != 'arm64')";
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    `${original.replace(`    "${spec}",\n`, '')}\n# "${spec}"\n`,
  );
  assert.throws(() => release.verifyDependencies(fixture), /missing bounded core dependency/);
});

test('dependency verification enforces the complete bounded dev test set', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-dev-dependency-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace('    "httpx2>=2,<3",\n', ''),
  );
  assert.throws(() => release.verifyDependencies(fixture), /invalid bounded dev dependencies/);
});

test('dependency verification rejects unreviewed extras and console scripts', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-extra-gate-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  const original = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace(
      '[project.scripts]',
      'unreviewed = ["totally-unbounded"]\n\n[project.scripts]',
    ),
  );
  assert.throws(
    () => release.verifyDependencies(fixture),
    /optional dependency groups must be exactly/,
  );

  fs.writeFileSync(
    path.join(fixture, 'pyproject.toml'),
    original.replace(
      'custback = "custback.__main__:main"',
      'custback = "custback.__main__:main"\nunreviewed = "evil:main"',
    ),
  );
  assert.throws(
    () => release.verifyDependencies(fixture),
    /console scripts must be exactly/,
  );
});

test('stale artifact reporting is deterministic and non-mutating', (t) => {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-release-test-'));
  t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
  for (const name of [
    'custback-0.1.0.tgz',
    'debug.txt',
    'onnxruntime_profile__2026-07-16_12-34-56.json',
    'uninstall.log',
    'keep.txt',
  ]) {
    fs.writeFileSync(path.join(fixture, name), name);
  }
  fs.mkdirSync(path.join(fixture, 'build'));
  fs.mkdirSync(path.join(fixture, 'src'));
  fs.mkdirSync(path.join(fixture, 'src', 'custback.egg-info'));
  assert.deepEqual(
    release.staleArtifacts(fixture).sort(),
    [
      'build',
      'custback-0.1.0.tgz',
      'debug.txt',
      'onnxruntime_profile__2026-07-16_12-34-56.json',
      'src/custback.egg-info',
      'uninstall.log',
    ],
  );
  assert.equal(fs.readFileSync(path.join(fixture, 'keep.txt'), 'utf8'), 'keep.txt');
});
