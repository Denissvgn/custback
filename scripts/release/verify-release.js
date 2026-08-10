#!/usr/bin/env node
/** Non-mutating release metadata and npm payload gate. */

'use strict';

const { spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { isDeepStrictEqual } = require('util');
const zlib = require('zlib');

const ROOT = path.resolve(__dirname, '..', '..');
const managed = require(path.join(ROOT, 'packaging', 'npm', 'managed-venv'));
const installer = require(path.join(ROOT, 'packaging', 'npm', 'install'));
const phase6Evidence = require(path.join(ROOT, 'scripts', 'release', 'phase6-evidence'));
const evidenceAssembly = require(path.join(ROOT, 'scripts', 'release', 'assemble-evidence'));
const configuredTimeout = Number(process.env.CUSTBACK_RELEASE_TIMEOUT_MS);
const COMMAND_TIMEOUT_MS = Number.isSafeInteger(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 15 * 60 * 1000;
const LINUX_MEMORY_BACKED_FILESYSTEM_MAGICS = new Set([
  0x01021994, // TMPFS_MAGIC
  0x858458f6, // RAMFS_MAGIC
]);
const RELEASE_SOURCE_BRIDGE_ENV = Object.freeze({
  root: 'CUSTBACK_RELEASE_GIT_ROOT',
  commit: 'CUSTBACK_RELEASE_SOURCE_COMMIT',
  tree: 'CUSTBACK_RELEASE_SOURCE_TREE',
});
const REVIEWED_PYTHON_MODULES = [
  'custback/__init__.py',
  'custback/__main__.py',
  'custback/_platform/__init__.py',
  'custback/_platform/base.py',
  'custback/_platform/paths.py',
  'custback/_platform/posix.py',
  'custback/_platform/windows.py',
  'custback/acceleration.py',
  'custback/api/__init__.py',
  'custback/api/avatar_proxy.py',
  'custback/api/security.py',
  'custback/api/server.py',
  'custback/api/streaming.py',
  'custback/api/webui.py',
  'custback/avatar/__init__.py',
  'custback/avatar/__main__.py',
  'custback/avatar/avatar.yaml',
  'custback/avatar/api.py',
  'custback/avatar/audio2face.py',
  'custback/avatar/config.py',
  'custback/avatar/drivers.py',
  'custback/avatar/renderer.py',
  'custback/avatar/rig.py',
  'custback/avatar/service.py',
  'custback/avatar/state.py',
  'custback/avatar/store.py',
  'custback/backgrounds.py',
  'custback/cadence.py',
  'custback/camera_devices.py',
  'custback/capture.py',
  'custback/capture_diagnostics.py',
  'custback/color.py',
  'custback/compositor.py',
  'custback/config.py',
  'custback/config_merge.py',
  'custback/default.yaml',
  'custback/diagnostics.py',
  'custback/geometry.py',
  'custback/gpu_probe.py',
  'custback/hub.py',
  'custback/light_wrap.py',
  'custback/matte_ablation.py',
  'custback/matte_attribution.py',
  'custback/matte_diagnostics.py',
  'custback/matte_live_diagnostics.py',
  'custback/matte_policy.py',
  'custback/matte_performance.py',
  'custback/matte_platform_qualification.py',
  'custback/matte_quality.py',
  'custback/matte_rvm_qualification.py',
  'custback/matte_rollout.py',
  'custback/matte_visual_qualification.py',
  'custback/migration.py',
  'custback/output_scheduler.py',
  'custback/pipeline.py',
  'custback/preview.py',
  'custback/remote_protocol.py',
  'custback/runtime_performance.py',
  'custback/segmentation.py',
  'custback/storage_tx.py',
  'custback/vcam.py',
  'custback/vcam_native.py',
  'custback/video_decoder.py',
];
const REVIEWED_PYTHON_TESTS = [
  'tests/test_acceleration.py',
  'tests/test_api.py',
  'tests/test_api_lifecycle.py',
  'tests/test_api_security.py',
  'tests/test_audio2face_protocol.py',
  'tests/test_avatar_api.py',
  'tests/test_avatar_config.py',
  'tests/test_avatar_drivers.py',
  'tests/test_avatar_proxy.py',
  'tests/test_avatar_rig.py',
  'tests/test_avatar_service.py',
  'tests/test_avatar_store.py',
  'tests/test_background_geometry.py',
  'tests/test_background_asset_fallback.py',
  'tests/test_background_video_lifetime.py',
  'tests/test_backend_selection.py',
  'tests/test_camera_devices.py',
  'tests/test_canonical_canvas.py',
  'tests/test_cadence.py',
  'tests/test_capture.py',
  'tests/test_capture_diagnostics.py',
  'tests/test_capture_geometry.py',
  'tests/test_color.py',
  'tests/test_config.py',
  'tests/test_config_merge.py',
  'tests/test_diagnostics.py',
  'tests/test_geometry.py',
  'tests/test_gpu_probe.py',
  'tests/test_light_wrap.py',
  'tests/test_matte_ablation.py',
  'tests/test_matte_attribution.py',
  'tests/test_matte_diagnostics.py',
  'tests/test_matte_interpolation_decision.py',
  'tests/test_matte_live_diagnostics.py',
  'tests/test_matte_operator_mitigations.py',
  'tests/test_matte_performance.py',
  'tests/test_matte_platform_qualification.py',
  'tests/test_matte_policy.py',
  'tests/test_matte_quality.py',
  'tests/test_matte_rvm_qualification.py',
  'tests/test_matte_rollout.py',
  'tests/test_matte_visual_qualification.py',
  'tests/test_matte_stabilizer.py',
  'tests/test_mediapipe_segmenter.py',
  'tests/test_model_acquisition.py',
  'tests/test_observability.py',
  'tests/test_output_geometry.py',
  'tests/test_output_scheduler.py',
  'tests/test_pipeline.py',
  'tests/test_platform_seam.py',
  'tests/test_phase5_lifecycle.py',
  'tests/test_phase5_storage.py',
  'tests/test_phase6_migration.py',
  'tests/test_phase6_stress.py',
  'tests/test_phase6_two_host_system.py',
  'tests/test_preview.py',
  'tests/test_processing.py',
  'tests/test_remediation_runtime.py',
  'tests/test_remediation_security.py',
  'tests/test_remote_protocol.py',
  'tests/test_runtime_performance.py',
  'tests/test_runtime_performance_status.py',
  'tests/test_segmentation_rvm.py',
  'tests/test_segmentation_timeline.py',
  'tests/test_spatial_edge_refinement.py',
  'tests/test_streaming.py',
  'tests/test_visual_consistency_e2e.py',
  'tests/test_visual_consistency_evidence.py',
  'tests/test_visual_consistency_qualification.py',
  'tests/test_video_color.py',
  'tests/test_vcam.py',
  'tests/test_vcam_timing.py',
  'tests/test_webui.py',
  'tests/test_windows_acceleration_gate.py',
  'tests/test_windows_packaging.py',
  'tests/test_windows_vcam.py',
  'tests/matte_ablation_evidence.py',
  'tests/matte_quality_evidence.py',
  'tests/matte_rvm_qualification_evidence.py',
  'tests/matte_platform_qualification_evidence.py',
  'tests/matte_visual_qualification_evidence.py',
  'tests/visual_consistency_evidence.py',
  'tests/fixtures/migration/expected-0.4.0-local-camera.yaml',
  'tests/fixtures/migration/legacy-0.3.0-default.yaml',
  'tests/fixtures/migration/legacy-0.3.0-local-camera.yaml',
  'tests/fixtures/migration/provenance.json',
];
const REVIEWED_PYTHON_SDIST_DATA = [
  'config/default.yaml',
  'docs/adr/0002-output-rate-matte-interpolation.md',
  'docs/adr/0003-720p-compositor-budget.md',
  'docs/adr/0004-matte-quality-rollout.md',
  'docs/cadence-observability.md',
  'docs/capture-cadence-diagnostics.md',
  'docs/capture-native-evidence-local-template.json',
  'docs/capture-runtime-evidence-local-template.json',
  'docs/matte-ablation.md',
  'docs/matte-ablation-local-reference-template.json',
  'docs/matte-backend-policies.md',
  'docs/matte-boundary-stabilization.md',
  'docs/matte-deterministic-regression-gate.md',
  'docs/matte-light-wrap.md',
  'docs/matte-live-diagnostics.md',
  'docs/matte-operator-mitigations.md',
  'docs/matte-performance.md',
  'docs/matte-performance-local-template.json',
  'docs/matte-platform-qualification.md',
  'docs/matte-platform-qualification-local-template.json',
  'docs/matte-alpha-attribution.md',
  'docs/matte-alpha-attribution-local-template.json',
  'docs/matte-quality-baseline.md',
  'docs/matte-quality-local-qualification-template.json',
  'docs/matte-quality-metrics.md',
  'docs/matte-quality-rollout.md',
  'docs/matte-replay-bundle.md',
  'docs/matte-rvm-profiles.md',
  'docs/matte-rvm-qualification-local-template.json',
  'docs/matte-spatial-refinement.md',
  'docs/matte-visual-qualification.md',
  'docs/matte-visual-qualification-local-template.json',
  'docs/visual-consistency-phase4-qualification-runbook.md',
  'docs/visual-consistency-phase4-qualification-template.json',
  'docs/visual-consistency-rollout.md',
  'scripts/release/visual-qualification-manifest.json',
  'scripts/release/visual-policy-rollout.json',
  'scripts/release/matte-policy-rollout.json',
  'scripts/release/visual_consistency_qualification.py',
];
const REVIEWED_NPM_PAYLOAD = [
  '.github/workflows/ci.yml',
  '.github/workflows/release.yml',
  'LICENSE',
  'MANIFEST.in',
  'README.md',
  'REMEDIATION_PLAN.md',
  'config/avatar.yaml',
  'config/default.yaml',
  'docs/adr/0001-visual-consistency-contract.md',
  'docs/adr/0002-output-rate-matte-interpolation.md',
  'docs/adr/0003-720p-compositor-budget.md',
  'docs/adr/0004-matte-quality-rollout.md',
  'docs/camera-control-characterization.md',
  'docs/cadence-observability.md',
  'docs/capture-cadence-diagnostics.md',
  'docs/capture-native-evidence-local-template.json',
  'docs/capture-runtime-evidence-local-template.json',
  'docs/matte-ablation.md',
  'docs/matte-ablation-local-reference-template.json',
  'docs/matte-backend-policies.md',
  'docs/matte-boundary-stabilization.md',
  'docs/matte-deterministic-regression-gate.md',
  'docs/matte-light-wrap.md',
  'docs/matte-live-diagnostics.md',
  'docs/matte-operator-mitigations.md',
  'docs/matte-performance.md',
  'docs/matte-performance-local-template.json',
  'docs/matte-platform-qualification.md',
  'docs/matte-platform-qualification-local-template.json',
  'docs/matte-alpha-attribution.md',
  'docs/matte-alpha-attribution-local-template.json',
  'docs/matte-quality-baseline.md',
  'docs/matte-quality-local-qualification-template.json',
  'docs/matte-quality-metrics.md',
  'docs/matte-quality-rollout.md',
  'docs/matte-replay-bundle.md',
  'docs/matte-rvm-profiles.md',
  'docs/matte-rvm-qualification-local-template.json',
  'docs/matte-spatial-refinement.md',
  'docs/matte-visual-qualification.md',
  'docs/matte-visual-qualification-local-template.json',
  'docs/remote-deployment.md',
  'docs/visual-consistency-phase0-baseline.json',
  'docs/visual-consistency-phase0-contact-sheet.png',
  'docs/visual-consistency-phase0-evidence.md',
  'docs/visual-consistency-phase0-implementation-review.md',
  'docs/visual-consistency-phase1-implementation-review.md',
  'docs/visual-consistency-phase2-implementation-review.md',
  'docs/visual-consistency-phase3-implementation-review.md',
  'docs/visual-consistency-phase3-video-color-qualification.md',
  'docs/visual-consistency-phase4-implementation-review.md',
  'docs/visual-consistency-phase4-local-observation-contact-sheet.png',
  'docs/visual-consistency-phase4-local-observation.json',
  'docs/visual-consistency-phase4-qualification-runbook.md',
  'docs/visual-consistency-phase4-qualification-template.json',
  'docs/visual-consistency-rollout.md',
  'examples/avatar_client.py',
  'package.json',
  'packaging/npm/custback.js',
  'packaging/npm/install.js',
  'packaging/npm/managed-venv.js',
  'packaging/npm/migrate-legacy.js',
  'packaging/npm/test/install-policy.test.js',
  'packaging/npm/test/managed-venv.test.js',
  'packaging/npm/test/candidate-build.test.js',
  'packaging/npm/test/clean-tree.test.js',
  'packaging/npm/test/evidence-assembly.test.js',
  'packaging/npm/test/migration-qualification.test.js',
  'packaging/npm/test/phase6-evidence.test.js',
  'packaging/npm/test/phase6-migration.test.js',
  'packaging/npm/test/release-check.test.js',
  'packaging/npm/test/release-workflow.test.js',
  'packaging/npm/test/windows-release-check.test.js',
  'packaging/npm/test/remediation-phase0.test.js',
  'pyproject.toml',
  'scripts/install_linux.sh',
  'scripts/install_macos.sh',
  'scripts/release/package-smoke.js',
  'scripts/release/assemble-evidence.js',
  'scripts/release/build-candidate.js',
  'scripts/release/phase6-evidence.js',
  'scripts/release/qualify-migrations.js',
  'scripts/release/required-gates.json',
  'scripts/release/two-host-system-test.py',
  'scripts/release/two-host/Dockerfile',
  'scripts/release/two-host/probe.py',
  'scripts/release/visual-qualification-manifest.json',
  'scripts/release/visual-policy-rollout.json',
  'scripts/release/matte-policy-rollout.json',
  'scripts/release/visual_consistency_qualification.py',
  'scripts/release/windows-acceleration-gate.py',
  'scripts/release/verify-clean-tree.js',
  'scripts/release/verify-release.js',
  'scripts/release/remediation-blockers.json',
  ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
  ...REVIEWED_PYTHON_TESTS,
];
const REVIEWED_BUILD_REQUIREMENTS = ['setuptools>=77,<84'];
const REVIEWED_CORE_DEPENDENCIES = [
  'numpy>=1.24,<3',
  'opencv-contrib-python>=4.8,<6',
  "av>=17,<18; python_version < '3.11'",
  "av>=18,<19; python_version >= '3.11'",
  'pillow>=10,<13',
  'pydantic>=2.7,<3',
  "pyvirtualcam>=0.11,<1; sys_platform != 'win32' or (platform_machine != 'ARM64' and platform_machine != 'arm64')",
  'fastapi>=0.110,<1',
  'uvicorn>=0.29,<1',
  'pyyaml>=6.0,<7',
  'websockets>=12.0,<17',
  'python-multipart>=0.0.9,<1',
  'httpx>=0.27,<0.29',
];
const REVIEWED_OPTIONAL_DEPENDENCIES = {
  mediapipe: ['mediapipe>=0.10.14,<0.11'],
  rvm: ['onnxruntime>=1.17,<2'],
  gpu: ['onnxruntime-gpu>=1.17,<1.27'],
  // Windows-only DirectML acceleration profile (WIN-6.2). Mutually exclusive
  // with `gpu` in any one environment (same `onnxruntime` package); the
  // freeze script and the WIN-6.2 gate both enforce the either-or.
  directml: ['onnxruntime-directml>=1.17,<2'],
  // Windows filesystem-security backend (custback._platform.windows): LockFileEx,
  // owner-only DACLs, reparse-point rejection, SID ownership, MoveFileEx.
  windows: ['pywin32>=306'],
  audio2face: [
    'grpcio>=1.67,<1.67.2',
    'nvidia-ace==1.0.0',
    'nvidia-audio2face-3d==1.3.0',
    'protobuf>=5.29.3,<6',
    'sounddevice>=0.4,<0.6',
  ],
  dev: [
    'build>=1.2,<2',
    'pytest>=8.0,<10',
    'pytest-timeout>=2.3,<3',
    'httpx>=0.27,<0.29',
    'httpx2>=2,<3',
    // Exact pin: ruff's pre-1.0 formatter style is version-specific, so the
    // `ruff format --check` gate is only deterministic when the version is
    // fixed (see pyproject dev extra + [tool.ruff] required-version).
    'ruff==0.15.22',
    // Exact pin: Pyright diagnostics are versioned CI policy even though this
    // source-only basic-mode check is advisory to the publication gate.
    'pyright==1.1.411',
  ],
};
const REVIEWED_CONSOLE_SCRIPTS = {
  custback: 'custback.__main__:main',
  'custback-avatar': 'custback.avatar.__main__:main',
};
const REVIEWED_ACTIONS = new Set([
  'actions/checkout@08eba0b27e820071cde6df949e0beb9ba4906955',
  'actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020',
  'actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065',
  'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02',
]);
const REVIEWED_LICENSE_COPYRIGHT = 'Copyright (c) 2026 Bramen';
const REVIEWED_REMEDIATION_CONTRACT_SHA256 =
  'daf3165058e28fbbd91a7d90aac8804cd67fa2a9e8af2e1b69fa67d5a8a91db0';
const REVIEWED_NPM_METADATA = {
  name: 'custback',
  description: 'Virtual camera with background replacement for meeting apps (Ubuntu / Debian / macOS)',
  license: 'MIT',
  bin: {
    custback: 'packaging/npm/custback.js',
    'custback-avatar': 'packaging/npm/custback.js',
    'custback-npm-migrate': 'packaging/npm/migrate-legacy.js',
  },
  scripts: {
    postinstall: 'node packaging/npm/install.js',
    test: 'node --test packaging/npm/test/*.test.js',
    doctor: 'node packaging/npm/custback.js doctor',
    'release:check': 'node scripts/release/verify-release.js',
    prepack: 'node scripts/release/verify-release.js --prepack',
  },
  files: [
    '.github/workflows/*.yml',
    'LICENSE',
    'REMEDIATION_PLAN.md',
    'packaging/npm/*.js',
    'packaging/npm/test/*.test.js',
    'src/**/*.py',
    'src/**/*.yaml',
    'tests/*.py',
    'tests/fixtures/migration/*',
    'config/*.yaml',
    'docs/*.md',
    'docs/*.json',
    'docs/*.png',
    'docs/adr/*.md',
    'scripts/*.sh',
    'scripts/release/*.js',
    'scripts/release/*.json',
    'scripts/release/*.py',
    'scripts/release/two-host/Dockerfile',
    'scripts/release/two-host/*.py',
    'examples/*.py',
    'MANIFEST.in',
    'pyproject.toml',
  ],
  os: ['linux', 'darwin'],
  engines: { node: '^18.15.0 || ^20.0.0 || ^22.0.0' },
  keywords: [
    'virtual-camera',
    'background-replacement',
    'webcam',
    'v4l2loopback',
    'obs',
  ],
};

function fail(message) {
  throw new Error(message);
}

function filesystemIsMemoryBacked(
  directory,
  statfs = fs.statfsSync,
  platform = process.platform,
) {
  if (platform !== 'linux') return false;
  if (typeof statfs !== 'function') {
    fail('cannot determine whether release temporary storage is memory-backed');
  }
  return LINUX_MEMORY_BACKED_FILESYSTEM_MAGICS.has(Number(statfs(directory).type));
}

function releaseBuildJobs(env = process.env) {
  const configured = env.CUSTBACK_RELEASE_BUILD_JOBS;
  if (configured === undefined) return '2';
  const value = String(configured);
  if (!/^[1-9][0-9]*$/.test(value) || Number(value) > 32) {
    fail('CUSTBACK_RELEASE_BUILD_JOBS must be an integer from 1 through 32');
  }
  return value;
}

function createReleaseTemporaryRoot(options = {}) {
  const env = options.env || process.env;
  const configured = env.CUSTBACK_RELEASE_TMPDIR;
  const candidate = configured || options.defaultBase || os.tmpdir();
  let base;
  try {
    base = fs.realpathSync(path.resolve(candidate));
    if (!fs.statSync(base).isDirectory()) {
      fail('release temporary base must be an existing directory');
    }
  } catch (err) {
    if (err.message === 'release temporary base must be an existing directory') throw err;
    fail(`release temporary base is unavailable: ${err.message}`);
  }
  const sourceRoot = fs.realpathSync(options.root || ROOT);
  const sourceRelative = path.relative(sourceRoot, base);
  if (!sourceRelative ||
      (!sourceRelative.startsWith(`..${path.sep}`) && sourceRelative !== '..' &&
       !path.isAbsolute(sourceRelative))) {
    fail('release temporary base must be outside the source checkout');
  }
  const allowTmpfs = env.CUSTBACK_RELEASE_ALLOW_TMPFS === '1';
  if (filesystemIsMemoryBacked(base, options.statfs, options.platform) && !allowTmpfs) {
    fail(
      `${base} is a memory-backed filesystem; set CUSTBACK_RELEASE_TMPDIR to an ` +
      'existing disk-backed directory (or explicitly set CUSTBACK_RELEASE_ALLOW_TMPFS=1)'
    );
  }
  return fs.mkdtempSync(path.join(base, 'custback-release-'));
}

function withDisposableDirectory(directory, callback) {
  if (fs.existsSync(directory)) {
    fail(`refusing to reuse disposable release directory: ${directory}`);
  }
  try {
    return callback(directory);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
}

function withTemporaryEnvironment(updates, callback, env = process.env) {
  const previous = new Map();
  for (const [name, value] of Object.entries(updates)) {
    previous.set(name, Object.prototype.hasOwnProperty.call(env, name) ? env[name] : undefined);
    env[name] = value;
  }
  try {
    return callback();
  } finally {
    for (const [name, value] of previous) {
      if (value === undefined) delete env[name];
      else env[name] = value;
    }
  }
}

function canonicalLicense(root = ROOT) {
  const licensePath = path.join(root, 'LICENSE');
  let text;
  try {
    const stat = fs.lstatSync(licensePath);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      fail('LICENSE must be a regular, non-symlink file');
    }
    text = fs.readFileSync(licensePath, 'utf8');
  } catch (err) {
    if (err.message.startsWith('LICENSE must')) throw err;
    fail(`LICENSE is missing or unreadable: ${err.message}`);
  }
  const required = [
    'MIT License',
    REVIEWED_LICENSE_COPYRIGHT,
    'Permission is hereby granted, free of charge, to any person obtaining a copy',
    'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND',
  ];
  if (!text.endsWith('\n') || required.some((line) => !text.includes(line))) {
    fail(`LICENSE must contain the reviewed MIT grant and ${REVIEWED_LICENSE_COPYRIGHT}`);
  }
  return Buffer.from(text, 'utf8');
}

function verifyLicenseMetadata(root = ROOT) {
  canonicalLicense(root);
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const project = tableBody(pyproject, 'project');
  if (stringValue(project, 'license') !== 'MIT' ||
      !isDeepStrictEqual(arrayValue(project, 'license-files'), ['LICENSE'])) {
    fail('pyproject.toml must declare MIT and ship only the canonical LICENSE file');
  }
  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  if (pkg.license !== 'MIT' || !Array.isArray(pkg.files) || !pkg.files.includes('LICENSE')) {
    fail('package.json must declare MIT and explicitly ship LICENSE');
  }
}

function verifyAvatarConfigTemplate(root = ROOT) {
  const canonicalPath = path.join(root, 'config', 'avatar.yaml');
  const packagePath = path.join(root, 'src', 'custback', 'avatar', 'avatar.yaml');
  if (!fs.existsSync(canonicalPath) || !fs.existsSync(packagePath) ||
      !fs.readFileSync(canonicalPath).equals(fs.readFileSync(packagePath))) {
    fail('Python package avatar template must be byte-identical to config/avatar.yaml');
  }
}

function verifyCoreConfigTemplate(root = ROOT) {
  const canonicalPath = path.join(root, 'config', 'default.yaml');
  const packagePath = path.join(root, 'src', 'custback', 'default.yaml');
  const canonical = fs.readFileSync(canonicalPath);
  const packaged = fs.readFileSync(packagePath);
  if (!canonical.equals(packaged)) {
    fail('Python package default template must be byte-identical to config/default.yaml');
  }
}

function remediationRegistry(root = ROOT) {
  const registryPath = path.join(root, 'scripts', 'release', 'remediation-blockers.json');
  if (!fs.existsSync(registryPath)) {
    fail('release remediation blocker registry is missing');
  }
  let registry;
  try {
    registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
  } catch (err) {
    fail(`release remediation blocker registry is invalid JSON: ${err.message}`);
  }
  if (!registry || registry.schema_version !== 2 ||
      !Number.isSafeInteger(registry.phase) || registry.phase < 1 ||
      !Array.isArray(registry.blockers)) {
    fail(
      'release remediation blocker registry must use schema_version 2 with a phase and blockers array'
    );
  }
  const seen = new Set();
  const blockers = registry.blockers.map((entry) => {
    if (!entry || typeof entry !== 'object' ||
        typeof entry.id !== 'string' || !/^[A-Z][A-Z0-9]+-\d{2}$/.test(entry.id) ||
        !Number.isSafeInteger(entry.phase) || entry.phase < 1 ||
        !['open', 'resolved'].includes(entry.status) ||
        typeof entry.title !== 'string' || entry.title.trim() === '' ||
        !entry.regression || typeof entry.regression !== 'object' ||
        !['pytest', 'node'].includes(entry.regression.runner) ||
        typeof entry.regression.test !== 'string' ||
        entry.regression.test.trim() !== entry.regression.test ||
        entry.regression.test === '') {
      fail('release remediation blocker registry contains an invalid entry');
    }
    const regression = { ...entry.regression };
    if (regression.runner === 'pytest') {
      const parts = regression.test.split('::');
      if (parts.length !== 2 || !parts[0].endsWith('.py') ||
          !/^test_[A-Za-z0-9_]+$/.test(parts[1]) ||
          path.isAbsolute(parts[0]) || parts[0].split(/[\\/]/).includes('..') ||
          Object.hasOwn(regression, 'file') || Object.hasOwn(regression, 'guard')) {
        fail(`${entry.id} has an invalid pytest regression node id`);
      }
      regression.file = parts[0];
    } else {
      if (typeof regression.file !== 'string' ||
          !regression.file.endsWith('.test.js') || path.isAbsolute(regression.file) ||
          regression.file.split(/[\\/]/).includes('..') ||
          (Object.hasOwn(regression, 'guard') &&
           (entry.status !== 'open' || typeof regression.guard !== 'string' ||
            regression.guard.trim() !== regression.guard || regression.guard === ''))) {
        fail(`${entry.id} has an invalid Node regression identifier`);
      }
    }
    if (seen.has(entry.id)) {
      fail(`release remediation blocker registry contains duplicate id ${entry.id}`);
    }
    seen.add(entry.id);
    return { ...entry, regression };
  });
  const open = blockers.filter((entry) => entry.status === 'open');
  if (registry.release_blocked !== (open.length > 0)) {
    fail('release_blocked must be true exactly while remediation blockers remain open');
  }
  return { ...registry, blockers };
}

function remediationBlockers(root = ROOT) {
  return remediationRegistry(root).blockers;
}

function remediationContractDigest(registry) {
  const contract = {
    phase: registry.phase,
    release_blocked: registry.release_blocked,
    blockers: registry.blockers.map((blocker) => ({
      id: blocker.id,
      phase: blocker.phase,
      status: blocker.status,
      title: blocker.title,
      regression: blocker.regression.runner === 'pytest'
        ? {
          runner: blocker.regression.runner,
          test: blocker.regression.test,
        }
        : {
          runner: blocker.regression.runner,
          file: blocker.regression.file,
          test: blocker.regression.test,
          ...(blocker.regression.guard ? { guard: blocker.regression.guard } : {}),
        },
    })),
  };
  return crypto.createHash('sha256').update(JSON.stringify(contract)).digest('hex');
}

function verifyBlockerRegressionCoverage(
  root = ROOT,
  { enforceReviewedContract = true } = {},
) {
  const registry = remediationRegistry(root);
  if (enforceReviewedContract &&
      remediationContractDigest(registry) !== REVIEWED_REMEDIATION_CONTRACT_SHA256) {
    fail('remediation registry does not match the reviewed Phase 5 blocker contract');
  }
  for (const blocker of registry.blockers) {
    const regressionPath = path.join(root, blocker.regression.file);
    if (!fs.existsSync(regressionPath) || !fs.statSync(regressionPath).isFile()) {
      fail(`${blocker.id} regression file is missing: ${blocker.regression.file}`);
    }
    const source = fs.readFileSync(regressionPath, 'utf8');
    if (blocker.regression.runner === 'pytest') {
      const functionName = blocker.regression.test.split('::')[1];
      const definition = new RegExp(
        `^(?:async\\s+)?def\\s+${functionName}\\s*\\(`,
        'm',
      );
      if (!definition.test(source)) {
        fail(`${blocker.id} pytest node is not defined: ${blocker.regression.test}`);
      }
    } else {
      for (const testName of [blocker.regression.test, blocker.regression.guard].filter(Boolean)) {
        if (!source.includes(JSON.stringify(testName)) &&
            !source.includes(`'${testName.replaceAll("'", "\\'")}'`)) {
          fail(`${blocker.id} Node test is not defined exactly: ${testName}`);
        }
      }
    }
  }
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function exactNodeTapOutcome(blockerId, testName, stdout) {
  const linePattern = new RegExp(
    `^(?:not )?ok\\s+\\d+\\s+-\\s+${escapeRegex(testName)}(?:\\s+#.*)?$`,
  );
  const outcomes = String(stdout || '').split(/\r?\n/)
    .map((line) => line.trim()).filter((line) => linePattern.test(line));
  if (outcomes.length !== 1) {
    fail(
      `${blockerId} Node regression produced ${outcomes.length} exact TAP outcomes: ${testName}`
    );
  }
  return outcomes[0];
}

function runNodeRegression(blocker, root = ROOT) {
  const runNamed = (testName) => {
    const result = spawnSync(process.execPath, [
      '--test-reporter=tap',
      '--test-name-pattern', `^${escapeRegex(testName)}$`,
      blocker.regression.file,
    ], {
      cwd: root,
      encoding: 'utf8',
      timeout: COMMAND_TIMEOUT_MS,
      maxBuffer: 16 * 1024 * 1024,
    });
    if (result.error) fail(`${blocker.id} Node regression could not start: ${result.error.message}`);
    return {
      result,
      outcome: exactNodeTapOutcome(blocker.id, testName, result.stdout),
    };
  };

  const acceptance = runNamed(blocker.regression.test);
  if (blocker.status === 'resolved') {
    if (acceptance.result.status !== 0 || /#\s*(?:TODO|SKIP)\b/i.test(acceptance.outcome) ||
        acceptance.outcome.startsWith('not ok')) {
      fail(`${blocker.id} resolved Node regression did not pass normally:\n${acceptance.result.stdout}\n${acceptance.result.stderr}`);
    }
    return;
  }
  if (acceptance.result.status !== 0 || !/#\s*TODO\b/i.test(acceptance.outcome)) {
    fail(`${blocker.id} open Node regression must produce a TODO result`);
  }
  if (!blocker.regression.guard) {
    fail(`${blocker.id} open Node TODO is missing an unexpected-pass guard`);
  }
  const guard = runNamed(blocker.regression.guard);
  if (guard.result.status !== 0 || !guard.outcome || guard.outcome.startsWith('not ok') ||
      /#\s*(?:TODO|SKIP)\b/i.test(guard.outcome)) {
    fail(`${blocker.id} Node TODO guard did not prove the known failure:\n${guard.result.stdout}\n${guard.result.stderr}`);
  }
}

function runPytestRegression(blocker, root = ROOT) {
  const python = process.env.PYTHON || 'python3';
  const result = spawnSync(
    python,
    ['-m', 'pytest', '-q', blocker.regression.test],
    {
      cwd: root,
      encoding: 'utf8',
      timeout: COMMAND_TIMEOUT_MS,
      maxBuffer: 16 * 1024 * 1024,
    },
  );
  if (result.error) fail(`${blocker.id} pytest regression could not start: ${result.error.message}`);
  const output = `${result.stdout || ''}\n${result.stderr || ''}`;
  if (blocker.status === 'resolved') {
    if (result.status !== 0 || /\b(?:skipped|xfailed|xpassed)\b/i.test(output) ||
        !/\bpassed\b/i.test(output)) {
      fail(`${blocker.id} resolved pytest regression did not pass normally:\n${output}`);
    }
  } else if (result.status !== 0 || !/\bxfailed\b/i.test(output) || /\bxpassed\b/i.test(output)) {
    fail(`${blocker.id} open pytest regression did not produce a strict expected failure:\n${output}`);
  }
}

function runBlockerRegressionTests(runner = 'all', root = ROOT) {
  if (!['all', 'pytest', 'node'].includes(runner)) {
    fail(`unknown remediation regression runner: ${runner}`);
  }
  verifyBlockerRegressionCoverage(root);
  for (const blocker of remediationBlockers(root)) {
    if (runner !== 'all' && blocker.regression.runner !== runner) continue;
    if (blocker.regression.runner === 'pytest') runPytestRegression(blocker, root);
    else runNodeRegression(blocker, root);
  }
}

function verifyNoReleaseBlockers(root = ROOT) {
  verifyBlockerRegressionCoverage(root);
  const open = remediationBlockers(root).filter((entry) => entry.status === 'open');
  if (open.length) {
    fail(
      `release blocked by ${open.length} open remediation blocker(s): ` +
      `${open.map((entry) => entry.id).join(', ')}; see REMEDIATION_PLAN.md`,
    );
  }
}

function projectVersion(pyproject) {
  const project = pyproject.match(/\[project\]([\s\S]*?)(?:\n\[|$)/);
  if (!project) fail('pyproject.toml has no [project] table');
  const version = project[1].match(/^version\s*=\s*"([^"]+)"/m);
  if (!version) fail('pyproject.toml [project] has no static version');
  return version[1];
}

function tableBody(toml, name) {
  const header = `[${name}]`;
  const start = toml.indexOf(header);
  if (start < 0) fail(`pyproject.toml has no ${header} table`);
  const contentStart = toml.indexOf('\n', start + header.length);
  if (contentStart < 0) return '';
  const rest = toml.slice(contentStart + 1);
  const nextTable = rest.search(/^\[/m);
  return nextTable < 0 ? rest : rest.slice(0, nextTable);
}

function arrayValue(table, key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = table.match(new RegExp(`^${escaped}\\s*=\\s*\\[([\\s\\S]*?)\\]`, 'm'));
  if (!match) fail(`pyproject.toml has no ${key} array in the expected table`);
  const uncommented = match[1].split('\n').map((line) => line.split('#', 1)[0]).join('\n');
  return [...uncommented.matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
}

function assignmentKeys(table) {
  return [...table.matchAll(/^([A-Za-z0-9_-]+)\s*=/gm)].map((entry) => entry[1]);
}

function stringValue(table, key) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = table.match(new RegExp(`^${escaped}\\s*=\\s*"([^"]+)"\\s*$`, 'm'));
  if (!match) fail(`pyproject.toml has no exact ${key} string in the expected table`);
  return match[1];
}

function sourceFallbackVersion(filename) {
  const python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
  const parseCode = `
import ast, json, pathlib, sys
source_path = pathlib.Path(sys.argv[1])
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
values = []
for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
    if not isinstance(handler.type, ast.Name) or handler.type.id != "PackageNotFoundError":
        continue
    for node in handler.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise RuntimeError("__version__ fallback must be a literal string")
        values.append(value)
if len(values) != 1:
    raise RuntimeError(f"expected exactly one PackageNotFoundError __version__ fallback, found {len(values)}")
print(json.dumps(values[0]))
`;
  const result = spawnSync(python, ['-c', parseCode, filename], {
    encoding: 'utf8',
    timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
  });
  if (result.error && result.status === null) {
    fail(`could not inspect source-tree __version__: ${result.error.message}`);
  }
  if (result.status !== 0) {
    fail(`invalid source-tree __version__ fallback: ${(result.stderr || '').trim()}`);
  }
  try {
    return JSON.parse(result.stdout);
  } catch (err) {
    fail(`source-tree __version__ probe returned invalid JSON: ${err.message}`);
  }
}

function verifyNpmMetadata(root = ROOT) {
  const packagePath = path.join(root, 'package.json');
  const lockPath = path.join(root, 'package-lock.json');
  const pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
  const lock = JSON.parse(fs.readFileSync(lockPath, 'utf8'));
  const { version, ...metadata } = pkg;
  if (typeof version !== 'string' || version === '' ||
      !isDeepStrictEqual(metadata, REVIEWED_NPM_METADATA)) {
    fail('package.json metadata must exactly match the reviewed release allowlist');
  }
  const expectedLock = {
    name: REVIEWED_NPM_METADATA.name,
    version,
    lockfileVersion: 3,
    requires: true,
    packages: {
      '': {
        name: REVIEWED_NPM_METADATA.name,
        version,
        hasInstallScript: true,
        license: REVIEWED_NPM_METADATA.license,
        os: REVIEWED_NPM_METADATA.os,
        bin: REVIEWED_NPM_METADATA.bin,
        engines: REVIEWED_NPM_METADATA.engines,
      },
    },
  };
  if (!isDeepStrictEqual(lock, expectedLock)) {
    fail('package-lock.json must exactly mirror reviewed package metadata');
  }
}

function verifyVersions(root = ROOT) {
  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const lock = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8'));
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const initPath = path.join(root, 'src', 'custback', '__init__.py');
  const versions = {
    package: pkg.version,
    lock: lock.version,
    lockRoot: lock.packages && lock.packages[''] && lock.packages[''].version,
    python: projectVersion(pyproject),
  };
  const unique = new Set(Object.values(versions));
  if (unique.size !== 1) fail(`release version mismatch: ${JSON.stringify(versions)}`);
  if (sourceFallbackVersion(initPath) !== pkg.version) {
    fail('source-tree __version__ fallback does not match package version');
  }
  verifyNpmMetadata(root);
  return pkg.version;
}

function verifyDependencies(root = ROOT) {
  const pyproject = fs.readFileSync(path.join(root, 'pyproject.toml'), 'utf8');
  const buildSystem = arrayValue(tableBody(pyproject, 'build-system'), 'requires');
  if (JSON.stringify(buildSystem) !== JSON.stringify(REVIEWED_BUILD_REQUIREMENTS)) {
    fail('build-system requirements must be exactly setuptools>=77,<84');
  }
  const core = arrayValue(tableBody(pyproject, 'project'), 'dependencies');
  for (const spec of REVIEWED_CORE_DEPENDENCIES) {
    if (!core.includes(spec)) fail(`missing bounded core dependency: ${spec}`);
  }
  const unexpectedCore = core.filter((spec) => !REVIEWED_CORE_DEPENDENCIES.includes(spec));
  if (unexpectedCore.length || core.length !== REVIEWED_CORE_DEPENDENCIES.length) {
    fail(`unexpected core dependencies: ${unexpectedCore.join(', ') || 'duplicate entries'}`);
  }
  const optional = tableBody(pyproject, 'project.optional-dependencies');
  const optionalKeys = assignmentKeys(optional);
  const reviewedOptionalKeys = Object.keys(REVIEWED_OPTIONAL_DEPENDENCIES);
  if (new Set(optionalKeys).size !== optionalKeys.length ||
      optionalKeys.length !== reviewedOptionalKeys.length ||
      optionalKeys.some((key) => !reviewedOptionalKeys.includes(key))) {
    fail(`optional dependency groups must be exactly ${reviewedOptionalKeys.join(', ')}`);
  }
  for (const [extra, expected] of Object.entries(REVIEWED_OPTIONAL_DEPENDENCIES)) {
    const dependencies = arrayValue(optional, extra);
    const unexpected = dependencies.filter((spec) => !expected.includes(spec));
    const missing = expected.filter((spec) => !dependencies.includes(spec));
    if (unexpected.length || missing.length || dependencies.length !== expected.length) {
      fail(
        `invalid bounded ${extra} dependencies; missing: ${missing.join(', ') || 'none'}; ` +
        `unexpected: ${unexpected.join(', ') || 'none'}`
      );
    }
  }
  const scripts = tableBody(pyproject, 'project.scripts');
  const scriptKeys = assignmentKeys(scripts);
  const expectedScripts = Object.keys(REVIEWED_CONSOLE_SCRIPTS);
  if (scriptKeys.length !== expectedScripts.length ||
      scriptKeys.some((key) => !expectedScripts.includes(key))) {
    fail(`console scripts must be exactly ${expectedScripts.join(', ')}`);
  }
  for (const [name, entrypoint] of Object.entries(REVIEWED_CONSOLE_SCRIPTS)) {
    if (stringValue(scripts, name) !== entrypoint) {
      fail(`console script ${name} must map exactly to ${entrypoint}`);
    }
  }
  if (!pyproject.includes('requires-python = ">=3.10,<3.15"')) {
    fail('Python support range must be >=3.10,<3.15');
  }
  if (/"opencv-python[<>=]/.test(pyproject)) {
    fail('opencv-python conflicts with the selected opencv-contrib-python distribution');
  }
}

function verifyReviewedSourceFiles(root, names) {
  for (const name of names) {
    let current = root;
    for (const part of name.split('/')) {
      current = path.join(current, part);
      let stat;
      try {
        stat = fs.lstatSync(current);
      } catch (err) {
        fail(`reviewed release source is missing: ${name} (${err.message})`);
      }
      if (stat.isSymbolicLink()) fail(`release payload source must not be a symlink: ${name}`);
    }
    if (!fs.lstatSync(current).isFile()) fail(`reviewed release source is not a file: ${name}`);
  }
}

function expectedNpmPayload(root) {
  verifyReviewedSourceFiles(root, REVIEWED_NPM_PAYLOAD);
  return new Set(REVIEWED_NPM_PAYLOAD);
}

function verifyNpmPayload(names, root) {
  const forbidden = names.filter((name) =>
    /(^|\/)\.venv(?:\/|$)/.test(name) || name.includes('custback-generations') ||
    name.endsWith('.tgz') || name.endsWith('.whl') ||
    name.endsWith('.tar.gz') || name.includes('__pycache__') || name.endsWith('.pyc') ||
    /(^|\/)onnxruntime_profile__.*\.json$/.test(name) ||
    name === 'debug.txt' || name === 'uninstall.log');
  if (forbidden.length) fail(`npm artifact contains forbidden files: ${forbidden.join(', ')}`);
  const expected = expectedNpmPayload(root);
  const actual = new Set(names);
  const unexpected = names.filter((name) => !expected.has(name));
  const missing = [...expected].filter((name) => !actual.has(name));
  if (unexpected.length || missing.length) {
    fail(
      `npm artifact manifest mismatch; missing: ${missing.join(', ') || 'none'}; ` +
      `unexpected: ${unexpected.join(', ') || 'none'}`
    );
  }
}

function parseNpmPackPayload(stdout, version) {
  let payload;
  try {
    payload = JSON.parse(stdout);
  } catch (err) {
    fail(`npm pack returned invalid JSON: ${err.message}`);
  }
  if (!Array.isArray(payload) || payload.length !== 1 || !payload[0] ||
      typeof payload[0] !== 'object' || Array.isArray(payload[0])) {
    fail('npm pack returned an unexpected payload');
  }
  const artifact = payload[0];
  const expectedFilename = `custback-${version}.tgz`;
  if (artifact.name !== 'custback' || artifact.version !== version ||
      artifact.filename !== expectedFilename) {
    fail('npm pack artifact identity is inconsistent');
  }
  if (!Array.isArray(artifact.files) || artifact.files.length === 0) {
    fail('npm pack artifact has no file manifest');
  }
  const names = artifact.files.map((entry) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry) ||
        typeof entry.path !== 'string' || entry.path === '' || entry.path === '.' ||
        entry.path.endsWith('/') || entry.path.includes('\\') ||
        path.posix.isAbsolute(entry.path) || path.posix.normalize(entry.path) !== entry.path ||
        entry.path.split('/').includes('..')) {
      fail('npm pack artifact contains an invalid file entry');
    }
    return entry.path;
  });
  if (new Set(names).size !== names.length) {
    fail('npm pack artifact contains duplicate file entries');
  }
  return { artifact, names };
}

function tarString(block, start, length) {
  const end = block.indexOf(0, start);
  const limit = end >= start && end < start + length ? end : start + length;
  return block.subarray(start, limit).toString('utf8');
}

function npmTarballFile(tarball, wanted) {
  let archive;
  try {
    archive = zlib.gunzipSync(fs.readFileSync(tarball));
  } catch (err) {
    fail(`npm artifact is not a readable gzip stream: ${err.message}`);
  }
  const seen = new Set();
  for (let offset = 0; offset + 512 <= archive.length;) {
    const header = archive.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) break;
    const name = tarString(header, 0, 100);
    const prefix = tarString(header, 345, 155);
    const pathname = prefix ? `${prefix}/${name}` : name;
    const rawSize = tarString(header, 124, 12).trim();
    if (!/^[0-7]+$/.test(rawSize)) {
      fail(`npm artifact has an invalid tar size for ${pathname || '<unnamed>'}`);
    }
    const size = Number.parseInt(rawSize, 8);
    if (!Number.isSafeInteger(size)) fail(`npm artifact tar entry is too large: ${pathname}`);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;
    if (dataEnd > archive.length) fail(`npm artifact tar entry is truncated: ${pathname}`);
    const type = String.fromCharCode(header[156] || 48);
    if (type === '0') {
      if (!pathname || path.posix.isAbsolute(pathname) ||
          path.posix.normalize(pathname) !== pathname ||
          pathname.split('/').includes('..') || seen.has(pathname)) {
        fail(`npm artifact has an unsafe or duplicate tar entry: ${pathname}`);
      }
      seen.add(pathname);
      if (pathname === wanted) return Buffer.from(archive.subarray(dataStart, dataEnd));
    }
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  fail(`npm artifact is missing ${wanted}`);
}

function verifyNpmArtifactLicense(tarball, root = ROOT) {
  const expected = canonicalLicense(root);
  const actual = npmTarballFile(tarball, 'package/LICENSE');
  if (!actual.equals(expected)) {
    fail('npm artifact LICENSE is not byte-identical to the canonical LICENSE');
  }
}

function staleArtifacts(root = ROOT) {
  const staleNames = new Set(['debug.txt', 'uninstall.log']);
  const stale = [];
  for (const name of fs.readdirSync(root)) {
    const full = path.join(root, name);
    if (staleNames.has(name) || /^onnxruntime_profile__.*\.json$/.test(name) ||
        name.endsWith('.tgz') || name.endsWith('.whl') ||
        name.endsWith('.tar.gz') || (['build', 'dist'].includes(name) && fs.statSync(full).isDirectory())) {
      stale.push(name);
    }
  }
  const sourceRoot = path.join(root, 'src');
  if (fs.existsSync(sourceRoot)) {
    for (const name of fs.readdirSync(sourceRoot)) {
      if (name.endsWith('.egg-info')) stale.push(path.join('src', name));
    }
  }
  return stale.sort();
}

function verifyDocs(root = ROOT) {
  const readme = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
  if (/custback-\d+\.\d+\.\d+\.tgz/.test(readme)) {
    fail('README hard-codes a versioned npm tarball');
  }
  if (!readme.includes('TARBALL=$(npm pack --silent)')) {
    fail('README local npm tarball capture must use npm pack --silent');
  }
  if (!readme.includes('(docs/remote-deployment.md)')) {
    fail('README must link the two-host remote deployment guide');
  }
  if (!readme.includes('(docs/visual-consistency-rollout.md)')) {
    fail('README must link the visual-consistency rollout and rollback guide');
  }
  const deployment = fs.readFileSync(
    path.join(root, 'docs', 'remote-deployment.md'), 'utf8',
  );
  const requirements = [
    [/renderer-scoped token/i, 'renderer-scoped token'],
    [/avatar-control token/i, 'avatar-control token'],
    [/\bwss:\/\//i, 'WSS renderer endpoint'],
    [/\bhttps:\/\//i, 'HTTPS avatar-control endpoint'],
    [/source\.tls_ca_file/, 'renderer CA configuration'],
    [/avatar\.tls_ca_file/, 'avatar-control CA configuration'],
    [/firewall rule/i, 'firewall direction'],
    [/\brotation\b/i, 'credential rotation'],
    [/privacy slate/i, 'renderer-outage privacy behavior'],
    [/avatar_auth_failed/, 'control-token failure behavior'],
    [/avatar_unreachable/, 'control-plane outage behavior'],
  ];
  for (const [pattern, description] of requirements) {
    if (!pattern.test(deployment)) {
      fail(`remote deployment guide is missing ${description}`);
    }
  }
  const rollout = fs.readFileSync(
    path.join(root, 'docs', 'visual-consistency-rollout.md'), 'utf8',
  );
  const rolloutRequirements = [
    [/active stage is `compatibility`/i, 'active compatibility stage'],
    [/versionless.*schema-1/is, 'versionless schema-1 migration semantics'],
    [/camera-cover.*linear-compositing.*automatic-correction/is, 'ordered stages'],
    [/one clean commit/i, 'one-commit-per-default policy'],
    [
      /historical Git tree.*ancestor of\s+the clean release checkout/is,
      'historical default-change evidence policy',
    ],
    [
      /standalone Git-less archive.*fails closed/is,
      'Git-less prepack failure policy',
    ],
    [/physical-camera/i, 'physical-camera prerequisite'],
    [/consumer-sink/i, 'consumer-sink prerequisite'],
    [/external.*full-range.*sRGB.*BGR/is, 'external-frame color assumption'],
    [/`passthrough`.*`blur`.*`color`.*`remote`/is, 'mode exclusions'],
    [/subject is cropped/i, 'crop troubleshooting'],
    [/black bars appear/i, 'contain-bar troubleshooting'],
    [/low confidence/i, 'low-confidence troubleshooting'],
    [/auto-exposure\/WB loop/i, 'camera auto-control troubleshooting'],
    [/tagged media is rejected/i, 'tagged-media troubleshooting'],
    [/untagged image or video/i, 'untagged-media troubleshooting'],
    [
      /does not lower the highest recognized\s+schema version/i,
      'schema-preserving rollback policy',
    ],
  ];
  for (const [pattern, description] of rolloutRequirements) {
    if (!pattern.test(rollout)) {
      fail(`visual-consistency rollout guide is missing ${description}`);
    }
  }
  if (!readme.includes('(docs/matte-quality-rollout.md)') ||
      !readme.includes('(docs/adr/0004-matte-quality-rollout.md)')) {
    fail('README must link the MATTE-5.4 rollout guide and ADR');
  }
  const matteRollout = fs.readFileSync(
    path.join(root, 'docs', 'matte-quality-rollout.md'), 'utf8',
  );
  const matteAdr = fs.readFileSync(
    path.join(root, 'docs', 'adr', '0004-matte-quality-rollout.md'), 'utf8',
  );
  const matteRolloutRequirements = [
    [/compatibility hold/i, 'compatibility hold'],
    [/versionless.*schema-1/is, 'legacy schema semantics'],
    [/generated.*pending/is, 'generated-evidence limitation'],
    [/baseline.*ablation.*visual.*performance.*platform.*privacy.*migration/is,
      'complete release-evidence chain'],
    [/sanitized rollout telemetry/i, 'sanitized canary telemetry'],
    [/`matte_rollout`/, 'runtime rollout status'],
    [/one-patch rollback/i, 'one-patch rollback'],
    [/does not.*delete.*config/is, 'non-destructive config rollback'],
    [/model cache/i, 'non-destructive model-cache rollback'],
    [/Reactions are outside/i, 'reaction separation'],
    [/27638e419a0dcf5955d52e2eb4ead2dafbdca7f2bbe0535108aa7c56c1f2f60d/,
      'canonical rollback digest'],
  ];
  for (const [pattern, description] of matteRolloutRequirements) {
    if (!pattern.test(matteRollout)) {
      fail(`matte-quality rollout guide is missing ${description}`);
    }
  }
  for (const [pattern, description] of [
    [/timestamp and reset ownership/i, 'timestamp/reset contract'],
    [/backend-specific effective policy/i, 'backend-effective policies'],
    [/light-wrap temporal policy/i, 'light-wrap temporal policy'],
    [/quality tiers and default disposition/i, 'quality/default decision'],
    [/performance, privacy, and evidence gates/i, 'performance/privacy gates'],
  ]) {
    if (!pattern.test(matteAdr)) {
      fail(`MATTE-5.4 ADR is missing ${description}`);
    }
  }
}

function trustedReleaseSourceContext(
  root = ROOT,
  env = process.env,
  gitRunner = spawnSync,
) {
  const values = Object.fromEntries(
    Object.entries(RELEASE_SOURCE_BRIDGE_ENV)
      .map(([name, variable]) => [name, env[variable]]),
  );
  const present = Object.values(values).filter(
    (value) => typeof value === 'string' && value.length > 0,
  ).length;
  if (present === 0) return null;
  if (present !== Object.keys(values).length) {
    fail('trusted release source bridge is incomplete');
  }
  if (!/^[0-9a-f]{40}$/.test(values.commit) ||
      !/^[0-9a-f]{40}$/.test(values.tree)) {
    fail('trusted release source bridge has invalid commit metadata');
  }

  const requestedRoot = path.resolve(values.root);
  let metadata;
  try {
    metadata = fs.lstatSync(requestedRoot);
  } catch (err) {
    fail(`trusted release Git root is unavailable: ${err.message}`);
  }
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail('trusted release Git root must be a regular directory');
  }
  const sourceRoot = fs.realpathSync(requestedRoot);
  if (sourceRoot !== requestedRoot) {
    fail('trusted release Git root must not traverse symlinks');
  }

  const git = (args, encoding = 'utf8') => {
    const result = gitRunner('git', ['-C', sourceRoot, ...args], {
      encoding,
      timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
      maxBuffer: 64 * 1024 * 1024,
    });
    if (result.error) {
      fail(`trusted release Git query could not start: ${result.error.message}`);
    }
    if (result.status !== 0) {
      fail(
        'trusted release Git query failed: ' +
        `${encoding === null ? '' : (result.stderr || '').trim()}`
      );
    }
    return result.stdout;
  };
  const topLevel = fs.realpathSync(git(
    ['rev-parse', '--show-toplevel'],
  ).trim());
  const head = git(['rev-parse', '--verify', 'HEAD']).trim();
  const tree = git(['rev-parse', '--verify', 'HEAD^{tree}']).trim();
  const status = git([
    'status', '--porcelain=v1', '--untracked-files=all', '--ignored=no',
  ]);
  if (topLevel !== sourceRoot ||
      head !== values.commit ||
      tree !== values.tree ||
      status !== '') {
    fail('trusted release Git root is not the exact clean source commit');
  }
  return {
    root: sourceRoot,
    commit: values.commit,
    tree: values.tree,
    stagedRoot: fs.realpathSync(root),
    git,
  };
}

function verifyTrustedReleaseSourceFile(context, relativePath) {
  if (!context) return;
  verifyReviewedSourceFiles(context.stagedRoot, [relativePath]);
  verifyReviewedSourceFiles(context.root, [relativePath]);
  const staged = fs.readFileSync(path.join(context.stagedRoot, relativePath));
  const working = fs.readFileSync(path.join(context.root, relativePath));
  const committed = context.git(
    ['show', `${context.commit}:${relativePath}`],
    null,
  );
  if (!Buffer.isBuffer(committed) ||
      !staged.equals(working) ||
      !staged.equals(committed)) {
    fail(`staged release source differs from trusted commit: ${relativePath}`);
  }
}

function validateVisualQualificationApproval(reportPath, root = ROOT, options = {}) {
  const expectedCommit = options.expectedCommit;
  if (typeof expectedCommit !== 'string' ||
      !/^[0-9a-f]{40}$/.test(expectedCommit)) {
    fail('VIS-4.2 approval validation requires the rollout change commit');
  }
  if (typeof options.qualificationValidator === 'function') {
    return options.qualificationValidator(reportPath, {
      evidenceRoot: path.dirname(reportPath),
      expectedCommit,
    });
  }
  const python = options.python || process.env.CUSTBACK_RELEASE_PYTHON ||
    (process.platform === 'win32' ? 'python' : 'python3');
  const script = path.join(
    root, 'scripts', 'release', 'visual_consistency_qualification.py',
  );
  const manifest = path.join(
    root, 'scripts', 'release', 'visual-qualification-manifest.json',
  );
  verifyReviewedSourceFiles(root, [
    'scripts/release/visual_consistency_qualification.py',
    'scripts/release/visual-qualification-manifest.json',
  ]);
  const result = spawnSync(
    python,
    [
      script,
      '--manifest', manifest,
      'validate',
      '--report', reportPath,
      '--claim', 'release',
      '--evidence-root', path.dirname(reportPath),
      '--expected-commit', expectedCommit,
    ],
    {
      cwd: root,
      env: options.env || process.env,
      encoding: 'utf8',
      timeout: Math.min(COMMAND_TIMEOUT_MS, 2 * 60 * 1000),
    },
  );
  if (result.error) {
    fail(`VIS-4.2 approval validator could not start: ${result.error.message}`);
  }
  if (result.status !== 0) {
    fail(
      'VIS-4.2 approval report failed strict release validation: ' +
      `${(result.stderr || result.stdout || '').trim()}`
    );
  }
  return JSON.parse(fs.readFileSync(reportPath, 'utf8'));
}

function verifyVisualPolicyRollout(root = ROOT, options = {}) {
  const sourceContext = trustedReleaseSourceContext(
    root,
    options.env || process.env,
    options.gitRunner || spawnSync,
  );
  const manifestPath = path.join(
    root, 'scripts', 'release', 'visual-policy-rollout.json',
  );
  if (sourceContext) {
    for (const relativePath of [
      'config/default.yaml',
      'scripts/release/visual-policy-rollout.json',
      'scripts/release/visual-qualification-manifest.json',
      'scripts/release/visual_consistency_qualification.py',
      'src/custback/default.yaml',
    ]) {
      verifyTrustedReleaseSourceFile(sourceContext, relativePath);
    }
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const expectedTopKeys = [
    'active_stage',
    'legacy_switch_retirement',
    'qualification_authority',
    'schema_version',
    'stages',
  ];
  if (!isDeepStrictEqual(Object.keys(manifest).sort(), expectedTopKeys) ||
      manifest.schema_version !== 1 ||
      typeof manifest.qualification_authority !== 'string' ||
      !/VIS-4\.2/.test(manifest.qualification_authority) ||
      !/physical-camera/.test(manifest.qualification_authority) ||
      !/consumer-sink/.test(manifest.qualification_authority)) {
    fail('visual-policy rollout manifest header is invalid');
  }

  const expectedProfiles = [
    {
      id: 'compatibility',
      default_schema_version: 1,
      defaults: {
        camera_fit_mode: 'stretch',
        blend_space: 'srgb_legacy',
        color_correction_mode: 'off',
      },
    },
    {
      id: 'camera-cover',
      default_schema_version: 2,
      defaults: {
        camera_fit_mode: 'cover',
        blend_space: 'srgb_legacy',
        color_correction_mode: 'off',
      },
    },
    {
      id: 'linear-compositing',
      default_schema_version: 3,
      defaults: {
        camera_fit_mode: 'cover',
        blend_space: 'linear_srgb',
        color_correction_mode: 'off',
      },
    },
    {
      id: 'automatic-correction',
      default_schema_version: 4,
      defaults: {
        camera_fit_mode: 'cover',
        blend_space: 'linear_srgb',
        color_correction_mode: 'auto',
      },
    },
  ];
  const expectedRollbackPins = [
    null,
    { 'camera.fit_mode': 'stretch' },
    { 'compositing.blend_space': 'srgb_legacy' },
    { 'compositing.color_correction.mode': 'off' },
  ];
  if (!Array.isArray(manifest.stages) ||
      manifest.stages.length !== expectedProfiles.length) {
    fail('visual-policy rollout must contain exactly four ordered stages');
  }
  const activeIndex = expectedProfiles.findIndex(
    (profile) => profile.id === manifest.active_stage,
  );
  if (activeIndex < 0) fail('visual-policy rollout active_stage is unknown');

  const commits = new Set();
  const evidencePaths = new Set();
  for (let index = 0; index < expectedProfiles.length; index += 1) {
    const stage = manifest.stages[index];
    const expected = expectedProfiles[index];
    const expectedStageKeys = [
      'change_commit',
      'default_schema_version',
      'defaults',
      'evidence',
      'id',
      'requires',
      'rollback',
      'sequence',
      'status',
    ];
    if (!stage || !isDeepStrictEqual(Object.keys(stage).sort(), expectedStageKeys) ||
        stage.id !== expected.id ||
        stage.sequence !== index + 1 ||
        stage.default_schema_version !== expected.default_schema_version ||
        !isDeepStrictEqual(stage.defaults, expected.defaults)) {
      fail(`visual-policy rollout stage ${index + 1} is invalid`);
    }
    const expectedStatus = index < activeIndex
      ? 'complete'
      : index === activeIndex ? 'active' : 'pending';
    if (stage.status !== expectedStatus) {
      fail(`visual-policy rollout stage ${stage.id} must be ${expectedStatus}`);
    }
    if (!Array.isArray(stage.requires) || !Array.isArray(stage.evidence)) {
      fail(`visual-policy rollout stage ${stage.id} has invalid evidence fields`);
    }

    if (index === 0) {
      if (stage.requires.length || stage.evidence.length ||
          stage.change_commit !== null || stage.rollback !== null) {
        fail('compatibility rollout stage must not fabricate qualification evidence');
      }
      continue;
    }
    if (!stage.requires.some((item) => /VIS-4\.2/.test(item)) ||
        !stage.requires.some((item) => /physical/i.test(item)) ||
        !stage.requires.some((item) => /consumer-sink/i.test(item)) ||
        !stage.rollback || typeof stage.rollback.instructions !== 'string' ||
        !stage.rollback.instructions.trim() ||
        !stage.rollback.retain_schema_version ||
        !isDeepStrictEqual(stage.rollback.pin, expectedRollbackPins[index])) {
      fail(`visual-policy rollout stage ${stage.id} lacks prerequisites or rollback`);
    }
    const [rollbackName, rollbackValue] = Object.entries(stage.rollback.pin)[0];
    if (!stage.rollback.instructions.includes(
      `${rollbackName}: ${rollbackValue}`,
    )) {
      fail(`visual-policy rollout stage ${stage.id} has incomplete rollback instructions`);
    }
    if (stage.status === 'pending') {
      if (stage.change_commit !== null || stage.evidence.length) {
        fail(`pending rollout stage ${stage.id} must not claim evidence or a commit`);
      }
      continue;
    }
    if (typeof stage.change_commit !== 'string' ||
        !/^[0-9a-f]{40}$/.test(stage.change_commit) ||
        commits.has(stage.change_commit) ||
        stage.evidence.length !== 1) {
      fail(`approved rollout stage ${stage.id} needs distinct commit-bound evidence`);
    }
    commits.add(stage.change_commit);
    for (const evidence of stage.evidence) {
      if (typeof evidence !== 'string' ||
          !/^docs\/[a-z0-9][a-z0-9._/-]*\.json$/.test(evidence) ||
          evidence.includes('..') ||
          evidencePaths.has(evidence)) {
        fail(`rollout stage ${stage.id} has an unsafe evidence path`);
      }
      verifyReviewedSourceFiles(root, [evidence]);
      verifyTrustedReleaseSourceFile(sourceContext, evidence);
      const report = validateVisualQualificationApproval(
        path.join(root, evidence), root,
        {
          ...options,
          env: options.env || process.env,
          expectedCommit: stage.change_commit,
        },
      );
      if (!report || report.release_qualified !== true ||
          !report.source || report.source.commit !== stage.change_commit ||
          report.source.clean !== true) {
        fail(
          `rollout stage ${stage.id} approval is not bound to its clean change commit`
        );
      }
      evidencePaths.add(evidence);
    }
  }

  const config = fs.readFileSync(path.join(root, 'config', 'default.yaml'), 'utf8');
  const topLevelBlock = (name) => {
    const start = config.search(new RegExp(`^${name}:\\s*(?:#.*)?$`, 'm'));
    if (start < 0) fail(`default config is missing section ${name}`);
    const remainder = config.slice(start);
    const firstNewline = remainder.indexOf('\n');
    const body = remainder.slice(firstNewline + 1);
    const next = body.search(/^[a-z][a-z0-9_]*:\s*(?:#.*)?$/m);
    return next < 0 ? body : body.slice(0, next);
  };
  const scalar = (source, indent, name) => {
    const match = source.match(
      new RegExp(`^${' '.repeat(indent)}${name}:\\s*([^#\\n]+)`, 'm'),
    );
    if (!match) fail(`default config is missing ${name}`);
    return match[1].trim().replace(/^["']|["']$/g, '');
  };
  const schemaMatch = config.match(/^schema_version:\s*([0-9]+)\s*$/m);
  const active = manifest.stages[activeIndex];
  const effective = {
    default_schema_version: schemaMatch ? Number(schemaMatch[1]) : null,
    defaults: {
      camera_fit_mode: scalar(topLevelBlock('camera'), 2, 'fit_mode'),
      blend_space: scalar(topLevelBlock('compositing'), 2, 'blend_space'),
      color_correction_mode: scalar(topLevelBlock('compositing'), 4, 'mode'),
    },
  };
  if (effective.default_schema_version !== active.default_schema_version ||
      !isDeepStrictEqual(effective.defaults, active.defaults)) {
    fail('distributed default config does not match the active rollout stage');
  }

  const retirement = manifest.legacy_switch_retirement;
  if (!retirement || typeof retirement.earliest !== 'string' ||
      !/later documented release/.test(retirement.earliest) ||
      !isDeepStrictEqual(retirement.switches, [
        'camera.fit_mode: stretch',
        'compositing.blend_space: srgb_legacy',
        'compositing.color_correction.mode: off',
      ])) {
    fail('visual-policy rollout must defer legacy-switch retirement');
  }
  return manifest;
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(
      (key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`
    ).join(',')}}`;
  }
  return JSON.stringify(value);
}

function yamlScalarAt(source, fieldPath) {
  const wanted = fieldPath.split('.');
  const stack = [];
  for (const line of source.split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;
    const match = line.match(/^(\s*)([a-z][a-z0-9_]*):(?:\s*(.*))?$/i);
    if (!match) continue;
    const indent = match[1].length;
    while (stack.length && stack[stack.length - 1].indent >= indent) stack.pop();
    const names = [...stack.map((entry) => entry.name), match[2]];
    const raw = (match[3] || '').split('#', 1)[0].trim();
    if (isDeepStrictEqual(names, wanted)) {
      const unquoted = raw.replace(/^(["'])(.*)\1$/, '$2');
      if (unquoted === 'true') return true;
      if (unquoted === 'false') return false;
      if (unquoted === 'null') return null;
      if (/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$/.test(unquoted)) {
        return Number(unquoted);
      }
      return unquoted;
    }
    if (!raw) stack.push({ indent, name: match[2] });
  }
  fail(`default config is missing ${fieldPath}`);
}

function verifyMattePolicyRollout(root = ROOT, options = {}) {
  const sourceContext = trustedReleaseSourceContext(
    root,
    options.env || process.env,
    options.gitRunner || spawnSync,
  );
  const relativeManifest = 'scripts/release/matte-policy-rollout.json';
  if (sourceContext) {
    for (const relativePath of [
      'config/default.yaml',
      relativeManifest,
      'src/custback/api/webui.py',
      'src/custback/config.py',
      'src/custback/default.yaml',
      'src/custback/matte_rollout.py',
    ]) {
      verifyTrustedReleaseSourceFile(sourceContext, relativePath);
    }
  }
  const manifest = JSON.parse(
    fs.readFileSync(path.join(root, relativeManifest), 'utf8'),
  );
  const expectedTopKeys = [
    'active_stage',
    'compatibility_policy',
    'decision',
    'default_schema_version',
    'preset_catalog',
    'promotion',
    'reactions',
    'rollback',
    'schema',
    'version',
  ];
  if (!isDeepStrictEqual(Object.keys(manifest).sort(), expectedTopKeys) ||
      manifest.schema !== 'custback.matte-quality-rollout' ||
      manifest.version !== 1 ||
      manifest.active_stage !== 'compatibility_hold' ||
      manifest.decision !== 'held_pending_physical_qualification' ||
      manifest.default_schema_version !== 1) {
    fail('matte-policy rollout manifest header is invalid');
  }

  const expectedPatch = {
    segmentation: {
      backend: 'auto',
      delegate: 'cpu',
      rvm_downsample: 0.0,
      threshold: 0.5,
      mask_blur: 7,
      edge_refine: true,
      mask_shift: 0,
      temporal_smoothing: 0.35,
      boundary_stabilization: {
        mode: 'off',
        time_constant_s: 0.1,
        max_motion_px_per_s: 720.0,
      },
      spatial_edge_refinement: {
        mode: 'legacy_watershed',
        reference_short_edge_px: 720,
        radius_at_reference_px: 8,
        min_radius_px: 2,
        max_radius_px: 12,
      },
    },
    acceleration: {
      mode: 'auto',
      provider: 'auto',
      device_id: 0,
    },
    compositing: {
      light_wrap: 0.25,
      use_model_foreground: true,
      blend_space: 'srgb_legacy',
      light_wrap_stabilization: {
        mode: 'off',
        time_constant_s: 0.12,
      },
      color_correction: {
        mode: 'off',
        strength: 0.5,
        exposure_limit_ev: 0.85,
        white_balance_strength: 0.5,
        adaptation_time_s: 0.8,
      },
    },
  };
  const expectedPatchDigest = crypto.createHash('sha256')
    .update(canonicalJson(expectedPatch)).digest('hex');
  let runtime = options.runtimeContract;
  if (runtime === undefined) {
    const python = options.python || process.env.CUSTBACK_RELEASE_PYTHON ||
      (process.platform === 'win32' ? 'python' : 'python3');
    const pythonPath = [
      path.join(root, 'src'),
      (options.env || process.env).PYTHONPATH || '',
    ].filter(Boolean).join(path.delimiter);
    const runtimeContract = spawnSync(
      python,
      [
        '-c',
      `import ast
import json
from pathlib import Path

def value(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Dict):
        return {value(k, names): value(v, names) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [value(item, names) for item in node.elts]
    raise ValueError(f"unsupported contract expression: {ast.dump(node)}")

def returned(tree, function_name, names):
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    result = next(node for node in function.body if isinstance(node, ast.Return))
    return value(result.value, names)

config_tree = ast.parse(Path("src/custback/config.py").read_text(encoding="utf-8"))
rollout_tree = ast.parse(Path("src/custback/matte_rollout.py").read_text(encoding="utf-8"))
names = {}
for node in rollout_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        try:
            names[node.targets[0].id] = value(node.value, names)
        except ValueError:
            pass
contract = {
    "patch": returned(config_tree, "legacy_matte_policy_patch", {}),
    "status": returned(rollout_tree, "empty_matte_rollout_status", names),
}
print(json.dumps(contract, sort_keys=True, separators=(",", ":")))`,
      ],
      {
        cwd: root,
        env: { ...(options.env || process.env), PYTHONPATH: pythonPath },
        encoding: 'utf8',
        timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
      },
    );
    if (runtimeContract.error || runtimeContract.status !== 0) {
      fail(
        'matte-policy runtime contract could not be evaluated: ' +
        (runtimeContract.error?.message || runtimeContract.stderr || '').trim()
      );
    }
    try {
      runtime = JSON.parse(runtimeContract.stdout);
    } catch (_error) {
      fail('matte-policy runtime contract did not return canonical JSON');
    }
  }
  const expectedStatus = {
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
  };
  if (!runtime || !isDeepStrictEqual(runtime.patch, expectedPatch) ||
      !isDeepStrictEqual(runtime.status, expectedStatus)) {
    fail('matte-policy Python rollback/status contract disagrees with the ledger');
  }
  const compatibility = manifest.compatibility_policy;
  if (!compatibility || !isDeepStrictEqual(
    Object.keys(compatibility).sort(), ['patch', 'patch_id', 'patch_sha256'],
  ) || compatibility.patch_id !== 'matte-legacy-v1' ||
      !isDeepStrictEqual(compatibility.patch, expectedPatch) ||
      compatibility.patch_sha256 !== expectedPatchDigest) {
    fail('matte-policy compatibility patch or digest is invalid');
  }

  const repositoryConfig = fs.readFileSync(
    path.join(root, 'config', 'default.yaml'), 'utf8',
  );
  const packagedConfig = fs.readFileSync(
    path.join(root, 'src', 'custback', 'default.yaml'), 'utf8',
  );
  if (repositoryConfig !== packagedConfig ||
      yamlScalarAt(repositoryConfig, 'schema_version') !== 1) {
    fail('matte-policy distributed default files are not byte-identical schema 1');
  }
  const visitPatch = (value, parts = []) => {
    for (const [name, child] of Object.entries(value)) {
      const next = [...parts, name];
      if (child && typeof child === 'object' && !Array.isArray(child)) {
        visitPatch(child, next);
      } else if (!isDeepStrictEqual(
        yamlScalarAt(repositoryConfig, next.join('.')), child,
      )) {
        fail(`distributed default config disagrees with matte policy at ${next.join('.')}`);
      }
    }
  };
  visitPatch(expectedPatch);

  if (!isDeepStrictEqual(manifest.preset_catalog, {
    schema: 'custback.matte-quality-presets',
    version: 1,
    evidence_status: 'not_qualified',
    preset_ids: [],
  })) {
    fail('matte-policy preset catalog must remain empty and not qualified');
  }
  const webui = fs.readFileSync(
    path.join(root, 'src', 'custback', 'api', 'webui.py'), 'utf8',
  );
  if (!/schema:\s*"custback\.matte-quality-presets"/.test(webui) ||
      !/evidenceStatus:\s*"not_qualified"/.test(webui) ||
      !/presets:\s*Object\.freeze\(\{\}\)/.test(webui)) {
    fail('WebUI matte preset catalog disagrees with the held rollout decision');
  }

  const promotion = manifest.promotion;
  const expectedPromotionKeys = [
    'authority',
    'candidate_build_sha256',
    'candidate_id',
    'candidate_patch_sha256',
    'change_commit',
    'evidence',
    'generated_evidence_can_qualify',
    'limitations',
    'physical_provenance_required',
    'status',
  ];
  const expectedEvidence = [
    ['baseline', 'custback.matte-quality-report'],
    ['ablation', 'custback.matte-ablation-report'],
    ['visual', 'custback.matte-visual-qualification-report'],
    ['performance', 'custback.matte-performance-report'],
    ['platform', 'custback.matte-platform-qualification-report'],
    ['privacy', 'custback.matte-rollout-privacy-review'],
    ['migration', 'custback.matte-rollout-migration-report'],
  ];
  if (!promotion || !isDeepStrictEqual(
    Object.keys(promotion).sort(), expectedPromotionKeys,
  ) || promotion.status !== 'pending' ||
      promotion.authority !== 'none-v1-physical-evidence-pending' ||
      promotion.physical_provenance_required !== true ||
      promotion.generated_evidence_can_qualify !== false ||
      promotion.candidate_id !== null ||
      promotion.candidate_build_sha256 !== null ||
      promotion.candidate_patch_sha256 !== null ||
      promotion.change_commit !== null ||
      !isDeepStrictEqual(promotion.limitations, [
        'upstream-build-binding-owner-attested',
        'physical-origin-owner-attested',
      ]) || !Array.isArray(promotion.evidence) ||
      promotion.evidence.length !== expectedEvidence.length) {
    fail('matte-policy promotion must remain fail-closed pending physical evidence');
  }
  promotion.evidence.forEach((evidence, index) => {
    const [id, schema] = expectedEvidence[index];
    if (!evidence || !isDeepStrictEqual(Object.keys(evidence).sort(), [
      'evidence_sha256', 'file_sha256', 'id', 'report', 'schema', 'status',
    ]) || evidence.id !== id || evidence.schema !== schema ||
        evidence.status !== 'pending' || evidence.report !== null ||
        evidence.file_sha256 !== null || evidence.evidence_sha256 !== null) {
      fail(`matte-policy ${id} evidence slot must remain explicit and pending`);
    }
  });

  if (!isDeepStrictEqual(manifest.rollback, {
    patch_id: 'matte-legacy-v1',
    patch_sha256: expectedPatchDigest,
    retain_schema_version: true,
    delete_user_config: false,
    delete_model_cache: false,
    minimum_stable_releases: 1,
  }) || !isDeepStrictEqual(manifest.reactions, {
    included: false,
    required_for_matte_qualification: false,
  })) {
    fail('matte-policy rollback or reaction separation is invalid');
  }
  return manifest;
}

function verifyCiWorkflow(root = ROOT) {
  const workflow = fs.readFileSync(path.join(root, '.github', 'workflows', 'ci.yml'), 'utf8');
  const uses = [...workflow.matchAll(/^\s*-\s+uses:\s+([^\s#]+)/gm)]
    .map((match) => match[1]);
  if (!uses.length || uses.some((action) => !REVIEWED_ACTIONS.has(action))) {
    fail(`CI actions must use the reviewed commit SHA pins: ${uses.join(', ') || 'none'}`);
  }
  if (/\bnpm\s+install\b/.test(workflow) || !/\bnpm\s+ci\b/.test(workflow)) {
    fail('CI must use npm ci and must not use npm install');
  }
  if (!/^\s{2}package-smoke:\s*$/m.test(workflow) ||
      !workflow.includes('node scripts/release/package-smoke.js')) {
    fail('CI must require the non-authorizing installed package smoke job');
  }
  if (!/^\s{2}ruff:\s*$/m.test(workflow) ||
      !workflow.includes('python -m ruff check src tests examples scripts/release') ||
      !workflow.includes('python -m ruff format --check src tests examples scripts/release')) {
    fail('CI must require the reviewed Ruff lint and format gates');
  }
  if (!/^\s{2}pyright:\s*$/m.test(workflow) ||
      !workflow.includes('python -m venv .venv') ||
      !workflow.includes(".venv/bin/python -m pip install -e '.[dev]'") ||
      !workflow.includes('.venv/bin/python -m pyright')) {
    fail('CI must require pinned Pyright in the configured project virtualenv');
  }
  if (!/^\s{2}stress:\s*$/m.test(workflow) ||
      !workflow.includes('tests/test_phase6_stress.py') ||
      !workflow.includes('CUSTBACK_STRESS_ITERATIONS: "100"')) {
    fail('CI must require the deterministic Phase 6 stress gate');
  }
}

function verifyPlatformScope(root = ROOT) {
  const scriptPath = path.join(root, 'scripts', 'install_linux.sh');
  const source = fs.readFileSync(scriptPath, 'utf8');
  const guard = source.search(/unsupported Linux distribution/i);
  const identity = source.search(/DISTRO_ID=/);
  const identityLike = source.search(/DISTRO_ID_LIKE=/);
  const apt = source.search(/\bapt-get\b/);
  if (guard < 0 || identity < 0 || identityLike < 0 || apt < 0 ||
      guard > apt || identity > apt || identityLike > apt ||
      !/ubuntu\|debian/.test(source)) {
    fail('Linux installer must allow only Ubuntu/Debian before apt-get');
  }

  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-platform-check-'));
  try {
    const releasePath = path.join(fixture, 'os-release');
    const markerPath = path.join(fixture, 'mutation-attempted');
    const fakeBin = path.join(fixture, 'bin');
    fs.mkdirSync(fakeBin);
    fs.writeFileSync(releasePath, 'ID=fedora\nID_LIKE="rhel centos"\n');
    const fakeSudo = path.join(fakeBin, 'sudo');
    fs.writeFileSync(fakeSudo, '#!/bin/sh\n: > "$CUSTBACK_MUTATION_MARKER"\nexit 99\n');
    fs.chmodSync(fakeSudo, 0o700);
    const result = spawnSync('/bin/bash', [scriptPath], {
      encoding: 'utf8',
      timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
      env: {
        ...process.env,
        CUSTBACK_OS_RELEASE_FILE: releasePath,
        CUSTBACK_MUTATION_MARKER: markerPath,
        PATH: `${fakeBin}${path.delimiter}${process.env.PATH || ''}`,
      },
    });
    if ((result.error && result.status === null) || result.status === 0 ||
        !/unsupported Linux distribution/i.test(result.stderr || '') ||
        fs.existsSync(markerPath)) {
      fail('Linux installer did not reject an unsupported distro before mutation');
    }
  } finally {
    fs.rmSync(fixture, { recursive: true, force: true });
  }
}

function verifyPhase6Contracts(root = ROOT) {
  const manifestPath = path.join(root, 'scripts', 'release', 'required-gates.json');
  const manifest = phase6Evidence.loadManifest(manifestPath);
  const workflowPath = path.join(root, '.github', 'workflows', 'release.yml');
  let metadata;
  try {
    metadata = fs.lstatSync(workflowPath);
  } catch (err) {
    fail(`Phase 6 release workflow is missing: ${err.message}`);
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail('Phase 6 release workflow must be a regular, non-symlink file');
  }
  verifyReleaseWorkflow(root, manifest);
  const registry = remediationRegistry(root);
  if (registry.phase !== 6 || !registry.blockers.some((entry) => entry.id === 'REL-01')) {
    fail('Phase 6 registry and required-gate manifest are inconsistent');
  }
  return manifest;
}

function verifyReleaseWorkflow(root = ROOT, manifest = phase6Evidence.loadManifest()) {
  const workflowPath = path.join(root, '.github', 'workflows', 'release.yml');
  const source = fs.readFileSync(workflowPath, 'utf8');
  const jobsOffset = source.indexOf('\njobs:\n');
  if (jobsOffset < 0) fail('Phase 6 workflow has no jobs mapping');
  const jobsSource = source.slice(jobsOffset + 1);
  const jobIds = [...jobsSource.matchAll(/^  ([a-z0-9]+(?:-[a-z0-9]+)*):$/gm)]
    .map((match) => match[1]);
  const expected = [
    ...manifest.workflow.required_job_ids,
    manifest.workflow.aggregate_job_id,
    manifest.workflow.publish_job_id,
  ];
  if (!isDeepStrictEqual([...jobIds].sort(), [...expected].sort())) {
    fail('Phase 6 workflow job IDs do not exactly match required-gates.json');
  }
  const reviewedActions = new Set([
    ...REVIEWED_ACTIONS,
    'actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02',
    'actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093',
    'actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6',
  ]);
  const uses = [...source.matchAll(/^\s*-\s+uses:\s+([^\s#]+)/gm)]
    .map((match) => match[1]);
  if (!uses.length || uses.some((action) => !reviewedActions.has(action))) {
    fail('Phase 6 workflow contains an unreviewed or unpinned action');
  }
  const requiredSnippets = [
    'node scripts/release/build-candidate.js --output',
    'actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6',
    'node scripts/release/assemble-evidence.js assemble',
    'node scripts/release/qualify-migrations.js',
    'merge-multiple: true',
    '--result "$RUNNER_TEMP/two-host-',
    'node scripts/release/phase6-evidence.js validate-trusted',
    'gh attestation verify',
    'if: ${{ always() }}',
    'needs: [release-gate]',
    'python -m twine upload --non-interactive "$wheel" "$sdist"',
    'npm publish "$tarball" --access public --provenance',
  ];
  const missing = requiredSnippets.filter((snippet) => !source.includes(snippet));
  if (missing.length) {
    fail(`Phase 6 workflow is missing release enforcement: ${missing.join(', ')}`);
  }
}

function readRegularJson(filename, label) {
  if (typeof filename !== 'string' || !path.isAbsolute(filename)) {
    fail(`${label} path must be absolute`);
  }
  let metadata;
  try {
    metadata = fs.lstatSync(filename);
  } catch (err) {
    fail(`${label} is unavailable: ${err.message}`);
  }
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`${label} must be a regular, non-symlink file`);
  }
  try {
    return JSON.parse(fs.readFileSync(filename, 'utf8'));
  } catch (err) {
    fail(`${label} is invalid JSON: ${err.message}`);
  }
}

function verifyQualifiedCandidate(version, root = ROOT, env = process.env) {
  const artifactDirectory = env.CUSTBACK_RELEASE_CANDIDATE_DIR;
  const evidencePath = env.CUSTBACK_PHASE6_EVIDENCE;
  if (!artifactDirectory || !evidencePath) {
    fail(
      'full release:check requires CUSTBACK_RELEASE_CANDIDATE_DIR and ' +
      'CUSTBACK_PHASE6_EVIDENCE from the current release workflow run'
    );
  }
  const resolvedArtifacts = path.resolve(artifactDirectory);
  const directoryMetadata = fs.lstatSync(resolvedArtifacts);
  if (!directoryMetadata.isDirectory() || directoryMetadata.isSymbolicLink()) {
    fail('release candidate directory must be a regular, non-symlink directory');
  }
  const manifest = verifyPhase6Contracts(root);
  const candidatePath = path.join(resolvedArtifacts, 'candidate-manifest.json');
  const candidate = readRegularJson(candidatePath, 'release candidate manifest');
  evidenceAssembly.validateCandidate(candidate, manifest, env);
  if (candidate.source.version !== version) {
    fail('release candidate version does not match source metadata');
  }
  const head = spawnSync('git', ['-C', root, 'rev-parse', '--verify', 'HEAD'], {
    encoding: 'utf8', timeout: Math.min(COMMAND_TIMEOUT_MS, 30 * 1000),
  });
  if (head.error || head.status !== 0 || head.stdout.trim() !== candidate.source.commit) {
    fail('release candidate commit does not match the checked-out source commit');
  }
  const evidence = phase6Evidence.loadEvidence(evidencePath);
  phase6Evidence.validateEvidence(evidence, manifest, {
    trusted: true,
    env,
    artifactPaths: phase6Evidence.artifactPathsFromDirectory(
      evidence,
      resolvedArtifacts,
    ),
  });
  if (!isDeepStrictEqual(evidence.artifacts, candidate.artifacts) ||
      evidence.provenance.commit !== candidate.source.commit) {
    fail('trusted evidence does not describe the exact release candidate');
  }
  phase6Evidence.verifyGithubAttestations(
    evidence,
    phase6Evidence.artifactPathsFromDirectory(evidence, resolvedArtifacts),
    { env, evidencePath: path.resolve(evidencePath) },
  );
  return candidate;
}

function verifyPack(version, root = ROOT) {
  const cache = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-npm-cache-'));
  try {
    const result = spawnSync(
      'npm',
      ['pack', '--dry-run', '--json', '--ignore-scripts', '--cache', cache],
      { cwd: root, encoding: 'utf8', timeout: Math.min(COMMAND_TIMEOUT_MS, 2 * 60 * 1000) },
    );
    if (result.error) fail(`npm pack dry-run could not start: ${result.error.message}`);
    if (result.status !== 0) fail(`npm pack dry-run failed: ${(result.stderr || '').trim()}`);
    const { names } = parseNpmPackPayload(result.stdout, version);
    for (const required of [
      'LICENSE',
      'README.md',
      'REMEDIATION_PLAN.md',
      '.github/workflows/ci.yml',
      'docs/remote-deployment.md',
      'docs/visual-consistency-phase4-qualification-runbook.md',
      'docs/visual-consistency-phase4-qualification-template.json',
      'docs/visual-consistency-rollout.md',
      'package.json',
      'packaging/npm/custback.js',
      'packaging/npm/install.js',
      'packaging/npm/managed-venv.js',
      'packaging/npm/migrate-legacy.js',
      'config/default.yaml',
      'pyproject.toml',
      'scripts/install_linux.sh',
      'scripts/install_macos.sh',
      'scripts/release/package-smoke.js',
      'scripts/release/visual-qualification-manifest.json',
      'scripts/release/visual-policy-rollout.json',
      'scripts/release/matte-policy-rollout.json',
      'scripts/release/visual_consistency_qualification.py',
      'scripts/release/verify-release.js',
      'scripts/release/remediation-blockers.json',
      'src/custback/__init__.py',
      'src/custback/__main__.py',
      'src/custback/default.yaml',
    ]) {
      if (!names.includes(required)) fail(`npm artifact is missing ${required}`);
    }
    verifyNpmPayload(names, root);
  } finally {
    fs.rmSync(cache, { recursive: true, force: true });
  }
}

function runChecked(command, args, options = {}) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
    timeout: COMMAND_TIMEOUT_MS,
    ...options,
  });
  if (result.error) fail(`${command} could not start: ${result.error.message}`);
  if (result.status !== 0) {
    fail(
      `${command} ${args.join(' ')} failed:\n${(result.stdout || '').trim()}\n${(result.stderr || '').trim()}`
    );
  }
  return result;
}

function extraArtifactProfiles(platform = process.platform, arch = process.arch) {
  const profiles = [
    {
      name: 'mediapipe',
      extras: ['mediapipe'],
      probe: 'import importlib.metadata as m; import mediapipe; m.version("mediapipe")',
    },
    {
      name: 'rvm',
      extras: ['rvm'],
      probe: 'import importlib.metadata as m; import onnxruntime; m.version("onnxruntime")',
    },
    {
      name: 'audio2face',
      extras: ['audio2face', 'dev'],
      test: 'tests/test_audio2face_protocol.py',
    },
    {
      name: 'dev',
      extras: ['dev'],
      probe: 'import importlib.metadata as m; import build, pytest; m.version("httpx2")',
    },
  ];
  // onnxruntime-gpu publishes Linux x86-64 wheels. Generic CI runners can
  // validate resolution and metadata without claiming CUDA execution.
  if (platform === 'linux' && arch === 'x64') {
    profiles.splice(2, 0, {
      name: 'gpu',
      extras: ['gpu'],
      probe: 'import importlib.metadata as m; m.version("onnxruntime-gpu")',
    });
  }
  return profiles;
}

function installAndProbeExtra(python, temporaryRoot, artifact, profile, source) {
  const venv = path.join(temporaryRoot, `${profile.name}-artifact-venv`);
  return withDisposableDirectory(venv, () => {
    runChecked(python, ['-m', 'venv', venv]);
    const venvPython = path.join(venv, 'bin', 'python');
    const spec = `${artifact}[${profile.extras.join(',')}]`;
    runChecked(venvPython, [
      '-m', 'pip', 'install', '--disable-pip-version-check', spec,
    ]);
    runChecked(venvPython, ['-m', 'pip', 'check']);
    if (profile.probe) runChecked(venvPython, ['-c', profile.probe]);
    if (profile.test) {
      runChecked(venvPython, [
        '-m', 'pytest', '-q', path.join(source, profile.test),
      ], { cwd: source });
    }
  });
}

function stageCleanSource(root, destination) {
  const excludedNames = new Set([
    '.agents', '.codex', '.git', '.venv', '.pytest_cache', 'build', 'dist', '__pycache__',
    'debug.txt', 'uninstall.log',
  ]);
  fs.cpSync(root, destination, {
    recursive: true,
    filter(source) {
      const relative = path.relative(root, source);
      if (relative === '') return true;
      const parts = relative.split(path.sep);
      if (parts.some((part) => excludedNames.has(part) || part.endsWith('.egg-info') ||
          part.endsWith('.custback-generations'))) return false;
      if (relative.endsWith('.tgz') || relative.endsWith('.whl') ||
          relative.endsWith('.tar.gz') || relative.endsWith('.pyc') ||
          /(^|[\\/])onnxruntime_profile__.*\.json$/.test(relative)) return false;
      return true;
    },
  });
}

function verifyPythonArtifacts(version, temporaryRoot, root = ROOT) {
  const python = process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
  const source = path.join(temporaryRoot, 'source');
  const output = path.join(temporaryRoot, 'python-dist');
  const buildTools = path.join(temporaryRoot, 'build-tools-venv');
  verifyReviewedSourceFiles(root, [
    'LICENSE',
    'MANIFEST.in',
    'README.md',
    'pyproject.toml',
    ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
    ...REVIEWED_PYTHON_TESTS,
    ...REVIEWED_PYTHON_SDIST_DATA,
  ]);
  stageCleanSource(root, source);
  fs.mkdirSync(output);
  withDisposableDirectory(buildTools, () => {
    runChecked(python, ['-m', 'venv', buildTools]);
    const buildPython = path.join(buildTools, 'bin', 'python');
    runChecked(buildPython, [
      '-m', 'pip', 'install', '--disable-pip-version-check',
      'pip>=23,<27', 'build>=1.2,<2',
    ]);
    runChecked(buildPython, [
      '-m', 'build', '--sdist', '--wheel', '--outdir', output, source,
    ]);
  });
  const files = fs.readdirSync(output);
  const wheels = files.filter((name) => name.endsWith('.whl'));
  const sdists = files.filter((name) => name.endsWith('.tar.gz'));
  if (files.length !== 2 || wheels.length !== 1 || sdists.length !== 1) {
    fail(`Python build produced an unexpected artifact set: ${files.join(', ')}`);
  }
  const [wheel] = wheels;
  const [sdist] = sdists;
  const normalized = version.replace(/-/g, '_');
  const escaped = normalized.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const wheelPattern = new RegExp(`^custback-${escaped}-(?:\\d+-)?[^-]+-[^-]+-[^-]+\\.whl$`);
  if (!wheelPattern.test(wheel) || sdist !== `custback-${version}.tar.gz`) {
    fail(`Python artifact filenames do not match ${version}: ${wheel}, ${sdist}`);
  }

  const inspectCode = `
import configparser, email.parser, io, json, pathlib, re, sys, tarfile, zipfile
(
    version, wheel_path, sdist_path, module_json, test_json, sdist_data_json,
    core_json, optional_json, scripts_json, license_path,
) = sys.argv[1:]
bad = ("/.venv/", "__pycache__", ".pyc", ".tgz", "/debug.txt", "/uninstall.log", "/build/")
source_modules = set(json.loads(module_json))
source_tests = set(json.loads(test_json))
source_sdist_data = set(json.loads(sdist_data_json))
core_dependencies = json.loads(core_json)
optional_dependencies = json.loads(optional_json)
console_scripts = json.loads(scripts_json)
license_bytes = pathlib.Path(license_path).read_bytes()

def require(condition, detail):
    if not condition:
        raise RuntimeError(repr(detail))

def normalize_requirement(value):
    requirement, separator, marker = value.partition(";")
    match = re.fullmatch(r"\\s*([A-Za-z0-9_.-]+)\\s*(.*?)\\s*", requirement)
    require(match, value)
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    specifier = match.group(2).strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1]
    specs = tuple(sorted(part.replace(" ", "") for part in specifier.split(",") if part.strip()))
    normalized_marker = ""
    if separator:
        # Setuptools may render PEP 508 marker quotes differently from
        # pyproject.toml. Preserve the reviewed marker while normalizing only
        # insignificant whitespace and quote style.
        normalized_marker = re.sub(
            r"\\s+", " ", marker.replace("'", '"')
        ).strip()
    return name, specs, normalized_marker

expected_requires = {normalize_requirement(item) for item in core_dependencies}
for extra, dependencies in optional_dependencies.items():
    expected_requires |= {
        normalize_requirement(f"{item}; extra == '{extra}'") for item in dependencies
    }

def verify_metadata(text):
    metadata = email.parser.Parser().parsestr(text)
    require(metadata["Name"].lower() == "custback", metadata["Name"])
    require(metadata["Version"] == version, metadata["Version"])
    require(metadata.get_all("License-File", []) == ["LICENSE"], metadata.items())
    requires_python = {
        item.strip() for item in metadata["Requires-Python"].split(",")
    }
    require(requires_python == {">=3.10", "<3.15"}, metadata["Requires-Python"])
    provides = metadata.get_all("Provides-Extra", [])
    require(len(provides) == len(set(provides)), provides)
    require(set(provides) == set(optional_dependencies), provides)
    requires = metadata.get_all("Requires-Dist", [])
    normalized = [normalize_requirement(item) for item in requires]
    require(len(normalized) == len(set(normalized)), requires)
    require(set(normalized) == expected_requires, {
        "missing": sorted(expected_requires - set(normalized)),
        "unexpected": sorted(set(normalized) - expected_requires),
    })

def verify_entry_points(text):
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_file(io.StringIO(text))
    require(parser.sections() == ["console_scripts"], parser.sections())
    require(dict(parser.items("console_scripts")) == console_scripts, dict(
        parser.items("console_scripts")
    ))

with zipfile.ZipFile(wheel_path) as archive:
    names = archive.namelist()
    require(len(names) == len(set(names)), names)
    require(not [n for n in names if any(x in "/" + n for x in bad)], names)
    require(not [n for n in names if n.endswith("/")], names)
    dist_info = f"custback-{version}.dist-info"
    expected = source_modules | {
        f"{dist_info}/licenses/LICENSE",
        f"{dist_info}/METADATA",
        f"{dist_info}/WHEEL",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/top_level.txt",
        f"{dist_info}/RECORD",
    }
    require(set(names) == expected, {
        "missing": sorted(expected - set(names)),
        "unexpected": sorted(set(names) - expected),
    })
    metadata = f"{dist_info}/METADATA"
    require(archive.read(f"{dist_info}/licenses/LICENSE") == license_bytes, "wheel LICENSE")
    text = archive.read(metadata).decode()
    require(f"Version: {version}\\n" in text, text[:500])
    verify_metadata(text)
    verify_entry_points(archive.read(f"{dist_info}/entry_points.txt").decode())
with tarfile.open(sdist_path, "r:gz") as archive:
    members = archive.getmembers()
    member_names = [m.name.rstrip("/") for m in members]
    require(len(member_names) == len(set(member_names)), member_names)
    require(not [m.name for m in members if not (m.isfile() or m.isdir())], [m.name for m in members])
    names = {m.name for m in members if m.isfile()}
    require(not [n for n in names if any(x in "/" + n for x in bad)], sorted(names))
    prefix = f"custback-{version}"
    expected = {
        f"{prefix}/LICENSE",
        f"{prefix}/MANIFEST.in",
        f"{prefix}/PKG-INFO",
        f"{prefix}/README.md",
        f"{prefix}/pyproject.toml",
        f"{prefix}/setup.cfg",
        f"{prefix}/src/custback.egg-info/PKG-INFO",
        f"{prefix}/src/custback.egg-info/SOURCES.txt",
        f"{prefix}/src/custback.egg-info/dependency_links.txt",
        f"{prefix}/src/custback.egg-info/entry_points.txt",
        f"{prefix}/src/custback.egg-info/requires.txt",
        f"{prefix}/src/custback.egg-info/top_level.txt",
    }
    expected |= {f"{prefix}/src/{name}" for name in source_modules}
    expected |= {f"{prefix}/{name}" for name in source_tests}
    expected |= {f"{prefix}/{name}" for name in source_sdist_data}
    require(names == expected, {
        "missing": sorted(expected - names),
        "unexpected": sorted(names - expected),
    })
    allowed_directories = {prefix}
    for name in expected:
        parent = pathlib.PurePosixPath(name).parent
        while str(parent) != ".":
            allowed_directories.add(str(parent))
            parent = parent.parent
    actual_directories = {m.name.rstrip("/") for m in members if m.isdir()}
    require(not (actual_directories - allowed_directories), sorted(
        actual_directories - allowed_directories
    ))
    pkg_info = f"{prefix}/PKG-INFO"
    require(archive.extractfile(f"{prefix}/LICENSE").read() == license_bytes, "sdist LICENSE")
    text = archive.extractfile(pkg_info).read().decode()
    require(f"Version: {version}\\n" in text, text[:500])
    verify_metadata(text)
    egg_info = f"{prefix}/src/custback.egg-info"
    verify_metadata(archive.extractfile(f"{egg_info}/PKG-INFO").read().decode())
    verify_entry_points(archive.extractfile(f"{egg_info}/entry_points.txt").read().decode())
`;
  runChecked(python, [
    '-c', inspectCode, version, path.join(output, wheel), path.join(output, sdist),
    JSON.stringify(REVIEWED_PYTHON_MODULES), JSON.stringify(REVIEWED_PYTHON_TESTS),
    JSON.stringify(REVIEWED_PYTHON_SDIST_DATA),
    JSON.stringify(REVIEWED_CORE_DEPENDENCIES),
    JSON.stringify(REVIEWED_OPTIONAL_DEPENDENCIES),
    JSON.stringify(REVIEWED_CONSOLE_SCRIPTS),
    path.join(root, 'LICENSE'),
  ]);

  const importProbe = [
    'import importlib.metadata as m',
    'import av, custback, custback.api.server, custback.color, custback.video_decoder, cv2, fastapi, numpy, pydantic, PIL, pyvirtualcam, uvicorn, websockets, yaml',
    `expected = ${JSON.stringify(version)}`,
    'metadata_version = m.version("custback")',
    'if metadata_version != expected:\n    raise RuntimeError(f"metadata version {metadata_version!r} != {expected!r}")',
    'if custback.__version__ != expected:\n    raise RuntimeError(f"source version {custback.__version__!r} != {expected!r}")',
    'video_frame = av.VideoFrame.from_ndarray(numpy.stack((numpy.full((2, 2), 126, dtype=numpy.uint8), numpy.full((2, 2), 128, dtype=numpy.uint8), numpy.full((2, 2), 128, dtype=numpy.uint8))), format="yuv444p")',
    'video_frame.colorspace, video_frame.color_range, video_frame.color_primaries, video_frame.color_trc = 1, 1, 1, 13',
    'normalized, color_contract = custback.video_decoder.normalize_video_frame(video_frame, video_frame, custback.video_decoder.VideoColorOverrides())',
    'if normalized.shape != (2, 2, 3) or normalized.dtype != numpy.uint8 or not normalized.flags.c_contiguous or int(numpy.max(numpy.abs(normalized.astype(numpy.int16) - 128))) > 2 or color_contract.declared_input != "bt709/limited/bt709/srgb" or color_contract.status != "tagged" or color_contract.output != "srgb-full-bgr" or color_contract.assumed_fields != () or color_contract.overridden_fields != ():\n    raise RuntimeError("installed tagged video normalization probe failed")',
  ].join('\n');
  for (const [kind, artifact] of [
    ['wheel', path.join(output, wheel)],
    ['sdist', path.join(output, sdist)],
  ]) {
    const venv = path.join(temporaryRoot, `${kind}-smoke-venv`);
    withDisposableDirectory(venv, () => {
      runChecked(python, ['-m', 'venv', venv]);
      const venvPython = path.join(venv, 'bin', 'python');
      runChecked(venvPython, [
        '-m', 'pip', 'install', '--disable-pip-version-check', artifact,
      ]);
      runChecked(venvPython, ['-m', 'pip', 'check']);
      runChecked(venvPython, ['-c', importProbe]);
      const installedCustback = path.join(venv, 'bin', 'custback');
      const installedAvatar = path.join(venv, 'bin', 'custback-avatar');
      const exportedConfig = path.join(temporaryRoot, `${kind}-avatar.yaml`);
      runChecked(installedCustback, ['--help']);
      runChecked(installedCustback, ['avatar', '--help']);
      runChecked(installedAvatar, ['--help']);
      runChecked(installedCustback, ['avatar', '--smoke']);
      runChecked(installedCustback, ['avatar', 'config', 'export', exportedConfig]);
      if (!fs.readFileSync(exportedConfig).equals(
        fs.readFileSync(path.join(root, 'config', 'avatar.yaml'))
      )) {
        fail(`${kind} installed avatar config export differs from the canonical template`);
      }
      if ((fs.statSync(exportedConfig).mode & 0o777) !== 0o600) {
        fail(`${kind} installed avatar config export is not mode 0600`);
      }
      const overwrite = spawnSync(
        installedAvatar,
        ['config', 'export', exportedConfig],
        { encoding: 'utf8', timeout: COMMAND_TIMEOUT_MS },
      );
      if (overwrite.error || overwrite.status !== 2 ||
          !/(?:exist|EEXIST|cannot export)/i.test(overwrite.stderr || '') ||
          !fs.readFileSync(exportedConfig).equals(
            fs.readFileSync(path.join(root, 'config', 'avatar.yaml'))
          )) {
        fail(`${kind} installed avatar config did not refuse overwrite safely`);
      }
      fs.rmSync(exportedConfig);
    });
  }

  const audio2faceProfile = extraArtifactProfiles()
    .find((profile) => profile.name === 'audio2face');
  if (!audio2faceProfile) fail('audio2face artifact profile is missing');
  const sourceProfile = { ...audio2faceProfile, name: 'audio2face-source' };
  installAndProbeExtra(python, temporaryRoot, source, sourceProfile, source);
  for (const profile of extraArtifactProfiles()) {
    installAndProbeExtra(
      python, temporaryRoot, path.join(output, wheel), profile, source,
    );
  }
  return { wheel: path.join(output, wheel), sdist: path.join(output, sdist) };
}

function verifyNpmArtifactInstall(version, temporaryRoot, root = ROOT) {
  const packDirectory = path.join(temporaryRoot, 'npm-pack');
  const cache = path.join(temporaryRoot, 'npm-cache');
  const prefix = path.join(temporaryRoot, 'npm-prefix');
  const managedVenv = path.join(temporaryRoot, 'npm-managed-venv');
  fs.mkdirSync(packDirectory);
  const packed = runChecked('npm', [
    'pack', '--json', '--ignore-scripts', '--pack-destination', packDirectory, '--cache', cache,
  ], { cwd: root });
  const { artifact, names } = parseNpmPackPayload(packed.stdout, version);
  verifyNpmPayload(names, root);
  const tarball = path.join(packDirectory, artifact.filename);
  verifyNpmArtifactLicense(tarball, root);
  const smokeEnv = {
    ...process.env,
    CUSTBACK_VENV: managedVenv,
    CUSTBACK_EXTRAS: '',
    CUSTBACK_SKIP_INSTALL: '0',
    CUSTBACK_FORCE_REBUILD: '0',
  };
  runChecked('npm', [
    'install', '--global', tarball, '--prefix', prefix, '--cache', cache,
  ], {
    cwd: temporaryRoot,
    env: smokeEnv,
  });
  const launcher = path.join(prefix, 'bin', 'custback');
  const avatarLauncher = path.join(prefix, 'bin', 'custback-avatar');
  const exportedConfig = path.join(temporaryRoot, 'npm-avatar.yaml');
  runChecked(launcher, ['--help'], { env: smokeEnv });
  runChecked(launcher, ['avatar', '--help'], { env: smokeEnv });
  runChecked(avatarLauncher, ['--help'], { env: smokeEnv });
  runChecked(launcher, ['avatar', '--smoke'], { env: smokeEnv });
  runChecked(launcher, ['avatar', 'config', 'export', exportedConfig], {
    env: smokeEnv,
  });
  if (!fs.readFileSync(exportedConfig).equals(
    fs.readFileSync(path.join(root, 'config', 'avatar.yaml'))
  )) {
    fail('npm installed avatar config export differs from the canonical template');
  }
  if ((fs.statSync(exportedConfig).mode & 0o777) !== 0o600) {
    fail('npm installed avatar config export is not mode 0600');
  }
  const npmOverwrite = spawnSync(
    avatarLauncher,
    ['config', 'export', exportedConfig],
    { encoding: 'utf8', timeout: COMMAND_TIMEOUT_MS, env: smokeEnv },
  );
  if (npmOverwrite.error || npmOverwrite.status !== 1 ||
      !/(?:exist|EEXIST|cannot export)/i.test(npmOverwrite.stderr || '') ||
      !fs.readFileSync(exportedConfig).equals(
        fs.readFileSync(path.join(root, 'config', 'avatar.yaml'))
      )) {
    fail('npm installed avatar config did not refuse overwrite safely');
  }
  runChecked(launcher, ['rebuild'], { env: smokeEnv });
  runChecked(launcher, ['doctor'], { env: smokeEnv });
  const stamp = JSON.parse(fs.readFileSync(path.join(managedVenv, managed.INSTALL_STAMP), 'utf8'));
  if (!installer.validInstallStamp(stamp) || stamp.packageVersion !== version ||
      stamp.sourceDigest !== installer.sourceDigest(root) || stamp.requestedExtras.length !== 0) {
    fail('npm smoke install stamp is inconsistent with the artifact');
  }
}

function verifyBuiltArtifacts(version, root = ROOT) {
  const buildJobs = releaseBuildJobs();
  const temporaryRoot = createReleaseTemporaryRoot({ root });
  try {
    return withTemporaryEnvironment({
      TMPDIR: temporaryRoot,
      TMP: temporaryRoot,
      TEMP: temporaryRoot,
      CMAKE_BUILD_PARALLEL_LEVEL: buildJobs,
      GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS: buildJobs,
      MAKEFLAGS: `-j${buildJobs}`,
      MAX_JOBS: buildJobs,
    }, () => {
      verifyPythonArtifacts(version, temporaryRoot, root);
      verifyNpmArtifactInstall(version, temporaryRoot, root);
    });
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
}

function releasePlan(argv) {
  const allowed = new Set(['--prepack', '--quick']);
  const unknown = argv.filter((arg) => !allowed.has(arg));
  if (unknown.length || new Set(argv).size !== argv.length ||
      (argv.includes('--prepack') && argv.includes('--quick'))) {
    fail(`invalid release verification arguments: ${argv.join(' ') || '<none>'}`);
  }
  const prepack = argv.includes('--prepack');
  const quick = argv.includes('--quick');
  return {
    builtArtifacts: !prepack && !quick,
    success: prepack
      ? 'metadata and non-recursive npm packlist verified'
      : quick
        ? 'metadata and npm packlist verified (quick mode)'
        : 'metadata, npm packlist, and exact qualified artifacts verified',
  };
}

function verifyPackageSmoke(root = ROOT) {
  // Installed-artifact acceptance remains runnable while the registry is open,
  // but this function never authorizes or publishes a release candidate.
  const version = verifyVersions(root);
  verifyDependencies(root);
  verifyLicenseMetadata(root);
  verifyAvatarConfigTemplate(root);
  verifyCoreConfigTemplate(root);
  verifyDocs(root);
  verifyVisualPolicyRollout(root);
  verifyMattePolicyRollout(root);
  verifyCiWorkflow(root);
  verifyPhase6Contracts(root);
  verifyPlatformScope(root);
  verifyBlockerRegressionCoverage(root);
  const stale = staleArtifacts(root);
  if (stale.length) {
    fail(`stale release artifacts must be removed before package smoke: ${stale.join(', ')}`);
  }
  verifyPack(version, root);
  verifyBuiltArtifacts(version, root);
  return version;
}

function main(argv = process.argv.slice(2)) {
  try {
    const regressionMode = argv.length === 1 &&
      argv[0].match(/^--regressions=(all|pytest|node)$/);
    if (regressionMode) {
      // This proves registry-linked outcomes but is deliberately
      // non-authorizing: an open blocker still makes every ordinary release
      // mode fail below.
      runBlockerRegressionTests(regressionMode[1]);
      console.log(
        `[custback release] ${regressionMode[1]} remediation regressions verified (diagnostic only)`
      );
      return 0;
    }
    const plan = releasePlan(argv);
    const version = verifyVersions();
    verifyDependencies();
    verifyLicenseMetadata();
    verifyAvatarConfigTemplate();
    verifyCoreConfigTemplate();
    verifyDocs();
    verifyVisualPolicyRollout();
    verifyMattePolicyRollout();
    verifyCiWorkflow();
    verifyPhase6Contracts();
    verifyPlatformScope();
    verifyNoReleaseBlockers();
    const stale = staleArtifacts();
    if (stale.length) {
      fail(`stale release artifacts must be removed before release: ${stale.join(', ')}`);
    }
    // --ignore-scripts makes this safe to call from prepack without invoking
    // the prepack lifecycle recursively. The full path consumes the exact
    // once-built candidate and trusted same-run evidence; it never rebuilds.
    verifyPack(version);
    if (plan.builtArtifacts) verifyQualifiedCandidate(version);
    const successLine = `[custback release] ${plan.success} for ${version}`;
    if (argv.includes('--prepack')) {
      // npm owns lifecycle stdout: `npm pack --silent` must emit only the
      // generated tarball filename so shell command substitution is safe.
      console.error(successLine);
    } else {
      console.log(successLine);
    }
    return 0;
  } catch (err) {
    console.error(`[custback release] ${err.message}`);
    return 1;
  }
}

module.exports = {
  canonicalLicense,
  createReleaseTemporaryRoot,
  extraArtifactProfiles,
  exactNodeTapOutcome,
  filesystemIsMemoryBacked,
  validateVisualQualificationApproval,
  main,
  parseNpmPackPayload,
  projectVersion,
  releaseBuildJobs,
  releasePlan,
  remediationBlockers,
  remediationContractDigest,
  remediationRegistry,
  runBlockerRegressionTests,
  staleArtifacts,
  trustedReleaseSourceContext,
  verifyDependencies,
  verifyDocs,
  verifyBuiltArtifacts,
  verifyAvatarConfigTemplate,
  verifyCoreConfigTemplate,
  verifyBlockerRegressionCoverage,
  verifyCiWorkflow,
  verifyLicenseMetadata,
  verifyMattePolicyRollout,
  verifyNpmArtifactLicense,
  verifyNpmArtifactInstall,
  verifyNpmMetadata,
  verifyNoReleaseBlockers,
  verifyPack,
  verifyPlatformScope,
  verifyPhase6Contracts,
  verifyPackageSmoke,
  verifyPythonArtifacts,
  verifyQualifiedCandidate,
  verifyReleaseWorkflow,
  verifyTrustedReleaseSourceFile,
  verifyVisualPolicyRollout,
  verifyVersions,
  withDisposableDirectory,
  withTemporaryEnvironment,
};

if (require.main === module) process.exit(main());
