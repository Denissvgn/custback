#!/usr/bin/env node
/** Reject release inputs not represented by the exact clean Git commit. */

'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..', '..');
const REQUIRED_TRACKED = [
  '.github/workflows/ci.yml',
  '.github/workflows/release.yml',
  'LICENSE',
  'MANIFEST.in',
  'README.md',
  'REMEDIATION_PLAN.md',
  'docs/remote-deployment.md',
  'package-lock.json',
  'package.json',
  'pyproject.toml',
  'scripts/release/build-candidate.js',
  'scripts/release/assemble-evidence.js',
  'scripts/release/phase6-evidence.js',
  'scripts/release/qualify-migrations.js',
  'scripts/release/remediation-blockers.json',
  'scripts/release/required-gates.json',
  'scripts/release/two-host-system-test.py',
  'scripts/release/two-host/Dockerfile',
  'scripts/release/two-host/probe.py',
  'packaging/npm/test/migration-qualification.test.js',
  'packaging/npm/test/release-workflow.test.js',
  'tests/test_audio2face_protocol.py',
  'tests/test_phase6_migration.py',
  'tests/test_phase6_stress.py',
  'tests/test_phase6_two_host_system.py',
];
const FORBIDDEN_NAMES = new Set([
  '.pytest_cache', '.ruff_cache', '.venv', '__pycache__', 'build', 'dist',
  'debug.txt', 'uninstall.log',
]);

function fail(message) {
  throw new Error(message);
}

function git(root, args) {
  const result = spawnSync('git', ['-C', root, ...args], {
    encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, timeout: 30 * 1000,
  });
  if (result.error) fail(`git could not start: ${result.error.message}`);
  if (result.status !== 0) {
    fail(`git ${args.join(' ')} failed: ${(result.stderr || result.stdout || '').trim()}`);
  }
  return result.stdout;
}

function forbiddenGeneratedPaths(root) {
  const found = [];
  const visit = (directory, relativeDirectory = '') => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const relative = relativeDirectory ? path.join(relativeDirectory, entry.name) : entry.name;
      if (relative === '.git' || relative.startsWith(`.git${path.sep}`)) continue;
      const artifact = entry.name.endsWith('.egg-info') || entry.name.endsWith('.tgz') ||
        entry.name.endsWith('.whl') || entry.name.endsWith('.tar.gz') ||
        /^onnxruntime_profile__.*\.json$/.test(entry.name);
      if (FORBIDDEN_NAMES.has(entry.name) || artifact) {
        found.push(relative);
        continue;
      }
      if (entry.isDirectory() && !entry.isSymbolicLink()) {
        visit(path.join(directory, entry.name), relative);
      }
    }
  };
  visit(root);
  return found.sort();
}

function verifyCleanTree(root = ROOT, options = {}) {
  const resolved = fs.realpathSync(root);
  const status = git(resolved, [
    'status', '--porcelain=v1', '--untracked-files=all', '--ignored=no',
  ]);
  if (status !== '') {
    fail(`release tree has modified or untracked paths: ${status.trimEnd().split('\n').join(', ')}`);
  }
  const forbidden = forbiddenGeneratedPaths(resolved);
  if (forbidden.length) {
    fail(`release tree contains generated/cache payloads: ${forbidden.join(', ')}`);
  }
  const required = options.requiredTracked || REQUIRED_TRACKED;
  for (const name of required) {
    const full = path.join(resolved, name);
    let metadata;
    try {
      metadata = fs.lstatSync(full);
    } catch (err) {
      fail(`delivery-critical file is missing: ${name}`);
    }
    if (!metadata.isFile() || metadata.isSymbolicLink()) {
      fail(`delivery-critical path must be a regular tracked file: ${name}`);
    }
    git(resolved, ['ls-files', '--error-unmatch', '--', name]);
  }
  return git(resolved, ['rev-parse', '--verify', 'HEAD']).trim();
}

function main(argv = process.argv.slice(2)) {
  try {
    if (argv.length > 1) fail('usage: verify-clean-tree.js [ROOT]');
    const commit = verifyCleanTree(argv[0] || ROOT);
    process.stdout.write(`[custback clean tree] ${commit}\n`);
    return 0;
  } catch (err) {
    process.stderr.write(`[custback clean tree] ${err.message}\n`);
    return 1;
  }
}

module.exports = { REQUIRED_TRACKED, forbiddenGeneratedPaths, main, verifyCleanTree };

if (require.main === module) process.exit(main());
