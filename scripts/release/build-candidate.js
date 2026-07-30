#!/usr/bin/env node
/** Build the three publishable files once and bind them to one clean commit. */

'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const evidence = require('./phase6-evidence');
const release = require('./verify-release');
const cleanTree = require('./verify-clean-tree');

const ROOT = path.resolve(__dirname, '..', '..');
const COMMAND_TIMEOUT_MS = 30 * 60 * 1000;
const RELEASE_SOURCE_BRIDGE_ENV = Object.freeze([
  'CUSTBACK_RELEASE_GIT_ROOT',
  'CUSTBACK_RELEASE_SOURCE_COMMIT',
  'CUSTBACK_RELEASE_SOURCE_TREE',
]);

function fail(message) {
  throw new Error(message);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd || ROOT,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
    timeout: options.timeout || COMMAND_TIMEOUT_MS,
    env: options.env || process.env,
    stdio: options.stdio,
  });
  if (result.error) fail(`${command} could not start: ${result.error.message}`);
  if (result.status !== 0) {
    fail(
      `${command} ${args.join(' ')} failed:\n` +
      `${(result.stdout || '').trim()}\n${(result.stderr || '').trim()}`,
    );
  }
  return result;
}

function gitOutput(root, args) {
  return run('git', ['-C', root, ...args]).stdout.trim();
}

function withoutReleaseSourceBridge(env = process.env) {
  const sanitized = { ...env };
  for (const name of RELEASE_SOURCE_BRIDGE_ENV) delete sanitized[name];
  return sanitized;
}

function trustedPrepackEnvironment(root, source, env = process.env) {
  if (!source ||
      !/^[0-9a-f]{40}$/.test(source.commit) ||
      !/^[0-9a-f]{40}$/.test(source.tree)) {
    fail('trusted prepack environment requires exact source commit metadata');
  }
  const sourceRoot = fs.realpathSync(root);
  return {
    ...withoutReleaseSourceBridge(env),
    CUSTBACK_SKIP_INSTALL: '1',
    CUSTBACK_RELEASE_GIT_ROOT: sourceRoot,
    CUSTBACK_RELEASE_SOURCE_COMMIT: source.commit,
    CUSTBACK_RELEASE_SOURCE_TREE: source.tree,
  };
}

function cleanCommit(root = ROOT) {
  const metadata = fs.lstatSync(root);
  if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
    fail('release source must be a regular, non-symlink directory');
  }
  const commit = gitOutput(root, ['rev-parse', '--verify', 'HEAD']);
  if (!/^[0-9a-f]{40}$/.test(commit)) fail('release source HEAD is not a full commit id');
  const tree = gitOutput(root, ['rev-parse', '--verify', 'HEAD^{tree}']);
  if (!/^[0-9a-f]{40}$/.test(tree)) fail('release source tree is not a full object id');
  const status = run('git', [
    '-C', root, 'status', '--porcelain=v1', '--untracked-files=all',
    '--ignored=no',
  ]).stdout;
  if (status !== '') {
    const paths = status.trimEnd().split('\n').slice(0, 20).join(', ');
    fail(`release source must be clean; changed paths: ${paths}`);
  }
  const tracked = gitOutput(root, ['ls-files', '-z']);
  if (!tracked) fail('release source commit contains no tracked files');
  return { commit, tree };
}

function sha256File(filename) {
  const metadata = fs.lstatSync(filename);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`candidate artifact must be a regular, non-symlink file: ${filename}`);
  }
  const descriptor = fs.openSync(filename, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW || 0));
  try {
    const hash = crypto.createHash('sha256');
    const buffer = Buffer.allocUnsafe(1024 * 1024);
    let position = 0;
    while (true) {
      const count = fs.readSync(descriptor, buffer, 0, buffer.length, position);
      if (count === 0) break;
      hash.update(buffer.subarray(0, count));
      position += count;
    }
    const opened = fs.fstatSync(descriptor);
    const after = fs.lstatSync(filename);
    if (opened.dev !== metadata.dev || opened.ino !== metadata.ino ||
        after.dev !== opened.dev || after.ino !== opened.ino || after.size !== opened.size) {
      fail(`candidate artifact changed while hashing: ${filename}`);
    }
    return { sha256: hash.digest('hex'), size: opened.size };
  } finally {
    fs.closeSync(descriptor);
  }
}

function artifactRecords(directory, manifest = evidence.loadManifest()) {
  const candidates = fs.readdirSync(directory)
    .filter((name) => !['candidate-manifest.json'].includes(name));
  const records = [];
  const consumed = new Set();
  for (const definition of manifest.artifacts) {
    const pattern = new RegExp(definition.filename_pattern);
    const matches = candidates.filter((name) => pattern.test(name));
    if (matches.length !== 1) {
      fail(
        `candidate directory must contain one ${definition.id}; found ` +
        `${matches.join(', ') || 'none'}`,
      );
    }
    const filename = matches[0];
    if (consumed.has(filename)) fail(`candidate artifact matched more than one id: ${filename}`);
    consumed.add(filename);
    const digest = sha256File(path.join(directory, filename));
    records.push({ id: definition.id, filename, ...digest });
  }
  const unexpected = candidates.filter((name) => !consumed.has(name));
  if (unexpected.length) fail(`candidate directory has unexpected files: ${unexpected.join(', ')}`);
  if (new Set(records.map((entry) => entry.sha256)).size !== records.length) {
    fail('candidate artifacts must have distinct SHA-256 digests');
  }
  return records;
}

function githubProvenance(commit, env = process.env) {
  const required = [
    'GITHUB_REPOSITORY', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_WORKFLOW_REF',
  ];
  if (env.GITHUB_ACTIONS === 'true') {
    for (const name of required) {
      if (!env[name]) fail(`release candidate build requires ${name} in GitHub Actions`);
    }
    if (env.GITHUB_SHA !== commit) fail('GitHub Actions SHA does not match the clean source commit');
    return {
      provider: 'github-actions',
      repository: env.GITHUB_REPOSITORY,
      commit,
      run_id: env.GITHUB_RUN_ID,
      run_attempt: Number(env.GITHUB_RUN_ATTEMPT),
      workflow_ref: env.GITHUB_WORKFLOW_REF,
    };
  }
  return {
    provider: 'local-diagnostic',
    repository: 'local/custback',
    commit,
    run_id: '0',
    run_attempt: 0,
    workflow_ref: 'local/custback/.github/workflows/release.yml@diagnostic',
  };
}

function prepareOutput(directory) {
  const resolved = path.resolve(directory);
  const parent = fs.realpathSync(path.dirname(resolved));
  const root = fs.realpathSync(ROOT);
  const relative = path.relative(root, resolved);
  if (!relative || (!relative.startsWith(`..${path.sep}`) && relative !== '..' &&
      !path.isAbsolute(relative))) {
    fail('candidate output directory must be outside the source checkout');
  }
  const existing = (() => {
    try { return fs.lstatSync(resolved); } catch (err) {
      if (err.code === 'ENOENT') return null;
      throw err;
    }
  })();
  if (existing) {
    if (!existing.isDirectory() || existing.isSymbolicLink() || fs.readdirSync(resolved).length) {
      fail('candidate output directory must be absent or an empty real directory');
    }
  } else {
    fs.mkdirSync(resolved, { mode: 0o700 });
  }
  if (fs.realpathSync(path.dirname(resolved)) !== parent) {
    fail('candidate output parent changed while preparing it');
  }
  return resolved;
}

function stageCommit(root, commit, destination) {
  const archive = path.join(path.dirname(destination), `source-${commit}.tar`);
  try {
    run('git', ['-C', root, 'archive', '--format=tar', `--output=${archive}`, commit]);
    fs.mkdirSync(destination, { mode: 0o700 });
    run('tar', ['-xf', archive, '-C', destination]);
  } finally {
    fs.rmSync(archive, { force: true });
  }
}

function buildCandidate(options = {}) {
  const root = path.resolve(options.root || ROOT);
  const output = prepareOutput(options.output);
  const diagnostic = options.diagnostic === true;
  const buildEnv = options.env || process.env;
  const source = cleanCommit(root);
  cleanTree.verifyCleanTree(root);
  if (!diagnostic) release.verifyNoReleaseBlockers(root);
  const version = release.verifyVersions(root);
  release.verifyNpmMetadata(root);
  release.verifyDependencies(root);
  release.verifyVisualPolicyRollout(root, {
    env: withoutReleaseSourceBridge(buildEnv),
  });
  const manifest = evidence.loadManifest(options.manifest);
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-candidate-'));
  try {
    const staged = path.join(scratch, 'source');
    stageCommit(root, source.commit, staged);
    const python = options.python || process.env.CUSTBACK_RELEASE_PYTHON || 'python3';
    run(python, ['-m', 'build', '--sdist', '--wheel', '--outdir', output], { cwd: staged });
    const npmArguments = ['pack', '--pack-destination', output];
    if (diagnostic) npmArguments.push('--ignore-scripts');
    run('npm', npmArguments, {
      cwd: staged,
      env: trustedPrepackEnvironment(root, source, buildEnv),
    });
    const artifacts = artifactRecords(output, manifest);
    const candidate = {
      schema_version: 1,
      authorization: diagnostic ? 'diagnostic' : 'release',
      manifest_id: manifest.manifest_id,
      manifest_sha256: evidence.manifestDigest(manifest),
      generated_at: new Date().toISOString(),
      source: { commit: source.commit, tree: source.tree, version },
      provenance: githubProvenance(source.commit, buildEnv),
      artifacts,
    };
    const candidatePath = path.join(output, 'candidate-manifest.json');
    fs.writeFileSync(candidatePath, `${JSON.stringify(candidate, null, 2)}\n`, {
      encoding: 'utf8', mode: 0o600, flag: 'wx',
    });
    return candidate;
  } catch (err) {
    for (const name of fs.readdirSync(output)) fs.rmSync(path.join(output, name), { force: true });
    throw err;
  } finally {
    fs.rmSync(scratch, { recursive: true, force: true });
  }
}

function main(argv = process.argv.slice(2)) {
  try {
    let diagnostic = false;
    let output;
    for (let index = 0; index < argv.length; index += 1) {
      if (argv[index] === '--diagnostic') diagnostic = true;
      else if (argv[index] === '--output' && index + 1 < argv.length) output = argv[index += 1];
      else fail('usage: build-candidate.js --output DIRECTORY [--diagnostic]');
    }
    if (!output) fail('usage: build-candidate.js --output DIRECTORY [--diagnostic]');
    const candidate = buildCandidate({ output, diagnostic });
    process.stdout.write(`${JSON.stringify(candidate)}\n`);
    return 0;
  } catch (err) {
    process.stderr.write(`[custback candidate] ${err.message}\n`);
    return 1;
  }
}

module.exports = {
  artifactRecords,
  buildCandidate,
  cleanCommit,
  githubProvenance,
  main,
  prepareOutput,
  sha256File,
  trustedPrepackEnvironment,
  withoutReleaseSourceBridge,
};

if (require.main === module) process.exit(main());
