'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const clean = require('../../../scripts/release/verify-clean-tree');

test('recursive clean-tree scan rejects caches, profiles, and release artifacts', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-clean-scan-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, 'nested', '.ruff_cache'), { recursive: true });
  fs.mkdirSync(path.join(root, 'src', 'custback.egg-info'), { recursive: true });
  fs.writeFileSync(path.join(root, 'nested', 'onnxruntime_profile__x.json'), '{}');
  fs.writeFileSync(path.join(root, 'candidate.whl'), 'artifact');
  assert.deepEqual(clean.forbiddenGeneratedPaths(root), [
    'candidate.whl',
    path.join('nested', '.ruff_cache'),
    path.join('nested', 'onnxruntime_profile__x.json'),
    path.join('src', 'custback.egg-info'),
  ]);
});

test('recursive clean-tree scan ignores Git internals but not nested Python caches', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-clean-git-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, '.git', 'objects'), { recursive: true });
  fs.mkdirSync(path.join(root, 'src', '__pycache__'), { recursive: true });
  fs.writeFileSync(path.join(root, '.git', 'objects', 'ignored.whl'), 'git object');
  assert.deepEqual(clean.forbiddenGeneratedPaths(root), [path.join('src', '__pycache__')]);
});
