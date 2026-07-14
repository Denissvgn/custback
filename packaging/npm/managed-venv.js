#!/usr/bin/env node
/** Safe lifecycle helpers for custback's private Python environment. */

'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OWNER = 'custback';
const MARKER_SCHEMA = 1;
const GENERATIONS_MARKER = '.custback-generations.json';
const VENV_MARKER = '.custback-venv.json';
const INSTALL_STAMP = '.custback-install.json';
const LEGACY_STAMP = '.custback-installed';
const LOCK_DIR = '.install-lock';
const LOCK_OWNER_PREFIX = '.install-lock-owner-';
const LOCK_BREAK_PREFIX = '.install-lock-break-';
const LOCK_QUARANTINE_PREFIX = '.install-lock-quarantine-';
const MIGRATION_JOURNAL = '.migration-pending.json';

function isWithin(parent, candidate) {
  const rel = path.relative(parent, candidate);
  return rel === '' || (!rel.startsWith(`..${path.sep}`) && rel !== '..' && !path.isAbsolute(rel));
}

function samePath(a, b) {
  return path.resolve(a) === path.resolve(b);
}

function symlinkPointsTo(link, destination) {
  const st = lstatOrNull(link);
  return Boolean(
    st && st.isSymbolicLink() &&
    samePath(path.resolve(path.dirname(link), fs.readlinkSync(link)), destination)
  );
}

function resolveThroughExistingParent(input) {
  const absolute = path.resolve(input);
  const missing = [];
  let current = path.dirname(absolute);
  while (!fs.existsSync(current)) {
    missing.unshift(path.basename(current));
    const next = path.dirname(current);
    if (next === current) break;
    current = next;
  }
  const realParent = fs.realpathSync(current);
  return path.join(realParent, ...missing, path.basename(absolute));
}

function resolveProtectedPath(input) {
  const absolute = path.resolve(input);
  try {
    return fs.realpathSync(absolute);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
    return resolveThroughExistingParent(absolute);
  }
}

function generationsRootFor(target) {
  const base = path.basename(target).replace(/^\.+/, '') || 'venv';
  return path.join(path.dirname(target), `.${base}.custback-generations`);
}

function assertSafeTarget(input, options = {}) {
  if (typeof input !== 'string' || input.trim() === '') {
    throw new Error('CUSTBACK_VENV must name a dedicated non-empty path');
  }
  const target = resolveThroughExistingParent(input);
  const pkgRoot = resolveProtectedPath(options.pkgRoot || path.join(__dirname, '..', '..'));
  const home = resolveProtectedPath(options.home || os.homedir());
  const cwd = resolveProtectedPath(options.cwd || process.cwd());
  const tmpRoot = resolveProtectedPath(options.tmpRoot || os.tmpdir());
  const defaultTarget = path.join(pkgRoot, '.venv');
  const root = path.parse(target).root;
  const forbiddenExact = new Set([root, home, pkgRoot, cwd, tmpRoot]);
  if (forbiddenExact.has(target)) {
    throw new Error(`refusing unsafe CUSTBACK_VENV target: ${target}`);
  }
  if (isWithin(target, pkgRoot) || isWithin(target, home) ||
      isWithin(target, cwd) || isWithin(target, tmpRoot) ||
      (isWithin(pkgRoot, target) && !samePath(target, defaultTarget))) {
    throw new Error(`refusing CUSTBACK_VENV target that contains a protected directory: ${target}`);
  }
  return target;
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`invalid custback metadata ${file}: ${err.message}`);
  }
}

function writeJson(file, value) {
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.tmp-${process.pid}-${crypto.randomBytes(8).toString('hex')}`,
  );
  let created = false;
  try {
    fs.writeFileSync(temporary, `${JSON.stringify(value, null, 2)}\n`, {
      mode: 0o600,
      flag: 'wx',
    });
    created = true;
    fs.renameSync(temporary, file);
    created = false;
  } finally {
    if (created) {
      try {
        fs.unlinkSync(temporary);
      } catch (err) {
        if (err.code !== 'ENOENT') throw err;
      }
    }
  }
}

function lstatOrNull(file) {
  try {
    return fs.lstatSync(file);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

function validateGenerationsRoot(root, target) {
  const st = fs.lstatSync(root);
  if (!st.isDirectory() || st.isSymbolicLink()) {
    throw new Error(`custback generations path is not an owned directory: ${root}`);
  }
  const marker = readJson(path.join(root, GENERATIONS_MARKER));
  if (marker.owner !== OWNER || marker.schema !== MARKER_SCHEMA ||
      !samePath(marker.logicalTarget, target)) {
    throw new Error(`custback generations marker does not match ${target}`);
  }
  return root;
}

function publishGenerationsRoot(root, target, options = {}) {
  const temporary = path.join(
    path.dirname(root),
    `.${path.basename(root)}.init-${process.pid}-${crypto.randomUUID()}`,
  );
  let created = false;
  try {
    fs.mkdirSync(temporary, { recursive: false, mode: 0o700 });
    created = true;
    writeJson(path.join(temporary, GENERATIONS_MARKER), {
      owner: OWNER,
      schema: MARKER_SCHEMA,
      logicalTarget: path.resolve(target),
    });
    if (options.replaceEmpty) {
      // rmdir is deliberately non-recursive: it succeeds only while the old
      // crash remnant is still an empty real directory. The canonical name is
      // then absent until the fully initialized replacement is published.
      try {
        fs.rmdirSync(root);
      } catch (err) {
        if (err.code !== 'ENOENT') throw err;
      }
    }
    fs.renameSync(temporary, root);
    created = false;
    return root;
  } catch (err) {
    if (!['EEXIST', 'ENOTEMPTY'].includes(err.code)) throw err;
    return validateGenerationsRoot(root, target);
  } finally {
    if (created) fs.rmSync(temporary, { recursive: true, force: false });
  }
}

function ensureGenerationsRoot(target) {
  const root = generationsRootFor(target);
  const existingRoot = lstatOrNull(root);
  if (existingRoot) {
    if (!existingRoot.isDirectory() || existingRoot.isSymbolicLink()) {
      throw new Error(`custback generations path is not an owned directory: ${root}`);
    }
    const markerPath = path.join(root, GENERATIONS_MARKER);
    // Another first-time installer may have won mkdir() but not yet flushed
    // its marker. Give that tiny initialization window time to finish.
    for (let attempt = 0; attempt < 20 && !fs.existsSync(markerPath); attempt += 1) {
      sleepSync(50);
    }
    if (fs.existsSync(markerPath)) return validateGenerationsRoot(root, target);
    // An older installer could crash after mkdir() but before writing its
    // marker. Only an empty directory is safe to replace atomically.
    if (fs.readdirSync(root).length === 0) {
      return publishGenerationsRoot(root, target, { replaceEmpty: true });
    }
    throw new Error(`custback generations path has no ownership marker: ${root}`);
  }
  return publishGenerationsRoot(root, target);
}

function validVenvShape(directory) {
  return fs.existsSync(path.join(directory, 'pyvenv.cfg')) &&
    fs.existsSync(path.join(directory, 'bin', 'python')) &&
    fs.existsSync(path.join(directory, 'bin', 'custback'));
}

function validateGeneration(directory, target, expectedRoot = generationsRootFor(target)) {
  const resolved = path.resolve(directory);
  if (path.dirname(resolved) !== path.resolve(expectedRoot)) {
    throw new Error(`generation is outside the managed root: ${resolved}`);
  }
  const st = fs.lstatSync(resolved);
  if (!st.isDirectory() || st.isSymbolicLink()) {
    throw new Error(`managed generation is not a real directory: ${resolved}`);
  }
  const marker = readJson(path.join(resolved, VENV_MARKER));
  if (marker.owner !== OWNER || marker.schema !== MARKER_SCHEMA ||
      !samePath(marker.logicalTarget, target) || marker.generationId !== path.basename(resolved)) {
    throw new Error(`generation marker does not match ${resolved}`);
  }
  if (!validVenvShape(resolved)) {
    throw new Error(`managed generation is not a complete custback venv: ${resolved}`);
  }
  return marker;
}

function inspectTarget(target) {
  const resolvedTarget = path.resolve(target);
  const generationRoot = generationsRootFor(resolvedTarget);
  const st = lstatOrNull(resolvedTarget);
  if (!st) {
    return { kind: 'absent', target: resolvedTarget, generationRoot };
  }
  if (st.isSymbolicLink()) {
    if (!fs.existsSync(generationRoot)) {
      throw new Error(`CUSTBACK_VENV points through an unowned symlink: ${resolvedTarget}`);
    }
    const actual = fs.realpathSync(resolvedTarget);
    validateGeneration(actual, resolvedTarget, generationRoot);
    return { kind: 'symlink', target: resolvedTarget, generationRoot, activeGeneration: actual };
  }
  if (!st.isDirectory()) {
    throw new Error(`CUSTBACK_VENV exists and is not a directory: ${resolvedTarget}`);
  }
  if (!validVenvShape(resolvedTarget)) {
    throw new Error(
      `refusing to replace non-custback directory ${resolvedTarget}; choose an empty/nonexistent path`
    );
  }
  const markerPath = path.join(resolvedTarget, VENV_MARKER);
  const legacyPath = path.join(resolvedTarget, LEGACY_STAMP);
  if (fs.existsSync(markerPath)) {
    const marker = readJson(markerPath);
    if (marker.owner !== OWNER || marker.schema !== MARKER_SCHEMA ||
        !samePath(marker.logicalTarget, resolvedTarget)) {
      throw new Error(`custback venv marker does not match ${resolvedTarget}`);
    }
    return { kind: 'direct', target: resolvedTarget, generationRoot, legacy: false };
  }
  if (fs.existsSync(legacyPath) && fs.statSync(legacyPath).isFile()) {
    const legacyStamp = fs.readFileSync(legacyPath, 'utf8').trim();
    if (/^\d+\.\d+\.\d+\s+\S+\s+3\.\d+(?:\.\d+)?\s+\[[^\]]*\]$/.test(legacyStamp)) {
      return { kind: 'direct', target: resolvedTarget, generationRoot, legacy: true };
    }
    throw new Error(`legacy custback ownership stamp is invalid: ${legacyPath}`);
  }
  throw new Error(
    `refusing to replace unowned venv ${resolvedTarget}; move it aside or choose another CUSTBACK_VENV`
  );
}

function markGeneration(generation, target) {
  const id = path.basename(generation);
  writeJson(path.join(generation, VENV_MARKER), {
    owner: OWNER,
    schema: MARKER_SCHEMA,
    generationId: id,
    logicalTarget: path.resolve(target),
  });
}

function createGeneration(generationRoot) {
  return fs.mkdtempSync(path.join(generationRoot, 'gen-'));
}

function removeCreatedGeneration(generation, generationRoot) {
  const resolved = path.resolve(generation);
  if (path.dirname(resolved) !== path.resolve(generationRoot) ||
      !path.basename(resolved).startsWith('gen-')) {
    throw new Error(`refusing to clean unexpected generation path: ${resolved}`);
  }
  if (fs.existsSync(resolved)) {
    const st = fs.lstatSync(resolved);
    if (!st.isDirectory() || st.isSymbolicLink()) {
      throw new Error(`refusing to recursively clean non-directory generation: ${resolved}`);
    }
    fs.rmSync(resolved, { recursive: true, force: false });
  }
}

function makeLink(linkPath, generation) {
  fs.symlinkSync(generation, linkPath, 'dir');
}

function replaceWithLink(target, generation) {
  const temporary = path.join(
    path.dirname(target),
    `.${path.basename(target)}.custback-link-${process.pid}-${Math.random().toString(16).slice(2)}`
  );
  let created = false;
  try {
    makeLink(temporary, generation);
    created = true;
    fs.renameSync(temporary, target);
  } finally {
    if (created) {
      try {
        fs.lstatSync(temporary);
        fs.unlinkSync(temporary);
      } catch (err) {
        if (err.code !== 'ENOENT') throw err;
      }
    }
  }
}

function promoteGeneration({ target, generation, inspection, validateActive }) {
  validateGeneration(generation, target, inspection.generationRoot);
  let previousGeneration = inspection.activeGeneration || null;
  let movedDirect = null;
  let journalPath = null;

  if (inspection.kind === 'direct') {
    const legacyId = `legacy-${Date.now()}-${process.pid}`;
    movedDirect = path.join(inspection.generationRoot, legacyId);
    writeJson(path.join(target, VENV_MARKER), {
      owner: OWNER,
      schema: MARKER_SCHEMA,
      generationId: legacyId,
      logicalTarget: path.resolve(target),
    });
    journalPath = path.join(inspection.generationRoot, MIGRATION_JOURNAL);
    writeJson(journalPath, {
      owner: OWNER,
      schema: MARKER_SCHEMA,
      phase: 'prepared',
      logicalTarget: path.resolve(target),
      previousGeneration: movedDirect,
      newGeneration: path.resolve(generation),
    });
    try {
      fs.renameSync(target, movedDirect);
      previousGeneration = movedDirect;
      replaceWithLink(target, generation);
    } catch (err) {
      if (!lstatOrNull(target) && lstatOrNull(movedDirect)) fs.renameSync(movedDirect, target);
      try { fs.rmSync(journalPath, { force: true }); } catch {}
      throw err;
    }
  } else {
    replaceWithLink(target, generation);
  }

  try {
    validateActive(target);
  } catch (err) {
    if (inspection.kind === 'symlink') {
      replaceWithLink(target, inspection.activeGeneration);
    } else if (inspection.kind === 'direct') {
      if (symlinkPointsTo(target, generation)) fs.unlinkSync(target);
      if (!lstatOrNull(target) && lstatOrNull(movedDirect)) {
        fs.renameSync(movedDirect, target);
      }
      try { fs.rmSync(journalPath, { force: true }); } catch {}
    } else if (inspection.kind === 'absent') {
      fs.unlinkSync(target);
    }
    throw err;
  }
  if (inspection.kind === 'direct') {
    try {
      const journal = readJson(journalPath);
      writeJson(journalPath, { ...journal, phase: 'committed' });
    } catch (err) {
      // Without a durable commit phase, recovery must treat the migration as
      // unvalidated. Restore the previous direct venv before reporting failure.
      if (symlinkPointsTo(target, generation)) fs.unlinkSync(target);
      if (!lstatOrNull(target) && lstatOrNull(movedDirect)) {
        fs.renameSync(movedDirect, target);
      }
      throw err;
    }
    // Cleanup after a durable commit is non-critical. Recovery verifies the
    // active generation and removes a journal left by a transient failure.
    try { fs.rmSync(journalPath, { force: true }); } catch {}
  }
  return { previousGeneration };
}

function readMigrationState(target, generationRoot) {
  const journalPath = path.join(generationRoot, MIGRATION_JOURNAL);
  if (!lstatOrNull(journalPath)) return null;
  const journal = readJson(journalPath);
  const previous = path.resolve(journal.previousGeneration || '');
  const next = path.resolve(journal.newGeneration || '');
  const phase = journal.phase || 'prepared';
  if (journal.owner !== OWNER || journal.schema !== MARKER_SCHEMA ||
      !['prepared', 'committed'].includes(phase) ||
      !samePath(journal.logicalTarget, target) || path.dirname(previous) !== path.resolve(generationRoot) ||
      path.dirname(next) !== path.resolve(generationRoot) || !path.basename(previous).startsWith('legacy-') ||
      !path.basename(next).startsWith('gen-')) {
    throw new Error(`invalid interrupted-migration journal: ${journalPath}`);
  }
  return { journal, journalPath, next, phase, previous };
}

function prepareInstallTarget(target) {
  try {
    inspectTarget(target);
    return ensureGenerationsRoot(target);
  } catch (err) {
    const targetStat = lstatOrNull(target);
    if (err.code !== 'ENOENT' || !targetStat || !targetStat.isSymbolicLink()) throw err;

    const generationRoot = generationsRootFor(target);
    const rootStat = lstatOrNull(generationRoot);
    if (!rootStat || !rootStat.isDirectory() || rootStat.isSymbolicLink()) {
      throw new Error(`CUSTBACK_VENV points through an unowned symlink: ${target}`);
    }
    ensureGenerationsRoot(target);
    const state = readMigrationState(target, generationRoot);
    if (!state || !symlinkPointsTo(target, state.next)) {
      throw new Error(`CUSTBACK_VENV is a dangling symlink without owned recovery state: ${target}`);
    }
    return generationRoot;
  }
}

function recoverInterruptedPromotion(target, generationRoot = generationsRootFor(target)) {
  const state = readMigrationState(target, generationRoot);
  if (!state) return false;
  const { journalPath, next, phase, previous } = state;

  const targetStat = lstatOrNull(target);
  const previousStat = lstatOrNull(previous);
  const targetPointsToNext = symlinkPointsTo(target, next);

  if (phase === 'committed') {
    const nextStat = lstatOrNull(next);
    if (nextStat) {
      validateGeneration(next, target, generationRoot);
      if (!targetStat) {
        makeLink(target, next);
      } else if (!targetPointsToNext) {
        throw new Error(`committed migration target changed unexpectedly: ${target}`);
      }
      fs.rmSync(journalPath, { force: true });
      return true;
    }
    // Repair the dangling-link state produced by older installers by falling
    // back to the retained, validated direct generation.
    if (!previousStat) {
      throw new Error('cannot recover committed migration; both generations are missing');
    }
    validateGeneration(previous, target, generationRoot);
    if (targetStat) {
      if (!targetPointsToNext) {
        throw new Error(`cannot recover committed migration over unexpected target: ${target}`);
      }
      fs.unlinkSync(target);
    }
    fs.renameSync(previous, target);
    fs.rmSync(journalPath, { force: true });
    return true;
  }

  if (targetStat && targetStat.isDirectory() && !targetStat.isSymbolicLink() && !previousStat) {
    // The journal was written but the legacy directory was never moved.
    fs.rmSync(journalPath, { force: true });
    return true;
  }
  if (!previousStat) {
    throw new Error(`cannot recover interrupted migration; previous venv is missing: ${previous}`);
  }
  validateGeneration(previous, target, generationRoot);
  if (targetStat) {
    if (!targetPointsToNext) {
      throw new Error(`cannot recover interrupted migration over unexpected target: ${target}`);
    }
    fs.unlinkSync(target);
  }
  fs.renameSync(previous, target);
  fs.rmSync(journalPath, { force: true });
  return true;
}

function generationIsReferenced(target, generation, generationRoot) {
  const expected = path.resolve(generation);
  try {
    if (symlinkPointsTo(target, expected)) return true;
    const journalPath = path.join(generationRoot, MIGRATION_JOURNAL);
    if (lstatOrNull(journalPath)) {
      const journal = readJson(journalPath);
      if (samePath(journal.newGeneration || '', expected)) return true;
    }
    return false;
  } catch {
    // Unreadable recovery state is never authority for recursive deletion.
    return true;
  }
}

function removeOwnedGeneration(generation, target, generationRoot) {
  validateGeneration(generation, target, generationRoot);
  fs.rmSync(generation, { recursive: true, force: false });
}

function cleanupGenerations(generationRoot, target, keep, retainPrevious = 1) {
  const candidates = [];
  for (const name of fs.readdirSync(generationRoot)) {
    if (!name.startsWith('gen-') && !name.startsWith('legacy-')) continue;
    const candidate = path.join(generationRoot, name);
    if (keep.has(path.resolve(candidate))) continue;
    try {
      validateGeneration(candidate, target, generationRoot);
      candidates.push({ path: candidate, mtime: fs.statSync(candidate).mtimeMs });
    } catch {
      // Foreign/corrupt entries are deliberately left untouched.
    }
  }
  candidates.sort((a, b) => b.mtime - a.mtime);
  for (const candidate of candidates.slice(retainPrevious)) {
    removeOwnedGeneration(candidate.path, target, generationRoot);
  }
}

function pidIsAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return err.code === 'EPERM';
  }
}

function sleepSync(milliseconds) {
  const cell = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(cell, 0, 0, milliseconds);
}

function sameInode(left, right) {
  try {
    const a = fs.statSync(left);
    const b = fs.statSync(right);
    return a.dev === b.dev && a.ino === b.ino;
  } catch {
    return false;
  }
}

function unlinkIfExists(file) {
  try {
    fs.unlinkSync(file);
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
}

function createLockRecord(generationRoot, prefix, { includePath = false } = {}) {
  const token = crypto.randomUUID();
  const file = path.join(generationRoot, `${prefix}${token}.json`);
  const record = {
    owner: OWNER,
    schema: MARKER_SCHEMA,
    token,
    pid: process.pid,
    createdAt: Date.now(),
  };
  if (includePath) record.recordPath = file;
  let descriptor;
  try {
    descriptor = fs.openSync(file, 'wx', 0o600);
    fs.writeFileSync(descriptor, `${JSON.stringify(record)}\n`);
    fs.fsyncSync(descriptor);
  } catch (err) {
    if (descriptor !== undefined) {
      try { fs.closeSync(descriptor); } catch {}
      descriptor = undefined;
    }
    unlinkIfExists(file);
    throw err;
  }
  fs.closeSync(descriptor);
  return { file, record };
}

function publishBreakBarrier(generationRoot) {
  const privateRecord = createLockRecord(generationRoot, LOCK_OWNER_PREFIX);
  const barrier = path.join(
    generationRoot,
    `${LOCK_BREAK_PREFIX}${privateRecord.record.token}.json`,
  );
  try {
    fs.linkSync(privateRecord.file, barrier);
  } finally {
    unlinkIfExists(privateRecord.file);
  }
  return barrier;
}

function activeBreakBarriers(generationRoot, staleMs) {
  const now = Date.now();
  const active = [];
  for (const name of fs.readdirSync(generationRoot)) {
    if (!name.startsWith(LOCK_BREAK_PREFIX)) continue;
    const barrier = path.join(generationRoot, name);
    let stat;
    try {
      stat = fs.lstatSync(barrier);
    } catch (err) {
      if (err.code === 'ENOENT') continue;
      throw err;
    }
    let record = null;
    try {
      record = readJson(barrier);
    } catch {}
    const valid = stat.isFile() && !stat.isSymbolicLink() && record &&
      record.owner === OWNER && record.schema === MARKER_SCHEMA &&
      typeof record.token === 'string' && record.token.length > 0 &&
      Number.isInteger(record.pid) && Number.isFinite(record.createdAt);
    // Filesystem mtimes may round a freshly-created inode a fraction into the
    // future relative to Date.now(). Treat that as age zero so a zero stale
    // threshold remains deterministic.
    const age = Math.max(0, now - Number(valid ? record.createdAt : stat.mtimeMs));
    if ((valid && !pidIsAlive(record.pid)) || (!valid && age >= staleMs)) {
      unlinkIfExists(barrier);
      continue;
    }
    active.push(barrier);
  }
  return active;
}

function inspectInstallLock(lock, generationRoot, staleMs) {
  const stat = lstatOrNull(lock);
  if (!stat) return null;
  const now = Date.now();
  if (stat.isDirectory() && !stat.isSymbolicLink()) {
    const entries = fs.readdirSync(lock);
    if (entries.length === 0) {
      return {
        kind: 'anonymous-directory',
        owned: true,
        stale: Math.max(0, now - stat.mtimeMs) >= staleMs,
        stat,
      };
    }
    if (entries.length !== 1 || entries[0] !== 'owner.json') {
      return { kind: 'foreign', owned: false, stale: false, stat };
    }
    const ownerPath = path.join(lock, 'owner.json');
    const ownerStat = lstatOrNull(ownerPath);
    if (!ownerStat || !ownerStat.isFile() || ownerStat.isSymbolicLink()) {
      return { kind: 'foreign', owned: false, stale: false, stat };
    }
    let owner = null;
    try { owner = readJson(ownerPath); } catch {}
    const valid = owner && owner.owner === OWNER && Number.isInteger(owner.pid) &&
      Number.isFinite(owner.createdAt);
    if (!valid) return { kind: 'foreign', owned: false, stale: false, stat };
    const age = Math.max(0, now - Number(valid ? owner.createdAt : stat.mtimeMs));
    return {
      kind: 'legacy-directory',
      owned: true,
      stale: age >= staleMs && (!valid || !pidIsAlive(owner.pid)),
      owner,
      ownerStat,
      stat,
    };
  }
  if (!stat.isFile() || stat.isSymbolicLink()) {
    return { kind: 'foreign', owned: false, stale: false, stat };
  }
  let owner;
  try {
    owner = readJson(lock);
  } catch {
    return { kind: 'corrupt-file', owned: false, stale: false, stat };
  }
  const recordPath = path.resolve(owner.recordPath || '');
  const validPath = path.dirname(recordPath) === path.resolve(generationRoot) &&
    path.basename(recordPath).startsWith(LOCK_OWNER_PREFIX);
  const valid = owner.owner === OWNER && owner.schema === MARKER_SCHEMA &&
    typeof owner.token === 'string' && owner.token.length > 0 &&
    Number.isInteger(owner.pid) && Number.isFinite(owner.createdAt) && validPath;
  if (!valid) return { kind: 'corrupt-file', owned: false, stale: false, stat };
  return {
    kind: 'owner-file',
    owned: true,
    stale: !pidIsAlive(owner.pid) && (now - owner.createdAt) >= staleMs,
    owner,
    recordPath,
    stat,
  };
}

function reclaimStaleLock(generationRoot, lock, staleMs, settleMs) {
  const barrier = publishBreakBarrier(generationRoot);
  const quarantine = path.join(
    generationRoot,
    `${LOCK_QUARANTINE_PREFIX}${crypto.randomUUID()}`,
  );
  try {
    sleepSync(settleMs);
    const current = inspectInstallLock(lock, generationRoot, staleMs);
    if (!current || !current.owned || !current.stale) return false;
    try {
      fs.renameSync(lock, quarantine);
    } catch (err) {
      if (err.code === 'ENOENT') return true;
      throw err;
    }
    const moved = fs.lstatSync(quarantine);
    if (moved.dev !== current.stat.dev || moved.ino !== current.stat.ino) {
      if (!lstatOrNull(lock)) fs.renameSync(quarantine, lock);
      return false;
    }
    if (current.kind === 'owner-file') {
      if (sameInode(current.recordPath, quarantine)) unlinkIfExists(current.recordPath);
      unlinkIfExists(quarantine);
    } else if (current.kind === 'legacy-directory') {
      const entries = fs.readdirSync(quarantine);
      if (entries.length !== 1 || entries[0] !== 'owner.json') {
        if (!lstatOrNull(lock)) fs.renameSync(quarantine, lock);
        throw new Error(`refusing to clean changed legacy install lock: ${quarantine}`);
      }
      const ownerPath = path.join(quarantine, 'owner.json');
      const ownerStat = fs.lstatSync(ownerPath);
      if (!ownerStat.isFile() || ownerStat.isSymbolicLink()) {
        if (!lstatOrNull(lock)) fs.renameSync(quarantine, lock);
        throw new Error(`refusing to clean invalid legacy install lock: ${quarantine}`);
      }
      let owner;
      try { owner = readJson(ownerPath); } catch {}
      if (ownerStat.dev !== current.ownerStat.dev || ownerStat.ino !== current.ownerStat.ino ||
          !owner || owner.owner !== OWNER || owner.pid !== current.owner.pid ||
          owner.createdAt !== current.owner.createdAt) {
        if (!lstatOrNull(lock)) fs.renameSync(quarantine, lock);
        throw new Error(`refusing to clean changed legacy install lock: ${quarantine}`);
      }
      unlinkIfExists(ownerPath);
      fs.rmdirSync(quarantine);
    } else {
      fs.rmdirSync(quarantine);
    }
    return true;
  } finally {
    unlinkIfExists(barrier);
  }
}

function withInstallLock(generationRoot, callback, options = {}) {
  const lock = path.join(generationRoot, LOCK_DIR);
  const deadline = Date.now() + (options.waitMs ?? 5000);
  const staleMs = options.staleMs ?? 15 * 60 * 1000;
  const settleMs = options.settleMs ?? 25;
  const privateOwner = createLockRecord(generationRoot, LOCK_OWNER_PREFIX, {
    includePath: true,
  });
  let acquired = false;
  let published = false;
  try {
    while (true) {
      const ownsCanonical = sameInode(lock, privateOwner.file);
      const barriers = activeBreakBarriers(generationRoot, staleMs);
      if (barriers.length) {
        if (!published && Date.now() >= deadline) {
          throw new Error(`another custback install/rebuild holds ${lock}`);
        }
        // A breaker may have quarantined the canonical name and later restore it.
        // Retain our private hard link until the barrier is withdrawn so that a
        // restored lock always has a verifiable owner record.
        sleepSync(ownsCanonical ? 25 : Math.min(
          100,
          Math.max(1, deadline - Date.now()),
        ));
        continue;
      }
      if (ownsCanonical) {
        acquired = true;
        break;
      }

      try {
        fs.linkSync(privateOwner.file, lock);
        published = true;
      } catch (err) {
        if (err.code !== 'EEXIST') {
          throw new Error(`cannot atomically publish custback install lock: ${err.message}`);
        }
        const current = inspectInstallLock(lock, generationRoot, staleMs);
        if (current && current.owned && current.stale) {
          reclaimStaleLock(generationRoot, lock, staleMs, settleMs);
          continue;
        }
        if (current && !current.owned) {
          throw new Error(`custback install lock is not owned metadata: ${lock}`);
        }
        if (Date.now() >= deadline) {
          throw new Error(`another custback install/rebuild holds ${lock}`);
        }
        sleepSync(100);
        continue;
      }

      sleepSync(settleMs);
    }
    return callback();
  } finally {
    if (acquired && !sameInode(lock, privateOwner.file)) {
      unlinkIfExists(privateOwner.file);
      throw new Error(`custback install lock ownership changed unexpectedly: ${lock}`);
    }
    if (acquired) unlinkIfExists(lock);
    unlinkIfExists(privateOwner.file);
  }
}

module.exports = {
  GENERATIONS_MARKER,
  INSTALL_STAMP,
  LEGACY_STAMP,
  MARKER_SCHEMA,
  MIGRATION_JOURNAL,
  OWNER,
  VENV_MARKER,
  assertSafeTarget,
  cleanupGenerations,
  createGeneration,
  ensureGenerationsRoot,
  generationIsReferenced,
  generationsRootFor,
  inspectTarget,
  isWithin,
  markGeneration,
  prepareInstallTarget,
  promoteGeneration,
  readJson,
  recoverInterruptedPromotion,
  removeCreatedGeneration,
  validateGeneration,
  withInstallLock,
  writeJson,
};
