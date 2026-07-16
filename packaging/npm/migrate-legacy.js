#!/usr/bin/env node
/** Move a pre-0.4 package-local venv out of npm's replaceable package tree. */

'use strict';

const fs = require('fs');
const path = require('path');

const installer = require('./install');
const managed = require('./managed-venv');

const LEGACY_DIRECTORY = '.venv';
const JOURNAL_SUFFIX = '.custback-preupgrade.json';
const MAX_REWRITABLE_SCRIPT_BYTES = 4 * 1024 * 1024;
const REFERENCE_INSTALL_STAMP_SCHEMAS = new Set([2, 3]);

function fail(message) {
  throw new Error(message);
}

function lstatOrNull(file) {
  try {
    return fs.lstatSync(file);
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

function samePath(left, right) {
  return path.resolve(left) === path.resolve(right);
}

function isWithin(parent, candidate) {
  const relative = path.relative(path.resolve(parent), path.resolve(candidate));
  return relative === '' || (
    !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative)
  );
}

function requireRegularFile(file, label) {
  const metadata = fs.lstatSync(file);
  if (!metadata.isFile() || metadata.isSymbolicLink()) {
    fail(`${label} must be a regular non-symlink file: ${file}`);
  }
  return metadata;
}

function parseLegacyExtras(target) {
  const stampPath = path.join(target, managed.LEGACY_STAMP);
  const stamp = fs.readFileSync(stampPath, 'utf8').trim();
  const match = stamp.match(/\[([^\]]*)\]$/);
  if (!match) fail(`legacy custback ownership stamp has no extras list: ${stampPath}`);
  return installer.parseExtras(match[1].replace(/\s+/g, ','));
}

function parseReferenceInstallExtras(generation) {
  const stampPath = path.join(generation, managed.INSTALL_STAMP);
  requireRegularFile(stampPath, 'reference install stamp');
  const stamp = managed.readJson(stampPath);
  if (!REFERENCE_INSTALL_STAMP_SCHEMAS.has(stamp.schema) ||
      !['0.3.0', '0.4.0'].includes(stamp.packageVersion) ||
      !Array.isArray(stamp.requestedExtras)) {
    fail(`package-local install stamp is not a reviewed schema 2/3 record: ${stampPath}`);
  }
  return installer.parseExtras(stamp.requestedExtras.join(','));
}

function replacePathInScript(file, oldRoot, newRoot) {
  const metadata = fs.lstatSync(file);
  if (!metadata.isFile() || metadata.isSymbolicLink()) return false;
  if (metadata.size > MAX_REWRITABLE_SCRIPT_BYTES) return false;
  const input = fs.readFileSync(file);
  if (input.includes(0)) return false;
  const before = Buffer.from(path.resolve(oldRoot));
  if (!input.includes(before)) return false;
  const output = Buffer.from(input.toString('utf8').split(before.toString()).join(path.resolve(newRoot)));
  const temporary = path.join(
    path.dirname(file),
    `.${path.basename(file)}.custback-migrate-${process.pid}-${Math.random().toString(16).slice(2)}`,
  );
  let created = false;
  try {
    const descriptor = fs.openSync(temporary, 'wx', metadata.mode & 0o777);
    created = true;
    try {
      fs.fchmodSync(descriptor, metadata.mode & 0o777);
      fs.writeFileSync(descriptor, output);
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    fs.renameSync(temporary, file);
    created = false;
    return true;
  } finally {
    if (created) fs.rmSync(temporary, { force: true });
  }
}

function rewriteRelocatedScripts(target, oldRoot) {
  const binaryDirectory = path.join(target, 'bin');
  const binaryMetadata = fs.lstatSync(binaryDirectory);
  if (!binaryMetadata.isDirectory() || binaryMetadata.isSymbolicLink()) {
    fail(`managed venv bin path is not a real directory: ${binaryDirectory}`);
  }
  let changed = 0;
  for (const name of fs.readdirSync(binaryDirectory)) {
    if (replacePathInScript(path.join(binaryDirectory, name), oldRoot, target)) changed += 1;
  }
  return changed;
}

function journalPathFor(target) {
  return `${path.resolve(target)}${JOURNAL_SUFFIX}`;
}

function readJournal(file, source, target) {
  requireRegularFile(file, 'npm pre-upgrade migration journal');
  const journal = managed.readJson(file);
  const layout = journal.layout || 'direct';
  if (journal.owner !== managed.OWNER || journal.schema !== 1 ||
      !samePath(journal.source, source) || !samePath(journal.target, target) ||
      !['direct', 'generational'].includes(layout) ||
      !Array.isArray(journal.requestedExtras)) {
    fail(`invalid npm pre-upgrade migration journal: ${file}`);
  }
  const requestedExtras = installer.parseExtras(journal.requestedExtras.join(','));
  if (layout === 'generational') {
    const expectedSourceRoot = managed.generationsRootFor(source);
    const expectedTargetRoot = managed.generationsRootFor(target);
    if (typeof journal.sourceGenerationRoot !== 'string' ||
        typeof journal.targetGenerationRoot !== 'string' ||
        !samePath(journal.sourceGenerationRoot, expectedSourceRoot) ||
        !samePath(journal.targetGenerationRoot, expectedTargetRoot) ||
        typeof journal.activeGeneration !== 'string' ||
        !/^(?:gen|legacy)-[A-Za-z0-9._-]+$/.test(journal.activeGeneration)) {
      fail(`invalid generational npm pre-upgrade migration journal: ${file}`);
    }
  }
  return { ...journal, layout, requestedExtras };
}

function ensureCompatibilityLink(source, target) {
  const existing = lstatOrNull(source);
  if (existing) {
    if (existing.isSymbolicLink() && samePath(
      path.resolve(path.dirname(source), fs.readlinkSync(source)),
      target,
    )) return;
    fail(`legacy package path changed during migration: ${source}`);
  }
  fs.symlinkSync(target, source, 'dir');
}

function ownedSymlinkDestination(link) {
  const metadata = lstatOrNull(link);
  if (!metadata || !metadata.isSymbolicLink()) return null;
  return path.resolve(path.dirname(link), fs.readlinkSync(link));
}

function ensureGenerationalLink(link, destination, allowedPrevious = null) {
  const metadata = lstatOrNull(link);
  if (!metadata) {
    fs.symlinkSync(destination, link, 'dir');
    return;
  }
  if (!metadata.isSymbolicLink()) {
    fail(`owned generational link changed during migration: ${link}`);
  }
  const current = ownedSymlinkDestination(link);
  if (samePath(current, destination)) return;
  if (!allowedPrevious || !samePath(current, allowedPrevious)) {
    fail(`owned generational link has an unexpected destination: ${link}`);
  }
  fs.unlinkSync(link);
  fs.symlinkSync(destination, link, 'dir');
}

function rewriteLogicalTarget(markerPath, generation, source, target) {
  requireRegularFile(markerPath, 'custback generation marker');
  const marker = managed.readJson(markerPath);
  if (marker.owner !== managed.OWNER || marker.schema !== managed.MARKER_SCHEMA ||
      marker.generationId !== path.basename(generation) ||
      (!samePath(marker.logicalTarget, source) && !samePath(marker.logicalTarget, target))) {
    fail(`generation marker is not owned by the package-local custback target: ${markerPath}`);
  }
  if (!samePath(marker.logicalTarget, target)) {
    managed.writeJson(markerPath, { ...marker, logicalTarget: path.resolve(target) });
  }
}

function relocatedGenerationPath(value, sourceRoot, targetRoot, label) {
  if (typeof value !== 'string' || !path.isAbsolute(value)) {
    fail(`${label} is not an absolute owned generation path`);
  }
  const resolved = path.resolve(value);
  const basename = path.basename(resolved);
  if (!/^(?:gen|legacy)-[A-Za-z0-9._-]+$/.test(basename)) {
    fail(`${label} is not a managed generation name`);
  }
  if (path.dirname(resolved) === path.resolve(targetRoot)) return resolved;
  if (path.dirname(resolved) !== path.resolve(sourceRoot)) {
    fail(`${label} escapes the package-local generation root`);
  }
  return path.join(targetRoot, basename);
}

function rewritePromotionJournal(root, sourceRoot, targetRoot, source, target) {
  const journalPath = path.join(root, managed.MIGRATION_JOURNAL);
  if (!lstatOrNull(journalPath)) return;
  requireRegularFile(journalPath, 'custback promotion journal');
  const journal = managed.readJson(journalPath);
  if (journal.owner !== managed.OWNER || journal.schema !== managed.MARKER_SCHEMA ||
      !['prepared', 'committed', undefined].includes(journal.phase) ||
      (!samePath(journal.logicalTarget, source) && !samePath(journal.logicalTarget, target))) {
    fail(`package-local promotion journal is not owned: ${journalPath}`);
  }
  const previousGeneration = relocatedGenerationPath(
    journal.previousGeneration, sourceRoot, targetRoot, 'promotion previousGeneration',
  );
  const newGeneration = relocatedGenerationPath(
    journal.newGeneration, sourceRoot, targetRoot, 'promotion newGeneration',
  );
  managed.writeJson(journalPath, {
    ...journal,
    logicalTarget: path.resolve(target),
    previousGeneration,
    newGeneration,
  });
}

function rewriteGenerationalMetadata(journal) {
  const source = path.resolve(journal.source);
  const target = path.resolve(journal.target);
  const sourceRoot = path.resolve(journal.sourceGenerationRoot);
  const targetRoot = path.resolve(journal.targetGenerationRoot);
  const rootMetadata = fs.lstatSync(targetRoot);
  if (!rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
    fail(`relocated generations root is not a real directory: ${targetRoot}`);
  }
  const rootMarkerPath = path.join(targetRoot, managed.GENERATIONS_MARKER);
  requireRegularFile(rootMarkerPath, 'custback generations marker');
  const rootMarker = managed.readJson(rootMarkerPath);
  if (rootMarker.owner !== managed.OWNER || rootMarker.schema !== managed.MARKER_SCHEMA ||
      (!samePath(rootMarker.logicalTarget, source) &&
       !samePath(rootMarker.logicalTarget, target))) {
    fail(`package-local generations marker is not owned: ${rootMarkerPath}`);
  }

  for (const name of fs.readdirSync(targetRoot).sort()) {
    if (!name.startsWith('gen-') && !name.startsWith('legacy-')) continue;
    const generation = path.join(targetRoot, name);
    const metadata = fs.lstatSync(generation);
    if (!metadata.isDirectory() || metadata.isSymbolicLink()) {
      fail(`managed generation is not a real directory: ${generation}`);
    }
    rewriteLogicalTarget(
      path.join(generation, managed.VENV_MARKER), generation, source, target,
    );
    rewriteRelocatedScripts(generation, path.join(sourceRoot, name));
  }
  rewritePromotionJournal(targetRoot, sourceRoot, targetRoot, source, target);
  if (!samePath(rootMarker.logicalTarget, target)) {
    managed.writeJson(rootMarkerPath, {
      ...rootMarker,
      logicalTarget: path.resolve(target),
    });
  }
}

function prepareGenerationalJournal(source, target, requestedExtras) {
  const sourceGenerationRoot = managed.generationsRootFor(source);
  const targetGenerationRoot = managed.generationsRootFor(target);
  const sourceRootMetadata = fs.lstatSync(sourceGenerationRoot);
  if (!sourceRootMetadata.isDirectory() || sourceRootMetadata.isSymbolicLink()) {
    fail(`package-local generations root is not a real owned directory: ${sourceGenerationRoot}`);
  }
  if (lstatOrNull(targetGenerationRoot)) {
    fail(`durable custback generations target already exists: ${targetGenerationRoot}`);
  }
  const active = fs.realpathSync(source);
  if (!isWithin(sourceGenerationRoot, active) || path.dirname(active) !== sourceGenerationRoot) {
    fail(`package-local venv points outside its owned generations root: ${source}`);
  }
  managed.validateGeneration(active, source, sourceGenerationRoot);
  if (sourceRootMetadata.dev !== fs.statSync(path.dirname(target)).dev) {
    fail('legacy npm venv migration requires source and prefix target on one filesystem');
  }
  return {
    owner: managed.OWNER,
    schema: 1,
    phase: 'prepared',
    layout: 'generational',
    source: path.resolve(source),
    target: path.resolve(target),
    sourceGenerationRoot: path.resolve(sourceGenerationRoot),
    targetGenerationRoot: path.resolve(targetGenerationRoot),
    activeGeneration: path.basename(active),
    requestedExtras,
  };
}

function migrateGenerationalVenv(journal, journalPath, failpoint) {
  const source = path.resolve(journal.source);
  const target = path.resolve(journal.target);
  const sourceRoot = path.resolve(journal.sourceGenerationRoot);
  const targetRoot = path.resolve(journal.targetGenerationRoot);
  const oldActive = path.join(sourceRoot, journal.activeGeneration);
  const newActive = path.join(targetRoot, journal.activeGeneration);
  const sourceRootMetadata = lstatOrNull(sourceRoot);
  const targetRootMetadata = lstatOrNull(targetRoot);
  if (sourceRootMetadata && targetRootMetadata) {
    fail('both package-local and durable generation roots exist during migration');
  }
  if (sourceRootMetadata) {
    if (!sourceRootMetadata.isDirectory() || sourceRootMetadata.isSymbolicLink()) {
      fail(`package-local generations root changed during migration: ${sourceRoot}`);
    }
    fs.renameSync(sourceRoot, targetRoot);
    failpoint('generation-root-renamed');
  }
  if (!lstatOrNull(targetRoot)) {
    fail(`relocated generations root is missing: ${targetRoot}`);
  }
  rewriteGenerationalMetadata(journal);
  failpoint('metadata-rewritten');
  ensureGenerationalLink(target, newActive);
  failpoint('target-linked');
  installer.writeInstallIntent(target, journal.requestedExtras);
  failpoint('intent-written');
  ensureGenerationalLink(source, target, oldActive);
  failpoint('compatibility-linked');
  managed.inspectTarget(target);
  fs.rmSync(journalPath, { force: false });
  return {
    migrated: true,
    layout: 'generational',
    requestedExtras: journal.requestedExtras,
    source,
    target,
  };
}

function migrateLegacyVenv(options) {
  const packageRoot = path.resolve(options.packageRoot);
  const source = path.join(packageRoot, LEGACY_DIRECTORY);
  const target = managed.assertSafeTarget(
    options.target || managed.defaultTargetForPackage(packageRoot),
    { pkgRoot: packageRoot },
  );
  const journalPath = journalPathFor(target);
  const failpoint = typeof options.failpoint === 'function' ? options.failpoint : () => {};
  let journal = lstatOrNull(journalPath) ? readJournal(journalPath, source, target) : null;

  const sourceMetadata = lstatOrNull(source);
  const targetMetadata = lstatOrNull(target);
  if (!journal) {
    if (!sourceMetadata) {
      if (targetMetadata) {
        managed.inspectTarget(target);
        return { migrated: false, requestedExtras: null, source, target };
      }
      return { migrated: false, requestedExtras: null, source, target };
    }
    if (sourceMetadata.isSymbolicLink()) {
      if (targetMetadata && samePath(
        path.resolve(path.dirname(source), fs.readlinkSync(source)), target,
      )) {
        return { migrated: false, requestedExtras: null, source, target };
      }
      if (targetMetadata) fail(`durable custback venv target already exists: ${target}`);
      const inspection = managed.inspectTarget(source);
      if (inspection.kind !== 'symlink') {
        fail(`package-local venv is not a recognized generational target: ${source}`);
      }
      const requestedExtras = parseReferenceInstallExtras(inspection.activeGeneration);
      journal = prepareGenerationalJournal(source, target, requestedExtras);
      managed.writeJson(journalPath, journal);
      failpoint('journal-written');
    } else {
      if (targetMetadata) fail(`durable custback venv target already exists: ${target}`);
      const inspection = managed.inspectTarget(source);
      if (inspection.kind !== 'direct' || !inspection.legacy) {
        fail(`package-local venv is not a recognized pre-0.4 custback environment: ${source}`);
      }
      const requestedExtras = parseLegacyExtras(source);
      journal = {
        owner: managed.OWNER,
        schema: 1,
        phase: 'prepared',
        layout: 'direct',
        source,
        target,
        requestedExtras,
      };
      managed.writeJson(journalPath, journal);
      failpoint('journal-written');
    }
  }

  const requestedExtras = installer.parseExtras(journal.requestedExtras.join(','));
  if ((journal.layout || 'direct') === 'generational') {
    return migrateGenerationalVenv(
      { ...journal, requestedExtras }, journalPath, failpoint,
    );
  }
  let currentSource = lstatOrNull(source);
  let currentTarget = lstatOrNull(target);
  if (currentSource && !currentSource.isSymbolicLink() && !currentTarget) {
    const sourceDevice = currentSource.dev;
    const targetParentDevice = fs.statSync(path.dirname(target)).dev;
    if (sourceDevice !== targetParentDevice) {
      fail('legacy npm venv migration requires source and prefix target on one filesystem');
    }
    fs.renameSync(source, target);
    currentSource = null;
    currentTarget = fs.lstatSync(target);
    failpoint('venv-renamed');
  }
  if (!currentTarget || currentTarget.isSymbolicLink()) {
    fail(`legacy npm venv is missing after migration rename: ${target}`);
  }
  managed.inspectTarget(target);

  rewriteRelocatedScripts(target, source);
  failpoint('scripts-rewritten');
  installer.writeInstallIntent(target, requestedExtras);
  failpoint('intent-written');
  ensureCompatibilityLink(source, target);
  failpoint('compatibility-linked');

  fs.rmSync(journalPath, { force: false });
  return { migrated: true, layout: 'direct', requestedExtras, source, target };
}

function packageRootForPrefix(prefix) {
  const resolved = path.resolve(prefix);
  const globalRoot = path.join(resolved, 'lib', 'node_modules', 'custback');
  const localRoot = path.join(resolved, 'node_modules', 'custback');
  if (lstatOrNull(globalRoot)) return globalRoot;
  if (lstatOrNull(localRoot)) return localRoot;
  return globalRoot;
}

function parseArgs(argv) {
  let packageRoot;
  let prefix;
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === '--package-root' && index + 1 < argv.length) {
      packageRoot = argv[index += 1];
    } else if (argument === '--prefix' && index + 1 < argv.length) {
      prefix = argv[index += 1];
    } else {
      fail('usage: custback-npm-migrate (--prefix PREFIX | --package-root PACKAGE_ROOT)');
    }
  }
  if (Boolean(packageRoot) === Boolean(prefix)) {
    fail('specify exactly one of --prefix or --package-root');
  }
  return path.resolve(packageRoot || packageRootForPrefix(prefix));
}

function main(argv = process.argv.slice(2)) {
  try {
    const packageRoot = parseArgs(argv);
    const result = migrateLegacyVenv({ packageRoot });
    if (result.migrated) {
      console.log(
        `[custback migrate] preserved the legacy npm venv at ${result.target} ` +
        `(extras: ${result.requestedExtras.join(', ') || 'core only'})`,
      );
    } else {
      console.log('[custback migrate] no package-local legacy npm venv was found');
    }
    return 0;
  } catch (err) {
    console.error(`[custback migrate] ${err.message}`);
    return 1;
  }
}

module.exports = {
  JOURNAL_SUFFIX,
  main,
  migrateLegacyVenv,
  parseReferenceInstallExtras,
  packageRootForPrefix,
  parseArgs,
  rewriteRelocatedScripts,
};

if (require.main === module) process.exit(main());
