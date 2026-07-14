'use strict';

const assert = require('node:assert/strict');
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const managed = require('../managed-venv');

function sleep(milliseconds) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

function waitForFile(file, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  while (!fs.existsSync(file) && Date.now() < deadline) sleep(10);
  assert.equal(fs.existsSync(file), true, `timed out waiting for ${file}`);
}

function sandbox(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-node-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

function fakeVenv(directory) {
  fs.mkdirSync(path.join(directory, 'bin'), { recursive: true });
  fs.writeFileSync(path.join(directory, 'pyvenv.cfg'), 'home = /test\n');
  fs.writeFileSync(path.join(directory, 'bin', 'python'), '#!/bin/sh\n');
  fs.writeFileSync(path.join(directory, 'bin', 'custback'), '#!/bin/sh\n');
}

function fakeGeneration(root, target) {
  const generation = managed.createGeneration(root);
  fakeVenv(generation);
  managed.markGeneration(generation, target);
  return generation;
}

test('dangerous logical targets are rejected before inspection', () => {
  const pkgRoot = path.resolve(__dirname, '..', '..', '..');
  for (const target of [path.parse(pkgRoot).root, os.homedir(), pkgRoot, os.tmpdir()]) {
    assert.throws(
      () => managed.assertSafeTarget(target, { pkgRoot }),
      /refusing unsafe|protected directory/,
    );
  }
  assert.throws(
    () => managed.assertSafeTarget(path.dirname(pkgRoot), { pkgRoot }),
    /protected directory/,
  );
});

test('ancestors of configured working and temporary roots are rejected', (t) => {
  const root = sandbox(t);
  const pkgRoot = path.join(root, 'package');
  const home = path.join(root, 'home');
  const cwd = path.join(root, 'workspace', 'checkout');
  const tmpRoot = path.join(root, 'private', 'tmp');
  for (const directory of [pkgRoot, home, cwd, tmpRoot]) {
    fs.mkdirSync(directory, { recursive: true });
  }
  const options = { pkgRoot, home, cwd, tmpRoot };
  for (const target of [path.dirname(cwd), path.dirname(tmpRoot)]) {
    assert.throws(
      () => managed.assertSafeTarget(target, options),
      /protected directory/,
    );
  }
});

test('an unrelated non-empty directory is never adopted', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'sentinel');
  fs.mkdirSync(target);
  fs.writeFileSync(path.join(target, 'keep-me'), 'safe');
  assert.throws(() => managed.inspectTarget(target), /refusing to replace non-custback directory/);
  assert.equal(fs.readFileSync(path.join(target, 'keep-me'), 'utf8'), 'safe');
});

test('a parent symlink cannot disguise a protected target', (t) => {
  const root = sandbox(t);
  const pkgRoot = path.join(root, 'package');
  fs.mkdirSync(pkgRoot);
  const alias = path.join(root, 'alias');
  fs.symlinkSync(pkgRoot, alias, 'dir');
  assert.throws(
    () => managed.assertSafeTarget(path.join(alias, 'venv'), { pkgRoot }),
    /protected directory/,
  );
});

test('a symlink spelling of a protected root cannot hide its real ancestor', (t) => {
  const root = sandbox(t);
  const actualParent = path.join(root, 'actual');
  const actualPackage = path.join(actualParent, 'package');
  fs.mkdirSync(actualPackage, { recursive: true });
  const packageAlias = path.join(root, 'package-alias');
  fs.symlinkSync(actualPackage, packageAlias, 'dir');
  assert.throws(
    () => managed.assertSafeTarget(actualParent, {
      pkgRoot: packageAlias,
      home: path.join(root, 'home'),
      cwd: path.join(root, 'cwd'),
      tmpRoot: path.join(root, 'tmp'),
    }),
    /protected directory/,
  );
});

test('legacy ownership requires the historical stamp shape', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), 'not a custback stamp');
  assert.throws(() => managed.inspectTarget(target), /legacy custback ownership stamp is invalid/);
  assert.ok(fs.existsSync(path.join(target, 'bin', 'custback')));
});

test('a dangling target symlink is not treated as an absent safe target', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  fs.symlinkSync(path.join(root, 'missing-foreign-venv'), target, 'dir');
  assert.throws(() => managed.inspectTarget(target), /unowned symlink/);
  assert.equal(fs.lstatSync(target).isSymbolicLink(), true);
});

test('an empty generations-root crash remnant is replaced by a complete root', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.generationsRootFor(target);
  fs.mkdirSync(generationRoot);

  assert.equal(managed.ensureGenerationsRoot(target), generationRoot);
  const marker = managed.readJson(path.join(generationRoot, managed.GENERATIONS_MARKER));
  assert.equal(marker.owner, managed.OWNER);
  assert.equal(marker.schema, managed.MARKER_SCHEMA);
  assert.equal(marker.logicalTarget, target);
});

test('a crash before generations-root publication leaves no partial canonical root', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.generationsRootFor(target);
  const modulePath = require.resolve('../managed-venv');
  const script = `
const fs = require('fs');
const managed = require(${JSON.stringify(modulePath)});
const realRename = fs.renameSync;
fs.renameSync = (source, destination) => {
  if (destination === ${JSON.stringify(generationRoot)}) process.exit(29);
  return realRename(source, destination);
};
managed.ensureGenerationsRoot(${JSON.stringify(target)});
`;
  const child = spawnSync(process.execPath, ['-e', script]);
  assert.equal(child.status, 29);
  assert.equal(fs.existsSync(generationRoot), false);
  assert.equal(managed.ensureGenerationsRoot(target), generationRoot);
  assert.equal(
    managed.readJson(path.join(generationRoot, managed.GENERATIONS_MARKER)).owner,
    managed.OWNER,
  );
});

test('candidate cleanup refuses generation-shaped symlinks', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const external = path.join(root, 'external');
  fs.mkdirSync(external);
  fs.writeFileSync(path.join(external, 'keep-me'), 'safe');
  const disguised = path.join(generationRoot, 'gen-disguised');
  fs.symlinkSync(external, disguised, 'dir');
  assert.throws(
    () => managed.removeCreatedGeneration(disguised, generationRoot),
    /refusing to recursively clean non-directory generation/,
  );
  assert.equal(fs.readFileSync(path.join(external, 'keep-me'), 'utf8'), 'safe');
});

test('owned generation promotion creates an inspectable logical symlink', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const generation = fakeGeneration(generationRoot, target);
  const inspection = managed.inspectTarget(target);
  managed.promoteGeneration({
    target,
    generation,
    inspection,
    validateActive(logical) {
      assert.equal(fs.realpathSync(logical), fs.realpathSync(generation));
    },
  });
  const active = managed.inspectTarget(target);
  assert.equal(active.kind, 'symlink');
  assert.equal(fs.realpathSync(active.activeGeneration), fs.realpathSync(generation));
});

test('promotion never removes a colliding unowned temporary path', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const generation = fakeGeneration(generationRoot, target);
  const originalRandom = Math.random;
  Math.random = () => 0.5;
  t.after(() => { Math.random = originalRandom; });
  const collision = path.join(root, `.custback-venv.custback-link-${process.pid}-8`);
  fs.writeFileSync(collision, 'unowned');

  assert.throws(
    () => managed.promoteGeneration({
      target,
      generation,
      inspection: managed.inspectTarget(target),
      validateActive() {},
    }),
    /EEXIST/,
  );
  assert.equal(fs.readFileSync(collision, 'utf8'), 'unowned');
  assert.equal(managed.inspectTarget(target).kind, 'absent');
});

test('failed post-promotion validation restores the previous generation', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const first = fakeGeneration(generationRoot, target);
  managed.promoteGeneration({
    target,
    generation: first,
    inspection: managed.inspectTarget(target),
    validateActive() {},
  });
  const before = managed.inspectTarget(target);
  const second = fakeGeneration(generationRoot, target);
  assert.throws(
    () => managed.promoteGeneration({
      target,
      generation: second,
      inspection: before,
      validateActive() { throw new Error('smoke failed'); },
    }),
    /smoke failed/,
  );
  assert.equal(fs.realpathSync(target), fs.realpathSync(first));
  assert.ok(fs.existsSync(second));
});

test('interrupted legacy promotion journal restores the previous venv', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 []');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const next = fakeGeneration(generationRoot, target);
  const legacyId = 'legacy-interrupted';
  const previous = path.join(generationRoot, legacyId);
  managed.writeJson(path.join(target, managed.VENV_MARKER), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    generationId: legacyId,
    logicalTarget: target,
  });
  managed.writeJson(path.join(generationRoot, managed.MIGRATION_JOURNAL), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    logicalTarget: target,
    previousGeneration: previous,
    newGeneration: next,
  });
  fs.renameSync(target, previous);
  fs.symlinkSync(next, target, 'dir');

  assert.equal(managed.recoverInterruptedPromotion(target, generationRoot), true);
  assert.equal(fs.lstatSync(target).isDirectory(), true);
  assert.equal(fs.existsSync(path.join(target, 'bin', 'custback')), true);
  assert.equal(fs.existsSync(path.join(generationRoot, managed.MIGRATION_JOURNAL)), false);
  assert.equal(fs.existsSync(next), true);
});

test('pre-move legacy journal recovery preserves the direct venv', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 []');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const next = fakeGeneration(generationRoot, target);
  const previous = path.join(generationRoot, 'legacy-never-moved');
  managed.writeJson(path.join(generationRoot, managed.MIGRATION_JOURNAL), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    logicalTarget: target,
    previousGeneration: previous,
    newGeneration: next,
  });

  assert.equal(managed.recoverInterruptedPromotion(target, generationRoot), true);
  assert.equal(fs.readFileSync(path.join(target, 'pyvenv.cfg'), 'utf8'), 'home = /test\n');
  assert.equal(fs.existsSync(path.join(generationRoot, managed.MIGRATION_JOURNAL)), false);
});

test('legacy custback venv is migrated without deleting it before promotion', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 [gpu]');
  const inspection = managed.inspectTarget(target);
  assert.equal(inspection.kind, 'direct');
  assert.equal(inspection.legacy, true);
  const generationRoot = managed.ensureGenerationsRoot(target);
  const next = fakeGeneration(generationRoot, target);
  const result = managed.promoteGeneration({
    target,
    generation: next,
    inspection,
    validateActive() {},
  });
  assert.equal(fs.realpathSync(target), fs.realpathSync(next));
  assert.ok(result.previousGeneration);
  assert.ok(fs.existsSync(path.join(result.previousGeneration, 'bin', 'custback')));
  managed.validateGeneration(result.previousGeneration, target, generationRoot);
});

test('committed legacy promotion survives journal cleanup failure', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 []');
  const inspection = managed.inspectTarget(target);
  const generationRoot = managed.ensureGenerationsRoot(target);
  const next = fakeGeneration(generationRoot, target);
  const realRmSync = fs.rmSync;
  let injected = false;
  fs.rmSync = (file, options) => {
    if (!injected && path.basename(file) === managed.MIGRATION_JOURNAL) {
      injected = true;
      throw new Error('injected journal cleanup failure');
    }
    return realRmSync(file, options);
  };
  try {
    assert.doesNotThrow(() => managed.promoteGeneration({
      target,
      generation: next,
      inspection,
      validateActive() {},
    }));
  } finally {
    fs.rmSync = realRmSync;
  }
  assert.equal(injected, true);
  assert.equal(fs.realpathSync(target), fs.realpathSync(next));
  assert.equal(fs.existsSync(next), true);
  const journalPath = path.join(generationRoot, managed.MIGRATION_JOURNAL);
  assert.equal(managed.readJson(journalPath).phase, 'committed');
  assert.equal(managed.generationIsReferenced(target, next, generationRoot), true);
  assert.equal(managed.recoverInterruptedPromotion(target, generationRoot), true);
  assert.equal(fs.realpathSync(target), fs.realpathSync(next));
  assert.equal(fs.existsSync(journalPath), false);
});

test('committed migration with a missing candidate restores retained legacy venv', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'legacy-venv');
  fakeVenv(target);
  fs.writeFileSync(path.join(target, managed.LEGACY_STAMP), '0.2.0 python3.12 3.12 []');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const next = fakeGeneration(generationRoot, target);
  const previous = path.join(generationRoot, 'legacy-retained');
  managed.writeJson(path.join(target, managed.VENV_MARKER), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    generationId: 'legacy-retained',
    logicalTarget: target,
  });
  fs.renameSync(target, previous);
  fs.symlinkSync(next, target, 'dir');
  managed.writeJson(path.join(generationRoot, managed.MIGRATION_JOURNAL), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    phase: 'committed',
    logicalTarget: target,
    previousGeneration: previous,
    newGeneration: next,
  });
  managed.removeCreatedGeneration(next, generationRoot);
  assert.equal(fs.existsSync(target), false); // dangling symlink
  assert.equal(managed.prepareInstallTarget(target), generationRoot);
  assert.equal(
    managed.withInstallLock(
      generationRoot,
      () => managed.recoverInterruptedPromotion(target, generationRoot),
      { settleMs: 0 },
    ),
    true,
  );
  assert.equal(fs.lstatSync(target).isDirectory(), true);
  assert.equal(fs.existsSync(path.join(target, 'bin', 'custback')), true);
});

test('install lock is exclusive and released after callback', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  managed.withInstallLock(generationRoot, () => {
    assert.throws(
      () => managed.withInstallLock(generationRoot, () => {}, { waitMs: 0 }),
      /another custback install/,
    );
  });
  assert.equal(managed.withInstallLock(generationRoot, () => 42), 42);
});

test('stale owned install lock is recovered', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const lock = path.join(generationRoot, '.install-lock');
  fs.mkdirSync(lock);
  managed.writeJson(path.join(lock, 'owner.json'), {
    owner: managed.OWNER,
    pid: 2147483647,
    createdAt: 0,
  });
  assert.equal(
    managed.withInstallLock(generationRoot, () => 7, { waitMs: 0, staleMs: 0 }),
    7,
  );
});

test('anonymous legacy lock is recoverable after its stale threshold', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const lock = path.join(generationRoot, '.install-lock');
  fs.mkdirSync(lock);
  assert.equal(
    managed.withInstallLock(
      generationRoot,
      () => 11,
      { waitMs: 0, staleMs: 0, settleMs: 0 },
    ),
    11,
  );
});

test('a non-empty foreign lock directory is preserved and never reclaimed', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const lock = path.join(generationRoot, '.install-lock');
  fs.mkdirSync(lock);
  managed.writeJson(path.join(lock, 'owner.json'), {
    owner: managed.OWNER,
    pid: 2147483647,
    createdAt: 0,
  });
  fs.writeFileSync(path.join(lock, 'keep-me'), 'foreign');

  assert.throws(
    () => managed.withInstallLock(
      generationRoot,
      () => assert.fail('foreign lock was reclaimed'),
      { waitMs: 0, staleMs: 0, settleMs: 0 },
    ),
    /install lock is not owned metadata/,
  );
  assert.equal(fs.readFileSync(path.join(lock, 'keep-me'), 'utf8'), 'foreign');
  assert.equal(fs.existsSync(path.join(lock, 'owner.json')), true);
});

test('crash before atomic lock publication leaves no canonical lock', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const modulePath = require.resolve('../managed-venv');
  const script = `
const fs = require('fs');
const path = require('path');
const managed = require(${JSON.stringify(modulePath)});
const realLink = fs.linkSync;
fs.linkSync = (source, destination) => {
  if (path.basename(destination) === '.install-lock') process.exit(23);
  return realLink(source, destination);
};
managed.withInstallLock(${JSON.stringify(generationRoot)}, () => {}, { settleMs: 0 });
`;
  const child = spawnSync(process.execPath, ['-e', script]);
  assert.equal(child.status, 23);
  assert.equal(fs.existsSync(path.join(generationRoot, '.install-lock')), false);
  assert.equal(
    managed.withInstallLock(generationRoot, () => 13, { settleMs: 0 }),
    13,
  );
});

test('crash after publication leaves a complete recoverable lock record', (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const modulePath = require.resolve('../managed-venv');
  const script = `
const managed = require(${JSON.stringify(modulePath)});
managed.withInstallLock(
  ${JSON.stringify(generationRoot)},
  () => process.exit(0),
  { settleMs: 0 },
);
`;
  const child = spawnSync(process.execPath, ['-e', script]);
  assert.equal(child.status, 0);
  const lock = path.join(generationRoot, '.install-lock');
  assert.equal(fs.lstatSync(lock).isFile(), true);
  const record = managed.readJson(lock);
  assert.equal(record.owner, managed.OWNER);
  assert.equal(typeof record.token, 'string');
  assert.equal(
    managed.withInstallLock(
      generationRoot,
      () => 17,
      { waitMs: 0, staleMs: 0, settleMs: 0 },
    ),
    17,
  );
});

test('lock owner survives a breaker quarantine-and-restore race', async (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const lock = path.join(generationRoot, '.install-lock');
  const barrier = path.join(generationRoot, '.install-lock-break-race');
  const quarantine = path.join(generationRoot, '.install-lock-quarantine-race');
  const start = path.join(root, 'start-breaker');
  const moved = path.join(root, 'lock-moved');
  const script = `
const fs = require('fs');
const sleep = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
while (!fs.existsSync(${JSON.stringify(start)})) sleep(5);
fs.writeFileSync(${JSON.stringify(barrier)}, JSON.stringify({
  owner: ${JSON.stringify(managed.OWNER)},
  schema: ${managed.MARKER_SCHEMA},
  token: 'race-test',
  pid: process.pid,
  createdAt: Date.now(),
}));
try {
  fs.renameSync(${JSON.stringify(lock)}, ${JSON.stringify(quarantine)});
  fs.writeFileSync(${JSON.stringify(moved)}, 'moved');
  sleep(100);
  fs.renameSync(${JSON.stringify(quarantine)}, ${JSON.stringify(lock)});
} finally {
  try { fs.unlinkSync(${JSON.stringify(barrier)}); } catch {}
}
`;
  const child = spawn(process.execPath, ['-e', script], { stdio: 'ignore' });
  const exited = new Promise((resolve) => child.once('exit', (code, signal) => {
    resolve({ code, signal });
  }));
  t.after(() => { if (child.exitCode === null) child.kill('SIGKILL'); });

  const realLink = fs.linkSync;
  let injected = false;
  fs.linkSync = (source, destination) => {
    const result = realLink(source, destination);
    if (!injected && destination === lock) {
      injected = true;
      fs.writeFileSync(start, 'start');
      waitForFile(moved);
    }
    return result;
  };
  try {
    assert.equal(
      managed.withInstallLock(
        generationRoot,
        () => 'entered',
        { waitMs: 0, staleMs: 1000, settleMs: 0 },
      ),
      'entered',
    );
  } finally {
    fs.linkSync = realLink;
  }
  assert.equal(injected, true);
  assert.deepEqual(await exited, { code: 0, signal: null });
  assert.equal(fs.existsSync(lock), false);
  assert.equal(fs.existsSync(quarantine), false);
  assert.equal(fs.existsSync(barrier), false);
  assert.equal(
    fs.readdirSync(generationRoot).some((name) => name.startsWith('.install-lock-owner-')),
    false,
  );
});

test('a live owner is never reclaimed even with zero stale threshold', async (t) => {
  const root = sandbox(t);
  const target = path.join(root, 'custback-venv');
  const generationRoot = managed.ensureGenerationsRoot(target);
  const ready = path.join(root, 'ready');
  const release = path.join(root, 'release');
  const modulePath = require.resolve('../managed-venv');
  const script = `
const fs = require('fs');
const managed = require(${JSON.stringify(modulePath)});
const sleep = (ms) => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
managed.withInstallLock(${JSON.stringify(generationRoot)}, () => {
  fs.writeFileSync(${JSON.stringify(ready)}, 'ready');
  while (!fs.existsSync(${JSON.stringify(release)})) sleep(10);
}, { settleMs: 0 });
`;
  const child = spawn(process.execPath, ['-e', script], { stdio: 'ignore' });
  t.after(() => { if (child.exitCode === null) child.kill('SIGKILL'); });
  waitForFile(ready);
  assert.throws(
    () => managed.withInstallLock(
      generationRoot,
      () => assert.fail('overlapping callback entered'),
      { waitMs: 0, staleMs: 0, settleMs: 0 },
    ),
    /another custback install/,
  );
  fs.writeFileSync(release, 'release');
  const code = await new Promise((resolve) => child.once('exit', resolve));
  assert.equal(code, 0);
});
