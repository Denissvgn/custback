'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const installer = require('../install');
const migration = require('../migrate-legacy');
const managed = require('../managed-venv');

function fixture(t) {
  const prefix = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-phase6-npm-'));
  t.after(() => fs.rmSync(prefix, { recursive: true, force: true }));
  const packageRoot = path.join(prefix, 'lib', 'node_modules', 'custback');
  const legacy = path.join(packageRoot, '.venv');
  fs.mkdirSync(path.join(legacy, 'bin'), { recursive: true });
  fs.writeFileSync(path.join(legacy, 'pyvenv.cfg'), 'home = /usr/bin\n');
  for (const name of ['python', 'custback']) {
    fs.writeFileSync(
      path.join(legacy, 'bin', name),
      `#!${legacy}/bin/python\n# ${legacy}\n`,
      { mode: 0o755 },
    );
  }
  fs.writeFileSync(
    path.join(legacy, managed.LEGACY_STAMP),
    '0.3.0 python3.12 3.12.9 [gpu]\n',
  );
  return { legacy, packageRoot, prefix };
}

function generationalFixture(t, { schema = 2, promotionPhase = null } = {}) {
  const prefix = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-phase6-generations-'));
  t.after(() => fs.rmSync(prefix, { recursive: true, force: true }));
  const packageRoot = path.join(prefix, 'lib', 'node_modules', 'custback');
  const legacy = path.join(packageRoot, '.venv');
  const generationRoot = managed.generationsRootFor(legacy);
  fs.mkdirSync(generationRoot, { recursive: true, mode: 0o700 });
  managed.writeJson(path.join(generationRoot, managed.GENERATIONS_MARKER), {
    owner: managed.OWNER,
    schema: managed.MARKER_SCHEMA,
    logicalTarget: legacy,
  });

  const makeGeneration = (name, withStamp = false) => {
    const generation = path.join(generationRoot, name);
    fs.mkdirSync(path.join(generation, 'bin'), { recursive: true });
    fs.writeFileSync(path.join(generation, 'pyvenv.cfg'), 'home = /usr/bin\n');
    for (const executable of ['python', 'custback']) {
      fs.writeFileSync(
        path.join(generation, 'bin', executable),
        `#!${generation}/bin/python\n# ${generation}\n`,
        { mode: 0o755 },
      );
    }
    managed.markGeneration(generation, legacy);
    if (withStamp) {
      managed.writeJson(path.join(generation, managed.INSTALL_STAMP), {
        schema,
        packageVersion: schema === 2 ? '0.3.0' : '0.4.0',
        sourceDigest: `sha256:${'a'.repeat(64)}`,
        python: {
          executable: '/usr/bin/python3',
          version: '3.12.9',
          cacheTag: 'cpython-312',
        },
        requestedExtras: ['rvm'],
        selectedExtras: ['rvm'],
        capabilities: {
          mediapipe: false,
          rvm: true,
          cuda_provider: false,
        },
        createdAt: '2026-01-01T00:00:00.000Z',
      });
    }
    return generation;
  };

  const active = makeGeneration('gen-active', true);
  const rollback = makeGeneration('legacy-rollback');
  const older = makeGeneration('gen-older');
  fs.symlinkSync(active, legacy, 'dir');
  if (promotionPhase) {
    managed.writeJson(path.join(generationRoot, managed.MIGRATION_JOURNAL), {
      owner: managed.OWNER,
      schema: managed.MARKER_SCHEMA,
      phase: promotionPhase,
      logicalTarget: legacy,
      previousGeneration: rollback,
      newGeneration: active,
    });
  }
  return { active, generationRoot, legacy, older, packageRoot, prefix, rollback };
}

test('Phase 6 pre-upgrade bridge survives npm package-directory replacement', (t) => {
  const { legacy, packageRoot } = fixture(t);
  const target = managed.defaultTargetForPackage(packageRoot);
  const result = migration.migrateLegacyVenv({ packageRoot });

  assert.equal(result.migrated, true);
  assert.deepEqual(result.requestedExtras, ['gpu']);
  assert.equal(fs.lstatSync(legacy).isSymbolicLink(), true);
  assert.equal(fs.realpathSync(legacy), fs.realpathSync(target));
  assert.match(fs.readFileSync(path.join(target, 'bin', 'custback'), 'utf8'),
    new RegExp(`^#!${target.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/bin/python`));
  assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['gpu']);

  fs.rmSync(packageRoot, { recursive: true });
  fs.mkdirSync(packageRoot, { recursive: true });
  assert.equal(fs.existsSync(target), true);
  assert.equal(fs.existsSync(path.join(target, 'bin', 'custback')), true);
  assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['gpu']);
});

test('Phase 6 pre-upgrade bridge converges after every durable boundary', async (t) => {
  for (const boundary of [
    'journal-written',
    'venv-renamed',
    'scripts-rewritten',
    'intent-written',
    'compatibility-linked',
  ]) {
    await t.test(boundary, (subtest) => {
      const { packageRoot } = fixture(subtest);
      let injected = false;
      assert.throws(
        () => migration.migrateLegacyVenv({
          packageRoot,
          failpoint(name) {
            if (!injected && name === boundary) {
              injected = true;
              throw new Error(`injected ${boundary}`);
            }
          },
        }),
        new RegExp(`injected ${boundary}`),
      );
      assert.equal(injected, true);
      const recovered = migration.migrateLegacyVenv({ packageRoot });
      const target = managed.defaultTargetForPackage(packageRoot);
      assert.equal(recovered.migrated, true);
      assert.equal(fs.realpathSync(path.join(packageRoot, '.venv')), fs.realpathSync(target));
      assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['gpu']);
      assert.equal(fs.existsSync(`${target}${migration.JOURNAL_SUFFIX}`), false);
    });
  }
});

test('actual 0.3 schema-2/schema-3 generations survive package replacement', async (t) => {
  for (const schema of [2, 3]) {
    await t.test(`install stamp schema ${schema}`, (subtest) => {
      const state = generationalFixture(subtest, {
        schema,
        promotionPhase: schema === 2 ? 'prepared' : 'committed',
      });
      const target = managed.defaultTargetForPackage(state.packageRoot);
      const targetRoot = managed.generationsRootFor(target);

      const result = migration.migrateLegacyVenv({ packageRoot: state.packageRoot });

      assert.equal(result.layout, 'generational');
      assert.deepEqual(result.requestedExtras, ['rvm']);
      assert.equal(fs.realpathSync(target), path.join(targetRoot, 'gen-active'));
      assert.equal(fs.realpathSync(state.legacy), fs.realpathSync(target));
      assert.equal(fs.existsSync(state.generationRoot), false);
      for (const name of ['gen-active', 'legacy-rollback', 'gen-older']) {
        const generation = path.join(targetRoot, name);
        assert.equal(managed.readJson(
          path.join(generation, managed.VENV_MARKER),
        ).logicalTarget, target);
        assert.match(
          fs.readFileSync(path.join(generation, 'bin', 'custback'), 'utf8'),
          new RegExp(targetRoot.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
        );
      }
      assert.equal(
        managed.readJson(path.join(targetRoot, managed.GENERATIONS_MARKER)).logicalTarget,
        target,
      );
      const promotion = managed.readJson(path.join(targetRoot, managed.MIGRATION_JOURNAL));
      assert.equal(promotion.logicalTarget, target);
      assert.equal(promotion.previousGeneration, path.join(targetRoot, 'legacy-rollback'));
      assert.equal(promotion.newGeneration, path.join(targetRoot, 'gen-active'));
      assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['rvm']);

      fs.rmSync(state.packageRoot, { recursive: true });
      fs.mkdirSync(state.packageRoot, { recursive: true });
      assert.equal(fs.realpathSync(target), path.join(targetRoot, 'gen-active'));
      assert.equal(fs.existsSync(path.join(targetRoot, 'legacy-rollback')), true);
      assert.equal(fs.existsSync(path.join(targetRoot, 'gen-older')), true);
    });
  }
});

test('generational bridge converges after every durable boundary', async (t) => {
  for (const boundary of [
    'journal-written',
    'generation-root-renamed',
    'metadata-rewritten',
    'target-linked',
    'intent-written',
    'compatibility-linked',
  ]) {
    await t.test(boundary, (subtest) => {
      const state = generationalFixture(subtest, { schema: 2 });
      let injected = false;
      assert.throws(
        () => migration.migrateLegacyVenv({
          packageRoot: state.packageRoot,
          failpoint(name) {
            if (!injected && name === boundary) {
              injected = true;
              throw new Error(`injected ${boundary}`);
            }
          },
        }),
        new RegExp(`injected ${boundary}`),
      );
      assert.equal(injected, true);

      const recovered = migration.migrateLegacyVenv({ packageRoot: state.packageRoot });
      const target = managed.defaultTargetForPackage(state.packageRoot);
      const targetRoot = managed.generationsRootFor(target);
      assert.equal(recovered.layout, 'generational');
      assert.equal(fs.realpathSync(target), path.join(targetRoot, 'gen-active'));
      assert.equal(fs.realpathSync(state.legacy), fs.realpathSync(target));
      assert.equal(fs.existsSync(`${target}${migration.JOURNAL_SUFFIX}`), false);
      assert.deepEqual(installer.readInstallIntent(target).requestedExtras, ['rvm']);
    });
  }
});

test('pre-upgrade bridge rejects symlinked or colliding legacy state', (t) => {
  const { legacy, packageRoot, prefix } = fixture(t);
  const outside = path.join(prefix, 'outside');
  fs.rmSync(legacy, { recursive: true });
  fs.mkdirSync(outside);
  fs.symlinkSync(outside, legacy, 'dir');
  assert.throws(
    () => migration.migrateLegacyVenv({ packageRoot }),
    /unowned symlink|real owned directory/,
  );
  assert.equal(fs.realpathSync(legacy), fs.realpathSync(outside));
});
