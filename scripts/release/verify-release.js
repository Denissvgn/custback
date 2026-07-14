#!/usr/bin/env node
/** Non-mutating release metadata and npm payload gate. */

'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { isDeepStrictEqual } = require('util');

const ROOT = path.resolve(__dirname, '..', '..');
const managed = require(path.join(ROOT, 'packaging', 'npm', 'managed-venv'));
const installer = require(path.join(ROOT, 'packaging', 'npm', 'install'));
const configuredTimeout = Number(process.env.CUSTBACK_RELEASE_TIMEOUT_MS);
const COMMAND_TIMEOUT_MS = Number.isSafeInteger(configuredTimeout) && configuredTimeout > 0
  ? configuredTimeout
  : 15 * 60 * 1000;
const REVIEWED_PYTHON_MODULES = [
  'custback/__init__.py',
  'custback/__main__.py',
  'custback/api/__init__.py',
  'custback/api/security.py',
  'custback/api/server.py',
  'custback/backgrounds.py',
  'custback/capture.py',
  'custback/compositor.py',
  'custback/config.py',
  'custback/hub.py',
  'custback/pipeline.py',
  'custback/preview.py',
  'custback/segmentation.py',
  'custback/vcam.py',
];
const REVIEWED_PYTHON_TESTS = [
  'tests/test_api.py',
  'tests/test_api_lifecycle.py',
  'tests/test_api_security.py',
  'tests/test_config.py',
  'tests/test_pipeline.py',
  'tests/test_preview.py',
  'tests/test_processing.py',
  'tests/test_segmentation_rvm.py',
];
const REVIEWED_NPM_PAYLOAD = [
  'README.md',
  'config/default.yaml',
  'examples/avatar_client.py',
  'package.json',
  'packaging/npm/custback.js',
  'packaging/npm/install.js',
  'packaging/npm/managed-venv.js',
  'pyproject.toml',
  'scripts/install_linux.sh',
  'scripts/install_macos.sh',
  'scripts/release/verify-release.js',
  ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
];
const REVIEWED_BUILD_REQUIREMENTS = ['setuptools>=68,<84'];
const REVIEWED_CORE_DEPENDENCIES = [
  'numpy>=1.24,<3',
  'opencv-contrib-python>=4.8,<6',
  'pillow>=10,<13',
  'pydantic>=2.7,<3',
  'pyvirtualcam>=0.11,<1',
  'fastapi>=0.110,<1',
  'uvicorn>=0.29,<1',
  'pyyaml>=6.0,<7',
  'websockets>=12.0,<17',
  'python-multipart>=0.0.9,<1',
];
const REVIEWED_OPTIONAL_DEPENDENCIES = {
  mediapipe: ['mediapipe>=0.10.14,<0.11'],
  rvm: ['onnxruntime>=1.17,<2'],
  gpu: ['onnxruntime-gpu>=1.17,<2'],
  dev: [
    'build>=1.2,<2',
    'pytest>=8.0,<10',
    'pytest-timeout>=2.3,<3',
    'httpx>=0.27,<0.29',
    'httpx2>=2,<3',
  ],
};
const REVIEWED_CONSOLE_SCRIPTS = { custback: 'custback.__main__:main' };
const REVIEWED_NPM_METADATA = {
  name: 'custback',
  description: 'Virtual camera with background replacement for meeting apps (Ubuntu / macOS)',
  license: 'MIT',
  bin: { custback: 'packaging/npm/custback.js' },
  scripts: {
    postinstall: 'node packaging/npm/install.js',
    test: 'node --test packaging/npm/test/*.test.js',
    doctor: 'node packaging/npm/custback.js doctor',
    'release:check': 'node scripts/release/verify-release.js',
    prepack: 'node scripts/release/verify-release.js --prepack',
  },
  files: [
    'packaging/npm/*.js',
    'src/**/*.py',
    'config/*.yaml',
    'scripts/*.sh',
    'scripts/release/*.js',
    'examples/*.py',
    'pyproject.toml',
  ],
  os: ['linux', 'darwin'],
  engines: { node: '>=18' },
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
    fail('build-system requirements must be exactly setuptools>=68,<84');
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
    name.startsWith('packaging/npm/test/') || name.endsWith('.tgz') || name.endsWith('.whl') ||
    name.endsWith('.tar.gz') || name.includes('__pycache__') || name.endsWith('.pyc') ||
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

function staleArtifacts(root = ROOT) {
  const staleNames = new Set(['debug.txt', 'uninstall.log']);
  const stale = [];
  for (const name of fs.readdirSync(root)) {
    const full = path.join(root, name);
    if (staleNames.has(name) || name.endsWith('.tgz') || name.endsWith('.whl') ||
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
      'README.md',
      'package.json',
      'packaging/npm/custback.js',
      'packaging/npm/install.js',
      'packaging/npm/managed-venv.js',
      'config/default.yaml',
      'pyproject.toml',
      'scripts/install_linux.sh',
      'scripts/install_macos.sh',
      'scripts/release/verify-release.js',
      'src/custback/__init__.py',
      'src/custback/__main__.py',
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
          relative.endsWith('.tar.gz') || relative.endsWith('.pyc')) return false;
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
    'README.md',
    'pyproject.toml',
    ...REVIEWED_PYTHON_MODULES.map((name) => `src/${name}`),
    ...REVIEWED_PYTHON_TESTS,
  ]);
  stageCleanSource(root, source);
  fs.mkdirSync(output);
  runChecked(python, ['-m', 'venv', buildTools]);
  const buildPython = path.join(buildTools, 'bin', 'python');
  runChecked(buildPython, [
    '-m', 'pip', 'install', '--disable-pip-version-check',
    'pip>=23,<27', 'build>=1.2,<2',
  ]);
  runChecked(buildPython, ['-m', 'build', '--sdist', '--wheel', '--outdir', output, source]);
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
    version, wheel_path, sdist_path, module_json, test_json,
    core_json, optional_json, scripts_json,
) = sys.argv[1:]
bad = ("/.venv/", "__pycache__", ".pyc", ".tgz", "/debug.txt", "/uninstall.log", "/build/")
source_modules = set(json.loads(module_json))
source_tests = set(json.loads(test_json))
core_dependencies = json.loads(core_json)
optional_dependencies = json.loads(optional_json)
console_scripts = json.loads(scripts_json)

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
    extra = ""
    if separator:
        marker_match = re.fullmatch(
            r"\\s*extra\\s*==\\s*['\\\"]([A-Za-z0-9_.-]+)['\\\"]\\s*",
            marker,
        )
        require(marker_match, value)
        extra = marker_match.group(1)
    return name, specs, extra

expected_requires = {normalize_requirement(item) for item in core_dependencies}
for extra, dependencies in optional_dependencies.items():
    expected_requires |= {
        normalize_requirement(f"{item}; extra == '{extra}'") for item in dependencies
    }

def verify_metadata(text):
    metadata = email.parser.Parser().parsestr(text)
    require(metadata["Name"].lower() == "custback", metadata["Name"])
    require(metadata["Version"] == version, metadata["Version"])
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
    JSON.stringify(REVIEWED_CORE_DEPENDENCIES),
    JSON.stringify(REVIEWED_OPTIONAL_DEPENDENCIES),
    JSON.stringify(REVIEWED_CONSOLE_SCRIPTS),
  ]);

  const importProbe = [
    'import importlib.metadata as m',
    'import custback, custback.api.server, cv2, fastapi, numpy, pydantic, PIL, pyvirtualcam, uvicorn, websockets, yaml',
    `expected = ${JSON.stringify(version)}`,
    'metadata_version = m.version("custback")',
    'if metadata_version != expected:\n    raise RuntimeError(f"metadata version {metadata_version!r} != {expected!r}")',
    'if custback.__version__ != expected:\n    raise RuntimeError(f"source version {custback.__version__!r} != {expected!r}")',
  ].join('\n');
  for (const [kind, artifact] of [
    ['wheel', path.join(output, wheel)],
    ['sdist', path.join(output, sdist)],
  ]) {
    const venv = path.join(temporaryRoot, `${kind}-smoke-venv`);
    runChecked(python, ['-m', 'venv', venv]);
    const venvPython = path.join(venv, 'bin', 'python');
    runChecked(venvPython, [
      '-m', 'pip', 'install', '--disable-pip-version-check', artifact,
    ]);
    runChecked(venvPython, ['-m', 'pip', 'check']);
    runChecked(venvPython, ['-c', importProbe]);
    runChecked(venvPython, ['-m', 'custback', '--help']);
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
  runChecked(launcher, ['--help'], { env: smokeEnv });
  runChecked(launcher, ['rebuild'], { env: smokeEnv });
  runChecked(launcher, ['doctor'], { env: smokeEnv });
  const stamp = JSON.parse(fs.readFileSync(path.join(managedVenv, managed.INSTALL_STAMP), 'utf8'));
  if (!installer.validInstallStamp(stamp) || stamp.packageVersion !== version ||
      stamp.sourceDigest !== installer.sourceDigest(root) || stamp.requestedExtras.length !== 0) {
    fail('npm smoke install stamp is inconsistent with the artifact');
  }
}

function verifyBuiltArtifacts(version, root = ROOT) {
  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'custback-release-'));
  try {
    verifyPythonArtifacts(version, temporaryRoot, root);
    verifyNpmArtifactInstall(version, temporaryRoot, root);
  } finally {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
}

function main(argv = process.argv.slice(2)) {
  try {
    const version = verifyVersions();
    verifyDependencies();
    verifyDocs();
    const stale = staleArtifacts();
    if (stale.length) {
      fail(`stale release artifacts must be removed before release: ${stale.join(', ')}`);
    }
    if (!argv.includes('--prepack')) {
      verifyPack(version);
      if (!argv.includes('--quick')) verifyBuiltArtifacts(version);
    }
    console.log(`[custback release] metadata and npm payload verified for ${version}`);
    return 0;
  } catch (err) {
    console.error(`[custback release] ${err.message}`);
    return 1;
  }
}

module.exports = {
  main,
  parseNpmPackPayload,
  projectVersion,
  staleArtifacts,
  verifyDependencies,
  verifyDocs,
  verifyBuiltArtifacts,
  verifyNpmArtifactInstall,
  verifyNpmMetadata,
  verifyPack,
  verifyPythonArtifacts,
  verifyVersions,
};

if (require.main === module) process.exit(main());
