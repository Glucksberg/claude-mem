#!/usr/bin/env node

const { execSync } = require('child_process');
const { existsSync, mkdirSync, readFileSync, writeFileSync } = require('fs');
const { createRequire } = require('module');
const path = require('path');
const os = require('os');

// Subpath imports the bundled worker requires transitively (via zod v4's compat
// exports). A stale/partial `bun install` can leave the `zod` directory present
// while these subpaths fail to resolve — surfacing later as a runtime
// `Cannot find module 'zod/v3'` that crashes the worker on startup and hangs
// every session waiting on it. Mirrors verifyCriticalModules() in
// src/npx-cli/install/setup-runtime.ts; keep them in sync.
const ZOD_REQUIRED_SUBPATHS = ['zod/v3', 'zod/v4', 'zod/v4-mini'];

// Assert that an install closure actually resolves its declared dependencies
// (and zod's subpath exports) from `targetDir`, not merely that the package
// directories exist on disk. Throws loud so a broken sync fails here instead of
// shipping a worker that crash-loops in the user's sessions.
function verifyCriticalModules(targetDir, label) {
  const pkgPath = path.join(targetDir, 'package.json');
  if (!existsSync(pkgPath)) return;
  const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));
  const dependencies = Object.keys(pkg.dependencies || {});

  const nodeModulesPath = path.join(targetDir, 'node_modules');
  const requireFromTarget = createRequire(path.join(nodeModulesPath, 'noop.js'));
  const resolvePaths = [nodeModulesPath];
  const unresolvable = [];

  for (const dep of dependencies) {
    try {
      requireFromTarget.resolve(dep, { paths: resolvePaths });
    } catch {
      // Some ESM-only packages expose only an `import` condition, and bin-only
      // packages have no importable entry point. A physical manifest still tells
      // us the dependency is installed; explicit subpath checks below cover the
      // exports the worker is known to require.
      if (!existsSync(path.join(nodeModulesPath, ...dep.split('/'), 'package.json'))) {
        unresolvable.push(dep);
      }
    }
  }

  if (dependencies.includes('zod')) {
    for (const subpath of ZOD_REQUIRED_SUBPATHS) {
      try {
        requireFromTarget.resolve(subpath, { paths: resolvePaths });
      } catch {
        unresolvable.push(subpath);
      }
    }
  }

  if (unresolvable.length > 0) {
    throw new Error(
      `Post-install check failed in ${label} (${targetDir}): unresolvable modules: ${unresolvable.join(', ')}`,
    );
  }
  console.log(`\x1b[32m%s\x1b[0m`, `✓ Verified critical modules resolve in ${label}`);
}

const INSTALLED_PATH = path.join(os.homedir(), '.claude', 'plugins', 'marketplaces', 'thedotmack');
const CACHE_BASE_PATH = path.join(os.homedir(), '.claude', 'plugins', 'cache', 'thedotmack', 'claude-mem');
const CODEX_CACHE_BASE_PATH = path.join(os.homedir(), '.codex', 'plugins', 'cache', 'claude-mem-local', 'claude-mem');

function parseWorkerPort(value) {
  const port = Number.parseInt(String(value ?? ''), 10);
  return Number.isInteger(port) && port >= 1 && port <= 65535 ? port : null;
}

function getCurrentBranch() {
  try {
    if (!existsSync(path.join(INSTALLED_PATH, '.git'))) {
      return null;
    }
    return execSync('git rev-parse --abbrev-ref HEAD', {
      cwd: INSTALLED_PATH,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim();
  } catch {
    return null;
  }
}

function getGitignoreExcludes(basePath) {
  const gitignorePath = path.join(basePath, '.gitignore');
  if (!existsSync(gitignorePath)) return '';

  const syncManagedFiles = new Set();

  const lines = readFileSync(gitignorePath, 'utf-8').split('\n');
  return lines
    .map(line => line.trim())
    .filter(line =>
      line &&
      !line.startsWith('#') &&
      !line.startsWith('!') &&
      !syncManagedFiles.has(line)
    )
    .map(pattern => `--exclude=${JSON.stringify(pattern)}`)
    .join(' ');
}

const branch = getCurrentBranch();
const isForce = process.argv.includes('--force');

if (branch && branch !== 'main' && !isForce) {
  console.log('');
  console.log('\x1b[33m%s\x1b[0m', `WARNING: Installed plugin is on beta branch: ${branch}`);
  console.log('\x1b[33m%s\x1b[0m', 'Running rsync would overwrite beta code.');
  console.log('');
  console.log('Options:');
  console.log('  1. Use UI at http://localhost:37777 to update beta');
  console.log('  2. Switch to stable in UI first, then run sync');
  console.log('  3. Force rsync: npm run sync-marketplace:force');
  console.log('');
  process.exit(1);
}

function getPluginVersion() {
  try {
    const pluginJsonPath = path.join(__dirname, '..', 'plugin', '.claude-plugin', 'plugin.json');
    const pluginJson = JSON.parse(readFileSync(pluginJsonPath, 'utf-8'));
    return pluginJson.version;
  } catch (error) {
    console.error('\x1b[31m%s\x1b[0m', 'Failed to read plugin version:', error.message);
    process.exit(1);
  }
}

function writeInstallMarker(pluginRoot, version) {
  writeFileSync(
    path.join(pluginRoot, '.install-version'),
    JSON.stringify({ version, installedAt: new Date().toISOString() }, null, 2) + '\n',
  );
}

function syncPluginCache(label, destinationPath, pluginGitignoreExcludes) {
  mkdirSync(destinationPath, { recursive: true });
  console.log(`Syncing to ${label} (${destinationPath})...`);
  execSync(
    `rsync -av --delete --exclude=.git --exclude=node_modules ${pluginGitignoreExcludes} plugin/ "${destinationPath}/"`,
    { stdio: 'inherit' }
  );

  console.log(`Running bun install in ${label}...`);
  execSync(`bun install`, { cwd: destinationPath, stdio: 'inherit' });
  verifyCriticalModules(destinationPath, label);
  writeInstallMarker(destinationPath, getPluginVersion());
}

function detectInstalledVersion(buildVersion) {
  const dataDir = process.env.CLAUDE_MEM_DATA_DIR || path.join(os.homedir(), '.claude-mem');
  const settingsPath = path.join(dataDir, 'settings.json');
  let port = parseWorkerPort(process.env.CLAUDE_MEM_WORKER_PORT);
  if (!port && existsSync(settingsPath)) {
    try {
      const s = JSON.parse(readFileSync(settingsPath, 'utf8'));
      const settingsPort = parseWorkerPort(s.CLAUDE_MEM_WORKER_PORT);
      if (settingsPort) port = settingsPort;
    } catch {}
  }
  if (!port) {
    const uid = typeof process.getuid === 'function' ? process.getuid() : 77;
    port = 37700 + (uid % 100);
  }
  let healthBody;
  try {
    healthBody = execSync(`curl -s --max-time 2 http://127.0.0.1:${port}/api/health`, {
      stdio: ['ignore', 'pipe', 'ignore'],
    }).toString().trim();
  } catch {
    return null;
  }
  if (!healthBody) return null;
  let installedVersion;
  let installedPath;
  try {
    const j = JSON.parse(healthBody);
    installedVersion = j.version;
    installedPath = j.workerPath;
  } catch {
    return null;
  }
  if (!installedVersion || installedVersion === buildVersion) return null;
  return { installedVersion, installedPath };
}

const installedMismatch = detectInstalledVersion(getPluginVersion());
if (installedMismatch) {
  console.log('');
  console.log('\x1b[33m%s\x1b[0m', 'Version mismatch detected:');
  console.log(`  Building:   ${getPluginVersion()}`);
  console.log(`  Installed:  ${installedMismatch.installedVersion}`);
  if (installedMismatch.installedPath) console.log(`  Worker path: ${installedMismatch.installedPath}`);
  console.log('');
  console.log('Claude Code is pinned to the installed version, so the worker loads from');
  console.log(`its cache dir. Mirroring this build into the installed-version cache so the`);
  console.log('worker restart picks up new code without a Claude Code session restart.');
  console.log('');
  console.log('\x1b[36m%s\x1b[0m', `For a formal version bump, run \`claude plugin update thedotmack/claude-mem\``);
  console.log('\x1b[36m%s\x1b[0m', `and restart Claude Code so it loads the ${getPluginVersion()} cache dir.`);
  console.log('');
}

console.log('Syncing to marketplace...');
try {
  const rootDir = path.join(__dirname, '..');
  const gitignoreExcludes = getGitignoreExcludes(rootDir);

  execSync(
    `rsync -av --delete --exclude=.git --exclude=bun.lock --exclude=package-lock.json --exclude=scripts/package.json --exclude=scripts/node_modules ${gitignoreExcludes} ./ ~/.claude/plugins/marketplaces/thedotmack/`,
    { stdio: 'inherit' }
  );

  console.log('Running bun install in marketplace...');
  execSync(
    'cd ~/.claude/plugins/marketplaces/thedotmack/ && bun install',
    { stdio: 'inherit' }
  );
  // The marketplace worker (plugin/scripts/worker-service.cjs) resolves zod via
  // upward traversal into this root node_modules, so verify the closure here.
  verifyCriticalModules(INSTALLED_PATH, 'marketplace');

  const version = getPluginVersion();
  const CACHE_VERSION_PATH = path.join(CACHE_BASE_PATH, version);

  const pluginDir = path.join(rootDir, 'plugin');
  const pluginGitignoreExcludes = getGitignoreExcludes(pluginDir);

  syncPluginCache(`Claude cache folder (version ${version})`, CACHE_VERSION_PATH, pluginGitignoreExcludes);

  const CODEX_CACHE_VERSION_PATH = path.join(CODEX_CACHE_BASE_PATH, version);
  syncPluginCache(`Codex cache folder (version ${version})`, CODEX_CACHE_VERSION_PATH, pluginGitignoreExcludes);

  if (installedMismatch && installedMismatch.installedVersion !== version) {
    const INSTALLED_CACHE_PATH = path.join(CACHE_BASE_PATH, installedMismatch.installedVersion);
    syncPluginCache(
      `installed-version Claude cache (${installedMismatch.installedVersion}) for hot reload`,
      INSTALLED_CACHE_PATH,
      pluginGitignoreExcludes,
    );
  }

  console.log('\x1b[32m%s\x1b[0m', 'Sync complete!');

} catch (error) {
  console.error('\x1b[31m%s\x1b[0m', 'Sync failed:', error.message);
  process.exit(1);
}
