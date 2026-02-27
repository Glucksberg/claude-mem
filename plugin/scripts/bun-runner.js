#!/usr/bin/env node
/**
 * Bun Runner - Finds and executes Bun even when not in PATH
 *
 * This script solves the fresh install problem where:
 * 1. smart-install.js installs Bun to ~/.bun/bin/bun
 * 2. But Bun isn't in PATH until terminal restart
 * 3. Subsequent hooks fail because they can't find `bun`
 *
 * Usage: node bun-runner.js <script> [args...]
 *
 * Fixes #818: Worker fails to start on fresh install
 */
import { spawnSync, spawn } from 'child_process';
import { existsSync } from 'fs';
import { join, dirname } from 'path';
import { homedir } from 'os';
import { fileURLToPath } from 'url';

const IS_WINDOWS = process.platform === 'win32';

// Self-resolve plugin root when CLAUDE_PLUGIN_ROOT is not set by Claude Code.
// Upstream bug: anthropics/claude-code#24529 — Stop hooks (and on Linux, all hooks)
// don't receive CLAUDE_PLUGIN_ROOT, causing script paths to resolve to /scripts/...
// which doesn't exist. This fallback derives the plugin root from bun-runner.js's
// own filesystem location (this file lives in <plugin-root>/scripts/).
const __bun_runner_dirname = dirname(fileURLToPath(import.meta.url));
const RESOLVED_PLUGIN_ROOT = process.env.CLAUDE_PLUGIN_ROOT || resolve(__bun_runner_dirname, '..');

/**
 * Fix script path arguments that were broken by empty CLAUDE_PLUGIN_ROOT.
 * When CLAUDE_PLUGIN_ROOT is empty, "${CLAUDE_PLUGIN_ROOT}/scripts/foo.cjs"
 * expands to "/scripts/foo.cjs" which doesn't exist. Detect this and rewrite
 * the path using our self-resolved plugin root.
 */
function fixBrokenScriptPath(argPath) {
  if (argPath.startsWith('/scripts/') && !existsSync(argPath)) {
    const fixedPath = join(RESOLVED_PLUGIN_ROOT, argPath);
    if (existsSync(fixedPath)) {
      return fixedPath;
    }
  }
  return argPath;
}

/**
 * Find Bun executable - checks PATH first, then common install locations
 */
function findBun() {
  // Try PATH first
  const pathCheck = spawnSync(IS_WINDOWS ? 'where' : 'which', ['bun'], {
    encoding: 'utf-8',
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: IS_WINDOWS
  });

  if (pathCheck.status === 0 && pathCheck.stdout.trim()) {
    return 'bun'; // Found in PATH
  }

  // Check common installation paths (handles fresh installs before PATH reload)
  // Windows: Bun installs to ~/.bun/bin/bun.exe (same as smart-install.js)
  // Unix: Check default location plus common package manager paths
  const bunPaths = IS_WINDOWS
    ? [join(homedir(), '.bun', 'bin', 'bun.exe')]
    : [
        join(homedir(), '.bun', 'bin', 'bun'),
        '/usr/local/bin/bun',
        '/opt/homebrew/bin/bun',
        '/home/linuxbrew/.linuxbrew/bin/bun'
      ];

  for (const bunPath of bunPaths) {
    if (existsSync(bunPath)) {
      return bunPath;
    }
  }

  return null;
}

// Early exit if plugin is disabled in Claude Code settings (#781).
// Sync read + JSON parse — fastest possible check before spawning Bun.
function isPluginDisabledInClaudeSettings() {
  try {
    const configDir = process.env.CLAUDE_CONFIG_DIR || join(homedir(), '.claude');
    const settingsPath = join(configDir, 'settings.json');
    if (!existsSync(settingsPath)) return false;
    const settings = JSON.parse(readFileSync(settingsPath, 'utf-8'));
    return settings?.enabledPlugins?.['claude-mem@thedotmack'] === false;
  } catch {
    return false;
  }
}

if (isPluginDisabledInClaudeSettings()) {
  process.exit(0);
}

// Get args: node bun-runner.js <script> [args...]
const args = process.argv.slice(2);

if (args.length === 0) {
  console.error('Usage: node bun-runner.js <script> [args...]');
  process.exit(1);
}

// Fix broken script paths caused by empty CLAUDE_PLUGIN_ROOT (#1215)
args[0] = fixBrokenScriptPath(args[0]);

const bunPath = findBun();

if (!bunPath) {
  console.error('Error: Bun not found. Please install Bun: https://bun.sh');
  console.error('After installation, restart your terminal.');
  process.exit(1);
}

// Extract the script path and its directory for cwd
// This prevents EPERM errors when hooks run from C:\WINDOWS\system32
// by ensuring Bun always runs from a valid working directory
const scriptPath = args[0];
const scriptDir = scriptPath ? dirname(scriptPath) : homedir();

// Spawn Bun with the provided script and args
// Use spawn (not spawnSync) to properly handle stdio
// Note: Don't use shell mode on Windows - it breaks paths with spaces in usernames
// Set cwd to script directory to avoid inheriting C:\WINDOWS\system32
// Set CLAUDE_MEM_SHOW_CONSOLE=1 to show Bun console window (Windows only)
const showConsole = process.env.CLAUDE_MEM_SHOW_CONSOLE === '1';

const child = spawn(bunPath, args, {
  stdio: 'inherit',
  windowsHide: !showConsole,  // Show console if env var is set
  env: process.env,
  cwd: scriptDir
});

// Write buffered stdin to child's pipe, then close it so the child sees EOF
if (stdinData && child.stdin) {
  child.stdin.write(stdinData);
  child.stdin.end();
}

child.on('error', (err) => {
  console.error(`Failed to start Bun: ${err.message}`);
  process.exit(1);
});

child.on('close', (code) => {
  process.exit(code || 0);
});
