import { describe, it, expect, beforeEach, afterEach, afterAll, mock } from 'bun:test';
import fs from 'node:fs';
import path from 'node:path';

const ORIGINAL_CLAUDE_MEM_DATA_DIR = process.env.CLAUDE_MEM_DATA_DIR;
const FAKE_DATA_DIR = `/tmp/fake-claude-mem-ssl-${process.pid}`;
const FAKE_CHROMA_DIR = path.join(FAKE_DATA_DIR, 'chroma');
const FAKE_CHROMA_LOCK = path.join(FAKE_CHROMA_DIR, '.claude-mem-chroma-mcp.lock');

process.env.CLAUDE_MEM_DATA_DIR = FAKE_DATA_DIR;

let currentSettings: Record<string, string> = {};
const defaultSettings: Record<string, string> = {
  CLAUDE_MEM_QUEUE_ENGINE: 'sqlite',
  CLAUDE_MEM_REDIS_MODE: 'external',
  CLAUDE_MEM_REDIS_URL: '',
  CLAUDE_MEM_REDIS_HOST: '127.0.0.1',
  CLAUDE_MEM_REDIS_PORT: '6379',
  CLAUDE_MEM_QUEUE_REDIS_PREFIX: 'claude_mem',
  CLAUDE_MEM_WELCOME_HINT_ENABLED: 'true',
  CLAUDE_MEM_WORKER_PORT: '37777',
};

let capturedTransportOpts: { command: string; args: string[] } | null = null;

mock.module('@modelcontextprotocol/sdk/client/stdio.js', () => ({
  StdioClientTransport: class FakeTransport {
    onclose: (() => void) | null = null;
    constructor(opts: { command: string; args: string[] }) {
      capturedTransportOpts = { command: opts.command, args: opts.args };
    }
    async close() {}
  },
}));

mock.module('@modelcontextprotocol/sdk/client/index.js', () => ({
  Client: class FakeClient {
    constructor() {}
    async connect() {}
    async callTool() {
      return { content: [{ type: 'text', text: '{}' }] };
    }
    async close() {}
  },
}));

mock.module('../../../src/shared/SettingsDefaultsManager.js', () => ({
  SettingsDefaultsManager: {
    get: (key: string) => currentSettings[key] ?? defaultSettings[key] ?? '',
    getInt: () => 0,
    loadFromFile: () => ({ ...defaultSettings, ...currentSettings }),
  },
}));

mock.module('../../../src/shared/paths.js', () => ({
  USER_SETTINGS_PATH: '/tmp/fake-settings.json',
  paths: {
    chroma: () => FAKE_CHROMA_DIR,
    combinedCerts: () => '/tmp/fake-combined-certs.pem',
    supervisorRegistry: () => path.join(FAKE_DATA_DIR, 'supervisor.json'),
  },
}));

mock.module('../../../src/supervisor/index.ts', () => ({
  getSupervisor: () => ({
    assertCanSpawn: () => {},
    registerProcess: () => {},
    unregisterProcess: () => {},
  }),
}));

mock.module('../../../src/supervisor/env-sanitizer.js', () => ({
  sanitizeEnv: (env: NodeJS.ProcessEnv) => env,
}));

mock.module('../../../src/utils/logger.js', () => ({
  logger: {
    info: () => {},
    debug: () => {},
    warn: () => {},
    error: () => {},
    failure: () => {},
    dataIn: () => {},
    dataOut: () => {},
    success: () => {},
  },
}));

const { ChromaMcpManager } = await import('../../../src/services/sync/ChromaMcpManager.js');
type ChromaMcpManagerType = InstanceType<typeof ChromaMcpManager>;

async function assertSslFlag(sslSetting: string | undefined, expectedValue: string) {
  currentSettings = { CLAUDE_MEM_CHROMA_MODE: 'remote' };
  if (sslSetting !== undefined) currentSettings.CLAUDE_MEM_CHROMA_SSL = sslSetting;

  await mgr.callTool('chroma_list_collections', {});

  expect(capturedTransportOpts).not.toBeNull();
  const sslIdx = capturedTransportOpts!.args.indexOf('--ssl');
  expect(sslIdx).not.toBe(-1);
  expect(capturedTransportOpts!.args[sslIdx + 1]).toBe(expectedValue);
}

let mgr: ChromaMcpManagerType;

describe('ChromaMcpManager SSL flag regression (#1286)', () => {
  beforeEach(async () => {
    await ChromaMcpManager.reset();
    try { fs.unlinkSync(FAKE_CHROMA_LOCK); } catch { /* absent */ }
    fs.mkdirSync(FAKE_CHROMA_DIR, { recursive: true });
    capturedTransportOpts = null;
    currentSettings = {};
    mgr = ChromaMcpManager.getInstance();
  });

  afterEach(async () => {
    await ChromaMcpManager.reset();
    try { fs.unlinkSync(FAKE_CHROMA_LOCK); } catch { /* absent */ }
  });

  afterAll(() => {
    try { fs.rmSync(FAKE_DATA_DIR, { recursive: true, force: true }); } catch { /* best-effort */ }
    if (ORIGINAL_CLAUDE_MEM_DATA_DIR === undefined) {
      delete process.env.CLAUDE_MEM_DATA_DIR;
    } else {
      process.env.CLAUDE_MEM_DATA_DIR = ORIGINAL_CLAUDE_MEM_DATA_DIR;
    }
  });

  it('emits --ssl false when CLAUDE_MEM_CHROMA_SSL=false', async () => {
    await assertSslFlag('false', 'false');
  });

  it('emits --ssl true when CLAUDE_MEM_CHROMA_SSL=true', async () => {
    await assertSslFlag('true', 'true');
  });

  it('defaults --ssl false when CLAUDE_MEM_CHROMA_SSL is not set', async () => {
    await assertSslFlag(undefined, 'false');
  });

  it('omits --ssl entirely in local mode', async () => {
    currentSettings = {
      CLAUDE_MEM_CHROMA_MODE: 'local',
    };

    await mgr.callTool('chroma_list_collections', {});

    expect(capturedTransportOpts).not.toBeNull();
    const args = capturedTransportOpts!.args;
    expect(args).not.toContain('--ssl');
    expect(args).toContain('--client-type');
    expect(args[args.indexOf('--client-type') + 1]).toBe('persistent');
  });
});
