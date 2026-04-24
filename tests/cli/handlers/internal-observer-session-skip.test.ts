/**
 * Tests for internal observer-session hook short-circuiting.
 *
 * Claude-Mem runs its own Claude Agent SDK sessions from OBSERVER_SESSIONS_DIR.
 * Hooks must ignore that directory before touching the worker, otherwise the
 * memory agent observes its own prompts and recursively stores observer prompts.
 */
import { afterAll, afterEach, beforeEach, describe, expect, it, mock, spyOn } from 'bun:test';
import { OBSERVER_SESSIONS_DIR } from '../../../src/shared/paths.js';
import { logger } from '../../../src/utils/logger.js';

const workerCallLog: Array<{ type: string; path?: string }> = [];

mock.module('../../../src/shared/worker-utils.js', () => ({
  ensureWorkerRunning: () => {
    workerCallLog.push({ type: 'ensureWorkerRunning' });
    return Promise.resolve(true);
  },
  getWorkerPort: () => 37777,
  workerHttpRequest: (apiPath: string) => {
    workerCallLog.push({ type: 'workerHttpRequest', path: apiPath });
    throw new Error(`worker must not be called for internal observer sessions: ${apiPath}`);
  },
}));

mock.module('../../../src/shared/SettingsDefaultsManager.js', () => ({
  SettingsDefaultsManager: {
    get: (key: string) => process.env[key] ?? '',
    getInt: () => 0,
    loadFromFile: () => ({
      CLAUDE_MEM_EXCLUDED_PROJECTS: '',
      CLAUDE_MEM_SEMANTIC_INJECT: 'false',
    }),
  },
}));

let loggerSpies: ReturnType<typeof spyOn>[] = [];

beforeEach(() => {
  workerCallLog.length = 0;
  loggerSpies = [
    spyOn(logger, 'info').mockImplementation(() => {}),
    spyOn(logger, 'debug').mockImplementation(() => {}),
    spyOn(logger, 'warn').mockImplementation(() => {}),
    spyOn(logger, 'error').mockImplementation(() => {}),
    spyOn(logger, 'failure').mockImplementation(() => {}),
    spyOn(logger, 'dataIn').mockImplementation(() => {}),
  ];
});

afterEach(() => {
  loggerSpies.forEach(spy => spy.mockRestore());
});

afterAll(() => {
  mock.restore();
});

describe('internal observer session hook skip', () => {
  it('skips session init before starting or calling the worker', async () => {
    const { sessionInitHandler } = await import('../../../src/cli/handlers/session-init.js');

    const result = await sessionInitHandler.execute({
      sessionId: 'observer-session-init',
      cwd: OBSERVER_SESSIONS_DIR,
      prompt: 'internal observer prompt',
      platform: 'claude-code',
    });

    expect(result.continue).toBe(true);
    expect(result.suppressOutput).toBe(true);
    expect(workerCallLog).toEqual([]);
  });

  it('skips tool observations before starting or calling the worker', async () => {
    const { observationHandler } = await import('../../../src/cli/handlers/observation.js');

    const result = await observationHandler.execute({
      sessionId: 'observer-session-observation',
      cwd: OBSERVER_SESSIONS_DIR,
      platform: 'claude-code',
      toolName: 'Read',
      toolInput: { file_path: '/tmp/example.txt' },
      toolResponse: { content: 'example' },
    });

    expect(result.continue).toBe(true);
    expect(result.suppressOutput).toBe(true);
    expect(workerCallLog).toEqual([]);
  });

  it('skips summaries before starting or calling the worker', async () => {
    const { summarizeHandler } = await import('../../../src/cli/handlers/summarize.js');

    const result = await summarizeHandler.execute({
      sessionId: 'observer-session-summary',
      cwd: OBSERVER_SESSIONS_DIR,
      platform: 'claude-code',
      transcriptPath: '/tmp/unused.jsonl',
    });

    expect(result.continue).toBe(true);
    expect(result.suppressOutput).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(workerCallLog).toEqual([]);
  });

  it('skips session completion before starting or calling the worker', async () => {
    const { sessionCompleteHandler } = await import('../../../src/cli/handlers/session-complete.js');

    const result = await sessionCompleteHandler.execute({
      sessionId: 'observer-session-complete',
      cwd: OBSERVER_SESSIONS_DIR,
      platform: 'claude-code',
    });

    expect(result.continue).toBe(true);
    expect(result.suppressOutput).toBe(true);
    expect(workerCallLog).toEqual([]);
  });

  it('skips file edit observations before starting or calling the worker', async () => {
    const { fileEditHandler } = await import('../../../src/cli/handlers/file-edit.js');

    const result = await fileEditHandler.execute({
      sessionId: 'observer-session-file-edit',
      cwd: OBSERVER_SESSIONS_DIR,
      platform: 'cursor',
      filePath: '/tmp/example.txt',
      edits: [{ start: 0, end: 1, replacement: 'x' }],
    });

    expect(result.continue).toBe(true);
    expect(result.suppressOutput).toBe(true);
    expect(result.exitCode).toBe(0);
    expect(workerCallLog).toEqual([]);
  });
});
