import { afterEach, describe, expect, mock, test } from 'bun:test';

describe('redis queue config', () => {
  const previousEnv = new Map<string, string | undefined>();

  afterEach(() => {
    for (const [key, value] of previousEnv.entries()) {
      if (value === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = value;
      }
    }
    previousEnv.clear();
    mock.restore();
  });

  test('loads queue settings from environment overrides', async () => {
    setEnv('CLAUDE_MEM_QUEUE_ENGINE', 'bullmq');
    setEnv('CLAUDE_MEM_REDIS_MODE', 'external');
    setEnv('CLAUDE_MEM_REDIS_HOST', 'env-host');
    setEnv('CLAUDE_MEM_REDIS_PORT', '6381');
    setEnv('CLAUDE_MEM_REDIS_URL', '');
    setEnv('CLAUDE_MEM_QUEUE_REDIS_PREFIX', 'settings-prefix');

    const { getRedisQueueConfig, getObservationQueueEngineName } = await import('../../../src/server/queue/redis-config.js');

    expect(getObservationQueueEngineName()).toBe('bullmq');
    const config = getRedisQueueConfig();
    expect(config.host).toBe('env-host');
    expect(config.port).toBe(6381);
    expect(config.prefix).toBe('settings-prefix');
  });

  function setEnv(key: string, value: string): void {
    if (!previousEnv.has(key)) {
      previousEnv.set(key, process.env[key]);
    }
    process.env[key] = value;
  }
});
