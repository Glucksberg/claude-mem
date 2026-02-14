/**
 * Mock Telegram Bot API Server
 *
 * Minimal implementation of Telegram Bot API for E2E testing.
 * Implements: getMe, getUpdates, getChat, sendMessage
 * Records all received messages for verification.
 *
 * Usage: bun docker/mock-telegram-server.ts
 * Listens on port 8443 by default (overridable via MOCK_PORT env var).
 */

const PORT = parseInt(process.env.MOCK_PORT || '8443', 10);
const MOCK_BOT_USERNAME = 'claudemem_test_bot';
const MOCK_CHAT_ID = '-1001234567890';
const MOCK_CHAT_TITLE = 'Claude-Mem Test Group';

// Store received messages for verification
const receivedMessages: Array<{ chatId: string; text: string; timestamp: number }> = [];

function jsonResponse(result: unknown) {
  return new Response(JSON.stringify({ ok: true, result }), {
    headers: { 'Content-Type': 'application/json' },
  });
}

function errorResponse(description: string, status: number = 400) {
  return new Response(JSON.stringify({ ok: false, description }), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

const server = Bun.serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;

    // Extract method from path: /bot<token>/<method>
    const match = path.match(/^\/bot[^/]+\/(\w+)$/);
    if (!match) {
      return errorResponse('Invalid path', 404);
    }

    const method = match[1];
    let body: Record<string, unknown> = {};

    if (req.method === 'POST') {
      try {
        body = await req.json() as Record<string, unknown>;
      } catch {
        // Empty body is OK for some methods
      }
    }

    switch (method) {
      case 'getMe':
        return jsonResponse({
          id: 123456789,
          is_bot: true,
          first_name: 'Claude-Mem Test',
          username: MOCK_BOT_USERNAME,
        });

      case 'getUpdates':
        return jsonResponse([
          {
            update_id: 1,
            message: {
              message_id: 1,
              from: { id: 999, first_name: 'Test User', is_bot: false },
              chat: {
                id: parseInt(MOCK_CHAT_ID),
                title: MOCK_CHAT_TITLE,
                type: 'supergroup',
              },
              date: Math.floor(Date.now() / 1000),
              text: 'Hello bot',
            },
          },
        ]);

      case 'getChat': {
        const chatId = String(body.chat_id || '');
        if (chatId === MOCK_CHAT_ID) {
          return jsonResponse({
            id: parseInt(MOCK_CHAT_ID),
            title: MOCK_CHAT_TITLE,
            type: 'supergroup',
          });
        }
        return errorResponse(`Chat not found: ${chatId}`, 400);
      }

      case 'sendMessage': {
        const chatId = String(body.chat_id || '');
        const text = String(body.text || '');
        receivedMessages.push({ chatId, text, timestamp: Date.now() });
        console.log(`[MOCK] Message received for chat ${chatId}: ${text.slice(0, 80)}...`);
        return jsonResponse({
          message_id: receivedMessages.length,
          chat: { id: parseInt(chatId), type: 'supergroup' },
          text,
          date: Math.floor(Date.now() / 1000),
        });
      }

      // Test helper: get all received messages
      case 'getReceivedMessages':
        return jsonResponse(receivedMessages);

      default:
        return errorResponse(`Unknown method: ${method}`, 404);
    }
  },
});

console.log(`[MOCK] Telegram API server listening on port ${PORT}`);
console.log(`[MOCK] Bot username: @${MOCK_BOT_USERNAME}`);
console.log(`[MOCK] Chat ID: ${MOCK_CHAT_ID}`);
