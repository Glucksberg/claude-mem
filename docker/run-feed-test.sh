#!/bin/bash
# Feed E2E test runner
# Starts mock Telegram, runs wizard in non-interactive mode, verifies results.

set -e

echo "=== Claude-Mem Feed E2E Test ==="
echo ""

# 1. Start mock Telegram API
echo "Starting mock Telegram API..."
bun docker/mock-telegram-server.ts &
MOCK_PID=$!
sleep 1

# Verify mock is running
if ! curl -s http://localhost:8443/botTEST/getMe > /dev/null 2>&1; then
  echo "FAIL: Mock Telegram server did not start"
  kill $MOCK_PID 2>/dev/null
  exit 1
fi
echo "Mock Telegram API started on port 8443"

# 2. Run wizard in non-interactive mode
echo ""
echo "Running feed setup wizard (non-interactive)..."
TELEGRAM_API_BASE=http://localhost:8443 \
  bun plugin/scripts/worker-service.cjs feed setup \
    --non-interactive \
    --bot-token=123456789:ABCdefGHIjklMNOpqrsTUVwxyz1234567890abc \
    --chat-id=-1001234567890

echo "Wizard completed"

# 3. Verify config was saved
echo ""
echo "Verifying config..."
if [ -f /tmp/claude-mem/settings.json ]; then
  echo "Settings file exists"
  if grep -q "CLAUDE_MEM_FEED_ENABLED" /tmp/claude-mem/settings.json; then
    echo "Feed settings found in config"
  else
    echo "FAIL: Feed settings not found in config"
    kill $MOCK_PID 2>/dev/null
    exit 1
  fi
else
  echo "FAIL: Settings file not created"
  kill $MOCK_PID 2>/dev/null
  exit 1
fi

# 4. Verify test message was sent to mock
echo ""
echo "Verifying test message..."
MESSAGES=$(curl -s http://localhost:8443/botTEST/getReceivedMessages)
if echo "$MESSAGES" | grep -q "Claude-Mem Feed connected"; then
  echo "Test message verified!"
else
  echo "FAIL: Test message not found in mock server"
  kill $MOCK_PID 2>/dev/null
  exit 1
fi

# Cleanup
kill $MOCK_PID 2>/dev/null

echo ""
echo "=== All tests passed! ==="
