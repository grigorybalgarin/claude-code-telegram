#!/bin/bash
# MCP Filesystem Server + Cloudflare Tunnel + VPS sync

set -e

PORT=8811
VPS_HOST="194.87.44.239"
VPS_USER="root"
VPS_MCP_CONFIG="/root/ClaudeBot/config/mcp.json"
VPS_BOT_NAME="claude-bot"
ALLOWED_DIR="/Users/grigorybalgarin"
TUNNEL_LOG="/tmp/cf-tunnel.log"

echo "🚀 Запуск MCP файлового сервера..."

# Запуск supergateway (MCP -> HTTP/SSE)
supergateway \
  --stdio "npx -y @modelcontextprotocol/server-filesystem $ALLOWED_DIR" \
  --port $PORT \
  --baseUrl "http://localhost:$PORT" \
  --ssePath /sse \
  --messagePath /message &

SUPERGATEWAY_PID=$!
echo "✓ supergateway запущен (PID: $SUPERGATEWAY_PID, порт: $PORT)"

sleep 2

echo "🌐 Запуск Cloudflare туннеля..."

# Запуск туннеля, пишем вывод в лог
cloudflared tunnel --url "http://localhost:$PORT" > "$TUNNEL_LOG" 2>&1 &
CF_PID=$!

echo "⏳ Ожидание URL туннеля..."

# Ждём пока появится URL
TUNNEL_URL=""
for i in $(seq 1 30); do
  TUNNEL_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
  if [ -n "$TUNNEL_URL" ]; then
    break
  fi
  sleep 1
done

if [ -z "$TUNNEL_URL" ]; then
  echo "❌ Не удалось получить URL туннеля"
  cat "$TUNNEL_LOG"
  kill $SUPERGATEWAY_PID $CF_PID 2>/dev/null
  exit 1
fi

echo "✓ Туннель: $TUNNEL_URL"

# Обновляем конфиг на VPS
echo "🔄 Обновляем конфиг на VPS..."

sshpass -p 'xHEN3hy-k_4eX3' ssh -o StrictHostKeyChecking=no "$VPS_USER@$VPS_HOST" \
  "cat > $VPS_MCP_CONFIG << 'JSON'
{
  \"mcpServers\": {
    \"filesystem\": {
      \"type\": \"sse\",
      \"url\": \"${TUNNEL_URL}/sse\"
    }
  }
}
JSON
pm2 restart $VPS_BOT_NAME --update-env"

echo "✓ VPS обновлён, бот перезапущен"
echo ""
echo "======================================"
echo "✅ Всё готово!"
echo "   MCP URL: $TUNNEL_URL/sse"
echo "   Доступ к: $ALLOWED_DIR"
echo "   Бот на VPS подключён"
echo "======================================"
echo ""
echo "Нажми Ctrl+C для остановки"

# Ждём завершения
wait $CF_PID
