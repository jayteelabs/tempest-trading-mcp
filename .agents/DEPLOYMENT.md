# Deployment Notes

Generated: 2026-04-20

## Container Deployment

**Service:** `tempest-tradingview-mcp` (systemd-managed Docker container)

**Restart command:**
```bash
bash /home/tempest/bin/docker-cmd restart tempest-tradingview-mcp
```
or via systemd:
```bash
sudo systemctl restart tempest-tradingview-mcp
```

**Image:** `tempest-tradingview-mcp:latest` — built from `Dockerfile` in project root.

**Container name:** `tempest-tradingview-mcp`

**Important Docker networking note:** the MCP server must bind to `0.0.0.0` inside the container. Binding to `127.0.0.1` only makes SSE/MCP work from inside the container itself and breaks host-side access through Docker port publishing on `9001`.

**Expected live endpoint behavior:**
- SSE endpoint: `http://127.0.0.1:9001/sse`
- Message endpoint advertised by SSE: `/messages/`
- Host-side sanity check should use a real MCP handshake, not just a TCP port check

## Docker Socket Access

The `tempest` user is in the `docker` group, but shell sessions don't dynamically inherit group membership after login. Use `sg docker -c "docker ..."` to run docker commands in a fresh group context.

For convenience, use the wrapper at `/home/tempest/bin/docker-cmd`:
```bash
bash /home/tempest/bin/docker-cmd <docker args>
```

This is already allowed under Souei's `bash ~/bin/*` permission.

## Smoke Test

After restart, verify the container is healthy:
```bash
bash /home/tempest/bin/docker-cmd exec tempest-tradingview-mcp python -c "import tempest_mcp"
```

## Rebuild Image (if needed)

```bash
cd /home/tempest/apps/tempest-tradingview-mcp
bash /home/tempest/bin/docker-cmd build -t tempest-tradingview-mcp:latest .
sudo systemctl restart tempest-tradingview-mcp
```
