# Deployment Notes

Generated: 2026-04-22

## Container Deployment

**Service:** `tempest-tradingview-mcp` (Docker Compose-managed container)

**Start command:**
```bash
cd /home/tempest/apps/tempest-tradingview-mcp
cp .env.example .env
sg docker -c "docker compose up -d --build"
```

**Stop command:**
```bash
cd /home/tempest/apps/tempest-tradingview-mcp
sg docker -c "docker compose down"
```

**View logs:**
```bash
cd /home/tempest/apps/tempest-tradingview-mcp
sg docker -c "docker compose logs -f"
```

**Image:** `tempest-tradingview-mcp:latest` — built from `Dockerfile` in project root.

**Container naming:** Docker Compose uses its default project-scoped container names. This avoids conflicts with older standalone `docker run --name tempest-tradingview-mcp ...` containers during migration/restarts.

**Important Docker networking note:** the MCP server must bind to `0.0.0.0` inside the container. Binding to `127.0.0.1` only makes SSE/MCP work from inside the container itself and breaks host-side access through Docker port publishing on `9001`.

**Exposure posture:** the repo-owned Compose config publishes `127.0.0.1:9001:9001` by default so the unauthenticated SSE + `/messages` surface stays local to the host. If remote access is needed, expose intentionally behind a reverse proxy, Tailscale, firewall rules, or equivalent trusted-network controls.

**Expected live endpoint behavior:**
- SSE endpoint: `http://127.0.0.1:9001/sse`
- Message endpoint: `http://127.0.0.1:9001/messages`
- Host-side sanity check should use a real MCP handshake, not just a TCP port check

**Fresh-clone validation (without creating `.env`):**
```bash
cd /home/tempest/apps/tempest-tradingview-mcp
sg docker -c "TEMPEST_ENV_FILE=.env.example docker compose config"
```

## Docker Socket Access

The `tempest` user is in the `docker` group, but shell sessions don't dynamically inherit group membership after login. Use `sg docker -c "docker ..."` to run docker commands in a fresh group context.

## Smoke Test

After startup, verify the container is healthy:
```bash
sg docker -c "docker compose ps"
```

The service includes a Docker healthcheck that verifies the SSE endpoint responds on `http://localhost:9001/sse`.

## Rebuild Image (if needed)

```bash
cd /home/tempest/apps/tempest-tradingview-mcp
sg docker -c "docker compose build"
sg docker -c "docker compose up -d"
```
