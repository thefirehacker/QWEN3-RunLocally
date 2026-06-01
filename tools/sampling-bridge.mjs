#!/usr/bin/env node
/**
 * Sampling Bridge: spawns ./run with -v 1, parses JSONL sampling events from
 * stderr, and broadcasts them over WebSocket to the Next.js dashboard.
 *
 * Usage:
 *   node tools/sampling-bridge.mjs ./run Qwen3-0.6B-FP32.gguf -v 1 -t 0.6 -p 0.95
 *
 * The bridge:
 *  - Pipes stdout (chat text) directly to your terminal
 *  - Parses stderr for lines matching the \x1eSAMPLE\x1e{...} marker
 *  - Non-SAMPLE stderr lines (TPS, TTFT, errors) pass through to terminal
 *  - Broadcasts parsed JSON to all WebSocket clients on port 3847
 *  - stdin is forwarded to the child process (for chat interaction)
 */

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { WebSocketServer } from "ws";

const MARKER = "\x1eSAMPLE\x1e";
const WS_PORT = parseInt(process.env.VIZ_PORT || "3847", 10);

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: node tools/sampling-bridge.mjs <run-binary> [args...]");
  console.error("Example: node tools/sampling-bridge.mjs ./run model.gguf -v 1 -t 0.6");
  process.exit(1);
}

if (!args.includes("-v")) {
  args.push("-v", "1");
}

const httpServer = createServer((_, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("sampling-bridge ws://localhost:" + WS_PORT);
});
const wss = new WebSocketServer({ server: httpServer });

const clients = new Set();
wss.on("connection", (ws) => {
  clients.add(ws);
  ws.on("close", () => clients.delete(ws));
  ws.on("error", () => clients.delete(ws));
});

function broadcast(jsonStr) {
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      ws.send(jsonStr);
    }
  }
}

httpServer.listen(WS_PORT, () => {
  console.error(`[bridge] WebSocket server on ws://localhost:${WS_PORT}`);
});

const child = spawn(args[0], args.slice(1), {
  stdio: ["pipe", "pipe", "pipe"],
  cwd: process.cwd(),
});

process.stdin.pipe(child.stdin);
child.stdout.pipe(process.stdout);

let stderrBuffer = "";
child.stderr.on("data", (chunk) => {
  stderrBuffer += chunk.toString();
  let lines = stderrBuffer.split("\n");
  stderrBuffer = lines.pop();

  for (const line of lines) {
    if (line.startsWith(MARKER)) {
      const jsonStr = line.slice(MARKER.length);
      broadcast(jsonStr);
    } else if (line.length > 0) {
      process.stderr.write(line + "\n");
    }
  }
});

child.on("close", (code) => {
  console.error(`[bridge] ./run exited with code ${code}`);
  httpServer.close();
  process.exit(code || 0);
});

child.on("error", (err) => {
  console.error(`[bridge] Failed to start: ${err.message}`);
  process.exit(1);
});
