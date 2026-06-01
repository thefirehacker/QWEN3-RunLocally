#!/usr/bin/env node
/**
 * Sampling Bridge: spawns ./run with -v 1, parses JSONL sampling events from
 * stderr, and broadcasts them over WebSocket to the Next.js dashboard.
 *
 * Features:
 *  - Pipes stdout (chat text) directly to your terminal
 *  - Parses stderr for lines matching the \x1eSAMPLE\x1e{...} marker
 *  - Non-SAMPLE stderr lines (TPS, TTFT, errors) pass through to terminal
 *  - Broadcasts parsed JSON to all WebSocket clients on port 3847
 *  - Saves all events to sampling-session.json (rewritten each run) for offline replay
 *  - Parses stdout for prompt context (system prompt, user question) and broadcasts
 *  - stdin is forwarded to the child process (for chat interaction)
 *
 * Usage:
 *   node tools/sampling-bridge.mjs ./run Qwen3-0.6B-FP32.gguf -v 1 -t 0.6 -p 0.95
 */

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MARKER = "\x1eSAMPLE\x1e";
const WS_PORT = parseInt(process.env.VIZ_PORT || "3847", 10);
const SESSION_PATH = process.env.SESSION_PATH || resolve(__dirname, "../apps/sampling-viz/public/sampling-session.json");

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: node tools/sampling-bridge.mjs <run-binary> [args...]");
  console.error("Example: node tools/sampling-bridge.mjs ./run model.gguf -v 1 -t 0.6");
  process.exit(1);
}

if (!args.includes("-v")) {
  args.push("-v", "1");
}

const sessionEvents = [];
let promptContext = { system_prompt: "", user_prompt: "" };

function saveSession() {
  try {
    writeFileSync(SESSION_PATH, JSON.stringify(sessionEvents, null, 0));
  } catch {}
}

const httpServer = createServer((_, res) => {
  res.writeHead(200, { "Content-Type": "text/plain" });
  res.end("sampling-bridge ws://localhost:" + WS_PORT);
});
const wss = new WebSocketServer({ server: httpServer });

const clients = new Set();
wss.on("connection", (ws) => {
  clients.add(ws);
  if (promptContext.user_prompt) {
    ws.send(JSON.stringify({ type: "prompt_context", ...promptContext }));
  }
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
  console.error(`[bridge] Session file: ${SESSION_PATH}`);
});

const child = spawn(args[0], args.slice(1), {
  stdio: ["pipe", "pipe", "pipe"],
  cwd: process.cwd(),
});

process.stdin.pipe(child.stdin);

let stdoutBuffer = "";
child.stdout.on("data", (chunk) => {
  const text = chunk.toString();
  process.stdout.write(text);

  stdoutBuffer += text;
  const lines = stdoutBuffer.split("\n");
  stdoutBuffer = lines.pop();

  for (const line of lines) {
    if (line.startsWith("Enter system prompt")) {
      // Next user input will be system prompt — captured on Q: line
    } else if (line.startsWith("Q: ") || line.match(/^Q:\s/)) {
      // Capture everything after "Q: " but the actual input comes from stdin
    }
  }
});

let captureNextAsSystem = false;
let captureNextAsUser = false;

const origStdinWrite = child.stdin.write.bind(child.stdin);
child.stdin.write = function (data, ...rest) {
  const str = data.toString();
  const trimmed = str.trim();

  if (captureNextAsSystem) {
    promptContext.system_prompt = trimmed;
    captureNextAsSystem = false;
  } else if (captureNextAsUser) {
    promptContext.user_prompt = trimmed;
    captureNextAsUser = false;
    const ctx = JSON.stringify({ type: "prompt_context", ...promptContext });
    broadcast(ctx);
    sessionEvents.push(JSON.parse(ctx));
    saveSession();
  }

  return origStdinWrite(data, ...rest);
};

child.stdout.on("data", (chunk) => {
  const text = chunk.toString();
  if (text.includes("Enter system prompt")) {
    captureNextAsSystem = true;
  } else if (text.includes("Q:")) {
    captureNextAsUser = true;
  }
});

let stderrBuffer = "";
child.stderr.on("data", (chunk) => {
  stderrBuffer += chunk.toString();
  let lines = stderrBuffer.split("\n");
  stderrBuffer = lines.pop();

  for (const line of lines) {
    if (line.startsWith(MARKER)) {
      const jsonStr = line.slice(MARKER.length);
      broadcast(jsonStr);
      try {
        sessionEvents.push(JSON.parse(jsonStr));
        saveSession();
      } catch {}
    } else if (line.length > 0) {
      process.stderr.write(line + "\n");
    }
  }
});

child.on("close", (code) => {
  saveSession();
  console.error(`[bridge] ./run exited with code ${code}`);
  console.error(`[bridge] Session saved: ${sessionEvents.length} events`);
  httpServer.close();
  process.exit(code || 0);
});

child.on("error", (err) => {
  console.error(`[bridge] Failed to start: ${err.message}`);
  process.exit(1);
});
