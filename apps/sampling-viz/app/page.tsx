"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface TopToken {
  id: number;
  piece: string;
  logit_raw: number;
  logit_scaled: number;
  prob: number;
  in_nucleus: boolean;
}

interface SampleEvent {
  type: "sample_step";
  pos: number;
  temperature: number;
  topp: number;
  chosen_id: number;
  chosen_piece: string;
  nucleus_count: number;
  vocab_size: number;
  nucleus_mass: number;
  top: TopToken[];
}

interface PromptContext {
  type: "prompt_context";
  system_prompt: string;
  user_prompt: string;
}

type SessionEvent = SampleEvent | PromptContext;

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<SampleEvent[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(-1);
  const [mode, setMode] = useState<"live" | "replay" | "file">("live");
  const [promptCtx, setPromptCtx] = useState<PromptContext | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  const event = selectedIdx === -1 ? history[history.length - 1] || null : history[selectedIdx];

  useEffect(() => {
    fetch("/sampling-session.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: SessionEvent[]) => {
        if (data && data.length > 0) {
          const samples = data.filter((e): e is SampleEvent => e.type === "sample_step");
          const ctx = data.find((e): e is PromptContext => e.type === "prompt_context");
          if (samples.length > 0) {
            setHistory(samples);
            setMode("file");
            setSelectedIdx(0);
            if (ctx) setPromptCtx(ctx);
          }
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:3847";
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

    function connect() {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setMode("live");
        setSelectedIdx(-1);
      };
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data.type === "sample_step") {
            setHistory((prev) => [...prev, data]);
            if (mode === "live") setSelectedIdx(-1);
          } else if (data.type === "prompt_context") {
            setPromptCtx(data);
          }
        } catch {}
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  useEffect(() => {
    if (timelineRef.current && mode === "live") {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  }, [history, mode]);

  const handleTokenClick = useCallback((idx: number) => {
    setSelectedIdx(idx);
    setMode("replay");
  }, []);

  const backToLive = useCallback(() => {
    setSelectedIdx(-1);
    setMode("live");
  }, []);

  const handleFileLoad = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data: SessionEvent[] = JSON.parse(ev.target?.result as string);
        const samples = data.filter((e): e is SampleEvent => e.type === "sample_step");
        const ctx = data.find((e): e is PromptContext => e.type === "prompt_context");
        if (samples.length > 0) {
          setHistory(samples);
          setMode("file");
          setSelectedIdx(0);
          if (ctx) setPromptCtx(ctx);
        }
      } catch {}
    };
    reader.readAsText(file);
  }, []);

  const statusText = mode === "live" ? "Live" : mode === "file" ? "File loaded" : `Step ${selectedIdx + 1} / ${history.length}`;
  const statusClass = connected ? "connected" : mode === "file" ? "file-mode" : "";

  if (!event) {
    return (
      <div className="container">
        <Header statusClass={statusClass} statusText={connected ? "Connected" : "Offline"} />
        <div className="waiting">
          <h2>Waiting for sampling events...</h2>
          <p>Start the bridge and inference:</p>
          <code>node tools/sampling-bridge.mjs ./run_viz model.gguf -v 1 -t 0.6 -p 0.95</code>
          <p style={{ marginTop: 8, fontSize: "0.8rem", color: "var(--text-dim)" }}>
            Or load a saved session file:
          </p>
          <div className="file-drop">
            <label>
              Drop or click to load sampling-session.json
              <input type="file" accept=".json" onChange={handleFileLoad} />
            </label>
          </div>
        </div>
        <AppFooter />
      </div>
    );
  }

  const maxProb = Math.max(...event.top.map((t) => t.prob), 0.01);

  return (
    <div className="container">
      <Header statusClass={statusClass} statusText={statusText} />

      {mode === "replay" && (
        <div className="replay-banner">
          <span className="step-info">
            Replaying step {selectedIdx + 1} of {history.length}
          </span>
          <button onClick={backToLive}>Back to Live</button>
        </div>
      )}
      {mode === "file" && (
        <div className="replay-banner">
          <span className="step-info">
            Loaded session: step {selectedIdx + 1} of {history.length} — click tokens to explore
          </span>
          <label style={{ cursor: "pointer", fontSize: "0.75rem", color: "var(--accent)" }}>
            Load another file
            <input type="file" accept=".json" onChange={handleFileLoad} style={{ display: "none" }} />
          </label>
        </div>
      )}

      {promptCtx && (promptCtx.system_prompt || promptCtx.user_prompt) && (
        <div className="prompt-context">
          {promptCtx.system_prompt && (
            <span>
              <span className="ctx-label">System:</span>
              <span className="ctx-value">{promptCtx.system_prompt}</span>
            </span>
          )}
          {promptCtx.user_prompt && (
            <span>
              <span className="ctx-label">Q:</span>
              <span className="ctx-value">{promptCtx.user_prompt}</span>
            </span>
          )}
        </div>
      )}

      <div className="params">
        <span>
          <span className="label">Temperature</span>
          <span className="value">{event.temperature.toFixed(2)}</span>
        </span>
        <span>
          <span className="label">Top-P</span>
          <span className="value">{event.topp.toFixed(2)}</span>
        </span>
        <span>
          <span className="label">Position</span>
          <span className="value">{event.pos}</span>
        </span>
        <span>
          <span className="label">Chosen</span>
          <span className="value" style={{ color: "var(--chosen)" }}>
            &quot;{event.chosen_piece}&quot;
          </span>
        </span>
      </div>

      <div className="grid">
        <div className="card">
          <h2>Top-{event.top.length} Token Probabilities (after temperature scaling)</h2>
          <div className="bar-chart">
            <div className="bar-row" style={{ fontWeight: 600, color: "var(--text-dim)", fontSize: "0.7rem" }}>
              <span>Token</span>
              <span></span>
              <span style={{ textAlign: "right" }}>Prob</span>
              <span style={{ textAlign: "right" }}>Logit</span>
            </div>
            {event.top.map((tok) => {
              const isChosen = tok.id === event.chosen_id;
              const barClass = isChosen
                ? "bar-fill chosen"
                : tok.in_nucleus
                ? "bar-fill in-nucleus"
                : "bar-fill out-nucleus";
              return (
                <div className="bar-row" key={tok.id}>
                  <span className="token" title={tok.piece}>
                    {isChosen ? ">>> " : ""}
                    {JSON.stringify(tok.piece)}
                  </span>
                  <div className="bar-container">
                    <div className={barClass} style={{ width: `${(tok.prob / maxProb) * 100}%` }} />
                  </div>
                  <span className="prob">{(tok.prob * 100).toFixed(1)}%</span>
                  <span className="logit">{tok.logit_raw.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
          <div className="legend">
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: "var(--chosen)" }} />
              Chosen
            </div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: "var(--nucleus)" }} />
              In Nucleus
            </div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: "var(--tail)" }} />
              Tail (excluded by top-p)
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div className="card">
            <h2>Nucleus Stats</h2>
            <div className="nucleus-info">
              <div className="nucleus-stat">
                <span className="stat-label">Nucleus size</span>
                <span className="stat-value">
                  {event.nucleus_count} / {event.vocab_size.toLocaleString()}
                </span>
              </div>
              <div className="nucleus-stat">
                <span className="stat-label">Nucleus mass</span>
                <span className="stat-value">{(event.nucleus_mass * 100).toFixed(1)}%</span>
              </div>
              <div className="nucleus-stat">
                <span className="stat-label">Top-p cutoff</span>
                <span className="stat-value">{(event.topp * 100).toFixed(0)}%</span>
              </div>
              <div className="nucleus-stat">
                <span className="stat-label">Tail excluded</span>
                <span className="stat-value">
                  {(event.vocab_size - event.nucleus_count).toLocaleString()} tokens
                </span>
              </div>
              <div className="nucleus-stat">
                <span className="stat-label">Tail mass</span>
                <span className="stat-value">
                  {((1 - event.nucleus_mass) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          <div className="card">
            <h2>How Temperature Works Here</h2>
            <div style={{ fontSize: "0.8rem", color: "var(--text-dim)", lineHeight: 1.6 }}>
              <p>
                <strong style={{ color: "var(--text)" }}>T = {event.temperature.toFixed(2)}</strong>:
                each raw logit is divided by {event.temperature.toFixed(2)} before softmax.
              </p>
              <p style={{ marginTop: 8 }}>
                Lower T {"->"} sharper distribution (top token dominates).
                Higher T {"->"} flatter (more randomness).
              </p>
              <p style={{ marginTop: 8 }}>
                Top token raw logit: <strong>{event.top[0]?.logit_raw.toFixed(2)}</strong>,
                scaled: <strong>{event.top[0]?.logit_scaled.toFixed(2)}</strong>
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="card timeline">
        <h2>Generated Tokens (click to replay)</h2>
        <div className="timeline-tokens" ref={timelineRef}>
          {history.map((evt, i) => (
            <span
              className={`timeline-token ${i === selectedIdx ? "active" : ""}`}
              key={i}
              onClick={() => handleTokenClick(i)}
              title={`Step ${i + 1}, pos=${evt.pos}`}
            >
              {evt.chosen_piece === " " ? "\u2423" : evt.chosen_piece === "\n" ? "\\n" : evt.chosen_piece}
            </span>
          ))}
        </div>
      </div>

      <AppFooter />
    </div>
  );
}

function Header({ statusClass, statusText }: { statusClass: string; statusText: string }) {
  return (
    <header>
      <div className="header-left">
        <img src="/favicon-32x32.png" alt="First Break AI" />
        <div>
          <h1>
            <span>First Break AI</span> — Sampling Visualizer
          </h1>
          <div className="header-subtitle">Step 2: Understanding How LLMs Choose Tokens</div>
        </div>
      </div>
      <div className="header-right">
        <div className="header-links">
          <a href="https://cohort.bubblnet.com/" target="_blank" rel="noopener">Cohort</a>
          <a href="https://discord.gg/hRPese4H3F" target="_blank" rel="noopener">Discord</a>
          <a href="https://github.com/thefirehacker/firstbreakai" target="_blank" rel="noopener">GitHub</a>
        </div>
        <div className="status">
          <div className={`status-dot ${statusClass}`} />
          {statusText}
        </div>
      </div>
    </header>
  );
}

function AppFooter() {
  return (
    <footer className="app-footer">
      &copy; 2026 <a href="https://cohort.bubblnet.com/">First Break AI</a> &mdash;
      Powered by <a href="https://fetchlens.ai">fetchlens.ai</a> |{" "}
      <a href="https://github.com/thefirehacker/qwen3.c">qwen3.c</a>
    </footer>
  );
}
