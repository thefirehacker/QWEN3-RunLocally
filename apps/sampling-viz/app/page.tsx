"use client";

import { useEffect, useRef, useState } from "react";

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

export default function Home() {
  const [connected, setConnected] = useState(false);
  const [event, setEvent] = useState<SampleEvent | null>(null);
  const [timeline, setTimeline] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:3847";
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

    function connect() {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (msg) => {
        try {
          const data: SampleEvent = JSON.parse(msg.data);
          if (data.type === "sample_step") {
            setEvent(data);
            setTimeline((prev) => [...prev.slice(-200), data.chosen_piece]);
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
    if (timelineRef.current) {
      timelineRef.current.scrollTop = timelineRef.current.scrollHeight;
    }
  }, [timeline]);

  if (!event) {
    return (
      <div className="container">
        <header>
          <h1>Qwen3 Sampling Visualizer</h1>
          <div className="status">
            <div className={`status-dot ${connected ? "connected" : ""}`} />
            {connected ? "Connected to bridge" : "Waiting for bridge..."}
          </div>
        </header>
        <div className="waiting">
          <h2>Waiting for sampling events...</h2>
          <p>Start the bridge and inference:</p>
          <code>node tools/sampling-bridge.mjs ./run model.gguf -v 1 -t 0.6 -p 0.95</code>
          <p style={{ marginTop: 8, fontSize: "0.8rem", color: "var(--text-dim)" }}>
            Then ask a question in the chat to see live sampling data here.
          </p>
        </div>
      </div>
    );
  }

  const maxProb = Math.max(...event.top.map((t) => t.prob), 0.01);

  return (
    <div className="container">
      <header>
        <h1>Qwen3 Sampling Visualizer</h1>
        <div className="status">
          <div className={`status-dot ${connected ? "connected" : ""}`} />
          {connected ? "Live" : "Disconnected"}
        </div>
      </header>

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
        <h2>Generated Tokens</h2>
        <div className="timeline-tokens" ref={timelineRef}>
          {timeline.map((tok, i) => (
            <span className="timeline-token" key={i}>
              {tok === " " ? "\u2423" : tok === "\n" ? "\\n" : tok}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
