import React, { useEffect, useMemo, useRef, useState } from "react";

const LEVELS = [
  { id: 0, label: "L0 local" },
  { id: 1, label: "L1 always-on" },
  { id: 3, label: "L3 selective" },
];

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  if (import.meta.env.DEV) {
    return `${proto}://${location.host}/ws`;
  }
  return `${proto}://${location.host}/ws`;
}

function layoutNodes(nodes, clusters, width, height) {
  const cx = width / 2;
  const cy = height / 2;
  const clusterR = Math.min(width, height) * 0.28;
  const positions = {};
  const byCluster = {};
  for (const n of nodes) {
    (byCluster[n.cluster] ??= []).push(n);
  }
  for (let c = 0; c < clusters; c++) {
    const angle = (2 * Math.PI * c) / clusters - Math.PI / 2;
    const clusterX = cx + clusterR * Math.cos(angle);
    const clusterY = cy + clusterR * Math.sin(angle);
    const members = byCluster[c] || [];
    const localR = 70 + members.length * 4;
    members.forEach((n, i) => {
      const a = (2 * Math.PI * i) / Math.max(members.length, 1) - Math.PI / 2;
      positions[n.id] = {
        x: clusterX + localR * Math.cos(a),
        y: clusterY + localR * Math.sin(a),
      };
    });
  }
  return positions;
}

function outcomeColor(outcome) {
  if (outcome === "hit") return "var(--hit)";
  if (outcome === "forward") return "var(--fwd)";
  if (outcome === "cloud") return "var(--cloud)";
  return "#5a6a80";
}

export default function App() {
  const [state, setState] = useState(null);
  const [log, setLog] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      const frame = JSON.parse(ev.data);
      setState(frame);
      if (frame.nodes) {
        const events = [];
        for (const n of frame.nodes) {
          if (n.outcome && n.outcome !== "none" && n.requested != null) {
            events.push({
              t: frame.timestep,
              text: `n${n.id} req ${n.requested} → ${n.outcome}`,
              cls: n.outcome,
            });
          }
          if (frame.comm_level === 3 && n.communicated) {
            events.push({
              t: frame.timestep,
              text: `n${n.id} communicated`,
              cls: "comm",
            });
          }
        }
        if (events.length) {
          setLog((prev) => [...events, ...prev].slice(0, 40));
        }
      }
    };
    return () => ws.close();
  }, []);

  const send = (payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  };

  const width = 1100;
  const height = 560;
  const positions = useMemo(() => {
    if (!state?.nodes) return {};
    return layoutNodes(state.nodes, state.clusters || 3, width, height);
  }, [state]);

  const stats = state?.stats || {};

  return (
    <div className="app">
      <header>
        <div>
          <h1>Edge Cache RL — live demo</h1>
          <p>
            10 nodes · 3 clusters · locality 0.3 · shifting traffic
            {connected ? " · connected" : " · connecting…"}
            {state ? ` · t=${state.timestep}` : ""}
          </p>
        </div>
        <div className="controls">
          {LEVELS.map((lvl) => (
            <button
              key={lvl.id}
              className={`level-btn ${state?.comm_level === lvl.id ? "active" : ""}`}
              onClick={() => send({ cmd: "set_level", level: lvl.id })}
            >
              {lvl.label}
            </button>
          ))}
          <button onClick={() => send({ cmd: state?.paused ? "play" : "pause" })}>
            {state?.paused === false ? "Pause" : "Play"}
          </button>
          <button onClick={() => send({ cmd: "step" })}>Step</button>
          <button onClick={() => send({ cmd: "reset" })}>Reset</button>
        </div>
      </header>

      <div className="stats">
        <div className="stat">
          <div className="label">Hit rate</div>
          <div className="value">{((stats.hit_rate || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="stat">
          <div className="label">Forward rate</div>
          <div className="value">{((stats.forward_rate || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="stat">
          <div className="label">Cloud rate</div>
          <div className="value">{((stats.cloud_rate || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="stat">
          <div className="label">Cache diversity</div>
          <div className="value">{stats.cache_diversity ?? "—"}</div>
        </div>
      </div>

      <div className="stage">
        <svg className="network" viewBox={`0 0 ${width} ${height}`}>
          {(state?.edges || []).map(([a, b]) => {
            const pa = positions[a];
            const pb = positions[b];
            if (!pa || !pb) return null;
            return (
              <line
                key={`${a}-${b}`}
                x1={pa.x}
                y1={pa.y}
                x2={pb.x}
                y2={pb.y}
                stroke="var(--edge)"
                strokeWidth="2"
              />
            );
          })}
          {(state?.nodes || []).map((n) => {
            const p = positions[n.id];
            if (!p) return null;
            const ring = outcomeColor(n.outcome);
            const comm = state.comm_level === 3 && n.communicated;
            return (
              <g key={n.id} transform={`translate(${p.x}, ${p.y})`}>
                <circle
                  r="42"
                  fill={
                    n.cluster === 0
                      ? "var(--cluster-0)"
                      : n.cluster === 1
                        ? "var(--cluster-1)"
                        : "var(--cluster-2)"
                  }
                  stroke={ring}
                  strokeWidth={n.outcome === "none" ? 2 : 4}
                />
                {comm && (
                  <circle
                    r="48"
                    fill="none"
                    stroke="var(--comm)"
                    strokeWidth="2"
                    strokeDasharray="4 3"
                  />
                )}
                <text
                  textAnchor="middle"
                  y="-18"
                  fill="var(--text)"
                  fontSize="12"
                  fontWeight="600"
                >
                  n{n.id}
                </text>
                <text textAnchor="middle" y="-4" fill="var(--muted)" fontSize="9">
                  c{n.cluster}
                  {n.requested != null ? ` · r${n.requested}` : ""}
                </text>
                {(n.cache || []).slice(0, 5).map((cid, i) => (
                  <text
                    key={`${n.id}-${cid}-${i}`}
                    className="cache-slot"
                    textAnchor="middle"
                    y={12 + i * 11}
                    fill={
                      n.requested === cid
                        ? "var(--hit)"
                        : "var(--text)"
                    }
                  >
                    [{cid}]
                  </text>
                ))}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="legend">
        <span>
          <i className="swatch" style={{ background: "var(--hit)" }} /> local hit
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--fwd)" }} /> forward
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--cloud)" }} /> cloud
        </span>
        <span>
          <i className="swatch" style={{ background: "var(--comm)" }} /> L3 communicate
        </span>
      </div>

      <div className="event-log">
        {log.length === 0 && <div>Step or press Play to stream events…</div>}
        {log.map((e, i) => (
          <div key={`${e.t}-${i}`} className={e.cls}>
            t={e.t} {e.text}
          </div>
        ))}
      </div>
    </div>
  );
}
