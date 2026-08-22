import React, { useEffect, useMemo, useRef, useState } from "react";

const LEVELS = [
  { id: 0, label: "L0 local" },
  { id: 1, label: "L1 always-on" },
  { id: 3, label: "L3 selective" },
];

const CACHE_CAP = 5;
const CLUSTER_FILL = ["#1e3348", "#1e3a32", "#3a2432"];
const CLUSTER_STROKE = ["#3d5f82", "#3d7a62", "#7a4a5e"];
const CLUSTER_GLOW = ["rgba(91,159,212,0.12)", "rgba(61,214,140,0.10)", "rgba(232,93,93,0.10)"];

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  if (import.meta.env.DEV) {
    return `${proto}://${location.host}/ws`;
  }
  return `${proto}://${location.host}/ws`;
}

function outcomeColor(outcome) {
  if (outcome === "hit") return "var(--hit)";
  if (outcome === "forward") return "var(--fwd)";
  if (outcome === "cloud") return "var(--cloud)";
  return "#4a5a72";
}

/**
 * Column layout: one vertical band per cluster, nodes packed on a grid that
 * fills the band. Bridge endpoints sit on facing edges so inter-cluster
 * links are short horizontal curves and do not cross.
 */
function layoutNetwork(nodes, edges, clusters, width, height) {
  const pad = 18;
  const gutter = 28;
  const bandW = (width - pad * 2 - gutter * (clusters - 1)) / clusters;
  const byCluster = {};
  for (const n of nodes) {
    (byCluster[n.cluster] ??= []).push(n);
  }
  for (const list of Object.values(byCluster)) {
    list.sort((a, b) => a.id - b.id);
  }

  // Prefer bridge nodes toward facing column edges.
  const bridgeRight = new Set(); // touches a higher-cluster neighbor
  const bridgeLeft = new Set();
  const clusterOf = Object.fromEntries(nodes.map((n) => [n.id, n.cluster]));
  for (const [a, b] of edges) {
    const ca = clusterOf[a];
    const cb = clusterOf[b];
    if (ca == null || cb == null || ca === cb) continue;
    if (ca < cb) {
      bridgeRight.add(a);
      bridgeLeft.add(b);
    } else {
      bridgeRight.add(b);
      bridgeLeft.add(a);
    }
  }

  function orderMembers(members, clusterId) {
    const score = (n) => {
      let s = 0;
      if (bridgeRight.has(n.id)) s += 2;
      if (bridgeLeft.has(n.id)) s -= 2;
      // Middle cluster: prefer dual-bridge node centered.
      if (clusterId === 1 && bridgeLeft.has(n.id) && bridgeRight.has(n.id)) s = 0;
      return s;
    };
    return [...members].sort((a, b) => score(a) - score(b) || a.id - b.id);
  }

  const bands = [];
  const positions = {};
  let nodeW = 110;
  let nodeH = 118;

  for (let c = 0; c < clusters; c++) {
    const x0 = pad + c * (bandW + gutter);
    const members = orderMembers(byCluster[c] || [], c);
    const n = members.length;
    const cols = n <= 1 ? 1 : n <= 2 ? 1 : 2;
    const rows = Math.ceil(n / cols);

    const innerPadX = 16;
    const innerPadY = 36; // room for cluster title
    const availW = bandW - innerPadX * 2;
    const availH = height - pad * 2 - innerPadY - 12;
    const gapX = 14;
    const gapY = 16;
    const cellW = (availW - gapX * (cols - 1)) / cols;
    const cellH = (availH - gapY * (rows - 1)) / rows;
    // Shared card size across clusters (use min so everything matches).
    const candW = Math.min(128, Math.max(84, cellW * 0.92));
    const candH = Math.min(136, Math.max(92, cellH * 0.88));
    if (c === 0) {
      nodeW = candW;
      nodeH = candH;
    } else {
      nodeW = Math.min(nodeW, candW);
      nodeH = Math.min(nodeH, candH);
    }

    bands.push({
      id: c,
      x: x0,
      y: pad,
      w: bandW,
      h: height - pad * 2,
      titleY: pad + 18,
    });

    // Place into grid; for odd last row, center the orphan.
    members.forEach((node, i) => {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const rowCount = row === rows - 1 ? n - row * cols : cols;
      const rowWidth = rowCount * nodeW + (rowCount - 1) * gapX;
      const startX = x0 + (bandW - rowWidth) / 2;
      const blockH = rows * nodeH + (rows - 1) * gapY;
      const startY = pad + innerPadY + (availH - blockH) / 2;
      positions[node.id] = {
        x: startX + col * (nodeW + gapX) + nodeW / 2,
        y: startY + row * (nodeH + gapY) + nodeH / 2,
        cluster: c,
      };
    });
  }

  // Second pass with finalized nodeW/H so packing is consistent.
  for (let c = 0; c < clusters; c++) {
    const x0 = pad + c * (bandW + gutter);
    const members = orderMembers(byCluster[c] || [], c);
    const n = members.length;
    const cols = n <= 1 ? 1 : n <= 2 ? 1 : 2;
    const rows = Math.ceil(n / cols);
    const gapX = 14;
    const gapY = 16;
    const innerPadY = 36;
    const availH = height - pad * 2 - innerPadY - 12;
    members.forEach((node, i) => {
      const row = Math.floor(i / cols);
      const col = i % cols;
      const rowCount = row === rows - 1 ? n - row * cols : cols;
      const rowWidth = rowCount * nodeW + (rowCount - 1) * gapX;
      const startX = x0 + (bandW - rowWidth) / 2;
      const blockH = rows * nodeH + (rows - 1) * gapY;
      const startY = pad + innerPadY + Math.max(0, (availH - blockH) / 2);
      positions[node.id] = {
        x: startX + col * (nodeW + gapX) + nodeW / 2,
        y: startY + row * (nodeH + gapY) + nodeH / 2,
        cluster: c,
      };
    });
  }

  return { positions, bands, nodeW, nodeH };
}

function edgePath(pa, pb, sameCluster) {
  if (sameCluster) {
    const mx = (pa.x + pb.x) / 2;
    const my = (pa.y + pb.y) / 2;
    // Slight outward bow so links don't sit under cards.
    const dx = pb.x - pa.x;
    const dy = pb.y - pa.y;
    const len = Math.hypot(dx, dy) || 1;
    const ox = (-dy / len) * 12;
    const oy = (dx / len) * 12;
    return `M ${pa.x} ${pa.y} Q ${mx + ox} ${my + oy} ${pb.x} ${pb.y}`;
  }
  // Inter-cluster: horizontal cubic through the gutter (no crossings).
  const mx = (pa.x + pb.x) / 2;
  return `M ${pa.x} ${pa.y} C ${mx} ${pa.y}, ${mx} ${pb.y}, ${pb.x} ${pb.y}`;
}

/** Cumulative episode-return sparkline; labels the running high. */
function ReturnChart({ series, width = 420, height = 72 }) {
  if (!series.length) {
    return (
      <div className="return-chart empty">
        <div className="label">Episode return</div>
        <div className="hint">Play to plot cumulative return</div>
      </div>
    );
  }
  const padL = 8;
  const padR = 52;
  const padT = 14;
  const padB = 6;
  const vals = series.map((p) => p.y);
  const ymin = Math.min(0, ...vals);
  const ymax = Math.max(1, ...vals);
  const span = ymax - ymin || 1;
  const xAt = (i) =>
    padL + (i / Math.max(series.length - 1, 1)) * (width - padL - padR);
  const yAt = (v) => padT + (1 - (v - ymin) / span) * (height - padT - padB);
  const d = series
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i).toFixed(1)} ${yAt(p.y).toFixed(1)}`)
    .join(" ");
  let hiIdx = 0;
  for (let i = 1; i < series.length; i++) {
    if (series[i].y > series[hiIdx].y) hiIdx = i;
  }
  const hi = series[hiIdx];
  const hx = xAt(hiIdx);
  const hy = yAt(hi.y);
  const current = series[series.length - 1].y;

  return (
    <div className="return-chart">
      <div className="return-chart-head">
        <div className="label">Episode return</div>
        <div className="value">{current.toFixed(1)}</div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <line
          x1={padL}
          y1={yAt(0)}
          x2={width - padR}
          y2={yAt(0)}
          stroke="#3a4a63"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
        <path d={d} fill="none" stroke="var(--comm)" strokeWidth="2" />
        <circle cx={hx} cy={hy} r="3.5" fill="var(--comm)" />
        <text
          x={Math.min(hx + 6, width - 4)}
          y={Math.max(hy - 4, 11)}
          fill="var(--comm)"
          fontSize="11"
          fontWeight="700"
        >
          high {hi.y.toFixed(1)}
        </text>
      </svg>
    </div>
  );
}

function CacheSlots({ cache, requested, nodeW }) {
  const slot = Math.max(14, Math.min(20, Math.floor((nodeW - 24) / CACHE_CAP) - 2));
  const gap = 3;
  const totalW = CACHE_CAP * slot + (CACHE_CAP - 1) * gap;
  const x0 = -totalW / 2;
  const y0 = 14;
  const slots = Array.from({ length: CACHE_CAP }, (_, i) =>
    i < cache.length ? cache[i] : null
  );
  return (
    <g>
      {slots.map((cid, i) => {
        const x = x0 + i * (slot + gap);
        const filled = cid != null;
        const isReq = filled && cid === requested;
        return (
          <g key={i}>
            <rect
              x={x}
              y={y0}
              width={slot}
              height={slot}
              rx="3"
              fill={isReq ? "var(--hit)" : filled ? "#243044" : "#0f141c"}
              stroke={isReq ? "#b8f0d0" : filled ? "#8aa0bc" : "#3a4a63"}
              strokeWidth="1.25"
            />
            {filled && (
              <text
                x={x + slot / 2}
                y={y0 + slot / 2 + 3.5}
                textAnchor="middle"
                fontSize={slot >= 17 ? 10 : 8}
                fontWeight="700"
                fill={isReq ? "#062016" : "var(--text)"}
              >
                {cid}
              </text>
            )}
          </g>
        );
      })}
    </g>
  );
}

export default function App() {
  const [state, setState] = useState(null);
  const [log, setLog] = useState([]);
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState({ w: 1180, h: 640 });
  const [returnSeries, setReturnSeries] = useState([]);
  const stageRef = useRef(null);
  const wsRef = useRef(null);
  const seriesKeyRef = useRef("");

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 40 && height > 40) {
        setFrame({ w: width, h: height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => {
      const next = JSON.parse(ev.data);
      setState(next);
      const key = `${next.policy_mode || "dqn"}-${next.comm_level}-${next.seed ?? ""}`;
      const ret = next.stats?.episode_return;
      const t = next.timestep ?? 0;
      if (typeof ret === "number") {
        setReturnSeries((prev) => {
          const reset =
            key !== seriesKeyRef.current ||
            t === 0 ||
            next.episode_done ||
            (prev.length && t < prev[prev.length - 1].t);
          seriesKeyRef.current = key;
          if (reset) {
            return t === 0 && !next.episode_done ? [] : [{ t, y: ret }];
          }
          const last = prev[prev.length - 1];
          if (last && last.t === t) {
            return [...prev.slice(0, -1), { t, y: ret }];
          }
          // Keep chart responsive: downsample if very long.
          const nextPts = [...prev, { t, y: ret }];
          if (nextPts.length > 800) {
            return nextPts.filter((_, i) => i % 2 === 0 || i === nextPts.length - 1);
          }
          return nextPts;
        });
      }
      if (next.nodes) {
        const events = [];
        for (const n of next.nodes) {
          if (n.outcome && n.outcome !== "none" && n.requested != null) {
            events.push({
              t: next.timestep,
              text: `n${n.id} requested: ${n.requested} → ${n.outcome}`,
              cls: n.outcome,
            });
          }
          if (next.comm_level === 3 && n.communicated) {
            events.push({
              t: next.timestep,
              text: `n${n.id} communicated`,
              cls: "comm",
            });
          }
        }
        if (events.length) {
          setLog((prev) => [...events, ...prev].slice(0, 40));
        }
      }
      if (next.episode_done && next.stats?.last_episode_return != null) {
        setLog((prev) =>
          [
            {
              t: next.timestep,
              text: `episode done — return ${next.stats.last_episode_return.toFixed(1)}`,
              cls: "reward",
            },
            ...prev,
          ].slice(0, 40)
        );
      }
    };
    return () => ws.close();
  }, []);

  const send = (payload) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    }
  };

  const layout = useMemo(() => {
    if (!state?.nodes) {
      return { positions: {}, bands: [], nodeW: 110, nodeH: 118 };
    }
    return layoutNetwork(
      state.nodes,
      state.edges || [],
      state.clusters || 3,
      frame.w,
      frame.h
    );
  }, [state, frame]);

  const stats = state?.stats || {};
  const { positions, bands, nodeW, nodeH } = layout;
  const policyLabel =
    state?.policy_mode === "lfu"
      ? "LFU heuristic"
      : state?.comm_level != null
        ? `L${state.comm_level} DQN`
        : "—";

  return (
    <div className="app">
      <header>
        <div>
          <h1>Edge Cache RL — live demo</h1>
          <p>
            10 nodes · 3 clusters · locality 0.3 · shifting traffic
            {connected ? " · connected" : " · connecting…"}
            {state ? ` · t=${state.timestep}` : ""}
            {state ? ` · ${policyLabel}` : ""}
          </p>
        </div>
        <div className="controls">
          {LEVELS.map((lvl) => (
            <button
              key={lvl.id}
              className={`level-btn ${
                state?.policy_mode !== "lfu" && state?.comm_level === lvl.id
                  ? "active"
                  : ""
              }`}
              onClick={() => send({ cmd: "set_level", level: lvl.id })}
            >
              {lvl.label}
            </button>
          ))}
          <button
            className={`level-btn heuristic ${
              state?.policy_mode === "lfu" ? "active" : ""
            }`}
            onClick={() => send({ cmd: "set_policy", policy: "lfu" })}
          >
            LFU heuristic
          </button>
          <button onClick={() => send({ cmd: state?.paused ? "play" : "pause" })}>
            {state?.paused === false ? "Pause" : "Play"}
          </button>
          <button onClick={() => send({ cmd: "step" })}>Step</button>
          <button onClick={() => send({ cmd: "reset" })}>Reset</button>
        </div>
      </header>

      <div className="stats">
        <ReturnChart series={returnSeries} />
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
        {state?.policy_mode === "dqn" && state?.comm_level === 3 && (
          <div className="stat">
            <div className="label">L3 comms</div>
            <div className="value">{stats.comm_events ?? 0}</div>
            <div className="sub">this episode</div>
          </div>
        )}
      </div>

      <div className="stage" ref={stageRef}>
        <svg
          className="network"
          viewBox={`0 0 ${frame.w} ${frame.h}`}
          preserveAspectRatio="none"
        >
          <defs>
            <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="8" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {bands.map((b) => (
            <g key={`band-${b.id}`}>
              <rect
                x={b.x}
                y={b.y}
                width={b.w}
                height={b.h}
                rx="16"
                fill={CLUSTER_GLOW[b.id] || CLUSTER_GLOW[0]}
                stroke={CLUSTER_STROKE[b.id] || CLUSTER_STROKE[0]}
                strokeWidth="1"
              />
              <text
                x={b.x + b.w / 2}
                y={b.titleY}
                textAnchor="middle"
                fill="var(--muted)"
                fontSize="12"
                fontWeight="600"
                letterSpacing="0.08em"
              >
                CLUSTER {b.id}
              </text>
            </g>
          ))}

          {(state?.edges || []).map(([a, b]) => {
            const pa = positions[a];
            const pb = positions[b];
            if (!pa || !pb) return null;
            const same = pa.cluster === pb.cluster;
            return (
              <path
                key={`${a}-${b}`}
                d={edgePath(pa, pb, same)}
                fill="none"
                stroke={same ? "#4a5d78" : "#6a8aaf"}
                strokeWidth={same ? 1.75 : 2.25}
                strokeOpacity={same ? 0.75 : 0.95}
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
                {comm && (
                  <rect
                    x={-nodeW / 2 - 6}
                    y={-nodeH / 2 - 6}
                    width={nodeW + 12}
                    height={nodeH + 12}
                    rx="14"
                    fill="none"
                    stroke="var(--comm)"
                    strokeWidth="2"
                    strokeDasharray="5 4"
                  />
                )}
                <rect
                  x={-nodeW / 2}
                  y={-nodeH / 2}
                  width={nodeW}
                  height={nodeH}
                  rx="12"
                  fill={CLUSTER_FILL[n.cluster] || CLUSTER_FILL[0]}
                  stroke={ring}
                  strokeWidth={n.outcome === "none" ? 1.75 : 3.25}
                  filter={n.outcome !== "none" ? "url(#softGlow)" : undefined}
                />
                <text
                  textAnchor="middle"
                  y={-nodeH / 2 + 24}
                  fill="var(--text)"
                  fontSize={Math.max(13, nodeW * 0.14)}
                  fontWeight="700"
                >
                  n{n.id}
                </text>
                <text
                  textAnchor="middle"
                  y={-nodeH / 2 + 42}
                  fill="var(--muted)"
                  fontSize="11"
                >
                  {n.requested != null ? `requested: ${n.requested}` : "idle"}
                </text>
                <CacheSlots
                  cache={n.cache || []}
                  requested={n.requested}
                  nodeW={nodeW}
                />
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
        <span>columns = clusters · square slots = cache</span>
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
