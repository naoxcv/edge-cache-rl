"""FastAPI + WebSocket demo server for the 10-node edge-cache simulation.

Serves live node caches, requests, forwards, and a communication-level toggle
(L0 / L1 / L3). Intended for presentation demos — not for training.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.baselines import LFUPolicy
from agents.multi_agent import resolve_multi_model_path, select_action_for_obs
from configs import load_config
from env.multi_agent_caching_env import MultiAgentCachingEnv
from stable_baselines3 import DQN

RUN_NAMES = {
    0: "dqn_evict_level0_scratch_loc0.3",
    1: "dqn_evict_level1_scratch_loc0.3",
    3: "dqn_evict_level3_scratch_loc0.3",
}

app = FastAPI(title="edge-cache-rl demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DemoSim:
    def __init__(self) -> None:
        self.comm_level = 1
        self.policy_mode = "dqn"  # "dqn" | "lfu"
        self.seed = 42
        self.paused = True
        self.step_delay_ms = 250
        self._models: dict[int, DQN] = {}
        self._lfu = LFUPolicy()
        self.env: MultiAgentCachingEnv | None = None
        self.obs: dict[str, np.ndarray] = {}
        self.last_frame: dict[str, Any] = {}
        self._episode_return = 0.0
        self._episode_task = 0.0
        self._last_step_return = 0.0
        self._last_step_task = 0.0
        self._last_episode_return: float | None = None
        self._load_models()
        self.reset()

    def _load_models(self) -> None:
        for level, name in RUN_NAMES.items():
            path = resolve_multi_model_path(name, prefer_best=True)
            if path is None:
                print(f"WARNING: missing model {name}")
                continue
            self._models[level] = DQN.load(str(path), device="cpu")
            print(f"Loaded L{level}: {path}")

    def _config(self, level: int) -> dict:
        cfg = load_config()
        cfg.update(
            {
                "num_nodes": 10,
                "num_clusters": 3,
                "traffic_pattern": "shifting",
                "locality_factor": 0.3,
                "comm_level": level,
                "overlap_penalty_weight": 0.0,
                "comm_penalty_lambda": 0.0,
                "forwarding_same_cluster_only": True,
                "enable_forwarding": True,
            }
        )
        return cfg

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = seed
        level = 0 if self.policy_mode == "lfu" else self.comm_level
        cfg = self._config(level)
        self.env = MultiAgentCachingEnv(cfg, seed=self.seed)
        self.obs, _ = self.env.reset(seed=self.seed)
        self._episode_return = 0.0
        self._episode_task = 0.0
        self._last_step_return = 0.0
        self._last_step_task = 0.0
        self.last_frame = self._frame(infos=None, actions=None, communicated={})
        return self.last_frame

    def set_level(self, level: int) -> dict[str, Any]:
        if level not in RUN_NAMES:
            raise ValueError(f"Unsupported demo level {level}; use 0, 1, or 3")
        if level not in self._models:
            raise ValueError(f"Model for level {level} not loaded")
        self.policy_mode = "dqn"
        # Preserve timestep seed family but rebuild env for obs width.
        self.comm_level = level
        return self.reset(seed=self.seed)

    def set_policy(self, mode: str) -> dict[str, Any]:
        if mode not in ("dqn", "lfu"):
            raise ValueError(f"Unsupported policy {mode}; use dqn or lfu")
        self.policy_mode = mode
        return self.reset(seed=self.seed)

    def _choose_actions(self) -> tuple[dict[str, int], dict[str, bool]]:
        assert self.env is not None
        env = self.env
        actions: dict[str, int] = {}
        communicated: dict[str, bool] = {}
        if self.policy_mode == "lfu":
            for aid, agent_obs in self.obs.items():
                node_id = int(aid)
                node = env.network.nodes[node_id]
                actions[aid] = self._lfu.act(
                    agent_obs,
                    env._pending_requests.get(node_id),
                    cache=list(node.cache),
                    cache_capacity=env.cache_capacity,
                    num_container_types=env.num_container_types,
                )
                communicated[aid] = False
            return actions, communicated

        model = self._models[self.comm_level]
        cfg = env.config
        for aid, agent_obs in self.obs.items():
            action, did_comm = select_action_for_obs(
                model, agent_obs, config=cfg, deterministic=True
            )
            actions[aid] = action
            communicated[aid] = bool(did_comm)
            if self.comm_level == 3 and did_comm:
                env.record_communication(int(aid), True)
        return actions, communicated

    def step_once(self) -> dict[str, Any]:
        assert self.env is not None
        actions, communicated = self._choose_actions()

        before = {
            i: (
                self.env.network.nodes[i].hits,
                self.env.network.nodes[i].forwards,
                self.env.network.nodes[i].misses,
            )
            for i in range(self.env.num_nodes)
        }
        self.obs, rewards, _, truncateds, infos = self.env.step(actions)
        outcomes: dict[str, str] = {}
        forward_from: dict[str, Optional[int]] = {}
        for i in range(self.env.num_nodes):
            aid = str(i)
            h0, f0, m0 = before[i]
            node = self.env.network.nodes[i]
            if node.hits > h0:
                outcomes[aid] = "hit"
            elif node.forwards > f0:
                outcomes[aid] = "forward"
                req = (infos or {}).get(aid, {}).get("requested")
                src = None
                if req is not None:
                    for nbr in self.env.neighbor_lists[i]:
                        if self.env.network.nodes[nbr].is_cached(req):
                            if (
                                self.env.same_cluster_only
                                and self.env.network.cluster_for_node[nbr]
                                != self.env.network.cluster_for_node[i]
                            ):
                                continue
                            src = nbr
                            break
                forward_from[aid] = src
            elif node.misses > m0:
                outcomes[aid] = "cloud"
            else:
                outcomes[aid] = "none"

        step_return = float(sum(rewards.values()))
        step_task = float(
            sum(float(infos[aid].get("task_reward", 0.0)) for aid in infos)
        )
        self._last_step_return = step_return
        self._last_step_task = step_task
        self._episode_return += step_return
        self._episode_task += step_task

        self.last_frame = self._frame(
            infos=infos,
            actions=actions,
            communicated=communicated,
            outcomes=outcomes,
            forward_from=forward_from,
        )
        self.last_frame["rewards"] = {k: float(v) for k, v in rewards.items()}
        if truncateds.get("__all__", False):
            self._last_episode_return = self._episode_return
            self.obs, _ = self.env.reset()
            self._episode_return = 0.0
            self._episode_task = 0.0
            self._last_step_return = 0.0
            self._last_step_task = 0.0
            self.last_frame["episode_done"] = True
        else:
            self.last_frame["episode_done"] = False
        return self.last_frame

    def _frame(
        self,
        *,
        infos: dict | None,
        actions: dict[str, int] | None,
        communicated: dict[str, bool],
        outcomes: dict[str, str] | None = None,
        forward_from: Optional[dict[str, Optional[int]]] = None,
    ) -> dict[str, Any]:
        assert self.env is not None
        env = self.env
        c = env.cache_capacity
        nodes = []
        for i in range(env.num_nodes):
            aid = str(i)
            node = env.network.nodes[i]
            info = (infos or {}).get(aid, {})
            requested = info.get("requested")
            action = None if actions is None else int(actions[aid])
            action_kind = None
            action_container = None
            if action is not None:
                needed = bool(info.get("needs_decision", False))
                if not needed:
                    action_kind = "none"
                elif action < c:
                    action_kind = "evict"
                    action_container = action
                else:
                    action_kind = "reject"
            default_comm = (
                self.policy_mode == "dqn" and self.comm_level in (1, 2)
            )
            nodes.append(
                {
                    "id": i,
                    "cluster": int(env.network.cluster_for_node[i]),
                    "cache": list(node.cache),
                    "requested": requested,
                    "outcome": (outcomes or {}).get(aid, "none"),
                    "forward_from": (forward_from or {}).get(aid),
                    "action_kind": action_kind,
                    "action_container": action_container,
                    "communicated": bool(communicated.get(aid, default_comm)),
                    "hit_rate": float(info.get("hit_rate", 0.0)),
                    "forward_rate": float(info.get("forward_rate", 0.0)),
                    "cloud_rate": float(info.get("cloud_rate", 0.0)),
                    "step_reward": float(info.get("task_reward", 0.0)),
                }
            )

        edges = []
        for i, nbrs in env.neighbor_lists.items():
            for j in nbrs:
                if i < j:
                    edges.append([i, j])

        stats = env.network_stats()
        steps = max(env.timestep, 1)
        return {
            "timestep": env.timestep,
            "comm_level": self.comm_level,
            "policy_mode": self.policy_mode,
            "paused": self.paused,
            "step_delay_ms": self.step_delay_ms,
            "nodes": nodes,
            "edges": edges,
            "clusters": int(env.config["num_clusters"]),
            "stats": {
                "hit_rate": float(stats["hit_rate"]),
                "forward_rate": float(stats["forward_rate"]),
                "cloud_rate": float(stats["cloud_rate"]),
                "cache_diversity": float(stats.get("cache_diversity") or 0.0),
                "comm_events": int(stats.get("comm_events") or 0),
                "episode_return": float(self._episode_return),
                "episode_task": float(self._episode_task),
                "step_return": float(self._last_step_return),
                "step_task": float(self._last_step_task),
                "avg_return_per_step": float(self._episode_return / steps),
                "last_episode_return": self._last_episode_return,
            },
        }


sim = DemoSim()


@app.get("/api/state")
def get_state():
    return sim.last_frame


@app.post("/api/reset")
def api_reset(seed: Optional[int] = None):
    return sim.reset(seed=seed)


@app.post("/api/level/{level}")
def api_level(level: int):
    return sim.set_level(level)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_json(sim.last_frame)
    try:
        while True:
            # Non-blocking receive with timeout so we can auto-step while playing.
            try:
                message = await asyncio.wait_for(ws.receive_text(), timeout=0.05)
                data = json.loads(message)
                cmd = data.get("cmd")
                if cmd == "play":
                    sim.paused = False
                elif cmd == "pause":
                    sim.paused = True
                elif cmd == "step":
                    frame = sim.step_once()
                    await ws.send_json(frame)
                    continue
                elif cmd == "reset":
                    frame = sim.reset(seed=data.get("seed"))
                    await ws.send_json(frame)
                    continue
                elif cmd == "set_level":
                    frame = sim.set_level(int(data["level"]))
                    await ws.send_json(frame)
                    continue
                elif cmd == "set_policy":
                    frame = sim.set_policy(str(data["policy"]))
                    await ws.send_json(frame)
                    continue
                elif cmd == "set_delay":
                    sim.step_delay_ms = max(50, int(data.get("ms", 250)))
                # last_frame is only rebuilt on step/reset/level — keep control
                # fields current so Play/Pause label updates immediately.
                sim.last_frame["paused"] = sim.paused
                sim.last_frame["step_delay_ms"] = sim.step_delay_ms
                await ws.send_json(sim.last_frame)
            except asyncio.TimeoutError:
                if not sim.paused:
                    frame = sim.step_once()
                    await ws.send_json(frame)
                    await asyncio.sleep(sim.step_delay_ms / 1000.0)
    except WebSocketDisconnect:
        return


STATIC_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "dashboard.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
