# Edge-cache demo dashboard

Live 10-node simulation for demos (Ishigaki). Not used for training.

## Run

```bash
# from repo root — build UI once
cd dashboard/frontend && npm install && npm run build && cd ../..

# serves API + built React app
python -m dashboard.server
# open http://127.0.0.1:8000
```

Dev mode (hot reload UI):

```bash
# terminal 1
python -m dashboard.server

# terminal 2
cd dashboard/frontend && npm run dev
# open http://127.0.0.1:5173
```

## Controls

- **L0 / L1 / L3** — swap the shared-policy checkpoint and rebuild the env obs layout
- **Play / Pause / Step / Reset** — stream timesteps over WebSocket
- Node rings: green hit · blue forward · red cloud · gold dashed = L3 communicate
