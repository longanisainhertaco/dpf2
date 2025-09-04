from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import asyncio
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from dpf2.dpf_config import DPFConfig
from dpf2.optimization.param_sweep import compute_sweep_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_LOG = BASE_DIR / "audit.log"
UPLOAD_DIR = BASE_DIR / "uploads"
logging.basicConfig(level=logging.INFO, filename=str(AUDIT_LOG), format="%(asctime)s %(message)s")
logger = logging.getLogger("dpf-web")

users = {
    "admin": {"username": "admin", "password": "secret", "role": "admin"},
    "user": {"username": "user", "password": "secret", "role": "user"},
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()

progress_clients: Dict[str, set[WebSocket]] = {}
diagnostic_clients: Dict[str, set[WebSocket]] = {}
sweep_clients: Dict[str, set[WebSocket]] = {}
sweep_results: Dict[str, Dict[float, Dict[str, float]]] = {}


def get_current_user(token: str = Depends(oauth2_scheme)):
    user = users.get(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    return user


def require_role(role: str):
    def _checker(user=Depends(get_current_user)):
        if user["role"] not in {role, "admin"}:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user

    return _checker


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}


class RunRequest(BaseModel):
    config: Dict[str, Any]


class UpdateRequest(BaseModel):
    voltage: float
    pressure: float


class SweepRequest(BaseModel):
    config: Dict[str, Any]
    parameter: str
    values: List[float]


async def broadcast_progress(run_id: str, progress: float) -> None:
    for ws in list(progress_clients.get(run_id, set())):
        await ws.send_json({"run_id": run_id, "progress": progress})


async def broadcast_diagnostics(run_id: str, data: Dict[str, Any]) -> None:
    for ws in list(diagnostic_clients.get(run_id, set())):
        await ws.send_json({"run_id": run_id, "diagnostics": data})


async def broadcast_sweep(run_id: str, param: float, metrics: Dict[str, float]) -> None:
    sweep_results.setdefault(run_id, {})[param] = metrics
    payload = {"run_id": run_id, "parameter": param, **metrics}
    for ws in list(sweep_clients.get(run_id, set())):
        await ws.send_json(payload)


@app.post("/run")
async def run_simulation(req: RunRequest, user=Depends(get_current_user)):
    cfg = DPFConfig.model_validate(req.config)
    run_id = dispatch_to_hpc(cfg, user["username"])
    logger.info("action=submit user=%s run_id=%s", user["username"], run_id)

    async def _mock_progress() -> None:
        for step in range(1, 11):
            await asyncio.sleep(0.1)
            await broadcast_progress(run_id, step / 10)
        await broadcast_diagnostics(run_id, {"status": "completed"})

    asyncio.create_task(_mock_progress())
    return {"run_id": run_id}


@app.post("/update/{run_id}")
async def update_simulation(run_id: str, req: UpdateRequest, user=Depends(get_current_user)):
    """Receive live parameter updates for a running simulation."""
    logger.info(
        "action=update user=%s run_id=%s voltage=%s pressure=%s",
        user["username"],
        run_id,
        req.voltage,
        req.pressure,
    )
    await broadcast_diagnostics(run_id, {"voltage": req.voltage, "pressure": req.pressure})
    return {"status": "updated"}


@app.post("/sweep")
async def run_sweep(req: SweepRequest, user=Depends(get_current_user)):
    cfg = DPFConfig.model_validate(req.config)
    run_id = dispatch_to_hpc(cfg, user["username"])
    logger.info(
        "action=sweep user=%s run_id=%s param=%s", user["username"], run_id, req.parameter
    )

    async def _mock_sweep() -> None:
        for val in req.values:
            await asyncio.sleep(0.1)
            t = [0.0, 1.0]
            current = [val, val]
            voltage = [cfg.charging_voltage, cfg.charging_voltage]
            metrics = compute_sweep_metrics(cfg, {val: (t, current, voltage)})[val]
            await broadcast_sweep(run_id, val, metrics)
        for ws in list(sweep_clients.get(run_id, set())):
            await ws.send_json({"run_id": run_id, "status": "completed"})

    asyncio.create_task(_mock_sweep())
    return {"run_id": run_id}


def dispatch_to_hpc(cfg: DPFConfig, username: str) -> str:
    run_id = f"run-{int(datetime.utcnow().timestamp())}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / f"{run_id}.json").write_text(cfg.model_dump_json())
    # Placeholder for real HPC dispatch
    return run_id


@app.get("/results/{run_id}")
def get_results(run_id: str, user=Depends(require_role("admin"))):
    path = UPLOAD_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    logger.info("action=get_results user=%s run_id=%s", user["username"], run_id)
    return json.loads(path.read_text())


@app.get("/sweep/{run_id}")
async def get_sweep(run_id: str, user=Depends(get_current_user)):
    return sweep_results.get(run_id, {})


@app.websocket("/ws/progress/{run_id}")
async def ws_progress(websocket: WebSocket, run_id: str):
    await websocket.accept()
    progress_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        progress_clients[run_id].discard(websocket)


@app.websocket("/ws/diagnostics/{run_id}")
async def ws_diagnostics(websocket: WebSocket, run_id: str):
    await websocket.accept()
    diagnostic_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        diagnostic_clients[run_id].discard(websocket)


@app.websocket("/ws/sweep/{run_id}")
async def ws_sweep(websocket: WebSocket, run_id: str):
    await websocket.accept()
    sweep_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        sweep_clients[run_id].discard(websocket)
