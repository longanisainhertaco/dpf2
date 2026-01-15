from __future__ import annotations

import json
import logging
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import random
import tempfile
from typing import Any, Dict, List, Optional

import bcrypt
import jwt
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from dpf2.dpf_config import DPFConfig
from dpf2.optimization.param_sweep import compute_sweep_metrics
from dpf2.diagnostics import RegimePanel
from dpf2.web.lab_mode_api import export_manifest_bundle

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_LOG = BASE_DIR / "audit.log"
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
logging.basicConfig(level=logging.INFO, filename=str(AUDIT_LOG), format="%(asctime)s %(message)s")
logger = logging.getLogger("dpf-web")

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Database Configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./dpf2.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False)


Base.metadata.create_all(bind=engine)


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed.encode("utf-8"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _init_admin_user():
    """Initialize admin user from environment variable if not exists."""
    admin_password = os.environ.get("DPF2_ADMIN_PASSWORD")
    if admin_password:
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.username == "admin").first()
            if not existing:
                admin_user = User(
                    username="admin",
                    password_hash=get_password_hash(admin_password),
                    role="admin",
                )
                db.add(admin_user)
                db.commit()
                logger.info("Admin user created from environment variable")
        finally:
            db.close()


_init_admin_user()

# Upload limits
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))  # 10MB default
ALLOWED_CONTENT_TYPES = {"application/json"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

progress_clients: Dict[str, set[WebSocket]] = {}
diagnostic_clients: Dict[str, set[WebSocket]] = {}
sweep_clients: Dict[str, set[WebSocket]] = {}
sweep_results: Dict[str, Dict[float, Dict[str, float]]] = {}
regime_clients: set[WebSocket] = set()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )
            return {"username": user.username, "role": user.role}
        finally:
            db.close()
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


def require_role(role: str):
    def _checker(user=Depends(get_current_user)):
        if user["role"] not in {role, "admin"}:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user

    return _checker


@app.post("/token")
@limiter.limit("5/minute")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user or not verify_password(form_data.password, user.password_hash):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()


@app.post("/token/refresh")
def refresh_token(user=Depends(get_current_user)):
    """Refresh the access token for an authenticated user."""
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


class RunRequest(BaseModel):
    config: Dict[str, Any]


class UpdateRequest(BaseModel):
    voltage: float
    pressure: float


class SweepRequest(BaseModel):
    config: Dict[str, Any]
    parameter: str
    values: List[float]


class SnapshotRequest(BaseModel):
    state: Dict[str, Any]


class LabBundleRequest(BaseModel):
    runs: List[Dict[str, Any]]


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


async def broadcast_regime(data: Dict[str, float]) -> None:
    for ws in list(regime_clients):
        await ws.send_json(data)


@app.post("/run")
@limiter.limit("10/minute")
async def run_simulation(request: Request, req: RunRequest, user=Depends(get_current_user)):
    cfg = DPFConfig.model_validate(req.config)
    run_id = dispatch_to_hpc(cfg, user["username"])
    logger.info("action=submit user=%s run_id=%s", user["username"], run_id)

    panel = RegimePanel(L=1.0)

    async def _mock_progress() -> None:
        for step in range(1, 11):
            await asyncio.sleep(0.1)
            await broadcast_progress(run_id, step / 10)
            reg = panel.log(
                step,
                n=1e20,
                T=1e3,
                B=1.0,
                v=1e5,
                eta=1e-6,
                mfp=0.01,
                tau_e=1e-9,
            )
            payload = {k: reg[k] for k in [
                "S",
                "beta",
                "M_A",
                "R_m",
                "K_n",
                "omega_ce_tau_e",
            ]}
            await broadcast_regime(payload)
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
    run_id = f"run-{uuid.uuid4().hex}"
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


@app.post("/snapshot/save")
@limiter.limit("20/minute")
async def save_snapshot(request: Request, req: SnapshotRequest, user=Depends(get_current_user)):
    """Persist a sandbox state and return a shareable reference."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_id = f"snap-{uuid.uuid4().hex}"
    path = SNAPSHOT_DIR / f"{snap_id}.json"
    path.write_text(json.dumps(req.state))
    logger.info("action=save_snapshot user=%s id=%s", user["username"], snap_id)
    return {"id": snap_id, "url": f"/snapshot/{snap_id}"}


@app.get("/snapshot/{snap_id}")
async def get_snapshot(snap_id: str, user=Depends(get_current_user)):
    # Validate snap_id format to prevent path traversal
    if not re.match(r"^snap-[a-f0-9]+$", snap_id):
        raise HTTPException(status_code=400, detail="Invalid snapshot ID format")
    path = SNAPSHOT_DIR / f"{snap_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    logger.info("action=get_snapshot user=%s id=%s", user["username"], snap_id)
    return json.loads(path.read_text())


@app.post("/snapshot/upload")
@limiter.limit("20/minute")
async def upload_snapshot(request: Request, file: UploadFile = File(...), user=Depends(get_current_user)):
    """Load a snapshot from a user-uploaded JSON file."""
    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only application/json allowed")
    
    # Read with size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE} bytes")
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    
    return data


@app.post("/lab-mode/manifests")
def create_lab_manifest_bundle(req: LabBundleRequest, user=Depends(get_current_user)):
    """Generate a manifest bundle for a batch of lab-mode configurations."""
    seeds = [random.randint(0, 2**32 - 1) for _ in req.runs]
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        export_manifest_bundle(req.runs, tmp.name, seeds=seeds)
        logger.info("action=lab_bundle user=%s runs=%d", user["username"], len(req.runs))
        return FileResponse(tmp.name, media_type="application/zip", filename="manifest_bundle.zip")


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


@app.websocket("/ws/regime")
async def ws_regime(websocket: WebSocket):
    await websocket.accept()
    regime_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        regime_clients.discard(websocket)
