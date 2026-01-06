from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import asyncio
import random
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from dpf2.dpf_config import DPFConfig
from dpf2.optimization.param_sweep import compute_sweep_metrics
from dpf2.diagnostics import RegimePanel
from dpf2.web.lab_mode_api import export_manifest_bundle

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_LOG = BASE_DIR / "audit.log"
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
RESULTS_DIR = BASE_DIR / "results"
CONFIG_DIR = BASE_DIR / "configs"

logging.basicConfig(level=logging.INFO, filename=str(AUDIT_LOG), format="%(asctime)s %(message)s")
logger = logging.getLogger("dpf-web")

# JWT Configuration - should be loaded from environment variables in production
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    # Generate a random key for development, but warn about it
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY not set in environment. Using randomly generated key. "
        "This is insecure for production! Set JWT_SECRET_KEY environment variable."
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load users from environment or use development defaults
# In production, users should be stored in a database
def get_hashed_password(password: str) -> str:
    return pwd_context.hash(password)

# Cache for hashed passwords to avoid re-hashing on every startup
_password_cache = {}

def get_or_hash_password(username: str, env_var: str, default: str) -> str:
    """Get cached password hash or hash the password if not cached."""
    cache_key = f"{username}:{env_var}"
    if cache_key not in _password_cache:
        password = os.getenv(env_var, default)
        _password_cache[cache_key] = get_hashed_password(password)
        if password == default:
            logger.warning(
                f"Using default password for {username}. "
                f"Set {env_var} environment variable in production."
            )
    return _password_cache[cache_key]

# Development users with hashed passwords
# TODO: Replace with database-backed user management
users = {
    "admin": {
        "username": "admin",
        "hashed_password": get_or_hash_password("admin", "ADMIN_PASSWORD", "secret"),
        "role": "admin"
    },
    "user": {
        "username": "user",
        "hashed_password": get_or_hash_password("user", "USER_PASSWORD", "secret"),
        "role": "user"
    },
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()

progress_clients: Dict[str, set[WebSocket]] = {}
diagnostic_clients: Dict[str, set[WebSocket]] = {}
sweep_clients: Dict[str, set[WebSocket]] = {}
sweep_results: Dict[str, Dict[float, Dict[str, float]]] = {}
regime_clients: set[WebSocket] = set()

# Locks for WebSocket client management to prevent race conditions
clients_lock = asyncio.Lock()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate a user by username and password."""
    user = users.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = users.get(token_data.username)
    if user is None:
        raise credentials_exception
    return user


def require_role(role: str):
    def _checker(user=Depends(get_current_user)):
        if user["role"] not in {role, "admin"}:
            raise HTTPException(status_code=403, detail="Not authorized")
        return user

    return _checker


@app.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning("action=login_failed username=%s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    logger.info("action=login_success username=%s", user["username"])
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
    """Broadcast progress updates to connected WebSocket clients."""
    async with clients_lock:
        clients = list(progress_clients.get(run_id, set()))
    
    for ws in clients:
        try:
            await ws.send_json({"run_id": run_id, "progress": progress})
        except Exception as e:
            logger.warning("action=broadcast_progress_failed run_id=%s error=%s", run_id, str(e))


async def broadcast_diagnostics(run_id: str, data: Dict[str, Any]) -> None:
    """Broadcast diagnostic data to connected WebSocket clients."""
    async with clients_lock:
        clients = list(diagnostic_clients.get(run_id, set()))
    
    for ws in clients:
        try:
            await ws.send_json({"run_id": run_id, "diagnostics": data})
        except Exception as e:
            logger.warning("action=broadcast_diagnostics_failed run_id=%s error=%s", run_id, str(e))


async def broadcast_sweep(run_id: str, param: float, metrics: Dict[str, float]) -> None:
    """Broadcast parameter sweep results to connected WebSocket clients."""
    sweep_results.setdefault(run_id, {})[param] = metrics
    payload = {"run_id": run_id, "parameter": param, **metrics}
    
    async with clients_lock:
        clients = list(sweep_clients.get(run_id, set()))
    
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception as e:
            logger.warning("action=broadcast_sweep_failed run_id=%s error=%s", run_id, str(e))


async def broadcast_regime(data: Dict[str, float]) -> None:
    """Broadcast regime panel data to connected WebSocket clients."""
    async with clients_lock:
        clients = list(regime_clients)
    
    for ws in clients:
        try:
            await ws.send_json(data)
        except Exception as e:
            logger.warning("action=broadcast_regime_failed error=%s", str(e))


@app.post("/run")
async def run_simulation(req: RunRequest, user=Depends(get_current_user)):
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
    """Save configuration for HPC dispatch and return unique run ID.
    
    Note: This is currently a placeholder for actual HPC job submission.
    In production, this should dispatch to a real job scheduler.
    """
    run_id = str(uuid.uuid4())
    
    # Create necessary directories
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = CONFIG_DIR / f"{run_id}.json"
    try:
        config_path.write_text(cfg.model_dump_json())
    except Exception as e:
        logger.error("action=save_config_failed run_id=%s error=%s", run_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to save configuration")
    
    # TODO: Implement actual HPC job dispatch here
    # For now, this is a placeholder that only saves the configuration
    
    return run_id


@app.get("/config/{run_id}")
def get_config(run_id: str, user=Depends(require_role("admin"))):
    """Retrieve the configuration for a specific run."""
    config_path = CONFIG_DIR / f"{run_id}.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    try:
        logger.info("action=get_config user=%s run_id=%s", user["username"], run_id)
        return json.loads(config_path.read_text())
    except Exception as e:
        logger.error("action=get_config_failed run_id=%s error=%s", run_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to read configuration")


@app.get("/results/{run_id}")
def get_results(run_id: str, user=Depends(require_role("admin"))):
    """Retrieve the simulation results for a specific run."""
    results_path = RESULTS_DIR / f"{run_id}.json"
    if not results_path.exists():
        # If results don't exist yet, check if the run exists
        config_path = CONFIG_DIR / f"{run_id}.json"
        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        else:
            raise HTTPException(status_code=202, detail="Results not ready yet")
    
    try:
        logger.info("action=get_results user=%s run_id=%s", user["username"], run_id)
        return json.loads(results_path.read_text())
    except Exception as e:
        logger.error("action=get_results_failed run_id=%s error=%s", run_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to read results")


@app.get("/sweep/{run_id}")
async def get_sweep(run_id: str, user=Depends(get_current_user)):
    return sweep_results.get(run_id, {})


@app.post("/snapshot/save")
async def save_snapshot(req: SnapshotRequest, user=Depends(get_current_user)):
    """Persist a sandbox state and return a shareable reference."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snap_id = str(uuid.uuid4())
    path = SNAPSHOT_DIR / f"{snap_id}.json"
    
    try:
        path.write_text(json.dumps(req.state))
        logger.info("action=save_snapshot user=%s id=%s", user["username"], snap_id)
        return {"id": snap_id, "url": f"/snapshot/{snap_id}"}
    except Exception as e:
        logger.error("action=save_snapshot_failed user=%s error=%s", user["username"], str(e))
        raise HTTPException(status_code=500, detail="Failed to save snapshot")


@app.get("/snapshot/{snap_id}")
async def get_snapshot(snap_id: str, user=Depends(get_current_user)):
    """Retrieve a previously saved snapshot. Requires authentication."""
    path = SNAPSHOT_DIR / f"{snap_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")
    
    try:
        logger.info("action=get_snapshot user=%s id=%s", user["username"], snap_id)
        return json.loads(path.read_text())
    except Exception as e:
        logger.error("action=get_snapshot_failed id=%s error=%s", snap_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to read snapshot")


@app.post("/snapshot/upload")
async def upload_snapshot(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    """Load a snapshot from a user-uploaded JSON file.
    
    Security: Requires authentication, validates file size and content type.
    """
    # Validate content type
    if file.content_type not in ["application/json", "text/json"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JSON files are allowed."
        )
    
    # Read with size limit (10 MB) - streaming to avoid DoS
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks
    
    try:
        # Read file in chunks and check size progressively
        content_chunks = []
        total_size = 0
        
        while True:
            chunk = await file.read(MAX_CHUNK_SIZE)
            if not chunk:
                break
            
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE:
                logger.warning("action=upload_snapshot_too_large user=%s size=%d", user["username"], total_size)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {MAX_FILE_SIZE} bytes."
                )
            
            content_chunks.append(chunk)
        
        # Combine chunks and parse JSON
        content = b''.join(content_chunks)
        data = json.loads(content)
        logger.info("action=upload_snapshot user=%s size=%d", user["username"], total_size)
        return data
    except HTTPException:
        # Re-raise HTTP exceptions (including 413) without wrapping
        raise
    except json.JSONDecodeError as e:
        logger.warning("action=upload_snapshot_invalid user=%s error=%s", user["username"], str(e))
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error("action=upload_snapshot_failed user=%s error=%s", user["username"], str(e))
        raise HTTPException(status_code=500, detail="Failed to process upload")


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
    """WebSocket endpoint for simulation progress updates."""
    await websocket.accept()
    async with clients_lock:
        progress_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            progress_clients[run_id].discard(websocket)


@app.websocket("/ws/diagnostics/{run_id}")
async def ws_diagnostics(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for diagnostic data updates."""
    await websocket.accept()
    async with clients_lock:
        diagnostic_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            diagnostic_clients[run_id].discard(websocket)


@app.websocket("/ws/sweep/{run_id}")
async def ws_sweep(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for parameter sweep updates."""
    await websocket.accept()
    async with clients_lock:
        sweep_clients.setdefault(run_id, set()).add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            sweep_clients[run_id].discard(websocket)


@app.websocket("/ws/regime")
async def ws_regime(websocket: WebSocket):
    """WebSocket endpoint for regime panel updates."""
    await websocket.accept()
    async with clients_lock:
        regime_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        async with clients_lock:
            regime_clients.discard(websocket)
