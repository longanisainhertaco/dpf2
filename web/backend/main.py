from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from dpf2.dpf_config import DPFConfig

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


@app.post("/run")
def run_simulation(req: RunRequest, user=Depends(get_current_user)):
    cfg = DPFConfig.model_validate(req.config)
    run_id = dispatch_to_hpc(cfg, user["username"])
    logger.info("action=submit user=%s run_id=%s", user["username"], run_id)
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
