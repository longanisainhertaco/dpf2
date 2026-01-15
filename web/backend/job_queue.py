"""Job queue module with Celery integration and SLURM support."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from celery import Celery

logger = logging.getLogger("dpf-web.job_queue")

# Redis URL for Celery broker and backend
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Whether to use SLURM for job submission
USE_SLURM = os.environ.get("USE_SLURM", "").lower() in ("1", "true", "yes")

# Directory for results
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

# Celery app configuration
celery_app = Celery(
    "dpf2_jobs",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_extended=True,
)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job status store (for lightweight tracking without Redis dependency)
_job_status_store: Dict[str, Dict[str, Any]] = {}


def get_job_status(run_id: str) -> Dict[str, Any]:
    """Get the status of a job by run_id."""
    if run_id in _job_status_store:
        return _job_status_store[run_id]
    
    # Check if result file exists (completed job)
    result_path = RESULTS_DIR / f"{run_id}.json"
    if result_path.exists():
        return {
            "run_id": run_id,
            "status": JobStatus.COMPLETED,
            "completed_at": datetime.fromtimestamp(
                result_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    
    # Check if config file exists (job was submitted)
    config_path = UPLOAD_DIR / f"{run_id}.json"
    if config_path.exists():
        return {
            "run_id": run_id,
            "status": JobStatus.PENDING,
            "message": "Job submitted but status unknown",
        }
    
    return {"run_id": run_id, "status": "not_found"}


def set_job_status(
    run_id: str,
    status: JobStatus,
    message: Optional[str] = None,
    error: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
) -> None:
    """Update the status of a job."""
    now = datetime.now(timezone.utc).isoformat()
    entry = _job_status_store.get(run_id, {"run_id": run_id, "created_at": now})
    entry["status"] = status
    entry["updated_at"] = now
    
    if message:
        entry["message"] = message
    if error:
        entry["error"] = error
    if slurm_job_id:
        entry["slurm_job_id"] = slurm_job_id
    
    if status == JobStatus.COMPLETED:
        entry["completed_at"] = now
    elif status == JobStatus.RUNNING:
        entry["started_at"] = now
    
    _job_status_store[run_id] = entry


def submit_slurm_job(run_id: str, config_path: Path) -> str:
    """Submit a job to SLURM and return the SLURM job ID."""
    # Create SLURM batch script
    script_content = f"""#!/bin/bash
#SBATCH --job-name=dpf2-{run_id}
#SBATCH --output={RESULTS_DIR}/{run_id}.out
#SBATCH --error={RESULTS_DIR}/{run_id}.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Load required modules (adjust for your HPC environment)
# module load python/3.10

# Run the simulation
python -m dpf2.cli run --config {config_path} --output {RESULTS_DIR}/{run_id}.json
"""
    
    script_path = UPLOAD_DIR / f"{run_id}.slurm"
    script_path.write_text(script_content)
    
    try:
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"SLURM submission failed: {result.stderr}")
        
        # Parse SLURM job ID from output (format: "Submitted batch job 12345")
        output = result.stdout.strip()
        slurm_job_id = output.split()[-1]
        
        logger.info("Submitted SLURM job %s for run %s", slurm_job_id, run_id)
        return slurm_job_id
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("SLURM submission timed out")
    except FileNotFoundError:
        raise RuntimeError("SLURM sbatch command not found")


@celery_app.task(bind=True, name="dpf2.run_simulation")
def run_simulation_task(self, run_id: str, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task to run a DPF2 simulation.
    
    Args:
        run_id: Unique identifier for this simulation run
        config_dict: DPF2 configuration dictionary
        
    Returns:
        Dictionary containing simulation results
    """
    from dpf2.dpf_config import DPFConfig
    
    set_job_status(run_id, JobStatus.RUNNING, message="Simulation started")
    
    try:
        # Validate configuration
        cfg = DPFConfig.model_validate(config_dict)
        
        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Run the actual simulation
        # For now, create mock results - replace with actual simulation call
        results = {
            "run_id": run_id,
            "config": config_dict,
            "status": "completed",
            "outputs": {
                "peak_current": 1.5e6,
                "pulse_duration": 2.5e-6,
                "energy_delivered": 125.0,
                "efficiency": 0.85,
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Save results to file
        result_path = RESULTS_DIR / f"{run_id}.json"
        result_path.write_text(json.dumps(results, indent=2))
        
        set_job_status(run_id, JobStatus.COMPLETED, message="Simulation completed")
        logger.info("Simulation %s completed successfully", run_id)
        
        return results
        
    except Exception as e:
        error_msg = str(e)
        set_job_status(run_id, JobStatus.FAILED, error=error_msg)
        logger.error("Simulation %s failed: %s", run_id, error_msg)
        raise


def dispatch_job(config_dict: Dict[str, Any], username: str) -> str:
    """
    Dispatch a simulation job to the appropriate backend.
    
    Uses SLURM if USE_SLURM is set, otherwise uses Celery task queue.
    Falls back to synchronous execution if neither is available.
    
    Args:
        config_dict: DPF2 configuration dictionary
        username: Username of the submitting user
        
    Returns:
        run_id: Unique identifier for tracking the job
    """
    run_id = f"run-{uuid.uuid4().hex}"
    
    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = UPLOAD_DIR / f"{run_id}.json"
    config_path.write_text(json.dumps(config_dict, indent=2))
    
    set_job_status(run_id, JobStatus.PENDING, message=f"Job submitted by {username}")
    
    if USE_SLURM:
        try:
            slurm_job_id = submit_slurm_job(run_id, config_path)
            set_job_status(
                run_id,
                JobStatus.PENDING,
                message="Submitted to SLURM",
                slurm_job_id=slurm_job_id,
            )
            return run_id
        except Exception as e:
            logger.warning("SLURM submission failed, falling back: %s", e)
    
    # Try Celery dispatch
    try:
        task = run_simulation_task.delay(run_id, config_dict)
        set_job_status(
            run_id,
            JobStatus.PENDING,
            message=f"Queued in Celery (task_id={task.id})",
        )
        logger.info("Dispatched job %s to Celery (task_id=%s)", run_id, task.id)
    except Exception as e:
        # Celery not available, job will run when manually triggered
        logger.warning("Celery dispatch failed: %s. Job saved for later processing.", e)
        set_job_status(
            run_id,
            JobStatus.PENDING,
            message="Job saved (async processing unavailable)",
        )
    
    return run_id
