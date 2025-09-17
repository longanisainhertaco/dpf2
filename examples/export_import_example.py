"""Demonstrate JSON export/import for simulation setups."""

from pathlib import Path

from dpf2.dpf_config import DPFConfig
from dpf2.io import export_config, import_config

cfg_path = Path(__file__).with_name("config.json")
config = DPFConfig.model_validate_json(cfg_path.read_text())

out_path = Path(__file__).with_name("shared_config.json")
export_config(config, out_path)
print(f"Exported configuration to {out_path}")

loaded = import_config(out_path)
print(f"Reloaded geometry: {loaded.simulation_control.geometry}")
