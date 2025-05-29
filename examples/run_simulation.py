"""Example script to run a DPF simulation and plot the results."""
from pathlib import Path

import matplotlib.pyplot as plt

from dpf_config import DPFConfig
from dpf2.simulation_engine import SimulationEngine


cfg = DPFConfig.with_defaults()
engine = SimulationEngine(cfg)
res = engine.run()

print(f"Neutron yield: {res.neutron_yield:.2e}")

plt.figure()
plt.plot(res.time * 1e6, res.current / 1e3)
plt.xlabel("Time [µs]")
plt.ylabel("Current [kA]")
plt.tight_layout()
plt.show()
