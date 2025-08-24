import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI  # type: ignore
except Exception as e:  # pragma: no cover - mpi4py optional
    MPI = None
    logger.warning(f"mpi4py not available: {e}")


class LoadBalanceMetrics:
    """Compute simple cell and particle balance metrics across MPI ranks."""

    def __init__(self, solver):
        self.solver = solver
        self.comm = MPI.COMM_WORLD if MPI else None
        self.last = {}

    def set_mpi_comm(self, comm):
        """Allow external injection of an MPI communicator."""
        self.comm = comm

    def _gather(self, value):
        if self.comm:
            return self.comm.allgather(value)
        return [value]

    def _get_cell_counts(self):
        if hasattr(self.solver, "get_cell_count"):
            local = self.solver.get_cell_count()
        else:
            local = 0
        return self._gather(local)

    def _get_particle_counts(self):
        if hasattr(self.solver, "get_particle_count"):
            local = self.solver.get_particle_count()
        else:
            local = 0
        return self._gather(local)

    def update(self):
        """Update cached metrics for the current iteration."""
        cells = self._get_cell_counts()
        parts = self._get_particle_counts()
        self.last = {
            "cell_min": int(np.min(cells)) if cells else 0,
            "cell_max": int(np.max(cells)) if cells else 0,
            "particle_min": int(np.min(parts)) if parts else 0,
            "particle_max": int(np.max(parts)) if parts else 0,
        }
        return self.last

    def get_metrics(self):
        """Return the most recently computed metrics."""
        if not self.last:
            return self.update()
        return self.last
