from dpf2.simulation.load_balance_metrics import LoadBalanceMetrics


class DummySolver:
    def get_cell_count(self):
        return 100

    def get_particle_count(self):
        return 50


def test_metrics_basic():
    lb = LoadBalanceMetrics(DummySolver())
    metrics = lb.get_metrics()
    assert metrics["cell_min"] == 100
    assert metrics["cell_max"] == 100
    assert metrics["particle_min"] == 50
    assert metrics["particle_max"] == 50
