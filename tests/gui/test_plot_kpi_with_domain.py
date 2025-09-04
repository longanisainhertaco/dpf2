from types import SimpleNamespace

from dpf2.gui.dashboard import plot_kpi_with_domain


class DummyAx:
    def __init__(self):
        self.scatter_calls = []
    def errorbar(self, *a, **k):
        pass
    def axvspan(self, *a, **k):
        pass
    def fill_between(self, *a, **k):
        pass
    def set_xlabel(self, *a, **k):
        pass
    def set_ylabel(self, *a, **k):
        pass
    def legend(self, *a, **k):
        pass
    def scatter(self, *a, **k):
        self.scatter_calls.append((a, k))


def test_plot_marks_ood_points():
    ax = DummyAx()
    plot_kpi_with_domain([1.0, 2.0], [1.0, 2.0], [0.1, 0.1], (0.5, 2.5), [False, True], ax=ax)
    assert len(ax.scatter_calls) == 1
