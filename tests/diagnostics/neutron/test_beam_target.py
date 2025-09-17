from diagnostics.neutron import beam_target


class DummyEDF:
    def energy_distribution(self, angle_deg: float):
        return [0.0, 1.0], [0.0, 1.0]


def test_yield_and_response():
    edf = DummyEDF()
    angles = [0.0]
    distance = 1.0
    time_bins = [0.0, 1.0, 2.0]

    y, hist = beam_target.compute_yield(edf, lambda e: 1.0, angles, distance, time_bins)
    assert len(y) == 1 and abs(y[0] - 0.5) < 1e-12
    assert hist == [[0.5, 0.0]]

    resp = lambda x: x * 2.0
    y2, hist2 = beam_target.compute_yield(
        edf, lambda e: 1.0, angles, distance, time_bins, response_fn=resp
    )
    assert y2 == [1.0]
    assert hist2 == [[1.0, 0.0]]


def test_tof_hook():
    edf = DummyEDF()
    angles = [0.0]
    distance = 1.0
    time_bins = [0.0, 1.0, 2.0]

    def hook(hist, ang):
        return [v * 3.0 for v in hist]

    _, hist = beam_target.compute_yield(
        edf, lambda e: 1.0, angles, distance, time_bins, tof_hook=hook
    )
    assert hist == [[1.5, 0.0]]
