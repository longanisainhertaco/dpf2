from diagnostics.neutron import thermonuclear


def test_isotropic_yield_and_response():
    reactivity = [1.0, 1.0]
    ion_density = [2.0, 2.0]
    dt = 0.5
    angles = [0.0, 90.0]
    yields, tofs = thermonuclear.compute_yield(reactivity, ion_density, dt, angles)
    assert yields == [2.0, 2.0]
    assert tofs == []

    resp = lambda x: x * 2.0
    yields2, _ = thermonuclear.compute_yield(
        reactivity, ion_density, dt, angles, response_fn=resp
    )
    assert yields2 == [4.0, 4.0]


def test_tof_hook_generation():
    reactivity = [1.0, 1.0]
    ion_density = [2.0, 2.0]
    dt = 0.5
    angles = [0.0]

    def hook(val, bins):
        return [val for _ in range(len(bins) - 1)]

    yields, tofs = thermonuclear.compute_yield(
        reactivity,
        ion_density,
        dt,
        angles,
        time_bins=[0.0, 1.0],
        tof_hook=hook,
    )
    assert yields == [4.0]
    assert tofs == [[4.0]]
