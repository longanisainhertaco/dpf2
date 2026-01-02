from dpf2.diagnostics import (
    IonBeamEDF,
    synthetic_neutron_diagnostics,
    synthetic_xray_diagnostics,
    synthetic_interferometer_diagnostics,
)


class _Beam(IonBeamEDF):
    def energy_distribution(self, angle_deg: float):
        base = 1.0 + angle_deg / 90.0
        return [1.0, 2.0, 3.0], [base, base, base]


def test_synthetic_neutron_channels_with_irf_and_alignment():
    irf = {"transfer_function": [0.5, 0.5]}
    diagnostics = synthetic_neutron_diagnostics(
        _Beam(),
        cross_section=lambda e: e,
        angles=[0.0, 90.0],
        distance_m=1.0,
        time_bins=[0.0, 1.0, 2.0],
        reactivity=[1.0, 1.0],
        ion_density=[1.0, 1.0],
        dt=1.0,
        detector_irf=irf,
        current_trace=[0.5, 0.25, 0.125],
        voltage_trace=[1.0, 0.5, 0.25],
        align_to_iv=True,
    )

    channels = diagnostics["channels"]
    assert channels["beam_target"]
    assert channels["thermonuclear"]
    assert diagnostics["anisotropy"] > 0.0
    processed = diagnostics["tof"]["processed"]
    assert processed and processed[0] != diagnostics["tof"]["raw"][0]
    assert diagnostics["alignment"] is not None


def test_synthetic_xray_and_interferometer_irf():
    centers, spec = synthetic_xray_diagnostics(
        [5.0, 15.0],
        [1.0, 1.0],
        [0.0, 10.0, 20.0],
        detector_irf={"transfer_function": [0.0, 1.0]},
    )
    assert centers == [5.0, 15.0]
    assert spec != [1.0, 1.0]

    phase = synthetic_interferometer_diagnostics(
        [1.0, 1.0],
        [0.01, 0.01],
        1e-6,
        detector_irf={"transfer_function": [0.5, 0.5]},
    )
    assert phase != 0.0
