from dpf2.diagnostics import assemble_diagnostic_outputs, IonBeamEDF


class _Beam(IonBeamEDF):
    def energy_distribution(self, angle_deg: float):
        base = 1.0 + angle_deg / 90.0
        return [1.0, 2.0], [base, base]


def test_output_bundle_includes_dual_channel_and_overlays():
    detector_response = lambda v: v * 2.0
    tof_irf = {"transfer_function": [0.0, 1.0]}
    outputs = assemble_diagnostic_outputs(
        ion_edf=_Beam(),
        cross_section=lambda e: e,
        angles=[0.0, 90.0],
        distance_m=1.0,
        time_bins=[0.0, 1.0, 2.0],
        reactivity=[1.0],
        ion_density=[1.0],
        dt=1.0,
        current_trace=[0.5, 0.25],
        voltage_trace=[1.0, 0.5],
        xray_energies_keV=[5.0, 15.0],
        xray_intensities=[1.0, 1.0],
        xray_bins_keV=[0.0, 10.0, 20.0],
        interferometer_density=[1.0],
        interferometer_path=[0.01],
        interferometer_wavelength=1e-6,
        detector_response=detector_response,
        cable_tau=0.5,
        tof_irf=tof_irf,
        align_tof_to_iv=True,
        benchmark_reference=[4.0, 7.0],
        benchmark_band=0.01,
        xray_response=lambda v: v * 3.0,
        tof_noise=lambda _: 0.1,
        azimuthal_field=[[1.0, 2.0], [3.0, 4.0]],
        azimuthal_axis=1,
        mode_acceptance={0: (0.0, 10.0), 1: (0.0, 10.0)},
        energy_partitions={"magnetic": 1.0, "kinetic": 2.0, "radiation": 0.5, "losses": 0.25},
        regime_panel=None,
    )

    dual = outputs["dual_channel_yield"]
    assert dual["energy_partition"]["beam_target"] > 0.0
    assert dual["energy_partition"]["thermonuclear_fraction"] < 1.0
    assert dual["anisotropy"] > 0.0
    assert dual["iv_phase"]

    angular = outputs["angular_distribution"]
    assert set(angular["per_angle_total"].keys()) == {0.0, 90.0}
    assert all(angular["benchmark_overlay"]["within_band"])

    tof = outputs["tof"]
    assert len(tof["aggregate"]) == 2
    assert tof["processed"]
    assert tof["processed"][0] != tof["raw"][0]
    assert tof["alignment"] is not None

    xray = outputs["xray"]
    assert xray["spectrum"] == [3.0, 3.0]

    interferometry = outputs["interferometry"]
    assert "phase_shift_rad" in interferometry

    modes = outputs["azimuthal_modes"]
    assert modes["m0"] > 0.0
    assert modes["overlay"]["gates"][0]["within"]

    energy = outputs["energy_partition"]
    assert energy["net"] > 0.0
