import pytest

from dpf2.diagnostics.streaming import NeutronYieldStreamer, XRayEmissionStreamer
from dpf2.core.bases import CouplingState


def test_streaming_diagnostics_emit_values():
    neutron_records = []
    xray_records = []

    n_stream = NeutronYieldStreamer(lambda t, v: neutron_records.append((t, v)))
    x_stream = XRayEmissionStreamer(lambda t, v: xray_records.append((t, v)))

    state = CouplingState(current=2.0, voltage=3.0)

    n_stream.record(state, 1.0)
    x_stream.record(state, 1.0)

    assert neutron_records[0][0] == 1.0
    assert neutron_records[0][1] == pytest.approx(1.0e5 * 4.0)

    assert xray_records[0][0] == 1.0
    assert xray_records[0][1] == pytest.approx(abs(2.0 * 3.0) * 1.0e-3)

    # total_yield accumulates
    n_stream.record(state, 2.0)
    assert n_stream.total_yield == pytest.approx(1.0e5 * 4.0 * 2)
