from dpf2.circuit_config import CircuitConfig, SegmentConfig, SwitchConfig


def test_build_distributed_model_parses_connections():
    cfg = CircuitConfig.with_defaults().model_copy(
        update={
            "segments": [
                SegmentConfig(
                    length=1.0,
                    L=1.0,
                    R=0.1,
                    C=0.01,
                    from_node=0,
                    to_node=1,
                    L_parasitic=0.1,
                    R_parasitic=0.2,
                    C_parasitic=0.3,
                    L_profile=[(0.0, 1.0)],
                    R_profile=[(0.0, 0.2)],
                    C_profile=[(0.0, 0.3)],
                )
            ],
            "switches": [
                SwitchConfig(
                    from_node=1,
                    to_node=2,
                    closed=True,
                    r_on=1.0,
                    r_off=1000.0,

                    trigger_times=[10.0],
                    L_parasitic=0.5,
                    R_parasitic=0.6,
                    C_parasitic=0.7,

                )
            ],
        }
    )
    segments, switches = cfg.build_distributed_model()
    seg = segments[0]
    assert seg.from_node == 0 and seg.to_node == 1
    assert seg.L_parasitic == 0.1e-6
    assert seg.L_profile == [(0.0, 1.0e-6)]
    assert seg.R_profile == [(0.0, 0.2e-3)]
    assert seg.C_profile == [(0.0, 0.3e-6)]
    sw = switches[0]
    assert sw.from_node == 1 and sw.to_node == 2

    assert sw.trigger_times == [10.0e-9]
    assert sw.L_parasitic == 0.5e-6
    assert sw.R_parasitic == 0.6e-3
    assert sw.C_parasitic == 0.7e-6

