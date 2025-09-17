def test_package_import():
    # The package should be importable via the top-level namespace
    from dpf2 import synthetic_diagnostics as sd

    assert sd is not None


def test_modes_submodule_import():
    # The modes submodule should still be accessible
    from dpf2.synthetic_diagnostics import modes

    assert hasattr(modes, "plot_growth_rates")
