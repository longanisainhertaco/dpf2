from dpf2.physics.material_interactions import (
    Species,
    sigmund_yield,
    yamamura_yield,
    impurity_source_terms,
    ImpurityState,
    get_material_properties,
)


def test_yields_and_impurity_tracking():
    d = Species("D", 1, 2.0)
    cu = Species("Cu", 29, 63.5)
    y0 = sigmund_yield(d, cu, 1000.0)
    y45 = yamamura_yield(d, cu, 1000.0, 45.0)
    assert 0.0 < y0 < y45

    flux = impurity_source_terms(1e20, y45, cu)
    state = ImpurityState()
    state.update(flux)
    charges = {cu.name: cu.Z}
    assert state.z_eff(charges) == cu.Z


def test_material_properties():
    props = get_material_properties("copper")
    assert props.resistivity_ohm_m > 0
    assert "Matula" in props.source
