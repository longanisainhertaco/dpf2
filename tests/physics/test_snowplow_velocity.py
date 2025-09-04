import numpy as np

from dpf2.axial_sheath import AxialSheathModel


def test_snowplow_velocity():
    """Axial snowplow model produces expected rundown velocity magnitude."""

    current = 0.6e6  # 0.6 MA
    radius = 0.01  # 1 cm
    area = np.pi * radius**2
    time = np.linspace(0.0, 1e-5, 1001)
    I = np.full_like(time, current)

    model = AxialSheathModel(
        area=area,
        mass=1e-3,
        length=1.0,
        upstream_density=0.02,
        upstream_pressure=133.0,
    )

    result = model.run(time, I)
    v_final = result.velocity[-1]

    assert 1e4 <= v_final <= 1e5

