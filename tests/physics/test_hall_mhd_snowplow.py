import math
import numpy as np

from dpf2.physics import HallMHD


def test_coaxial_inductance_reference():
    model = HallMHD()
    model.current = 1.0
    mu0 = 4e-7 * math.pi
    r_inner = 0.01
    r_outer = 0.02
    L_ref = mu0 / (2 * math.pi) * math.log(r_outer / r_inner)
    B = math.sqrt(L_ref) * model.current
    p = 1.0
    rho = 1.0
    energy = p / (model.gamma - 1.0) + 0.5 * B ** 2
    state = np.array([rho, 0.0, 0.0, 0.0, energy, 0.0, B, 0.0, 0.0])
    L_calc = model.plasma_inductance(state)
    assert abs(L_calc - L_ref) / L_ref < 0.05
