import numpy as np

from dpf2.mesh import plasma_gradient_refinement, wavefront_refinement


def test_plasma_gradient_refinement_tags_cells():
    density = np.array([[0.0, 0.0], [0.0, 1.0]])
    mask = plasma_gradient_refinement(density, threshold=0.5)
    assert mask[1, 1]


def test_wavefront_refinement_detects_change():
    prev = np.zeros((2, 2))
    curr = np.zeros((2, 2))
    curr[0, 0] = 1.0
    mask = wavefront_refinement(curr, prev, threshold=0.2)
    assert mask[0, 0]
