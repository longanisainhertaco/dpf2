"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from picongpu.pypicongpu.grid import Grid3D, BoundaryCondition

import unittest
import typeguard


class TestGrid3D(unittest.TestCase):
    def setUp(self):
        """setup default grid"""
        self.g = Grid3D()
        self.g.cell_size_si = (1.2, 2.3, 4.5)
        self.g.cell_cnt = (6, 7, 8)
        self.g.boundary_condition = (
            BoundaryCondition.PERIODIC,
            BoundaryCondition.ABSORBING,
            BoundaryCondition.PERIODIC,
        )
        self.g.n_gpus = (2, 4, 1)
        self.g.super_cell_size = (8, 8, 4)
        self.g.grid_dist = None

    def test_basic(self):
        """test default setup"""
        g = self.g
        self.assertSequenceEqual((1.2, 2.3, 4.5), g.cell_size_si)
        self.assertSequenceEqual((6, 7, 8), g.cell_cnt)
        self.assertSequenceEqual(
            (BoundaryCondition.PERIODIC, BoundaryCondition.ABSORBING, BoundaryCondition.PERIODIC), g.boundary_condition
        )

    def test_types(self):
        """test raising errors if types are wrong"""
        g = self.g
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_size_si = ("54.3", 2.3, 4.5)
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_size_si = (1.2, "2", 4.5)
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_size_si = (1.2, 2.3, "126")
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_cnt = (11.1, 7, 8)
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_cnt = (6, 11.412, 8)
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_cnt = (6, 7, 16781123173.12637183)
        with self.assertRaises(typeguard.TypeCheckError):
            g.boundary_condition = ("open", BoundaryCondition.ABSORBING, BoundaryCondition.PERIODIC)
        with self.assertRaises(typeguard.TypeCheckError):
            g.boundary_condition = (BoundaryCondition.PERIODIC, 1, BoundaryCondition.PERIODIC)
        with self.assertRaises(typeguard.TypeCheckError):
            g.boundary_condition = (BoundaryCondition.PERIODIC, BoundaryCondition.ABSORBING, {})
        with self.assertRaises(typeguard.TypeCheckError):
            # list not accepted - tuple needed
            g.n_gpus = [1, 1, 1]

    def test_gpu_and_cell_cnt_positive(self):
        """test if n_gpus and cell number s are >0"""
        g = self.g
        with self.assertRaisesRegex(Exception, ".*cell_cnt.*greater than 0.*"):
            g.cell_cnt = (-1, 7, 8)
            g.get_rendering_context()
        # revert changes
        g.cell_cnt = (6, 7, 8)

        with self.assertRaisesRegex(Exception, ".*cell_cnt.*greater than 0.*"):
            g.cell_cnt = (6, -2, 8)
            g.get_rendering_context()
        # revert changes
        g.cell_cnt = (6, 7, 8)

        with self.assertRaisesRegex(Exception, ".*cell_cnt.*greater than 0.*"):
            g.cell_cnt = (6, 7, 0)
            g.get_rendering_context()
        # revert changes
        g.cell_cnt = (6, 7, 8)

        for wrong_n_gpus in [tuple([-1, 1, 1]), tuple([1, 1, 0])]:
            with self.assertRaisesRegex(Exception, ".*greater than 0.*"):
                g.n_gpus = wrong_n_gpus
                g.get_rendering_context()

    def test_mandatory(self):
        """test if None as content fails"""
        # check that mandatory arguments can't be none
        g = self.g
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_size_si = None
        with self.assertRaises(typeguard.TypeCheckError):
            g.cell_cnt = None
        with self.assertRaises(typeguard.TypeCheckError):
            g.boundary_condition = None
        with self.assertRaises(typeguard.TypeCheckError):
            g.n_gpus = None

    def test_get_rendering_context(self):
        """object is correctly serialized"""
        # automatically checks against schema
        context = self.g.get_rendering_context()
        self.assertEqual(1.2, context["cell_size"]["x"])
        self.assertEqual(2.3, context["cell_size"]["y"])
        self.assertEqual(4.5, context["cell_size"]["z"])
        self.assertEqual(6, context["cell_cnt"]["x"])
        self.assertEqual(7, context["cell_cnt"]["y"])
        self.assertEqual(8, context["cell_cnt"]["z"])

        # boundary condition translated to numbers for cfgfiles
        self.assertEqual("1", context["boundary_condition"]["x"])
        self.assertEqual("0", context["boundary_condition"]["y"])
        self.assertEqual("1", context["boundary_condition"]["z"])

        # n_gpus ouput
        self.assertEqual(2, context["gpu_cnt"]["x"])
        self.assertEqual(4, context["gpu_cnt"]["y"])
        self.assertEqual(1, context["gpu_cnt"]["z"])


class TestBoundaryCondition(unittest.TestCase):
    def test_cfg_translation(self):
        """test boundary condition strings"""
        p = BoundaryCondition.PERIODIC
        a = BoundaryCondition.ABSORBING
        self.assertEqual("0", a.get_cfg_str())
        self.assertEqual("1", p.get_cfg_str())
