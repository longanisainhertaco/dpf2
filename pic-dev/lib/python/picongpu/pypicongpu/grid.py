"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

from . import util
import typeguard
import typing
import enum
from .rendering import RenderedObject


@typeguard.typechecked
class BoundaryCondition(enum.Enum):
    """
    Boundary Condition of PIConGPU

    Defines how particles that pass the simulation bounding box are treated.

    TODO: implement the other methods supported by PIConGPU
    (reflecting, thermal)
    """

    PERIODIC = 1
    ABSORBING = 2

    def get_cfg_str(self) -> str:
        """
        Get string equivalent for cfg files
        :return: string for --periodic
        """
        literal_by_boundarycondition = {
            BoundaryCondition.PERIODIC: "1",
            BoundaryCondition.ABSORBING: "0",
        }
        return literal_by_boundarycondition[self]


@typeguard.typechecked
class Grid3D(RenderedObject):
    """
    PIConGPU 3 dimensional (cartesian) grid

    Defined by the dimensions of each cell and the number of cells per axis.

    The bounding box is implicitly given as TODO.
    """

    cell_size_si = util.build_typesafe_property(tuple[float, float, float])
    """Width of individual cell in each direction"""

    cell_cnt = util.build_typesafe_property(tuple[int, int, int])
    """total number of cells in each direction"""

    boundary_condition = util.build_typesafe_property(tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition])
    """behavior towards particles crossing each boundary"""

    n_gpus = util.build_typesafe_property(typing.Tuple[int, int, int])
    """number of GPUs in x y and z direction as 3-integer tuple"""

    grid_dist = util.build_typesafe_property(typing.Tuple[list[int], list[int], list[int]] | None)
    """distribution of grid cells to GPUs for each axis"""

    super_cell_size = util.build_typesafe_property(typing.Tuple[int, int, int])
    """size of super cell in x y and z direction as 3-integer tuple in cells"""

    def _get_serialized(self) -> dict:
        """serialized representation provided for RenderedObject"""
        assert all(x > 0 for x in self.cell_cnt), "cell_cnt must be greater than 0"
        assert all(x > 0 for x in self.n_gpus), "all n_gpus entries must be greater than 0"
        if self.grid_dist is not None:
            assert sum(self.grid_dist[0]) == self.cell_cnt[0], "sum of grid_dists in x must be equal to number_of_cells"
            assert sum(self.grid_dist[1]) == self.cell_cnt[1], "sum of grid_dists in y must be equal to number_of_cells"
            assert sum(self.grid_dist[2]) == self.cell_cnt[2], "sum of grid_dists in z must be equal to number_of_cells"

        result_dict = {
            "cell_size": dict(zip("xyz", self.cell_size_si)),
            "cell_cnt": dict(zip("xyz", self.cell_cnt)),
            "boundary_condition": dict(zip("xyz", map(BoundaryCondition.get_cfg_str, self.boundary_condition))),
            "gpu_cnt": dict(zip("xyz", self.n_gpus)),
            "super_cell_size": dict(zip("xyz", self.super_cell_size)),
        }
        if self.grid_dist is not None:
            result_dict["grid_dist"] = {
                "x": [{"device_cells": x} for x in self.grid_dist[0]],
                "y": [{"device_cells": x} for x in self.grid_dist[1]],
                "z": [{"device_cells": x} for x in self.grid_dist[2]],
            }
        else:
            result_dict["grid_dist"] = None

        return result_dict
