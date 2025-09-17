"""
This file is part of the PIConGPU.
Copyright 2024-2024 PIConGPU contributors
Authors: Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import pydantic
import datetime

from .rendering import RenderedObject


class Walltime(RenderedObject, pydantic.BaseModel):
    walltime: datetime.timedelta
    """time after which the cluster scheduler will stop the simulation"""

    def check(self) -> None:
        if self.walltime.total_seconds() <= 0.0:
            raise ValueError("walltime must be > 0.")

    def _get_serialized(self) -> dict:
        HOUR = datetime.timedelta(hours=1.0)
        MINUTE = datetime.timedelta(minutes=1.0)
        SECOND = datetime.timedelta(seconds=1.0)

        hours, rest = divmod(self.walltime, HOUR)
        minutes, rest = divmod(rest, MINUTE)
        seconds, _ = divmod(rest, SECOND)
        return {"walltime": "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)}  # @todo: might be cluster specific
