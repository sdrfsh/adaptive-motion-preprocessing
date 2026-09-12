from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class MotionImage:
    data: Frame
    frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.data.dtype != np.uint8:
            raise TypeError(f"MotionImage.data must be uint8, got {self.data.dtype}")
