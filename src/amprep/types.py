from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]
"""A single video frame.

Contract:
    dtype:          ``numpy.uint8`` (values 0-255)
    shape:          ``(H, W, 3)``
    channel order:  **BGR**, the order OpenCV decodes a frame into

BGR is not a bug to be fixed. Frames are consumed by OpenCV operations
throughout the pipeline; converting to RGB here would silently change
what every downstream stage sees.
"""


@dataclass(frozen=True)
class MotionImage:
    """The encoded output handed to a downstream neural network.

    Contract:
        ``data`` dtype:  ``numpy.uint8``
        ``data`` shape:  fixed and identical for every ``MotionImage``
                         the pipeline produces, regardless of how many
                         frames were sampled or how fast the motion was.
                         The encoder resizes or pads to reach it.

    A fixed shape is the point of this type: a downstream model declares
    one input size, so a pipeline that sometimes emits a different shape
    must fail loudly rather than at inference time.

    Attributes:
        data: The encoded motion image.
        frame_indices: Indices, within the source window, of the frames
            the adaptive sampler selected. Kept so the sampler's
            behaviour is inspectable from its output alone.
    """

    data: Frame
    frame_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        """Ensure the image data uses 8-bit unsigned integer values."""
        if self.data.dtype != np.uint8:
            raise TypeError(f"MotionImage.data must be uint8, got {self.data.dtype}")
