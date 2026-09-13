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


@dataclass(frozen=True)
class Window:
    """A full run of consecutive frames, collected while motion lasted.

    The handover between the collector and the two stages that read it:
    the adaptive sampler measures how fast things moved from ``masks``,
    where the foreground centroid is trivially locatable, and the encoder
    stacks the ``frames`` the sampler picked. Both are carried because
    neither can be recovered from the other, and they are carried
    together because a mask is only meaningful beside the frame it came
    from.

    Contract:
        ``frames`` and ``masks`` are the same length, and the entries at
        any index were captured at the same moment. A window is never
        empty and never partial: the collector emits one only once it is
        full, so a stage reading this never has to ask whether it got a
        whole one.

    Attributes:
        frames: The frames, in capture order. See ``Frame``.
        masks: The foreground mask for each frame, in the same order.
    """

    frames: tuple[Frame, ...]
    masks: tuple[Frame, ...]

    def __post_init__(self) -> None:
        """Ensure every frame kept its mask, and that there is anything here."""
        if len(self.frames) != len(self.masks):
            raise ValueError(
                f"Window needs one mask per frame, got {len(self.frames)} "
                f"frames and {len(self.masks)} masks"
            )
        if not self.frames:
            raise ValueError("Window must not be empty")

    def __len__(self) -> int:
        """The number of frames in the window."""
        return len(self.frames)
