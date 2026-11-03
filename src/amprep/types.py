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
        ``data`` shape:  ``(H, W)``, one channel. The ``width`` and
                         ``height`` the preprocessor was given, or the
                         size of the incoming frames when it was given
                         none. Nothing is padded. It does not depend on
                         how many frames were sampled or how fast the
                         motion was, and the frame size cannot change
                         mid-stream without ``reset()``, so it holds for
                         every ``MotionImage`` of a stream.

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
    """A full run of consecutive foreground masks, collected while motion lasted.

    The handover between the collector and the two stages that read it:
    the adaptive sampler measures how fast the foreground changes across
    ``masks``, and the encoder paints the masks the sampler picked. Both
    work on masks alone, so the frames themselves are not carried: a
    window of full-colour frames would cost memory and nothing would
    read it.

    Contract:
        ``masks`` is in capture order, one per frame, each a ``uint8``
        ``(H, W)`` array. A window is never empty and never partial: the
        collector emits one only once it is full, so a stage reading this
        never has to ask whether it got a whole one.

    Attributes:
        masks: The foreground mask of each frame, in capture order.
    """

    masks: tuple[Frame, ...]

    def __post_init__(self) -> None:
        """Ensure there is anything here."""
        if not self.masks:
            raise ValueError("Window must not be empty")

    def __len__(self) -> int:
        """The number of frames in the window."""
        return len(self.masks)
