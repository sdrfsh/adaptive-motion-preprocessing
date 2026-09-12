from abc import ABC, abstractmethod

import numpy as np

from amprep.types import Frame


class NoiseReducer(ABC):
    """Removes sensor noise from a frame before background subtraction.

    Subclass this to plug in your own denoising. Pass an instance to
    ``AdaptiveMotionPreprocessor(noise_reducer=...)``; leave it unset and
    the package's default implementation is used instead.

    Contract:
        ``apply`` must return a frame with the **same shape and dtype**
        as its input — ``uint8``, ``(H, W, 3)``, BGR. Downstream stages
        size their buffers from the first frame they see, so a stage that
        changes shape mid-stream breaks them silently.
    """

    @staticmethod
    def validate_frame(frame: Frame) -> None:
        """Raise ``TypeError`` when ``frame`` is not a valid video frame."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(frame)}")
        if frame.dtype != np.uint8:
            raise TypeError(f"Expected uint8 frame, got {frame.dtype}")

    @abstractmethod
    def _apply(self, frame: Frame) -> Frame:
        """Return a denoised copy of ``frame``.

        Args:
            frame: The frame to denoise. See ``Frame`` for its contract.

        Returns:
            A frame of the same shape and dtype as the input.
        """
        pass

    def apply(self, frame: Frame) -> Frame:
        """Return a denoised copy of ``frame``.

        Args:
            frame: The frame to denoise. See ``Frame`` for its contract.

        Returns:
            A frame of the same shape and dtype as the input.
        """
        self.validate_frame(frame)
        result = self._apply(frame)
        self.validate_frame(result)
        return result