from abc import ABC, abstractmethod

import cv2
import numpy as np

from amprep.types import Frame


def _validate(frame: Frame, label: str) -> None:
    """Raise ``TypeError`` when ``frame`` is not a valid video frame."""
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected {label} to be a NumPy array, got {type(frame)}")
    if frame.dtype != np.uint8:
        raise TypeError(f"Expected {label} to be uint8, got {frame.dtype}")


class NoiseReducer(ABC):
    """Removes sensor noise from a frame before background subtraction.

    Subclass this and implement ``_apply`` to plug in your own denoising.
    Do not override ``apply``: it validates the input and the returned
    frame against the contract below. Pass an instance to
    ``AdaptiveMotionPreprocessor(noise_reducer=...)``; leave it unset and
    the package's default implementation is used instead.

    Contract:
        ``_apply`` must return a frame with the **same shape and dtype**
        as its input: ``uint8``, ``(H, W, 3)``, BGR. Downstream stages
        size their buffers from the first frame they see, so a stage that
        changes shape mid-stream breaks them silently.
    """

    @abstractmethod
    def _apply(self, frame: Frame) -> Frame:
        """Return a denoised copy of ``frame``.

        Args:
            frame: The frame to denoise. See ``Frame`` for its contract.

        Returns:
            A frame of the same shape and dtype as the input.
        """

    def apply(self, frame: Frame) -> Frame:
        """Return a denoised copy of ``frame``."""
        _validate(frame, "the input frame")

        result = self._apply(frame)

        _validate(result, "the frame returned by _apply")
        if result.shape != frame.shape:
            raise ValueError(
                f"_apply must preserve shape: {frame.shape} -> {result.shape}"
            )

        return result


class MedianNoiseReducer(NoiseReducer):
    """Removes sensor noise with a median filter.

    The package default. A median discards extreme pixel values instead
    of averaging them in, which suits the salt-and-pepper noise that
    would otherwise become phantom foreground in the mask, and it blurs
    edges far less than a Gaussian, so the silhouette the sampler and
    encoder depend on stays sharp.

    Args:
        ksize: Aperture size, an odd integer greater than 1. Larger
            removes more noise and costs more time. ``3`` is roughly
            seven times faster with sharper edges, at about twice the
            residual noise.
    """

    def __init__(self, ksize: int = 5) -> None:
        if not isinstance(ksize, int) or ksize <= 1 or ksize % 2 == 0:
            raise ValueError(
                f"ksize must be an odd integer greater than 1, got {ksize!r}"
            )
        self._ksize = ksize

    def _apply(self, frame: Frame) -> Frame:
        return cv2.medianBlur(frame, self._ksize)
