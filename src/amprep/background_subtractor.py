from abc import ABC, abstractmethod

import numpy as np

from amprep.types import Frame


def _validate(array: Frame, label: str, ndim: int) -> None:
    """Raise when ``array`` is not a ``uint8`` array of ``ndim`` dimensions.

    A three-dimensional array is additionally required to be BGR, i.e. to
    have exactly three channels.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected {label} to be a NumPy array, got {type(array)}")
    if array.dtype != np.uint8:
        raise TypeError(f"Expected {label} to be uint8, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(
            f"Expected {label} to be {ndim}-dimensional, got {array.shape}"
        )
    if ndim == 3 and array.shape[2] != 3:
        raise ValueError(f"Expected {label} to be (H, W, 3), got {array.shape}")


class BackgroundSubtractor(ABC):
    """Separates moving foreground from a learned background model.

    Subclass this and implement ``_apply`` to plug in your own subtractor.
    Do not override ``apply`` — it validates the input frame and the
    returned mask against the contract below. Pass an instance to
    ``AdaptiveMotionPreprocessor(background_subtractor=...)``; leave it
    unset and the package's default implementation is used instead.

    Contract:
        ``_apply`` takes a ``uint8`` ``(H, W, 3)`` BGR frame and must
        return a **single-channel** ``uint8`` mask of shape ``(H, W)`` —
        the same height and width as its input. Implementations are
        stateful: consecutive calls are expected to come from the same
        video, in order, so the background model can adapt.
    """

    @abstractmethod
    def _apply(self, frame: Frame) -> Frame:
        """Return the foreground mask for ``frame``.

        Args:
            frame: The input frame to process. See ``Frame`` for its
                contract.

        Returns:
            A single-channel ``uint8`` mask of shape ``(H, W)`` matching
            the input frame's height and width.
        """

    def apply(self, frame: Frame) -> Frame:
        """Return the foreground mask for ``frame``."""
        _validate(frame, "the input frame", ndim=3)

        mask = self._apply(frame)

        _validate(mask, "the mask returned by _apply", ndim=2)
        if mask.shape != frame.shape[:2]:
            raise ValueError(
                f"_apply must preserve frame H×W: {frame.shape[:2]} -> {mask.shape}"
            )

        return mask
