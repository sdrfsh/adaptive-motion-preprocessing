from abc import ABC, abstractmethod

import cv2
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

    Subclass this and implement ``_apply`` and ``reset`` to plug in your
    own subtractor. Do not override ``apply`` — it validates the input
    frame and the returned mask against the contract below. Pass an
    instance to ``AdaptiveMotionPreprocessor(background_subtractor=...)``;
    leave it unset and the package's default implementation is used
    instead.

    Contract:
        ``_apply`` takes a ``uint8`` ``(H, W, 3)`` BGR frame and must
        return a **single-channel** ``uint8`` mask of shape ``(H, W)`` —
        the same height and width as its input. Implementations are
        stateful: consecutive calls are expected to come from the same
        video, in order, so the background model can adapt.

        ``reset`` discards that accumulated state. It is the caller's way
        of saying the next frame belongs to a different scene, so a model
        learned from the preceding frames describes a background that is
        no longer there.

        ``warmup_frames`` says how many frames a subtractor needs before
        its masks mean anything. It is concrete and defaults to 0, so a
        subclass that is useful from its first frame writes nothing;
        override it only if yours is not.
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

    @abstractmethod
    def reset(self) -> None:
        """Discard the learned background model.

        Called when the scene changes — a cut, a camera move, or the
        start of a different video — so that state accumulated from the
        preceding frames does not leak into the next one. Implementations
        that hold no state between frames should implement this as a
        no-op.
        """

    @property
    def warmup_frames(self) -> int:
        """Frames before this subtractor's masks are meaningful. Default 0."""
        return 0

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


class KNNBackgroundSubtractor(BackgroundSubtractor):
    """Separates foreground with OpenCV's KNN background subtractor.

    The package default. KNN models each pixel's recent history as a
    cloud of samples and calls a new value foreground when too few of its
    neighbours sit close to it. That copes with the swaying leaves and
    rippling water a single-Gaussian model keeps flagging as motion,
    which matters here because every false foreground pixel becomes
    spurious motion in the encoded output.

    The mask is strictly binary: 0 for background, 255 for foreground.
    OpenCV marks shadows with an intermediate grey, and a downstream
    stage that thresholds a mask at anything other than 0 would silently
    treat those pixels as half-present, so they are folded into one of
    the two labels here instead.

    Args:
        history: Number of recent frames the model is built from. Longer
            tolerates slower background change but takes longer to
            forget an object that stops moving and becomes scenery.
        dist2_threshold: Squared distance between a pixel and a sample
            for that sample to count as its neighbour. Larger admits
            more variation as background, so the mask keeps less noise
            and fewer faint edges.
        detect_shadows: Whether to recognise shadows and exclude them
            from the foreground. Shadows move with their subject, so
            leaving this on keeps them out of the silhouette at a modest
            cost in speed. Turn it off and they are foreground like any
            other change.
    """

    def __init__(
        self,
        history: int = 500,
        dist2_threshold: float = 400.0,
        detect_shadows: bool = True,
    ) -> None:
        if not isinstance(history, int) or isinstance(history, bool) or history <= 0:
            raise ValueError(f"history must be a positive integer, got {history!r}")
        if (
            isinstance(dist2_threshold, bool)
            or not isinstance(dist2_threshold, (int, float))
            or dist2_threshold <= 0
        ):
            raise ValueError(
                f"dist2_threshold must be a positive number, got {dist2_threshold!r}"
            )
        if not isinstance(detect_shadows, bool):
            raise ValueError(
                f"detect_shadows must be a boolean, got {detect_shadows!r}"
            )

        self._history = history
        self._dist2_threshold = float(dist2_threshold)
        self._detect_shadows = detect_shadows
        self._subtractor = self._build()

    def _build(self) -> cv2.BackgroundSubtractorKNN:
        """Return a freshly constructed subtractor with no learned model."""
        return cv2.createBackgroundSubtractorKNN(
            history=self._history,
            dist2Threshold=self._dist2_threshold,
            detectShadows=self._detect_shadows,
        )

    @property
    def warmup_frames(self) -> int:
        """Frames before KNN's masks are meaningful.

        KNN calls a pixel foreground when too few of its recent samples
        sit close to the new value, so on the opening frames it has too
        few samples to judge against and marks most of the image as
        motion. Four is where that settles in practice.
        """
        return 4

    def _apply(self, frame: Frame) -> Frame:
        mask = self._subtractor.apply(frame)
        # OpenCV labels foreground 255 and everything else — background,
        # and shadows when detect_shadows is on — below it.
        return np.where(mask == 255, np.uint8(255), np.uint8(0))

    def reset(self) -> None:
        """Discard the learned background model.

        OpenCV exposes no way to clear a subtractor in place, so the
        model is thrown away and rebuilt. The next frame is treated as
        the opening frame of a new scene.
        """
        self._subtractor = self._build()
