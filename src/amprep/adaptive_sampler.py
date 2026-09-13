import logging

import numpy as np

from amprep.types import Frame, Window

_log = logging.getLogger(__name__)

DEFAULT_MIN_FRAMES = 3
"""Frames kept from a window when nothing is moving.

Never fewer than this, however still the scene is: a window was only
collected because the trigger saw motion, so there is something in it
worth showing.
"""

DEFAULT_MAX_FRAMES = 8
"""Frames kept from a window when motion is at or above ``full_speed``."""

DEFAULT_FULL_SPEED = 0.05
"""Velocity at which sampling saturates, as a fraction of the diagonal.

A centroid crossing 5% of the frame diagonal per frame is already fast —
a tenth of the picture every two frames. Beyond it there is nothing left
to give, because ``max_frames`` has been reached. Expressing it as a
fraction of the diagonal rather than in pixels keeps the same value
meaningful whatever resolution the caller feeds in.
"""


def _centroid(mask: Frame) -> tuple[float, float] | None:
    """Return the (row, column) centre of a mask's foreground, or ``None``.

    ``None`` means the mask has no foreground at all, which the caller
    has to treat as "velocity unknown" rather than as zero movement: an
    object that vanished did not stand still.
    """
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return float(rows.mean()), float(columns.mean())


class AdaptiveFrameSampler:
    """Picks which frames of a window to keep, more of them when motion is fast.

    Fast motion changes a lot between frames, so a fixed sampling rate
    either wastes frames on a slow scene or skips past a quick one. This
    measures how far the foreground travelled inside the window and keeps
    more frames when it travelled further.

    Velocity is read from the masks rather than the frames: a mask is
    already "where the moving thing is", so the centroid of its
    foreground moves with the subject, and the distance it covers between
    two known moments is a velocity in pixels per frame. Measuring the
    same thing from raw frames would need optical flow.

    The first mask is compared against the middle one, as the issue
    specifies. Half a window is enough to characterise a burst that the
    trigger already judged to be one continuous piece of motion, and
    stopping at the middle keeps the estimate away from the end of the
    window, where an object is most likely to be leaving the scene.

    The measured velocity is divided by ``full_speed`` to give a
    saturating fraction, and the kept-frame count moves linearly between
    ``min_frames`` and ``max_frames`` across it. Those frames are then
    spread evenly over the whole window, so the selection always spans
    the burst rather than clustering at its start.

    Stateless: each window is judged on its own, so there is nothing to
    reset between scenes.

    Args:
        min_frames: Frames kept when the scene is still. At least 2, so
            there is always a before and an after.
        max_frames: Frames kept at or above ``full_speed``. At least
            ``min_frames``; equal to it disables the adaptation.
        full_speed: Velocity at which ``max_frames`` is reached, as a
            fraction of the frame diagonal travelled per frame.
    """

    def __init__(
        self,
        min_frames: int = DEFAULT_MIN_FRAMES,
        max_frames: int = DEFAULT_MAX_FRAMES,
        full_speed: float = DEFAULT_FULL_SPEED,
    ) -> None:
        # bool is an int subclass, so it is refused by name before the
        # range checks, where True would otherwise read as 1.
        if isinstance(min_frames, bool) or not isinstance(min_frames, int):
            raise ValueError(f"min_frames must be an integer, got {min_frames!r}")
        if min_frames < 2:
            raise ValueError(f"min_frames must be at least 2, got {min_frames}")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            raise ValueError(f"max_frames must be an integer, got {max_frames!r}")
        if max_frames < min_frames:
            raise ValueError(
                f"max_frames must be at least min_frames ({min_frames}), "
                f"got {max_frames}"
            )
        if isinstance(full_speed, bool) or not isinstance(full_speed, (int, float)):
            raise ValueError(f"full_speed must be a number, got {full_speed!r}")
        if not full_speed > 0:
            raise ValueError(f"full_speed must be positive, got {full_speed!r}")

        self._min_frames = min_frames
        self._max_frames = max_frames
        self._full_speed = float(full_speed)

    @property
    def min_frames(self) -> int:
        """Frames kept from a still window."""
        return self._min_frames

    @property
    def max_frames(self) -> int:
        """Frames kept from a window at or above ``full_speed``."""
        return self._max_frames

    @property
    def full_speed(self) -> float:
        """Velocity at which the kept-frame count saturates."""
        return self._full_speed

    def velocity(self, window: Window) -> float:
        """Return the foreground's speed across the first half of ``window``.

        Measured as the distance the foreground centroid travels between
        the first mask and the middle one, per frame, as a fraction of
        the frame diagonal. Zero when the window is too short to have a
        middle, or when either mask holds no foreground to locate.

        Args:
            window: The window to measure. See ``Window``.

        Returns:
            Fraction of the frame diagonal covered per frame, never
            negative. Distance is unsigned, so a subject that doubles
            back reads as slower than one that keeps going — which is
            the intent, since what is being asked is how much changes
            between neighbouring frames.
        """
        middle = len(window) // 2
        if middle == 0:
            return 0.0

        start = _centroid(window.masks[0])
        end = _centroid(window.masks[middle])
        if start is None or end is None:
            return 0.0

        travelled = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        diagonal = float(np.hypot(*window.masks[0].shape[:2]))
        return travelled / diagonal / middle

    def select(self, window: Window) -> tuple[int, ...]:
        """Return the indices of the frames worth keeping from ``window``.

        Args:
            window: The window to sample. See ``Window``.

        Returns:
            Indices into the window, strictly increasing and within it.
            Always at least two, always spanning the window from its
            first frame to its last.
        """
        speed = self.velocity(window)
        # Saturating rather than proportional: past full_speed there is
        # no more resolution to buy, and an outlier centroid jump must
        # not ask for more frames than the window holds.
        fraction = min(1.0, speed / self._full_speed)
        wanted = round(
            self._min_frames + fraction * (self._max_frames - self._min_frames)
        )
        count = max(2, min(wanted, len(window)))

        # linspace spans both ends, and unique leaves the result sorted
        # and free of the repeats that rounding creates on short windows,
        # which is what makes the indices strictly increasing.
        indices = np.unique(np.linspace(0, len(window) - 1, count).round().astype(int))

        _log.debug("velocity %.4f -> %d of %d frames", speed, indices.size, len(window))
        return tuple(int(index) for index in indices)
