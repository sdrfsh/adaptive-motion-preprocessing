import logging

import numpy as np

from amprep.types import Frame, Window

_log = logging.getLogger(__name__)

DEFAULT_SAMPLE_FRAMES = 4
"""Frames kept from every window, whatever the motion did.

Fixed on purpose. The encoder behind this stage has to emit one shape
every time, and a constant number of frames going in is the cheapest way
to get there — it has nothing to pad or drop. Keep it at or below
``window_frames`` or short windows will yield fewer.
"""

DEFAULT_FULL_SPEED = 0.05
"""Velocity at which sampling is at its tightest, as a fraction of the diagonal.

A centroid crossing 5% of the frame diagonal per frame is already fast —
a tenth of the picture every two frames. At or above it the samples sit
on consecutive frames, which is as close together as they can get.
Expressing it as a fraction of the diagonal rather than in pixels keeps
the same value meaningful whatever resolution the caller feeds in.
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
    """Picks which frames of a window to keep, closer together when motion is fast.

    Always the same number of frames; what velocity changes is how far
    apart they sit. The aim is to hold the *distance* between consecutive
    samples roughly steady rather than the time between them::

        window of 10:   0 1 2 3 4 5 6 7 8 9

        slow motion     X . . X . . X . . X    spread across the window
        fast motion     X X X X . . . . . .    bunched at the start

    Both extremes are there for the same reason. Sampling a slow scene
    tightly gives four near-identical frames, which describe no motion at
    all; sampling a fast one across the whole window gives four frames
    the subject has jumped between, which describe motion that looks
    discontinuous. Moving the spacing keeps the step between samples in a
    useful middle band either way.

    The cost is that fast motion is sampled from the start of the window
    and the tail is not represented. That is the trade the fixed count
    buys: with a constant number of frames, coverage and density cannot
    both be held, and density is what carries the shape of the movement.

    Velocity is read from the masks rather than the frames: a mask is
    already "where the moving thing is", so the centroid of its
    foreground moves with the subject, and the distance it covers between
    two known moments is a velocity in pixels per frame. Measuring the
    same thing from raw frames would need optical flow.

    The first mask is compared against the middle one, as the issue
    specifies. Half a window is enough to characterise a burst the
    trigger already judged to be one continuous piece of motion, and
    stopping at the middle keeps the estimate away from the end of the
    window, where an object is most likely to be leaving the scene.

    Stateless: each window is judged on its own, so there is nothing to
    reset between scenes.

    Args:
        sample_frames: Frames kept from every window. At least 2, so
            there is always a before and an after. A window holding
            fewer frames than this yields only what it has, so keep it
            at or below the collector's ``window_frames``.
        full_speed: Velocity at which the samples are packed onto
            consecutive frames, as a fraction of the frame diagonal
            travelled per frame.
    """

    def __init__(
        self,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES,
        full_speed: float = DEFAULT_FULL_SPEED,
    ) -> None:
        # bool is an int subclass, so it is refused by name before the
        # range check, where True would otherwise read as 1.
        if isinstance(sample_frames, bool) or not isinstance(sample_frames, int):
            raise ValueError(f"sample_frames must be an integer, got {sample_frames!r}")
        if sample_frames < 2:
            raise ValueError(f"sample_frames must be at least 2, got {sample_frames}")
        if isinstance(full_speed, bool) or not isinstance(full_speed, (int, float)):
            raise ValueError(f"full_speed must be a number, got {full_speed!r}")
        if not full_speed > 0:
            raise ValueError(f"full_speed must be positive, got {full_speed!r}")

        self._sample_frames = sample_frames
        self._full_speed = float(full_speed)

    @property
    def sample_frames(self) -> int:
        """Frames kept from every window."""
        return self._sample_frames

    @property
    def full_speed(self) -> float:
        """Velocity at which the samples sit on consecutive frames."""
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
            Exactly ``sample_frames`` indices, or every index the window
            has if it holds fewer. Strictly increasing, within the
            window, and always starting at its first frame.
        """
        count = min(self._sample_frames, len(window))
        if count < 2:
            return (0,)

        speed = self.velocity(window)
        # Saturating rather than proportional: past full_speed the
        # samples are already on consecutive frames and cannot tighten
        # further, and an outlier centroid jump must not reach past it.
        fraction = min(1.0, speed / self._full_speed)

        # The span is how much of the window the samples cover. At its
        # widest they reach the last frame; at its narrowest they sit on
        # consecutive frames, which needs exactly count - 1 steps. Using
        # a span rather than an integer stride is what keeps the choice
        # fine-grained: a stride of 1 or 2 is all a ten-frame window can
        # offer, while the span moves through every value between.
        widest = len(window) - 1
        narrowest = count - 1
        span = round(widest - fraction * (widest - narrowest))

        # Spacing is at least one frame because span >= count - 1, so
        # rounding cannot collapse two samples onto the same index.
        indices = np.linspace(0, span, count).round().astype(int)

        _log.debug(
            "velocity %.4f -> %d frames spanning %d of %d",
            speed,
            count,
            span,
            len(window),
        )
        return tuple(int(index) for index in indices)
