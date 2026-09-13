import logging

import numpy as np

from amprep.types import Frame

_log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.01
"""Fraction of the frame that must be foreground to count as motion.

Sits between a speck (about 0.25%) and a small object (about 2%) on
synthetic scenes. Tune it on real footage using ``last_fraction``.
"""


class MotionTrigger:
    """Says whether something is moving, one mask at a time.

    Active when the foreground fraction is at or above ``threshold``,
    idle below it. Reacts from the very first mask; warm-up is decided
    by the background subtractor, not here.

    Args:
        threshold: Fraction of the frame, in (0, 1]. Defaults to
            ``DEFAULT_THRESHOLD`` (1%).
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"threshold must be a number, got {threshold!r}")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
        self._threshold = float(threshold)
        self.reset()

    @property
    def threshold(self) -> float:
        """The threshold in use."""
        return self._threshold

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def last_fraction(self) -> float:
        """Last measured fraction. Look at this to tune ``threshold``."""
        return self._last_fraction

    def reset(self) -> None:
        self._active = False
        self._last_fraction = 0.0

    def update(self, mask: Frame) -> bool:
        # ``count_nonzero`` hands back a NumPy integer, which would make
        # the fraction a ``float64`` and the comparison below a
        # ``np.bool_`` — and ``np.bool_`` is not a ``bool``, so a caller
        # writing ``if trigger.update(mask) is True`` would silently
        # never match. Coercing here keeps both annotations honest.
        self._last_fraction = float(np.count_nonzero(mask) / mask.size)
        self._active = self._last_fraction >= self._threshold
        _log.debug("foreground %.4f -> %s", self._last_fraction, self._active)
        return self._active
