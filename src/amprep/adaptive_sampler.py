import logging
from collections import deque

import numpy as np

from amprep.types import Window

_log = logging.getLogger(__name__)

DEFAULT_SAMPLE_FRAMES = 4
"""Frames kept from every window, whatever the motion did.

Fixed on purpose. The encoder behind this stage has to emit one shape
every time, and a constant number of frames going in is the cheapest way
to get there — it has nothing to pad or drop. Keep it at or below
``window_frames`` or short windows will yield fewer.
"""

DEFAULT_FULL_SPEED = 0.35
"""Change rate at which the samples sit on consecutive frames.

A share of the subject changing per frame. Normal walking measures about
0.17 and running about 0.55, so 0.35 puts walking mid-range and running
at the tightest spacing. Measured on synthetic masks. In auto mode this
is only the starting point, used until the scene has taught a better one.
"""

AUTO_PERCENTILE = 90
"""Where among the scene's recent change rates auto mode sets ``full_speed``.

Not the maximum: a single flicker or mask glitch would drag the line up
and leave every ordinary window spread out. The 90th percentile is the
fast end of what the scene really does.
"""

AUTO_WARMUP_WINDOWS = 5
"""Windows auto mode sees before it trusts what it has learned.

Until then ``DEFAULT_FULL_SPEED`` is used, because a percentile of one or
two readings is just those readings.
"""

AUTO_HISTORY_WINDOWS = 50
"""Most recent windows auto mode learns from.

Bounded, so a live stream holds a fixed amount of memory, and so a scene
that changes pace — a walk turning into a run — is followed rather than
averaged away. At the default ten-frame window and 30 fps this is about
seventeen seconds of motion.
"""

AUTO_MIN_FULL_SPEED = 0.15
"""The lowest ``full_speed`` auto mode will learn.

Just under a walk. Without a floor, a scene of barely-moving subjects
would learn a tiny line, count its own drift as fast, and pack the
samples onto consecutive frames that hardly differ — the very failure
the spacing exists to avoid.
"""


class AdaptiveFrameSampler:
    """Picks which frames of a window to keep, closer together when motion is fast.

    Always the same number of frames; what velocity changes is how far
    apart they sit. The aim is to hold the *amount of change* between
    consecutive samples roughly steady rather than the time between
    them::

        window of 10:   0 1 2 3 4 5 6 7 8 9

        slow motion     X . . X . . X . . X    spread across the window
        fast motion     . . . . . . X X X X    packed at the newest frames

    Both extremes are there for the same reason. Sampling a slow scene
    tightly gives four near-identical frames, which describe no motion at
    all; sampling a fast one across the whole window gives four frames
    the subject has jumped between, which describe motion that looks
    discontinuous. Moving the spacing keeps the step between samples in a
    useful middle band either way.

    The cost is that fast motion is sampled from the end of the window
    and the opening frames are not represented. That is the trade the
    fixed count buys: with a constant number of frames, coverage and
    density cannot both be held. Density is what carries the shape of the
    movement, and when it has to be bought, it is bought from the oldest
    frames rather than the freshest — the samples always reach the most
    recent frame in the window.

    Velocity is read from the masks rather than the frames: a mask is
    already "what is moving", so the share of it that changes between one
    frame and the next is a rate of change needing no optical flow.

    ``full_speed`` is the change rate at which the spacing bottoms out,
    and what counts as fast depends on the footage — distance, lens and
    subject all move it. Left as ``None`` it is learned: the sampler
    remembers the change rates of recent windows and uses the fast end of
    them (see ``AUTO_PERCENTILE``), so whatever is fast *for this scene*
    gets the tightest spacing. That makes the sampler stateful; call
    ``reset()`` when the scene changes. Pass a number to fix it instead,
    and every window is judged on its own.

    Args:
        sample_frames: Frames kept from every window. At least 2, so
            there is always a before and an after. A window holding
            fewer frames than this yields only what it has, so keep it
            at or below the collector's ``window_frames``.
        full_speed: Change rate at which the samples are packed onto
            consecutive frames, as a share of the subject changing per
            frame. ``None``, the default, learns it from the scene.
    """

    def __init__(
        self,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES,
        full_speed: float | None = None,
    ) -> None:
        # bool is an int subclass, so it is refused by name before the
        # range check, where True would otherwise read as 1.
        if isinstance(sample_frames, bool) or not isinstance(sample_frames, int):
            raise ValueError(f"sample_frames must be an integer, got {sample_frames!r}")
        if sample_frames < 2:
            raise ValueError(f"sample_frames must be at least 2, got {sample_frames}")
        if full_speed is not None:
            if isinstance(full_speed, bool) or not isinstance(full_speed, (int, float)):
                raise ValueError(f"full_speed must be a number, got {full_speed!r}")
            if not full_speed > 0:
                raise ValueError(f"full_speed must be positive, got {full_speed!r}")

        self._sample_frames = sample_frames
        self._auto = full_speed is None
        self._speeds: deque[float] = deque(maxlen=AUTO_HISTORY_WINDOWS)
        self._full_speed = (
            DEFAULT_FULL_SPEED if full_speed is None else float(full_speed)
        )

    @property
    def sample_frames(self) -> int:
        """Frames kept from every window."""
        return self._sample_frames

    @property
    def full_speed(self) -> float:
        """Change rate at which the samples sit on consecutive frames.

        In auto mode, the value learned so far — ``DEFAULT_FULL_SPEED``
        until warm-up ends. Look at this to see what the scene taught it.
        """
        return self._full_speed

    @property
    def auto(self) -> bool:
        """Whether ``full_speed`` is learned from the scene rather than fixed."""
        return self._auto

    def reset(self) -> None:
        """Forget what the scene taught and start again from the default.

        Only auto mode has anything to forget; a fixed ``full_speed`` is
        left as it is.
        """
        self._speeds.clear()
        if self._auto:
            self._full_speed = DEFAULT_FULL_SPEED

    def velocity(self, window: Window) -> float:
        """Return how fast the foreground changes, over the second half of ``window``.

        For each pair of consecutive masks, the share of the subject that
        flipped between them: pixels that changed, over pixels foreground
        in either. The result is the mean of those shares, so it stays in
        ``[0, 1]`` whatever the window length or the resolution — an
        average over pairs rather than a total divided by the number of
        steps, which would shrink as the window grew.

        The second half is measured because that is the half the samples
        are taken from when motion is fast.

        Unlike tracking the centre of the foreground, this sees a subject
        approaching the camera, which grows without moving sideways, and
        two subjects moving apart, whose combined centre stays put.

        It is relative to the subject's size: a small, distant subject
        reads faster than a large, close one moving at the same speed,
        because the same displacement flips a larger share of it.

        Args:
            window: The window to measure. See ``Window``.

        Returns:
            Mean share of the foreground changing between consecutive
            frames, in ``[0, 1]``. Zero when no measured pair holds any
            foreground at all.
        """
        start = (len(window) - 1) // 2
        shares = []
        for earlier, later in zip(
            window.masks[start:-1], window.masks[start + 1 :], strict=True
        ):
            was, now = earlier > 0, later > 0
            # Pixels foreground in either frame. Dividing by this rather
            # than by the frame area is what makes the reading relative
            # to the subject instead of to the picture.
            union = np.count_nonzero(was | now)
            if union:
                shares.append(np.count_nonzero(was ^ now) / union)
        return float(np.mean(shares)) if shares else 0.0

    def select(self, window: Window) -> tuple[int, ...]:
        """Return the indices of the frames worth keeping from ``window``.

        In auto mode this also teaches the sampler: the window's change
        rate joins the history before ``full_speed`` is read, so the
        window is judged against a line it helped draw.

        Args:
            window: The window to sample. See ``Window``.

        Returns:
            Exactly ``sample_frames`` indices, or every index the window
            has if it holds fewer. Strictly increasing, within the
            window, and always reaching its last frame.
        """
        count = min(self._sample_frames, len(window))
        if count < 2:
            return (0,)

        speed = self.velocity(window)
        if self._auto:
            self._learn(speed)

        # Saturating rather than proportional: past full_speed the
        # samples are already on consecutive frames and cannot tighten
        # further, and an outlier reading must not reach past it.
        fraction = min(1.0, speed / self._full_speed)

        # The span is how much of the window the samples cover. At its
        # widest they reach back to the first frame; at its narrowest
        # they sit on consecutive frames, which needs exactly count - 1
        # steps. Using a span rather than an integer stride is what keeps
        # the choice fine-grained: a stride of 1 or 2 is all a ten-frame
        # window can offer, while the span moves through every value
        # between.
        widest = len(window) - 1
        narrowest = count - 1
        span = round(widest - fraction * (widest - narrowest))

        # Anchored at the newest frame and reaching backwards, so
        # tightening the spacing drops the oldest frames rather than the
        # freshest. Spacing stays at least one frame because span is
        # never below count - 1, so rounding cannot collapse two samples
        # onto the same index.
        last = len(window) - 1
        indices = np.linspace(last - span, last, count).round().astype(int)

        _log.debug(
            "change rate %.4f (full speed %.4f) -> %d frames spanning %d of %d",
            speed,
            self._full_speed,
            count,
            span,
            len(window),
        )
        return tuple(int(index) for index in indices)

    def _learn(self, speed: float) -> None:
        """Fold one window's change rate into the learned ``full_speed``."""
        self._speeds.append(speed)
        if len(self._speeds) < AUTO_WARMUP_WINDOWS:
            return
        learned = float(np.percentile(self._speeds, AUTO_PERCENTILE))
        self._full_speed = max(AUTO_MIN_FULL_SPEED, learned)
