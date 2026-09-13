import logging

from amprep.types import Frame, Window

_log = logging.getLogger(__name__)

DEFAULT_WINDOW_FRAMES = 10
"""Frames gathered into one window.

Counts frames, not seconds: ten is about 0.33 s at 30 fps and 1 s at
10 fps. The package never reads the frame rate, so converting between
the two is the caller's arithmetic.
"""

MIN_WINDOW_FRAMES = 2
"""Fewest frames a window can be asked to hold.

The adaptive sampler compares the first frame of a window against the
middle one, which needs two distinct frames to be a comparison at all.
"""


class FrameWindowCollector:
    """Gathers frames into fixed-size windows while motion lasts.

    Fed one frame at a time along with the trigger's verdict on it. While
    the trigger says moving, frames are collected; on the frame that
    fills the window, that window is handed back and the next one begins
    immediately. There is no cooldown, so a long burst of motion produces
    back-to-back windows with no frame skipped between them::

        trigger:  idle   ACTIVE x 10        ACTIVE x 10        idle
        window:          [1..10] -> out     [1..10] -> out     (partial, dropped)

    Frames and their masks are kept together, because the two stages
    downstream need different halves of the same moment: the sampler
    reads frames, the encoder reads masks.

    A partial window is discarded when motion stops. Velocity is measured
    across a window, so a window cut short describes a slower movement
    than actually happened: a stage cannot tell "the object slowed" from
    "the recording stopped". Dropping the remainder loses a fragment of
    real motion; keeping it would report a false one, which is worse.

    Args:
        window_frames: Frames per window, at least ``MIN_WINDOW_FRAMES``.
            Defaults to ``DEFAULT_WINDOW_FRAMES`` (10).
    """

    def __init__(self, window_frames: int = DEFAULT_WINDOW_FRAMES) -> None:
        # bool is an int subclass, so it is turned away by name before the
        # range check, where True would otherwise read as 1.
        if (
            isinstance(window_frames, bool)
            or not isinstance(window_frames, int)
            or window_frames < MIN_WINDOW_FRAMES
        ):
            raise ValueError(
                f"window_frames must be an integer of at least "
                f"{MIN_WINDOW_FRAMES}, got {window_frames!r}"
            )
        self._window_frames = window_frames
        self._clear()

    @property
    def window_frames(self) -> int:
        """Frames per window, as given to the constructor."""
        return self._window_frames

    @property
    def pending(self) -> int:
        """Frames collected so far towards the window being built.

        Zero between windows as well as while idle: a window that is
        handed over leaves nothing behind it.
        """
        return len(self._frames)

    def reset(self) -> None:
        """Drop whatever was being collected and start again empty.

        The caller's way of saying the next frame belongs to a different
        scene, so frames gathered before a cut cannot end up in a window
        beside frames from after it.
        """
        self._clear()

    def _clear(self) -> None:
        """Empty the buffers, saying nothing about why they were emptied.

        Kept separate from ``reset`` because the two internal callers are
        not scene changes: one abandons a partial window, the other has
        just handed a full one over. Should ``reset`` ever grow behaviour
        that belongs to a new scene, it must not fire on those.
        """
        self._frames: list[Frame] = []
        self._masks: list[Frame] = []

    def update(self, frame: Frame, mask: Frame, active: bool) -> Window | None:
        """Take one frame, and return a window if this frame completed one.

        Args:
            frame: The frame captured at this moment. See ``Frame``.
            mask: Its foreground mask, from the background subtractor.
            active: The trigger's verdict on this frame. ``False`` drops
                this frame and discards anything part-collected with it.

        Returns:
            The completed ``Window`` if this frame filled one, otherwise
            ``None``. Collecting resumes from empty on the next frame,
            so consecutive windows share no frames.
        """
        if not active:
            if self._frames:
                _log.debug("motion ended, dropping %d frames", len(self._frames))
                self._clear()
            return None

        self._frames.append(frame)
        self._masks.append(mask)

        if len(self._frames) < self._window_frames:
            return None

        window = Window(frames=tuple(self._frames), masks=tuple(self._masks))
        # Emptied rather than kept, so the next frame opens the next
        # window: that is what "no cooldown" means here.
        self._clear()
        _log.debug("window complete at %d frames", len(window))
        return window
