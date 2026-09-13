import numpy as np
import pytest

from amprep import Frame
from amprep.frame_window import (
    DEFAULT_WINDOW_FRAMES,
    MIN_WINDOW_FRAMES,
    FrameWindowCollector,
)
from amprep.types import Window

SHAPE = (4, 4)
"""Small enough to be free, large enough to be a real array."""


def _frame(value: int) -> Frame:
    """A frame carrying ``value``, so its place in a window is readable."""
    return np.full((*SHAPE, 3), value, dtype=np.uint8)


def _mask(value: int) -> Frame:
    """The matching mask for ``_frame(value)``, carrying the same value."""
    return np.full(SHAPE, value, dtype=np.uint8)


def _feed(collector: FrameWindowCollector, count: int, active: bool = True, start=0):
    """Push ``count`` numbered frames through and return what came out."""
    return [
        collector.update(_frame(i), _mask(i), active)
        for i in range(start, start + count)
    ]


def _values(window: Window) -> list[int]:
    """The numbers the window's frames were built from, in order."""
    return [int(frame[0, 0, 0]) for frame in window.frames]


def test_default_window_is_ten_frames():
    """Ten frames, the packaged default."""
    assert DEFAULT_WINDOW_FRAMES == 10
    assert FrameWindowCollector().window_frames == 10


def test_window_frames_is_kept_as_given():
    """A caller-supplied size is the one used."""
    assert FrameWindowCollector(window_frames=3).window_frames == 3


@pytest.mark.parametrize("size", [None, 3.0, "3", [3], True, False])
def test_non_integer_window_frames_is_rejected(size):
    """A size that is not a count of frames fails at construction."""
    with pytest.raises(ValueError, match="window_frames must be an integer"):
        FrameWindowCollector(window_frames=size)


@pytest.mark.parametrize("size", [1, 0, -1])
def test_window_smaller_than_the_minimum_is_rejected(size):
    """One frame cannot be compared against anything, so two is the floor."""
    assert MIN_WINDOW_FRAMES == 2
    with pytest.raises(ValueError, match="at least 2"):
        FrameWindowCollector(window_frames=size)


def test_the_minimum_window_is_allowed():
    """Two frames is the smallest real comparison, and is accepted."""
    collector = FrameWindowCollector(window_frames=MIN_WINDOW_FRAMES)

    assert _feed(collector, 2)[-1] is not None


def test_starts_empty():
    """A fresh collector holds nothing."""
    assert FrameWindowCollector().pending == 0


def test_nothing_is_emitted_before_the_window_is_full():
    """Nine frames of a ten-frame window produce nothing at all."""
    collector = FrameWindowCollector(window_frames=10)

    assert _feed(collector, 9) == [None] * 9
    assert collector.pending == 9


def test_the_window_fills_at_exactly_n_frames():
    """The window arrives on the Nth frame, not the Nth minus or plus one."""
    collector = FrameWindowCollector(window_frames=4)

    results = _feed(collector, 4)

    assert results[:3] == [None, None, None]
    assert isinstance(results[3], Window)
    assert len(results[3]) == 4


def test_the_window_carries_its_frames_in_order():
    """The window is the run that filled it, in the order it arrived."""
    collector = FrameWindowCollector(window_frames=4)

    window = _feed(collector, 4)[-1]

    assert _values(window) == [0, 1, 2, 3]


def test_every_frame_keeps_its_mask():
    """Frames and masks line up by index, which is the whole point of pairing."""
    collector = FrameWindowCollector(window_frames=4)

    window = _feed(collector, 4)[-1]

    assert len(window.frames) == len(window.masks) == 4
    for frame, mask in zip(window.frames, window.masks, strict=True):
        assert int(mask[0, 0]) == int(frame[0, 0, 0])


def test_the_next_window_starts_on_the_very_next_frame():
    """No cooldown: twenty active frames are two back-to-back windows."""
    collector = FrameWindowCollector(window_frames=10)

    results = _feed(collector, 20)
    windows = [result for result in results if result is not None]

    assert len(windows) == 2
    assert _values(windows[0]) == list(range(0, 10))
    assert _values(windows[1]) == list(range(10, 20))


def test_consecutive_windows_share_no_frames():
    """A frame belongs to exactly one window; none is counted twice."""
    collector = FrameWindowCollector(window_frames=3)

    windows = [result for result in _feed(collector, 9) if result is not None]
    seen = [value for window in windows for value in _values(window)]

    assert seen == list(range(9))


def test_a_completed_window_leaves_nothing_pending():
    """Handing a window over empties the collector rather than half-filling it."""
    collector = FrameWindowCollector(window_frames=3)

    _feed(collector, 3)

    assert collector.pending == 0


def test_a_partial_window_is_discarded_when_motion_stops():
    """Motion ending mid-window throws the fragment away, as documented."""
    collector = FrameWindowCollector(window_frames=10)
    _feed(collector, 6)

    assert collector.pending == 6

    assert collector.update(_frame(99), _mask(99), active=False) is None
    assert collector.pending == 0


def test_a_discarded_window_is_never_emitted_later():
    """The dropped fragment does not reappear at the front of the next window."""
    collector = FrameWindowCollector(window_frames=4)
    _feed(collector, 3)
    collector.update(_frame(99), _mask(99), active=False)

    window = _feed(collector, 4, start=10)[-1]

    assert _values(window) == [10, 11, 12, 13]


def test_idle_frames_are_not_collected():
    """Frames arriving while the trigger is idle are not part of any window."""
    collector = FrameWindowCollector(window_frames=2)

    assert _feed(collector, 5, active=False) == [None] * 5
    assert collector.pending == 0


def test_motion_stopping_on_a_full_window_loses_nothing():
    """A window completed by the last active frame is still handed over."""
    collector = FrameWindowCollector(window_frames=3)

    results = _feed(collector, 3)
    after = collector.update(_frame(99), _mask(99), active=False)

    assert isinstance(results[-1], Window)
    assert after is None


def test_bursts_are_collected_independently():
    """Stop and start again and the second burst fills its own window."""
    collector = FrameWindowCollector(window_frames=3)

    _feed(collector, 2)
    collector.update(_frame(99), _mask(99), active=False)
    window = _feed(collector, 3, start=20)[-1]

    assert _values(window) == [20, 21, 22]


def test_reset_drops_what_was_being_collected():
    """Reset abandons the part-built window."""
    collector = FrameWindowCollector(window_frames=10)
    _feed(collector, 4)

    collector.reset()

    assert collector.pending == 0


def test_reset_keeps_the_window_size():
    """The size is configuration, not accumulated state."""
    collector = FrameWindowCollector(window_frames=5)

    collector.reset()

    assert collector.window_frames == 5


def test_reset_returns_none():
    """Reset is a command, not a query."""
    assert FrameWindowCollector().reset() is None


@pytest.mark.parametrize("name", ["window_frames", "pending"])
def test_the_properties_are_read_only(name):
    """Nothing outside the collector rewrites its size or its progress."""
    collector = FrameWindowCollector()

    with pytest.raises(AttributeError):
        setattr(collector, name, 3)


def test_window_rejects_a_mask_count_that_does_not_match():
    """A window with a frame missing its mask is not constructible."""
    with pytest.raises(ValueError, match="one mask per frame"):
        Window(frames=(_frame(0), _frame(1)), masks=(_mask(0),))


def test_window_rejects_being_empty():
    """An empty window describes nothing and is never emitted."""
    with pytest.raises(ValueError, match="must not be empty"):
        Window(frames=(), masks=())


def test_window_is_frozen():
    """A window is a handover, not a workspace for the stage that gets it."""
    window = Window(frames=(_frame(0),), masks=(_mask(0),))

    with pytest.raises(AttributeError):
        window.frames = ()
