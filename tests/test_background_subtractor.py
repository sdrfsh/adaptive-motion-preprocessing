import numpy as np
import pytest

from amprep import BackgroundSubtractor, Frame


def _frame(height: int = 64, width: int = 64) -> Frame:
    """Return a valid uint8 BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_background_subtractor_is_abstract():
    """BackgroundSubtractor requires subclasses to implement _apply."""
    with pytest.raises(TypeError):
        BackgroundSubtractor()


def test_subclass_without_apply_is_abstract():
    """A subclass without _apply cannot be instantiated."""

    class DummySubtractor(BackgroundSubtractor):
        pass

    with pytest.raises(TypeError):
        DummySubtractor()


class RecordingSubtractor(BackgroundSubtractor):
    """A subtractor that returns a valid mask and records whether it ran.

    Returning a valid mask keeps the output checks quiet, so a rejection
    is attributable to the input checks alone.
    """

    def __init__(self) -> None:
        self.called = False

    def _apply(self, frame: Frame) -> Frame:
        self.called = True
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def test_apply_rejects_non_array_input():
    """The public apply method rejects values that are not NumPy arrays."""
    subtractor = RecordingSubtractor()

    with pytest.raises(TypeError, match="the input frame to be a NumPy array"):
        subtractor.apply("not a frame")

    assert not subtractor.called


def test_apply_rejects_non_uint8_input():
    """The public apply method requires uint8 input data."""
    subtractor = RecordingSubtractor()
    frame = np.zeros((64, 64, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="the input frame to be uint8"):
        subtractor.apply(frame)

    assert not subtractor.called


def test_apply_rejects_single_channel_input():
    """The public apply method rejects a frame that is not 3-dimensional."""
    subtractor = RecordingSubtractor()
    frame = np.zeros((64, 64), dtype=np.uint8)

    with pytest.raises(ValueError, match="the input frame to be 3-dimensional"):
        subtractor.apply(frame)

    assert not subtractor.called


def test_apply_rejects_input_with_wrong_channel_count():
    """The public apply method requires three-channel BGR input."""
    subtractor = RecordingSubtractor()
    frame = np.zeros((64, 64, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"the input frame to be \(H, W, 3\)"):
        subtractor.apply(frame)

    assert not subtractor.called


def test_apply_accepts_valid_frame():
    """The public apply method returns a single-channel uint8 mask."""
    subtractor = RecordingSubtractor()

    mask = subtractor.apply(_frame())

    assert subtractor.called
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.shape == (64, 64)


def test_apply_accepts_non_square_frame():
    """The mask keeps the frame's height and width, which may differ."""
    mask = RecordingSubtractor().apply(_frame(height=8, width=5))

    assert mask.shape == (8, 5)


def test_apply_uses_custom_subtractor():
    """The public apply method delegates processing to the subclass."""

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return frame.max(axis=2)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0, 1] = 255

    mask = DummySubtractor().apply(frame)

    np.testing.assert_array_equal(mask, np.array([[255, 0], [0, 0]], dtype=np.uint8))


def test_apply_passes_the_frame_through_unchanged():
    """The subclass sees exactly the frame the caller passed in."""
    seen: list[Frame] = []

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            seen.append(frame)
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    frame = _frame(4, 4)

    DummySubtractor().apply(frame)

    assert seen[0] is frame


def test_apply_allows_state_to_persist_across_calls():
    """Consecutive calls reach the same instance, so a model can adapt."""

    class CountingSubtractor(BackgroundSubtractor):
        def __init__(self) -> None:
            self.calls = 0

        def _apply(self, frame: Frame) -> Frame:
            self.calls += 1
            return np.full(frame.shape[:2], self.calls, dtype=np.uint8)

    subtractor = CountingSubtractor()
    frame = _frame(2, 2)

    first = subtractor.apply(frame)
    second = subtractor.apply(frame)

    assert subtractor.calls == 2
    np.testing.assert_array_equal(first, np.full((2, 2), 1, dtype=np.uint8))
    np.testing.assert_array_equal(second, np.full((2, 2), 2, dtype=np.uint8))


def test_apply_rejects_non_array_output():
    """The public apply method validates the mask type."""

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return "not a mask"

    with pytest.raises(
        TypeError, match="the mask returned by _apply to be a NumPy array"
    ):
        DummySubtractor().apply(_frame())


def test_apply_rejects_non_uint8_output():
    """The public apply method validates the mask dtype."""

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.float32)

    with pytest.raises(TypeError, match="the mask returned by _apply to be uint8"):
        DummySubtractor().apply(_frame())


def test_apply_rejects_multi_channel_output():
    """The public apply method requires a single-channel mask."""

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape, dtype=np.uint8)

    with pytest.raises(
        ValueError, match="the mask returned by _apply to be 2-dimensional"
    ):
        DummySubtractor().apply(_frame())


def test_apply_rejects_size_changing_output():
    """The public apply method requires _apply to preserve frame H x W."""

    class DummySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros((frame.shape[0] // 2, frame.shape[1] // 2), dtype=np.uint8)

    expected = r"preserve frame H.W: \(64, 64\) -> \(32, 32\)"

    with pytest.raises(ValueError, match=expected):
        DummySubtractor().apply(_frame())
