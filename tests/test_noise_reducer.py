import numpy as np
import pytest

from amprep import Frame, NoiseReducer


def test_noise_reducer_is_abstract():
    """NoiseReducer requires subclasses to implement _apply."""
    with pytest.raises(TypeError):
        NoiseReducer()


def test_noise_reducer_apply_is_abstract():
    """A subclass without _apply cannot be instantiated."""

    class DummyReducer(NoiseReducer):
        pass

    with pytest.raises(TypeError):
        DummyReducer()


class RecordingReducer(NoiseReducer):
    """A reducer that returns a valid frame and records whether it ran.

    Returning a valid frame keeps the output checks quiet, so a rejection
    is attributable to the input checks alone.
    """

    def __init__(self) -> None:
        self.called = False

    def _apply(self, frame: Frame) -> Frame:
        self.called = True
        return np.zeros((64, 64, 3), dtype=np.uint8)


def test_apply_rejects_non_array_input():
    """The public apply method rejects values that are not NumPy arrays."""
    reducer = RecordingReducer()

    with pytest.raises(TypeError, match="NumPy array"):
        reducer.apply("not a frame")

    assert not reducer.called


def test_apply_rejects_non_uint8_input():
    """The public apply method requires uint8 input data."""
    reducer = RecordingReducer()
    frame = np.zeros((64, 64, 3), dtype=np.float32)

    with pytest.raises(TypeError, match="uint8"):
        reducer.apply(frame)

    assert not reducer.called


def test_apply_accepts_valid_frame():
    """The public apply method accepts a uint8 video frame."""

    class DummyReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return frame

    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    result = DummyReducer().apply(frame)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (64, 64, 3)


def test_apply_uses_custom_reducer():
    """The public apply method delegates processing to the subclass."""

    class DummyReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return frame + 1

    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = DummyReducer().apply(frame)

    np.testing.assert_array_equal(result, np.ones((2, 2, 3), dtype=np.uint8))


def test_apply_rejects_non_array_output():
    """The public apply method validates the reducer output type."""

    class DummyReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return "not a frame"

    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="NumPy array"):
        DummyReducer().apply(frame)


def test_apply_rejects_non_uint8_output():
    """The public apply method validates the reducer output dtype."""

    class DummyReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return frame.astype(np.float32)

    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="uint8"):
        DummyReducer().apply(frame)
