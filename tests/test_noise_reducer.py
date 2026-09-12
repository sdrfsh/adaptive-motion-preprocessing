import numpy as np
import pytest

from amprep import Frame, MedianNoiseReducer, NoiseReducer


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


def test_apply_rejects_shape_changing_output():
    """The public apply method requires _apply to preserve frame shape."""

    class DummyReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return frame[::2, ::2]

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    expected = r"preserve shape: \(64, 64, 3\) -> \(32, 32, 3\)"

    with pytest.raises(ValueError, match=expected):
        DummyReducer().apply(frame)


def test_median_reducer_rejects_even_ksize():
    """An even aperture is rejected at construction, not at the first frame."""
    with pytest.raises(ValueError, match="odd integer greater than 1"):
        MedianNoiseReducer(ksize=4)


@pytest.mark.parametrize("ksize", [1, 0, -3, 5.0, "5"])
def test_median_reducer_rejects_invalid_ksize(ksize):
    """Apertures that are not odd integers above 1 are rejected up front."""
    with pytest.raises(ValueError, match="odd integer greater than 1"):
        MedianNoiseReducer(ksize=ksize)


def test_median_reducer_accepts_odd_ksize():
    """An odd aperture greater than 1 constructs without complaint."""
    assert MedianNoiseReducer(ksize=3)._ksize == 3


def test_median_reducer_does_not_override_apply():
    """The validating apply is inherited, so its checks still run."""
    assert MedianNoiseReducer.apply is NoiseReducer.apply


def test_median_reducer_validates_its_input():
    """Inheriting apply means the input checks guard the median filter too."""
    with pytest.raises(TypeError, match="NumPy array"):
        MedianNoiseReducer().apply("not a frame")


def test_median_reducer_preserves_shape_and_dtype():
    """The filtered frame matches the input's shape and dtype."""
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    result = MedianNoiseReducer().apply(frame)

    assert result.shape == (32, 48, 3)
    assert result.dtype == np.uint8


def test_median_reducer_removes_salt_and_pepper_noise():
    """Isolated extreme pixels are discarded rather than averaged in."""
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    frame[10, 10] = 255
    frame[20, 20] = 0

    result = MedianNoiseReducer().apply(frame)

    np.testing.assert_array_equal(result, np.full((32, 32, 3), 100, dtype=np.uint8))


def test_median_reducer_leaves_clean_frames_alone():
    """A frame without outliers survives the filter unchanged in its interior."""
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    frame[:, 16:] = 200

    result = MedianNoiseReducer().apply(frame)

    np.testing.assert_array_equal(result[2:-2, 2:-2], frame[2:-2, 2:-2])


def test_median_reducer_does_not_mutate_input():
    """Filtering returns a new frame and leaves the caller's frame intact."""
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    frame[10, 10] = 255
    original = frame.copy()

    MedianNoiseReducer().apply(frame)

    np.testing.assert_array_equal(frame, original)
