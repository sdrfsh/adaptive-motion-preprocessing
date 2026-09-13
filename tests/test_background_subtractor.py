import numpy as np
import pytest

from amprep import BackgroundSubtractor, Frame, KNNBackgroundSubtractor


def _frame(height: int = 64, width: int = 64) -> Frame:
    """Return a valid uint8 BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_background_subtractor_is_abstract():
    """BackgroundSubtractor requires subclasses to implement _apply."""
    with pytest.raises(TypeError):
        BackgroundSubtractor()


def test_subclass_without_any_hook_is_abstract():
    """A subclass implementing neither hook cannot be instantiated."""

    class DummySubtractor(BackgroundSubtractor):
        pass

    with pytest.raises(TypeError):
        DummySubtractor()


def test_subclass_without_apply_is_abstract():
    """Implementing reset alone is not enough: _apply is still required."""

    class ResetOnlySubtractor(BackgroundSubtractor):
        def reset(self) -> None:
            pass

    with pytest.raises(TypeError, match="_apply"):
        ResetOnlySubtractor()


def test_subclass_without_reset_is_abstract():
    """Implementing _apply alone is not enough: reset is still required."""

    class ApplyOnlySubtractor(BackgroundSubtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    with pytest.raises(TypeError, match="reset"):
        ApplyOnlySubtractor()


class _Subtractor(BackgroundSubtractor):
    """Base for the apply-focused test doubles.

    Implements ``reset`` as a no-op so each double below declares only the
    ``_apply`` behaviour the test under it cares about.
    """

    def reset(self) -> None:
        pass


class RecordingSubtractor(_Subtractor):
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

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return frame.max(axis=2)

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0, 1] = 255

    mask = DummySubtractor().apply(frame)

    np.testing.assert_array_equal(mask, np.array([[255, 0], [0, 0]], dtype=np.uint8))


def test_apply_passes_the_frame_through_unchanged():
    """The subclass sees exactly the frame the caller passed in."""
    seen: list[Frame] = []

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            seen.append(frame)
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    frame = _frame(4, 4)

    DummySubtractor().apply(frame)

    assert seen[0] is frame


def test_apply_allows_state_to_persist_across_calls():
    """Consecutive calls reach the same instance, so a model can adapt."""

    class CountingSubtractor(_Subtractor):
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

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return "not a mask"

    with pytest.raises(
        TypeError, match="the mask returned by _apply to be a NumPy array"
    ):
        DummySubtractor().apply(_frame())


def test_apply_rejects_non_uint8_output():
    """The public apply method validates the mask dtype."""

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.float32)

    with pytest.raises(TypeError, match="the mask returned by _apply to be uint8"):
        DummySubtractor().apply(_frame())


def test_apply_rejects_multi_channel_output():
    """The public apply method requires a single-channel mask."""

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape, dtype=np.uint8)

    with pytest.raises(
        ValueError, match="the mask returned by _apply to be 2-dimensional"
    ):
        DummySubtractor().apply(_frame())


def test_apply_rejects_size_changing_output():
    """The public apply method requires _apply to preserve frame H x W."""

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros((frame.shape[0] // 2, frame.shape[1] // 2), dtype=np.uint8)

    expected = r"preserve frame H.W: \(64, 64\) -> \(32, 32\)"

    with pytest.raises(ValueError, match=expected):
        DummySubtractor().apply(_frame())


def test_reset_is_callable_and_returns_none():
    """The base contract asks nothing of reset beyond being callable."""
    assert RecordingSubtractor().reset() is None


def test_reset_discards_accumulated_state():
    """A stateful subtractor can clear its model without being rebuilt."""

    class CountingSubtractor(_Subtractor):
        def __init__(self) -> None:
            self.calls = 0

        def _apply(self, frame: Frame) -> Frame:
            self.calls += 1
            return np.full(frame.shape[:2], self.calls, dtype=np.uint8)

        def reset(self) -> None:
            self.calls = 0

    subtractor = CountingSubtractor()
    frame = _frame(2, 2)

    subtractor.apply(frame)
    subtractor.apply(frame)
    subtractor.reset()
    after_reset = subtractor.apply(frame)

    assert subtractor.calls == 1
    np.testing.assert_array_equal(after_reset, np.full((2, 2), 1, dtype=np.uint8))


def test_reset_does_not_prevent_further_use():
    """apply still validates and delegates normally after a reset."""
    subtractor = RecordingSubtractor()

    subtractor.reset()
    mask = subtractor.apply(_frame(3, 7))

    assert subtractor.called
    assert mask.shape == (3, 7)


def _train(subtractor: BackgroundSubtractor, frame: Frame, frames: int = 30) -> None:
    """Feed ``frame`` repeatedly so the background model learns it."""
    for _ in range(frames):
        subtractor.apply(frame)


def _scene() -> tuple[Frame, Frame]:
    """Return a flat background frame and the same frame with an object."""
    background = np.full((32, 32, 3), 50, dtype=np.uint8)
    occupied = background.copy()
    occupied[8:24, 8:24] = 200
    return background, occupied


@pytest.mark.parametrize("history", [0, -1, 5.0, "500", True])
def test_knn_rejects_invalid_history(history):
    """A history that is not a positive integer is rejected up front."""
    with pytest.raises(ValueError, match="history must be a positive integer"):
        KNNBackgroundSubtractor(history=history)


@pytest.mark.parametrize("threshold", [0, -1.0, "400", True])
def test_knn_rejects_invalid_dist2_threshold(threshold):
    """A threshold that is not a positive number is rejected up front."""
    with pytest.raises(ValueError, match="dist2_threshold must be a positive number"):
        KNNBackgroundSubtractor(dist2_threshold=threshold)


def test_knn_rejects_non_boolean_detect_shadows():
    """The shadow switch is a flag, not a truthy value."""
    with pytest.raises(ValueError, match="detect_shadows must be a boolean"):
        KNNBackgroundSubtractor(detect_shadows=1)


def test_knn_accepts_valid_arguments():
    """Legal tuning values construct without complaint."""
    subtractor = KNNBackgroundSubtractor(
        history=10, dist2_threshold=100, detect_shadows=False
    )

    assert subtractor._history == 10
    assert subtractor._dist2_threshold == 100.0
    assert subtractor._detect_shadows is False


def test_knn_does_not_override_apply():
    """The validating apply is inherited, so its checks still run."""
    assert KNNBackgroundSubtractor.apply is BackgroundSubtractor.apply


def test_knn_validates_its_input():
    """Inheriting apply means the input checks guard the KNN model too."""
    with pytest.raises(TypeError, match="the input frame to be a NumPy array"):
        KNNBackgroundSubtractor().apply("not a frame")


def test_knn_returns_uint8_mask_of_frame_size():
    """apply returns a single-channel uint8 mask matching the frame's H x W."""
    mask = KNNBackgroundSubtractor().apply(_frame(height=8, width=5))

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.uint8
    assert mask.shape == (8, 5)


def test_knn_mask_is_strictly_binary():
    """Only 0 and 255 reach the caller, never an intermediate shadow grey."""
    background, occupied = _scene()
    shadowed = background.copy()
    shadowed[8:24, 8:24] = 25  # A darkened copy of the background reads as shadow.

    subtractor = KNNBackgroundSubtractor()
    _train(subtractor, background)

    for frame in (occupied, shadowed):
        assert set(np.unique(subtractor.apply(frame))) <= {0, 255}


def test_knn_marks_a_new_object_as_foreground():
    """A learned background stays dark while an object arriving on it lights up."""
    background, occupied = _scene()
    subtractor = KNNBackgroundSubtractor()

    _train(subtractor, background)
    mask = subtractor.apply(occupied)

    assert mask[8:24, 8:24].all()
    assert not mask[:8].any()


def test_knn_excludes_shadows_by_default():
    """A region that merely darkens is not foreground when shadows are detected."""
    background, _ = _scene()
    shadowed = background.copy()
    shadowed[8:24, 8:24] = 25

    subtractor = KNNBackgroundSubtractor()
    _train(subtractor, background)

    assert not subtractor.apply(shadowed).any()


def test_knn_includes_shadows_when_detection_is_off():
    """Without shadow detection the same darkening counts as foreground."""
    background, _ = _scene()
    shadowed = background.copy()
    shadowed[8:24, 8:24] = 25

    subtractor = KNNBackgroundSubtractor(detect_shadows=False)
    _train(subtractor, background)

    assert subtractor.apply(shadowed)[8:24, 8:24].all()


def test_knn_does_not_mutate_input():
    """Subtraction leaves the caller's frame intact."""
    background, occupied = _scene()
    original = occupied.copy()

    subtractor = KNNBackgroundSubtractor()
    _train(subtractor, background)
    subtractor.apply(occupied)

    np.testing.assert_array_equal(occupied, original)


def test_knn_reset_discards_the_learned_model():
    """The same frame reads differently once the learned background is gone."""
    background, occupied = _scene()
    subtractor = KNNBackgroundSubtractor()

    _train(subtractor, background)
    before_reset = subtractor.apply(occupied)

    subtractor.reset()
    after_reset = subtractor.apply(occupied)

    # Against a learned background only the object is foreground; against an
    # empty model nothing is known to be background, so the frame is all new.
    assert not before_reset[:8].any()
    assert after_reset.all()
    assert (before_reset != after_reset).any()


def test_knn_reset_returns_none():
    """reset clears state rather than reporting anything."""
    assert KNNBackgroundSubtractor().reset() is None


def test_knn_reset_replaces_the_underlying_model():
    """The OpenCV model object itself is rebuilt, not reused."""
    subtractor = KNNBackgroundSubtractor()
    original = subtractor._subtractor

    subtractor.reset()

    assert subtractor._subtractor is not original


def test_knn_relearns_after_reset():
    """A reset subtractor is fully usable and builds a new model from scratch."""
    background, occupied = _scene()
    subtractor = KNNBackgroundSubtractor()

    _train(subtractor, background)
    subtractor.reset()
    _train(subtractor, background)
    mask = subtractor.apply(occupied)

    assert mask[8:24, 8:24].all()
    assert not mask[:8].any()


def test_warmup_frames_defaults_to_zero():
    """A subclass that says nothing is taken to be useful immediately."""

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    assert DummySubtractor().warmup_frames == 0


def test_warmup_frames_is_concrete_on_the_abc():
    """The property is not abstract: omitting it does not block a subclass."""

    class DummySubtractor(_Subtractor):
        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    DummySubtractor()  # would raise TypeError if warmup_frames were abstract

    assert "warmup_frames" not in BackgroundSubtractor.__abstractmethods__


def test_warmup_frames_can_be_overridden():
    """A subtractor that needs settling time says so, and is believed."""

    class SlowSubtractor(_Subtractor):
        @property
        def warmup_frames(self) -> int:
            return 12

        def _apply(self, frame: Frame) -> Frame:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    assert SlowSubtractor().warmup_frames == 12


def test_warmup_frames_is_a_property_not_a_method():
    """Reading it gives the count itself, not something to call."""
    assert isinstance(KNNBackgroundSubtractor().warmup_frames, int)
    assert isinstance(
        type(KNNBackgroundSubtractor()).warmup_frames,
        property,
    )


def test_knn_declares_a_warmup():
    """KNN needs a few frames of samples before its masks mean anything."""
    assert KNNBackgroundSubtractor().warmup_frames == 4


def test_knn_warmup_survives_reset():
    """Warmup is a property of the algorithm, not of the current model."""
    subtractor = KNNBackgroundSubtractor()
    subtractor.reset()

    assert subtractor.warmup_frames == 4


def test_knn_warmup_is_long_enough_to_settle():
    """The declared warmup is honest: past it, a still scene reads as still.

    KNN judges a pixel against its recent samples, so on the opening
    frames it has too few to judge with and calls almost everything
    foreground. This asserts the count is *sufficient* rather than exact
    — that masks are trustworthy once the warmup is spent — because how
    many frames it takes to get there is OpenCV's business and could
    reasonably shift between builds.
    """
    background, _ = _scene()
    subtractor = KNNBackgroundSubtractor()

    for _ in range(subtractor.warmup_frames):
        subtractor.apply(background)

    assert not subtractor.apply(background).any()
