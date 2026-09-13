import numpy as np
import pytest

from amprep import (
    AdaptiveMotionPreprocessor,
    BackgroundSubtractor,
    Frame,
    KNNBackgroundSubtractor,
    MotionImage,
)

H, W = 60, 80


def _scene(still_before=10, moving=30, still_after=10, seed=0):
    """Still, then a block sliding right 3px per frame, then still."""
    rng = np.random.default_rng(seed)
    frames, left = [], 0
    for i in range(still_before + moving + still_after):
        f = np.full((H, W, 3), 120, np.int16)
        if i >= still_before:
            if i < still_before + moving:
                left = (i - still_before) * 3
            f[20:40, left : left + 20] = 230
        f += rng.normal(0, 4, f.shape).astype(np.int16)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


# ---- it works ----


def test_motion_produces_motion_images():
    images = list(AdaptiveMotionPreprocessor().process(_scene()))

    assert len(images) == 2
    assert all(isinstance(image, MotionImage) for image in images)


def test_first_image_arrives_right_after_one_window_of_motion():
    pulled = 0

    def frames():
        nonlocal pulled
        for f in _scene():
            pulled += 1
            yield f

    next(AdaptiveMotionPreprocessor().process(frames()))

    assert pulled == 20  # 10 still + 10 that filled the window


def test_a_still_scene_produces_nothing():
    assert list(AdaptiveMotionPreprocessor().process(_scene(40, 0, 0))) == []


# ---- size ----


def test_output_keeps_the_frame_size_by_default():
    image = next(AdaptiveMotionPreprocessor().process(_scene()))

    assert image.data.shape == (H, W)
    assert image.data.dtype == np.uint8


def test_width_and_height_set_the_output_size():
    image = next(AdaptiveMotionPreprocessor(width=32, height=24).process(_scene()))

    assert image.data.shape == (24, 32)


# ---- warm-up ----


def test_warmup_is_not_mistaken_for_motion():
    """Without the skip, this emits one pure-white junk image."""
    assert (
        list(AdaptiveMotionPreprocessor(window_frames=4).process(_scene(8, 0, 0))) == []
    )


def test_a_subtractor_with_no_warmup_is_not_skipped():
    subtractor = KNNBackgroundSubtractor(warmup_frames=0)
    preprocessor = AdaptiveMotionPreprocessor(
        background_subtractor=subtractor, window_frames=4
    )

    assert len(list(preprocessor.process(_scene(8, 0, 0)))) == 1


# ---- input kinds and chunks ----


class _Threshold(BackgroundSubtractor):
    """A deterministic stand-in for KNN.

    OpenCV's KNN refreshes its sample model at random, so two fresh
    subtractors fed identical frames disagree on a handful of edge
    pixels. That randomness would drown out what the test below is
    actually asking, which is whether the *kind* of iterable changes
    anything. Bright pixels are foreground: same input, same mask, every
    time.
    """

    def _apply(self, frame: Frame) -> Frame:
        return np.where(frame.max(axis=2) > 180, np.uint8(255), np.uint8(0))

    def reset(self) -> None:
        pass


def test_list_generator_and_iterator_give_the_same_images():
    def run(frames):
        preprocessor = AdaptiveMotionPreprocessor(background_subtractor=_Threshold())
        return [image.data for image in preprocessor.process(frames)]

    expected = run(_scene())
    for other in (run(f for f in _scene()), run(iter(_scene()))):
        assert len(other) == len(expected)
        for a, b in zip(expected, other, strict=True):
            np.testing.assert_array_equal(a, b)


def test_splitting_the_stream_across_calls_changes_nothing():
    frames = _scene()
    preprocessor = AdaptiveMotionPreprocessor()

    images = list(preprocessor.process(frames[:15])) + list(
        preprocessor.process(frames[15:])
    )

    assert len(images) == 2


def test_you_can_stop_reading_an_endless_stream():
    pulled = 0

    def endless():
        nonlocal pulled
        while True:
            for f in _scene():
                pulled += 1
                yield f

    images = AdaptiveMotionPreprocessor().process(endless())
    next(images)
    images.close()

    assert pulled == 20


# ---- errors ----


def test_sample_frames_cannot_exceed_window_frames():
    with pytest.raises(ValueError, match="sample_frames"):
        AdaptiveMotionPreprocessor(window_frames=3, sample_frames=4)


def test_a_frame_size_change_is_an_error():
    frames = [np.zeros((H, W, 3), np.uint8)] * 3 + [np.zeros((30, 40, 3), np.uint8)]

    with pytest.raises(ValueError, match="reset"):
        list(AdaptiveMotionPreprocessor().process(frames))


def test_reset_allows_a_new_frame_size():
    preprocessor = AdaptiveMotionPreprocessor()
    list(preprocessor.process([np.zeros((H, W, 3), np.uint8)] * 3))

    preprocessor.reset()

    list(preprocessor.process([np.zeros((30, 40, 3), np.uint8)] * 3))  # no error
