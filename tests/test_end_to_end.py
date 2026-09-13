"""The whole pipeline, run on a clip built in memory.

No video file and no ``cv2.VideoWriter``: the package takes an iterable of
frames, so the test builds one. Every random value comes from a seeded
``numpy.random.default_rng``, so the frames are identical on every run.
"""

import cv2
import numpy as np
import pytest

from amprep import (
    AdaptiveMotionPreprocessor,
    BackgroundSubtractor,
    Frame,
    KNNBackgroundSubtractor,
    MedianNoiseReducer,
    NoiseReducer,
)
from amprep.adaptive_sampler import DEFAULT_SAMPLE_FRAMES
from amprep.frame_window import DEFAULT_WINDOW_FRAMES

HEIGHT, WIDTH = 64, 96

BOX = 16
"""Side of the square block that moves."""

TOP = 24
"""Row the block's top edge sits on."""

START_LEFT = 4
"""Column of the block's left edge on its first moving frame."""

STEP = 3
"""Pixels the block moves right each frame."""

STILL_BEFORE = 8
"""Background-only frames before anything moves.

KNN spends its first ``warmup_frames`` (4) building a model, so eight leaves
four more for it to settle before the object appears.
"""

MOVING = 25
"""Frames the object is in view and moving.

Two full windows of ten, plus five that start a third and are dropped when
the object leaves. That makes the expected count exact rather than a range.
"""

STILL_AFTER = 8
"""Background-only frames after the object has gone."""

EXPECTED_IMAGES = MOVING // DEFAULT_WINDOW_FRAMES  # 2

NOISE_SIGMA = 6
"""Gaussian sensor noise, in grey levels.

Strong enough that skipping the median filter puts foreground into the
masks of a still scene, so the denoising stage is doing real work here.
"""


SLOW_STEP = 1
"""Pixels per frame for slow motion.

About 12% of the block changes per frame, well under the sampler's full
speed, so its four samples spread across the window.
"""

FAST_STEP = 8
"""Pixels per frame for fast motion.

About 67% of the block changes per frame, past the sampler's full speed,
so its four samples pack onto the newest frames.
"""

FAST_MOVING = DEFAULT_WINDOW_FRAMES
"""Frames of fast motion: one window, the most that fits in the frame."""


def _left(frame: int, step: int = STEP) -> int:
    """Column of the block's left edge on clip frame ``frame``."""
    return START_LEFT + (frame - STILL_BEFORE) * step


def _clip(seed: int = 0, step: int = STEP, moving: int = MOVING) -> list[np.ndarray]:
    """Still background, a block sliding right ``step`` pixels a frame, still again."""
    # Slicing past the edge would quietly shrink the block instead of failing.
    assert _left(STILL_BEFORE + moving - 1, step) + BOX <= WIDTH, "block leaves frame"

    rng = np.random.default_rng(seed)
    frames = []
    for index in range(STILL_BEFORE + moving + STILL_AFTER):
        frame = np.full((HEIGHT, WIDTH, 3), 110.0)
        if STILL_BEFORE <= index < STILL_BEFORE + moving:
            left = _left(index, step)
            frame[TOP : TOP + BOX, left : left + BOX] = 220
        frame += rng.normal(0, NOISE_SIGMA, frame.shape)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def test_the_clip_is_the_same_every_run():
    """Seeded, so a failure here can be reproduced exactly."""
    for first, second in zip(_clip(), _clip(), strict=True):
        np.testing.assert_array_equal(first, second)


def test_a_clip_in_yields_the_expected_number_of_motion_images():
    """Two full windows of motion come out as two images."""
    images = list(AdaptiveMotionPreprocessor().process(_clip()))

    assert len(images) == EXPECTED_IMAGES


OUTPUT_SHAPE = (HEIGHT, WIDTH)
"""The one shape every ``MotionImage`` carries for this clip."""

OUTPUT_DTYPE = np.uint8
"""The one dtype every ``MotionImage`` carries."""


class _GaussianNoiseReducer(NoiseReducer):
    """A user-supplied denoiser that is not the packaged median filter."""

    def _apply(self, frame: Frame) -> Frame:
        return cv2.GaussianBlur(frame, (5, 5), 0)


class _BrightnessSubtractor(BackgroundSubtractor):
    """A user-supplied subtractor: bright pixels are foreground.

    Stateless and with no warm-up, so it differs from KNN in every way the
    preprocessor could plausibly care about.
    """

    def _apply(self, frame: Frame) -> Frame:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.where(grey > 165, np.uint8(255), np.uint8(0))

    def reset(self) -> None:
        pass


@pytest.mark.parametrize(
    ("clip", "make_preprocessor"),
    [
        pytest.param({}, AdaptiveMotionPreprocessor, id="default"),
        # Window length
        pytest.param(
            {},
            lambda: AdaptiveMotionPreprocessor(window_frames=DEFAULT_SAMPLE_FRAMES),
            id="window-shortest",
        ),
        pytest.param(
            {},
            lambda: AdaptiveMotionPreprocessor(window_frames=20),
            id="window-20",
        ),
        # Motion velocity
        pytest.param({"step": SLOW_STEP}, AdaptiveMotionPreprocessor, id="slow"),
        pytest.param(
            {"step": FAST_STEP, "moving": FAST_MOVING},
            AdaptiveMotionPreprocessor,
            id="fast",
        ),
        # Custom stages, passed by keyword
        pytest.param(
            {},
            lambda: AdaptiveMotionPreprocessor(noise_reducer=_GaussianNoiseReducer()),
            id="custom-noise-reducer",
        ),
        pytest.param(
            {},
            lambda: AdaptiveMotionPreprocessor(
                background_subtractor=_BrightnessSubtractor()
            ),
            id="custom-background-subtractor",
        ),
        pytest.param(
            {},
            lambda: AdaptiveMotionPreprocessor(
                noise_reducer=_GaussianNoiseReducer(),
                background_subtractor=_BrightnessSubtractor(),
            ),
            id="custom-both",
        ),
    ],
)
def test_every_configuration_hands_the_model_one_shape_and_dtype(
    clip, make_preprocessor
):
    """A model gets the same input whatever the pipeline was configured with.

    Window length, how fast the subject moved and which stages were plugged
    in all change what happens inside the pipeline. None of them may change
    the shape or dtype of what comes out: two configurations that each look
    fine on their own must not hand a network differently shaped input.

    Preprocessors are built per test rather than shared, because the stages
    hold state.
    """
    images = list(make_preprocessor().process(_clip(**clip)))

    assert images, "no motion images came out, so nothing was checked"
    for image in images:
        assert image.data.shape == OUTPUT_SHAPE
        assert image.data.dtype == OUTPUT_DTYPE


def test_slow_and_fast_motion_are_sampled_differently():
    """Guards the velocity cases above: they must reach different sampling.

    If slow and fast motion were sampled the same way, the velocity cases
    would repeat the default case without failing. Slow motion must reach
    further back into its window than fast motion does.
    """
    slow = next(AdaptiveMotionPreprocessor().process(_clip(step=SLOW_STEP)))
    fast = next(
        AdaptiveMotionPreprocessor().process(_clip(step=FAST_STEP, moving=FAST_MOVING))
    )

    assert slow.frame_indices[0] < fast.frame_indices[0]


def test_frame_indices_are_strictly_increasing_and_within_the_window():
    """Each image records which frames of its window it was built from."""
    images = list(AdaptiveMotionPreprocessor().process(_clip()))

    assert images
    for image in images:
        indices = image.frame_indices
        assert list(indices) == sorted(set(indices))
        assert indices[0] >= 0
        assert indices[-1] < DEFAULT_WINDOW_FRAMES


def test_the_newest_frame_is_painted_where_the_object_ended():
    """The image shows the motion, not just the right shape.

    Count, shape, dtype and indices can all be right for a picture with
    nothing in it: an encoder painting pure black passes every test above.
    The newest frame of a window is painted at full brightness, so the 255
    pixels must sit exactly where the block was on that window's last
    frame.

    The windows line up with the clip because motion is detected on the
    first frame the object appears. If detection ever lags, this fails
    rather than drifting.
    """
    images = list(AdaptiveMotionPreprocessor().process(_clip()))

    assert len(images) == EXPECTED_IMAGES
    for number, image in enumerate(images):
        last_frame = STILL_BEFORE + (number + 1) * DEFAULT_WINDOW_FRAMES - 1
        rows, cols = np.nonzero(image.data == 255)

        assert rows.size, f"image {number} has no full-brightness pixels"
        assert (rows.min(), rows.max()) == (TOP, TOP + BOX - 1)
        assert (cols.min(), cols.max()) == (
            _left(last_frame),
            _left(last_frame) + BOX - 1,
        )


def test_the_noise_is_strong_enough_to_need_the_denoiser():
    """Guards the scene itself: without enough noise, the denoiser does nothing.

    The still frames before the object appears are run through KNN twice,
    once denoised and once raw. Raw noise produces foreground and the median
    filter removes it. If the noise were ever turned down far enough that
    both read as clean, the pipeline test above would stop exercising the
    denoising stage without failing, so this makes that visible.
    """

    class PassThrough(NoiseReducer):
        def _apply(self, frame: np.ndarray) -> np.ndarray:
            return frame

    still = _clip()[:STILL_BEFORE]

    def foreground(reducer: NoiseReducer) -> int:
        subtractor = KNNBackgroundSubtractor()
        total = 0
        for index, frame in enumerate(still):
            mask = subtractor.apply(reducer.apply(frame))
            if index >= subtractor.warmup_frames:
                total += np.count_nonzero(mask)
        return total

    assert foreground(PassThrough()) > 0
    assert foreground(MedianNoiseReducer()) == 0
