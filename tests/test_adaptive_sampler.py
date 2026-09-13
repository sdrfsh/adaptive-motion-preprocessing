import numpy as np
import pytest

from amprep.adaptive_sampler import (
    DEFAULT_FULL_SPEED,
    DEFAULT_SAMPLE_FRAMES,
    AdaptiveFrameSampler,
)
from amprep.types import Window

HEIGHT, WIDTH = 120, 160
BOX = 20

WALKING_PX = 3
"""Sideways pixels per frame that read as a walk on these synthetic masks."""

RUNNING_PX = 10
"""Sideways pixels per frame well past ``full_speed``."""

FASTEST_PX = 14
"""The quickest a box can cross these masks without reaching the edge.

``(WIDTH - BOX - 10) / 9`` is 14.4, so anything above this parks against
the right wall part-way through and the rest of the window is a subject
standing still: a property of the scaffold, not of the sampler.
"""


def _window(masks: list[np.ndarray]) -> Window:
    """Wrap masks in a window, with blank frames the sampler never reads."""
    blanks = tuple(np.zeros((*mask.shape, 3), np.uint8) for mask in masks)
    return Window(frames=blanks, masks=tuple(masks))


def _moving_window(frames: int, pixels_per_frame: float) -> Window:
    """A window whose foreground box drifts sideways at a known speed."""
    masks = []
    for index in range(frames):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        left = int(round(10 + index * pixels_per_frame))
        left = max(0, min(WIDTH - BOX, left))
        mask[50 : 50 + BOX, left : left + BOX] = 255
        masks.append(mask)
    return _window(masks)


def _approaching_window(frames: int) -> Window:
    """A subject growing in place, as one walking toward the camera does.

    Its centre never moves, which is exactly the case a centroid-based
    velocity reads as perfectly still.
    """
    masks = []
    for index in range(frames):
        size = int(round(BOX * (1 + 0.10 * index)))
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        top, left = (HEIGHT - size) // 2, (WIDTH - size) // 2
        mask[top : top + size, left : left + size] = 255
        masks.append(mask)
    return _window(masks)


def _two_apart_window(frames: int) -> Window:
    """Two subjects separating, whose combined centre stays put.

    The second case a centroid cannot see: the halves cancel out.
    """
    masks = []
    for index in range(frames):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        gap = index * 4
        mask[50 : 50 + BOX, 60 - gap - BOX : 60 - gap] = 255
        mask[50 : 50 + BOX, 100 + gap : 100 + gap + BOX] = 255
        masks.append(mask)
    return _window(masks)


def _blank_window(frames: int) -> Window:
    """A window whose masks hold no foreground at all."""
    return _window([np.zeros((HEIGHT, WIDTH), np.uint8) for _ in range(frames)])


def _flickering_window(frames: int, pixels_per_frame: float, specks: int) -> Window:
    """A moving window with noise that lands somewhere new every frame."""
    rng = np.random.default_rng(1)
    masks = []
    for mask in _moving_window(frames, pixels_per_frame).masks:
        noisy = mask.copy()
        for _ in range(specks):
            row, column = rng.integers(0, HEIGHT - 2), rng.integers(0, WIDTH - 2)
            noisy[row : row + 2, column : column + 2] = 255
        masks.append(noisy)
    return _window(masks)


def _static_speck_window(frames: int, pixels_per_frame: float, specks: int) -> Window:
    """A moving window with noise that stays in the same places."""
    rng = np.random.default_rng(1)
    points = [
        (rng.integers(0, HEIGHT - 2), rng.integers(0, WIDTH - 2)) for _ in range(specks)
    ]
    masks = []
    for mask in _moving_window(frames, pixels_per_frame).masks:
        noisy = mask.copy()
        for row, column in points:
            noisy[row : row + 2, column : column + 2] = 255
        masks.append(noisy)
    return _window(masks)


def _span(indices: tuple[int, ...]) -> int:
    """How much of the window the selection reaches across."""
    return indices[-1] - indices[0]


def test_defaults_are_what_the_docstrings_claim():
    """The packaged defaults, pinned so a silent change is visible."""
    sampler = AdaptiveFrameSampler()

    assert sampler.sample_frames == DEFAULT_SAMPLE_FRAMES == 4
    assert sampler.full_speed == DEFAULT_FULL_SPEED == 0.35


# --- what the change-rate metric exists to catch ---


def test_approaching_the_camera_is_not_read_as_still():
    """A subject growing in place is moving, though its centre is not."""
    sampler = AdaptiveFrameSampler()

    assert sampler.velocity(_approaching_window(10)) > sampler.velocity(
        _moving_window(10, 0)
    )


def test_two_subjects_moving_apart_are_not_read_as_still():
    """Two subjects separating cancel out as a centre but not as change."""
    assert AdaptiveFrameSampler().velocity(_two_apart_window(10)) > 0.1


def test_velocity_does_not_depend_on_window_length():
    """Averaging over pairs, not dividing by steps, keeps N out of the answer."""
    sampler = AdaptiveFrameSampler()

    assert sampler.velocity(_moving_window(10, WALKING_PX)) == pytest.approx(
        sampler.velocity(_moving_window(20, WALKING_PX)), abs=0.02
    )


def test_velocity_is_resolution_independent():
    """The same motion on a bigger frame reads the same rate.

    The share is taken over the subject's own pixels, so scaling the
    whole scene scales numerator and denominator alike.
    """
    sampler = AdaptiveFrameSampler()

    def window_at(scale: int) -> Window:
        masks = []
        for index in range(10):
            mask = np.zeros((HEIGHT * scale, WIDTH * scale), np.uint8)
            left = (10 + index * WALKING_PX) * scale
            mask[50 * scale : 70 * scale, left : left + BOX * scale] = 255
            masks.append(mask)
        return _window(masks)

    assert sampler.velocity(window_at(1)) == pytest.approx(
        sampler.velocity(window_at(3)), abs=0.02
    )


def test_velocity_stays_within_zero_and_one():
    """A share of a subject cannot exceed the whole of it."""
    sampler = AdaptiveFrameSampler()

    for speed in (0, 1, 3, 10, 40):
        assert 0.0 <= sampler.velocity(_moving_window(10, speed)) <= 1.0


def test_velocity_rises_with_speed():
    """Faster sideways motion flips a larger share of the subject."""
    sampler = AdaptiveFrameSampler()

    readings = [sampler.velocity(_moving_window(10, px)) for px in (0, 1, 2, 3)]

    assert readings == sorted(readings)


def test_velocity_is_zero_for_a_still_window():
    """A foreground that does not change has no rate of change."""
    assert AdaptiveFrameSampler().velocity(_moving_window(10, 0)) == 0.0


def test_velocity_is_zero_when_the_foreground_is_missing():
    """No foreground in either mask means no share to measure."""
    assert AdaptiveFrameSampler().velocity(_blank_window(10)) == 0.0


def test_velocity_is_zero_for_a_single_frame_window():
    """One frame makes no pair, so there is nothing to compare."""
    assert AdaptiveFrameSampler().velocity(_moving_window(1, 10)) == 0.0


def test_static_specks_barely_move_the_reading():
    """Noise that stays put is shared by both masks, so it mostly cancels."""
    sampler = AdaptiveFrameSampler()

    clean = sampler.velocity(_moving_window(10, WALKING_PX))
    speckled = sampler.velocity(_static_speck_window(10, WALKING_PX, 60))

    assert abs(clean - speckled) < 0.1


def test_flickering_noise_inflates_the_reading():
    """The metric's known weakness, pinned rather than papered over.

    Noise that lands somewhere new each frame is change, and this
    measures change, so it reads as motion. A still scene under heavy
    flicker can read faster than a walking one under none. The defence
    is upstream (the noise reducer and the subtractor exist to remove
    this before it arrives), so this test documents the exposure rather
    than asserting it is handled here.
    """
    sampler = AdaptiveFrameSampler()

    still_but_noisy = sampler.velocity(_flickering_window(10, 0, 60))
    walking_and_clean = sampler.velocity(_moving_window(10, WALKING_PX))

    assert still_but_noisy > walking_and_clean


# --- the fixed count, which is what the encoder depends on ---


@pytest.mark.parametrize("speed", [0, 1, 2, 3, 4, 6, 10, 30])
def test_the_count_is_fixed_whatever_the_motion(speed):
    """Always ``sample_frames`` indices, so the encoder has nothing to pad."""
    sampler = AdaptiveFrameSampler()

    assert len(sampler.select(_moving_window(10, speed))) == sampler.sample_frames


def test_the_count_is_fixed_across_window_lengths():
    """Window length changes the spacing, never the number of samples."""
    sampler = AdaptiveFrameSampler()

    for length in (4, 6, 10, 25):
        assert len(sampler.select(_moving_window(length, WALKING_PX))) == 4


def test_a_window_shorter_than_the_sample_count_yields_what_it_has():
    """The one case the count is not fixed, and it cannot be."""
    sampler = AdaptiveFrameSampler(sample_frames=6)

    assert sampler.select(_moving_window(3, 2)) == (0, 1, 2)


def test_a_single_frame_window_yields_its_only_frame():
    """Degenerate but constructible, so it has a defined answer."""
    assert AdaptiveFrameSampler().select(_moving_window(1, 0)) == (0,)


# --- the spacing, which is what the change rate moves ---


def test_slow_motion_spreads_across_the_whole_window():
    """A still scene is sampled end to end, so the samples differ at all."""
    assert AdaptiveFrameSampler().select(_moving_window(10, 0)) == (0, 3, 6, 9)


def test_fast_motion_packs_onto_the_newest_frames():
    """Past ``full_speed`` the samples are the last four frames."""
    assert AdaptiveFrameSampler().select(_moving_window(10, RUNNING_PX)) == (6, 7, 8, 9)


def test_motion_beyond_full_speed_does_not_tighten_further():
    """Consecutive frames is the floor; the spacing saturates there."""
    sampler = AdaptiveFrameSampler()

    at_speed = sampler.select(_moving_window(10, RUNNING_PX))
    far_faster = sampler.select(_moving_window(10, FASTEST_PX))

    assert far_faster == at_speed == (6, 7, 8, 9)


def test_a_subject_that_stops_part_way_reads_as_still():
    """Motion is measured where it is sampled, not where it happened.

    A subject crossing fast enough to reach the far edge is stationary
    for the rest of the window, and only the second half is measured, so
    the reading is zero and the samples spread back out. Correct rather
    than surprising: by the time these frames are sampled there is
    nothing moving in them, and the wide spread is what shows the
    subject arriving and then holding still.
    """
    sampler = AdaptiveFrameSampler()
    window = _moving_window(10, 40)

    assert sampler.velocity(window) == 0.0
    assert sampler.select(window) == (0, 3, 6, 9)


def test_faster_motion_never_spreads_wider():
    """Across the velocity range the span is non-increasing."""
    sampler = AdaptiveFrameSampler()
    speeds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 10]

    spans = [_span(sampler.select(_moving_window(10, s))) for s in speeds]

    assert spans == sorted(spans, reverse=True)


def test_faster_and_slower_motion_produce_different_index_sets():
    """The acceptance criterion: velocity actually changes the selection."""
    sampler = AdaptiveFrameSampler()

    assert sampler.select(_moving_window(10, 0)) != sampler.select(
        _moving_window(10, RUNNING_PX)
    )


def test_the_velocity_term_is_not_a_no_op_at_ten_frames():
    """The issue's question, pinned: the spacing really does move at N=10.

    A plain integer stride would offer only 1, 2 or 3 on a ten-frame
    window. Moving a span instead keeps the choice fine-grained, and this
    asserts that breadth survives.
    """
    sampler = AdaptiveFrameSampler()
    speeds = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 6, 10]

    selections = {sampler.select(_moving_window(10, s)) for s in speeds}

    assert len(selections) == 7


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (0, (0, 3, 6, 9)),
        (0.5, (1, 4, 6, 9)),
        (1, (2, 4, 7, 9)),
        (2, (3, 5, 7, 9)),
        (3, (4, 6, 7, 9)),
        (3.5, (5, 6, 8, 9)),
        (4, (6, 7, 8, 9)),
    ],
)
def test_known_velocity_gives_the_expected_indices(speed, expected):
    """Exact selections for known speeds, so a change of formula shows up."""
    assert AdaptiveFrameSampler().select(_moving_window(10, speed)) == expected


# --- invariants that hold at every velocity ---


@pytest.mark.parametrize("speed", [0, 1, 2, 3, 4, 10, 25])
def test_indices_are_strictly_increasing(speed):
    """No repeats and no going backwards, at any velocity."""
    indices = AdaptiveFrameSampler().select(_moving_window(10, speed))

    assert list(indices) == sorted(set(indices))


@pytest.mark.parametrize("speed", [0, 1, 2, 3, 4, 10, 25])
def test_indices_are_within_the_window(speed):
    """Every index addresses a frame the window actually holds."""
    window = _moving_window(10, speed)

    assert all(
        0 <= index < len(window) for index in AdaptiveFrameSampler().select(window)
    )


@pytest.mark.parametrize("speed", [0, 2, 4, 10, 30])
def test_the_selection_always_ends_at_the_last_frame(speed):
    """The freshest frame is always kept; tightening drops the oldest."""
    window = _moving_window(10, speed)

    assert AdaptiveFrameSampler().select(window)[-1] == len(window) - 1


def test_select_returns_plain_ints():
    """Indices are Python ints, not numpy scalars.

    ``linspace`` yields numpy integers, and a ``np.int64`` reaching
    ``MotionImage.frame_indices`` would compare equal but serialise
    oddly and fail an identity check.
    """
    indices = AdaptiveFrameSampler().select(_moving_window(10, WALKING_PX))

    assert all(type(index) is int for index in indices)


# --- configuration ---


def test_full_speed_sets_where_the_spacing_bottoms_out():
    """A lower ``full_speed`` reaches the tightest spacing on slower motion."""
    gentle = AdaptiveFrameSampler(full_speed=0.005)
    strict = AdaptiveFrameSampler(full_speed=5.0)
    window = _moving_window(10, WALKING_PX)

    assert gentle.select(window) == (6, 7, 8, 9)
    assert strict.select(window) == (0, 3, 6, 9)


def test_sample_frames_sets_how_many_are_kept():
    """The count follows the kwarg, at every velocity."""
    sampler = AdaptiveFrameSampler(sample_frames=6)

    assert len(sampler.select(_moving_window(10, 0))) == 6
    assert len(sampler.select(_moving_window(10, RUNNING_PX))) == 6


def test_the_sampler_is_stateless():
    """Windows are judged independently; nothing carries between them."""
    sampler = AdaptiveFrameSampler()
    fast = _moving_window(10, RUNNING_PX)

    first = sampler.select(fast)
    sampler.select(_moving_window(10, 0))

    assert sampler.select(fast) == first


@pytest.mark.parametrize("value", [None, 4.5, "4", [4], True, False])
def test_non_integer_sample_frames_is_rejected(value):
    """A frame count that is not a whole number fails at construction."""
    with pytest.raises(ValueError, match="sample_frames must be an integer"):
        AdaptiveFrameSampler(sample_frames=value)


@pytest.mark.parametrize("value", [1, 0, -1])
def test_sample_frames_below_two_is_rejected(value):
    """One frame shows no movement, so two is the floor."""
    with pytest.raises(ValueError, match="sample_frames must be at least 2"):
        AdaptiveFrameSampler(sample_frames=value)


@pytest.mark.parametrize("value", [None, "0.35", [0.35], True, False])
def test_non_numeric_full_speed_is_rejected(value):
    """A saturation point that is not a number fails at construction."""
    with pytest.raises(ValueError, match="full_speed must be a number"):
        AdaptiveFrameSampler(full_speed=value)


@pytest.mark.parametrize("value", [0, 0.0, -0.1])
def test_non_positive_full_speed_is_rejected(value):
    """Zero would divide by zero; negative has no meaning."""
    with pytest.raises(ValueError, match="full_speed must be positive"):
        AdaptiveFrameSampler(full_speed=value)


@pytest.mark.parametrize("name", ["sample_frames", "full_speed"])
def test_the_properties_are_read_only(name):
    """Configuration is fixed once validated."""
    sampler = AdaptiveFrameSampler()

    with pytest.raises(AttributeError):
        setattr(sampler, name, 5)
