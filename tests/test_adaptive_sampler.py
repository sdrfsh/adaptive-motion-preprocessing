import numpy as np
import pytest

from amprep.adaptive_sampler import (
    DEFAULT_FULL_SPEED,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    AdaptiveFrameSampler,
)
from amprep.types import Window

HEIGHT, WIDTH = 120, 160
"""A 120x160 frame has a diagonal of exactly 200px, so a velocity in
fractions-of-the-diagonal converts to pixels in the head."""

BOX = 20
DIAGONAL = 200.0


def _moving_window(frames: int, pixels_per_frame: float) -> Window:
    """A window whose foreground box drifts right at a known speed."""
    masks, pictures = [], []
    for index in range(frames):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        left = int(round(10 + index * pixels_per_frame))
        left = max(0, min(WIDTH - BOX, left))
        mask[50 : 50 + BOX, left : left + BOX] = 255
        masks.append(mask)
        pictures.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    return Window(frames=tuple(pictures), masks=tuple(masks))


def _blank_window(frames: int) -> Window:
    """A window whose masks hold no foreground at all."""
    return Window(
        frames=tuple(np.zeros((HEIGHT, WIDTH, 3), np.uint8) for _ in range(frames)),
        masks=tuple(np.zeros((HEIGHT, WIDTH), np.uint8) for _ in range(frames)),
    )


def test_defaults_are_what_the_docstrings_claim():
    """The packaged defaults, pinned so a silent change is visible."""
    sampler = AdaptiveFrameSampler()

    assert (sampler.min_frames, sampler.max_frames) == (3, 8)
    assert (DEFAULT_MIN_FRAMES, DEFAULT_MAX_FRAMES) == (3, 8)
    assert sampler.full_speed == DEFAULT_FULL_SPEED == 0.05


def test_a_still_window_keeps_the_minimum():
    """Nothing moving means nothing to resolve, so the floor is enough."""
    sampler = AdaptiveFrameSampler()

    assert len(sampler.select(_moving_window(10, 0))) == sampler.min_frames


def test_motion_at_full_speed_keeps_the_maximum():
    """At ``full_speed`` the count saturates, by definition."""
    sampler = AdaptiveFrameSampler()
    # full_speed 0.05 of a 200px diagonal is 10px per frame.
    window = _moving_window(10, DEFAULT_FULL_SPEED * DIAGONAL)

    assert len(sampler.select(window)) == sampler.max_frames


def test_motion_beyond_full_speed_does_not_keep_more():
    """The count saturates rather than running past ``max_frames``."""
    sampler = AdaptiveFrameSampler()

    at_speed = sampler.select(_moving_window(10, 10))
    far_faster = sampler.select(_moving_window(10, 40))

    assert len(far_faster) == len(at_speed) == sampler.max_frames


def test_faster_motion_keeps_more_frames():
    """The headline behaviour: the count rises with velocity."""
    sampler = AdaptiveFrameSampler()

    counts = [len(sampler.select(_moving_window(10, speed))) for speed in (0, 4, 10)]

    assert counts[0] < counts[1] < counts[2]


def test_the_count_never_falls_as_motion_rises():
    """Across the whole velocity range the mapping is non-decreasing."""
    sampler = AdaptiveFrameSampler()
    speeds = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20]

    counts = [len(sampler.select(_moving_window(10, speed))) for speed in speeds]

    assert counts == sorted(counts)


def test_the_velocity_term_is_not_a_no_op_at_ten_frames():
    """The issue's question, pinned: the rate really does move at N=10.

    With ``window_frames=10`` there is little room between the floor and
    the ceiling, so this asserts the adaptation actually produces a range
    of counts rather than quietly collapsing to one.
    """
    sampler = AdaptiveFrameSampler()
    speeds = [0, 1, 4, 5, 8, 10]

    counts = {len(sampler.select(_moving_window(10, speed))) for speed in speeds}

    assert counts == {3, 4, 5, 6, 7, 8}


def test_faster_and_slower_motion_produce_different_index_sets():
    """Not merely different counts — different frames are chosen."""
    sampler = AdaptiveFrameSampler()

    slow = sampler.select(_moving_window(10, 0))
    fast = sampler.select(_moving_window(10, 10))

    assert slow != fast


@pytest.mark.parametrize("speed", [0, 1, 2, 4, 6, 10, 25])
def test_indices_are_strictly_increasing(speed):
    """No repeats and no going backwards, at any velocity."""
    indices = AdaptiveFrameSampler().select(_moving_window(10, speed))

    assert list(indices) == sorted(set(indices))


@pytest.mark.parametrize("speed", [0, 1, 2, 4, 6, 10, 25])
def test_indices_are_within_the_window(speed):
    """Every index addresses a frame the window actually holds."""
    window = _moving_window(10, speed)

    indices = AdaptiveFrameSampler().select(window)

    assert all(0 <= index < len(window) for index in indices)


@pytest.mark.parametrize("speed", [0, 4, 10])
def test_the_selection_spans_the_whole_window(speed):
    """The first and last frames are always kept, so the burst is bracketed."""
    window = _moving_window(10, speed)

    indices = AdaptiveFrameSampler().select(window)

    assert indices[0] == 0
    assert indices[-1] == len(window) - 1


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (0, (0, 4, 9)),
        (2, (0, 3, 6, 9)),
        (10, (0, 1, 3, 4, 5, 6, 8, 9)),
    ],
)
def test_known_velocity_gives_the_expected_indices(speed, expected):
    """Exact selections for known speeds, so a change of formula shows up."""
    assert AdaptiveFrameSampler().select(_moving_window(10, speed)) == expected


def test_velocity_is_zero_for_a_still_window():
    """A foreground that does not move has no speed."""
    assert AdaptiveFrameSampler().velocity(_moving_window(10, 0)) == 0.0


def test_velocity_matches_the_distance_travelled():
    """Velocity is the centroid's travel per frame over the diagonal."""
    sampler = AdaptiveFrameSampler()

    # 4px per frame across a 200px diagonal.
    assert sampler.velocity(_moving_window(10, 4)) == pytest.approx(4 / DIAGONAL)


def test_velocity_is_zero_when_the_foreground_is_missing():
    """An empty mask locates nothing, which is not the same as standing still."""
    assert AdaptiveFrameSampler().velocity(_blank_window(10)) == 0.0


def test_velocity_is_zero_for_a_single_frame_window():
    """A window with no middle frame has nothing to compare against."""
    assert AdaptiveFrameSampler().velocity(_moving_window(1, 10)) == 0.0


def test_velocity_is_resolution_independent():
    """The same motion, measured on a bigger frame, reads the same speed.

    Velocity is a fraction of the diagonal rather than a pixel count, so
    ``full_speed`` keeps its meaning whatever the caller feeds in.
    """
    sampler = AdaptiveFrameSampler()

    def window_at(scale: int) -> Window:
        masks, pictures = [], []
        for index in range(10):
            mask = np.zeros((HEIGHT * scale, WIDTH * scale), np.uint8)
            left = (10 + index * 4) * scale
            mask[50 * scale : 70 * scale, left : left + BOX * scale] = 255
            masks.append(mask)
            pictures.append(np.zeros((HEIGHT * scale, WIDTH * scale, 3), np.uint8))
        return Window(frames=tuple(pictures), masks=tuple(masks))

    assert sampler.velocity(window_at(1)) == pytest.approx(
        sampler.velocity(window_at(3))
    )


def test_a_short_window_is_never_oversampled():
    """A window holding fewer frames than wanted yields only what it has."""
    sampler = AdaptiveFrameSampler(min_frames=8, max_frames=8)

    indices = sampler.select(_moving_window(3, 10))

    assert len(indices) == 3
    assert indices == (0, 1, 2)


def test_equal_min_and_max_disables_the_adaptation():
    """Pinning both ends turns the sampler into a fixed-rate one."""
    sampler = AdaptiveFrameSampler(min_frames=4, max_frames=4)

    slow = sampler.select(_moving_window(10, 0))
    fast = sampler.select(_moving_window(10, 30))

    assert slow == fast
    assert len(slow) == 4


def test_full_speed_sets_where_saturation_lands():
    """A lower ``full_speed`` reaches the maximum on slower motion."""
    gentle = AdaptiveFrameSampler(full_speed=0.005)
    strict = AdaptiveFrameSampler(full_speed=0.5)
    window = _moving_window(10, 4)

    assert len(gentle.select(window)) == gentle.max_frames
    assert len(strict.select(window)) == strict.min_frames


def test_the_sampler_is_stateless():
    """Windows are judged independently; nothing carries between them."""
    sampler = AdaptiveFrameSampler()
    fast = _moving_window(10, 10)
    slow = _moving_window(10, 0)

    first = sampler.select(fast)
    sampler.select(slow)

    assert sampler.select(fast) == first


@pytest.mark.parametrize("value", [None, 3.5, "3", [3], True, False])
def test_non_integer_min_frames_is_rejected(value):
    """A frame count that is not a whole number fails at construction."""
    with pytest.raises(ValueError, match="min_frames must be an integer"):
        AdaptiveFrameSampler(min_frames=value)


@pytest.mark.parametrize("value", [1, 0, -1])
def test_min_frames_below_two_is_rejected(value):
    """One frame shows no movement, so two is the floor."""
    with pytest.raises(ValueError, match="min_frames must be at least 2"):
        AdaptiveFrameSampler(min_frames=value)


@pytest.mark.parametrize("value", [None, 8.5, "8", True])
def test_non_integer_max_frames_is_rejected(value):
    """Same rule at the other end."""
    with pytest.raises(ValueError, match="max_frames must be an integer"):
        AdaptiveFrameSampler(max_frames=value)


def test_max_frames_below_min_frames_is_rejected():
    """An inverted range would make faster motion keep fewer frames."""
    with pytest.raises(ValueError, match="max_frames must be at least min_frames"):
        AdaptiveFrameSampler(min_frames=6, max_frames=5)


@pytest.mark.parametrize("value", [None, "0.05", [0.05], True, False])
def test_non_numeric_full_speed_is_rejected(value):
    """A saturation point that is not a number fails at construction."""
    with pytest.raises(ValueError, match="full_speed must be a number"):
        AdaptiveFrameSampler(full_speed=value)


@pytest.mark.parametrize("value", [0, 0.0, -0.1])
def test_non_positive_full_speed_is_rejected(value):
    """Zero would divide by zero; negative has no meaning."""
    with pytest.raises(ValueError, match="full_speed must be positive"):
        AdaptiveFrameSampler(full_speed=value)


@pytest.mark.parametrize("name", ["min_frames", "max_frames", "full_speed"])
def test_the_properties_are_read_only(name):
    """Configuration is fixed once validated."""
    sampler = AdaptiveFrameSampler()

    with pytest.raises(AttributeError):
        setattr(sampler, name, 5)


def test_select_returns_plain_ints():
    """Indices are Python ints, not numpy scalars, so they index and compare.

    ``linspace`` and ``unique`` both yield numpy integers, and a
    ``np.int64`` leaking into ``MotionImage.frame_indices`` would compare
    equal but serialise oddly and fail an ``is``-style identity check.
    """
    indices = AdaptiveFrameSampler().select(_moving_window(10, 4))

    assert all(type(index) is int for index in indices)
