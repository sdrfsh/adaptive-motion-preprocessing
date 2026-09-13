import numpy as np
import pytest

from amprep.adaptive_sampler import (
    DEFAULT_FULL_SPEED,
    DEFAULT_SAMPLE_FRAMES,
    AdaptiveFrameSampler,
)
from amprep.types import Window

HEIGHT, WIDTH = 120, 160
"""A 120x160 frame has a diagonal of exactly 200px, so a velocity in
fractions-of-the-diagonal converts to pixels in the head."""

BOX = 20
DIAGONAL = 200.0
FULL_SPEED_PX = DEFAULT_FULL_SPEED * DIAGONAL  # 10 px per frame


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


def _span(indices: tuple[int, ...]) -> int:
    """How much of the window the selection reaches across."""
    return indices[-1] - indices[0]


def test_defaults_are_what_the_docstrings_claim():
    """The packaged defaults, pinned so a silent change is visible."""
    sampler = AdaptiveFrameSampler()

    assert sampler.sample_frames == DEFAULT_SAMPLE_FRAMES == 4
    assert sampler.full_speed == DEFAULT_FULL_SPEED == 0.05


# --- the fixed count, which is what the encoder depends on ---


@pytest.mark.parametrize("speed", [0, 1, 2, 4, 6, 8, 10, 30])
def test_the_count_is_fixed_whatever_the_motion(speed):
    """Always ``sample_frames`` indices, so the encoder has nothing to pad."""
    sampler = AdaptiveFrameSampler()

    assert len(sampler.select(_moving_window(10, speed))) == sampler.sample_frames


def test_the_count_is_fixed_across_window_lengths():
    """Window length changes the spacing, never the number of samples."""
    sampler = AdaptiveFrameSampler()

    for length in (4, 6, 10, 25):
        assert len(sampler.select(_moving_window(length, 3))) == 4


def test_a_window_shorter_than_the_sample_count_yields_what_it_has():
    """The one case the count is not fixed, and it cannot be."""
    sampler = AdaptiveFrameSampler(sample_frames=6)

    indices = sampler.select(_moving_window(3, 2))

    assert indices == (0, 1, 2)


def test_a_single_frame_window_yields_its_only_frame():
    """Degenerate but constructible, so it has a defined answer."""
    assert AdaptiveFrameSampler().select(_moving_window(1, 0)) == (0,)


# --- the spacing, which is what velocity actually moves ---


def test_slow_motion_spreads_across_the_whole_window():
    """A still scene is sampled end to end, so the samples differ at all."""
    indices = AdaptiveFrameSampler().select(_moving_window(10, 0))

    assert indices == (0, 3, 6, 9)
    assert indices[-1] == 9


def test_fast_motion_bunches_onto_consecutive_frames():
    """At ``full_speed`` the samples are as tight as they can be."""
    indices = AdaptiveFrameSampler().select(_moving_window(10, FULL_SPEED_PX))

    assert indices == (0, 1, 2, 3)


def test_motion_beyond_full_speed_does_not_tighten_further():
    """Consecutive frames is the floor; the spacing saturates there."""
    sampler = AdaptiveFrameSampler()

    at_speed = sampler.select(_moving_window(10, FULL_SPEED_PX))
    far_faster = sampler.select(_moving_window(10, FULL_SPEED_PX * 4))

    assert far_faster == at_speed == (0, 1, 2, 3)


def test_faster_motion_never_spreads_wider():
    """Across the whole velocity range the span is non-increasing."""
    sampler = AdaptiveFrameSampler()
    speeds = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15]

    spans = [_span(sampler.select(_moving_window(10, s))) for s in speeds]

    assert spans == sorted(spans, reverse=True)


def test_faster_and_slower_motion_produce_different_index_sets():
    """The acceptance criterion: velocity actually changes the selection."""
    sampler = AdaptiveFrameSampler()

    slow = sampler.select(_moving_window(10, 0))
    fast = sampler.select(_moving_window(10, FULL_SPEED_PX))

    assert slow != fast


def test_the_velocity_term_is_not_a_no_op_at_ten_frames():
    """The issue's question, pinned: the spacing really does move at N=10.

    A plain integer stride would offer only 1 or 2 on a ten-frame window —
    two possible selections. Moving a span instead keeps the choice
    fine-grained, and this asserts that breadth survives.
    """
    sampler = AdaptiveFrameSampler()
    speeds = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15]

    selections = {sampler.select(_moving_window(10, s)) for s in speeds}

    assert len(selections) == 7


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (0, (0, 3, 6, 9)),
        (1, (0, 3, 5, 8)),
        (3, (0, 2, 5, 7)),
        (5, (0, 2, 4, 6)),
        (8, (0, 1, 3, 4)),
        (10, (0, 1, 2, 3)),
    ],
)
def test_known_velocity_gives_the_expected_indices(speed, expected):
    """Exact selections for known speeds, so a change of formula shows up."""
    assert AdaptiveFrameSampler().select(_moving_window(10, speed)) == expected


# --- invariants that hold at every velocity ---


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


@pytest.mark.parametrize("speed", [0, 4, 10, 30])
def test_the_selection_always_starts_at_the_first_frame(speed):
    """The burst is read from its beginning, however tightly it is sampled."""
    assert AdaptiveFrameSampler().select(_moving_window(10, speed))[0] == 0


def test_select_returns_plain_ints():
    """Indices are Python ints, not numpy scalars.

    ``linspace`` yields numpy integers, and a ``np.int64`` reaching
    ``MotionImage.frame_indices`` would compare equal but serialise
    oddly and fail an identity check.
    """
    indices = AdaptiveFrameSampler().select(_moving_window(10, 4))

    assert all(type(index) is int for index in indices)


# --- velocity itself ---


def test_velocity_is_zero_for_a_still_window():
    """A foreground that does not move has no speed."""
    assert AdaptiveFrameSampler().velocity(_moving_window(10, 0)) == 0.0


def test_velocity_matches_the_distance_travelled():
    """Velocity is the centroid's travel per frame over the diagonal."""
    sampler = AdaptiveFrameSampler()

    assert sampler.velocity(_moving_window(10, 4)) == pytest.approx(4 / DIAGONAL)


def test_velocity_is_zero_when_the_foreground_is_missing():
    """An empty mask locates nothing, which is not the same as standing still."""
    assert AdaptiveFrameSampler().velocity(_blank_window(10)) == 0.0


def test_velocity_is_zero_for_a_single_frame_window():
    """A window with no middle frame has nothing to compare against."""
    assert AdaptiveFrameSampler().velocity(_moving_window(1, 10)) == 0.0


def test_velocity_is_resolution_independent():
    """The same motion on a bigger frame reads the same speed.

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


# --- configuration ---


def test_full_speed_sets_where_the_spacing_bottoms_out():
    """A lower ``full_speed`` reaches the tightest spacing on slower motion."""
    gentle = AdaptiveFrameSampler(full_speed=0.005)
    strict = AdaptiveFrameSampler(full_speed=0.5)
    window = _moving_window(10, 4)

    assert gentle.select(window) == (0, 1, 2, 3)
    assert strict.select(window) == (0, 3, 6, 9)


def test_sample_frames_sets_how_many_are_kept():
    """The count follows the kwarg, at every velocity."""
    sampler = AdaptiveFrameSampler(sample_frames=6)

    assert len(sampler.select(_moving_window(10, 0))) == 6
    assert len(sampler.select(_moving_window(10, 30))) == 6


def test_the_sampler_is_stateless():
    """Windows are judged independently; nothing carries between them."""
    sampler = AdaptiveFrameSampler()
    fast = _moving_window(10, 10)

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


@pytest.mark.parametrize("name", ["sample_frames", "full_speed"])
def test_the_properties_are_read_only(name):
    """Configuration is fixed once validated."""
    sampler = AdaptiveFrameSampler()

    with pytest.raises(AttributeError):
        setattr(sampler, name, 5)
