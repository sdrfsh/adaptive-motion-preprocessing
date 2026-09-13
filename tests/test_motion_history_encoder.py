import numpy as np
import pytest

from amprep.motion_history_encoder import MotionHistoryEncoder
from amprep.types import Window


def _window(masks):
    frames = tuple(np.zeros((*m.shape, 3), np.uint8) for m in masks)
    return Window(frames=frames, masks=tuple(masks))


def _full(h, w, count):
    return [np.full((h, w), 255, np.uint8)] * count


# ---- size is required ----


def test_no_size_keeps_the_source_size():
    """The default: your frames come back the size you sent them."""
    image = MotionHistoryEncoder().encode(_window(_full(200, 100, 4)), (0, 1, 2, 3))

    assert image.data.shape == (200, 100)


def test_no_size_does_not_resize_at_all():
    """Not merely scaled to the same size -- never scaled.

    A one-pixel line is the giveaway: any resize would blur or move it,
    so finding it exactly where it started proves nothing touched it.
    """
    mask = np.zeros((60, 80), np.uint8)
    mask[:, 37] = 255

    image = MotionHistoryEncoder().encode(_window([mask]), (0,))

    expected = np.zeros((60, 80), np.uint8)
    expected[:, 37] = 255
    np.testing.assert_array_equal(image.data, expected)


@pytest.mark.parametrize("half", [{"width": 64}, {"height": 48}])
def test_half_a_size_is_an_error(half):
    """One without the other is a mistake, not a request."""
    with pytest.raises(ValueError, match="must be given together"):
        MotionHistoryEncoder(**half)


def test_size_cannot_be_given_positionally():
    """Keyword-only, so the two cannot be swapped by accident."""
    with pytest.raises(TypeError):
        MotionHistoryEncoder(64, 48)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "64", True, None])
def test_bad_size_is_an_error(bad):
    with pytest.raises(ValueError):
        MotionHistoryEncoder(width=bad, height=48)


@pytest.mark.parametrize("bad", [0, -1, 2.5, "48", True, None])
def test_bad_height_is_an_error(bad):
    with pytest.raises(ValueError):
        MotionHistoryEncoder(width=64, height=bad)


# ---- output is always the same shape ----


@pytest.mark.parametrize(
    ("h", "w", "count"), [(40, 40, 2), (90, 160, 4), (240, 320, 10)]
)
def test_output_shape_never_changes(h, w, count):
    image = MotionHistoryEncoder(width=64, height=48).encode(
        _window(_full(h, w, count)), tuple(range(count))
    )

    assert image.data.shape == (48, 64)
    assert image.data.dtype == np.uint8


# ---- old = dim, new = bright ----


def test_newer_is_brighter():
    masks = []
    for i in range(4):
        m = np.zeros((40, 40), np.uint8)
        m[10:16, i * 8 : i * 8 + 6] = 255
        masks.append(m)

    image = MotionHistoryEncoder(width=40, height=40).encode(
        _window(masks), (0, 1, 2, 3)
    )

    assert [int(image.data[12, i * 8 + 2]) for i in range(4)] == [64, 128, 191, 255]


def test_brightness_follows_capture_time():
    """Brightness is when a frame was captured, not its place in the sample.

    The last frame is 255 in both selections, while the oldest frame each
    one reaches differs: frame 0 of ten is 26, frame 6 of ten is 178.
    """
    masks = []
    for i in range(10):
        m = np.zeros((20, 40), np.uint8)
        m[5:15, i * 4 : i * 4 + 2] = 255
        masks.append(m)
    window = _window(masks)
    encoder = MotionHistoryEncoder()

    spread = encoder.encode(window, (0, 3, 6, 9)).data
    tight = encoder.encode(window, (6, 7, 8, 9)).data

    assert int(spread[10, 0]) == 26  # frame 0 of 10
    assert int(tight[10, 24]) == 178  # frame 6 of 10
    assert int(spread[10, 36]) == int(tight[10, 36]) == 255


def test_the_newest_mask_wins_where_they_overlap():
    """Brightest wins, and brightest is newest, so the trail reads correctly."""
    masks = [np.full((20, 20), 255, np.uint8) for _ in range(4)]

    image = MotionHistoryEncoder(width=20, height=20).encode(
        _window(masks), (0, 1, 2, 3)
    )

    assert (image.data == 255).all()


def test_background_stays_black():
    """Nowhere a mask ever was is left at zero."""
    masks = []
    for _ in range(4):
        m = np.zeros((20, 20), np.uint8)
        m[0:4, 0:4] = 255
        masks.append(m)

    image = MotionHistoryEncoder(width=20, height=20).encode(
        _window(masks), (0, 1, 2, 3)
    )

    assert (image.data[10:, 10:] == 0).all()


def test_thin_person_is_not_lost_when_shrunk():
    masks = []
    for i in range(4):
        m = np.zeros((240, 320), np.uint8)
        m[60:180, 100 + i * 12 : 104 + i * 12] = 255
        masks.append(m)

    image = MotionHistoryEncoder(width=32, height=24).encode(
        _window(masks), (0, 1, 2, 3)
    )

    assert (image.data == 255).any()


def test_a_single_index_is_painted_at_full_brightness():
    """The last frame of the window is full brightness, even painted alone."""
    image = MotionHistoryEncoder(width=20, height=20).encode(
        _window(_full(20, 20, 4)), (3,)
    )

    assert (image.data == 255).all()


# ---- indices ----


def test_indices_are_saved():
    image = MotionHistoryEncoder(width=20, height=20).encode(
        _window(_full(20, 20, 10)), (3, 5, 7, 9)
    )

    assert image.frame_indices == (3, 5, 7, 9)


@pytest.mark.parametrize("bad", [(), (2, 1), (1, 1), (-1, 2), (0, 10)])
def test_bad_indices_are_an_error(bad):
    with pytest.raises(ValueError):
        MotionHistoryEncoder(width=20, height=20).encode(
            _window(_full(20, 20, 10)), bad
        )
