import numpy as np
import pytest

from amprep import Frame
from amprep.motion_trigger import DEFAULT_THRESHOLD, MotionTrigger

MASK_SHAPE = (10, 10)
"""One hundred pixels, so a pixel count reads directly as a percentage.

One pixel is exactly 1%, which is also ``DEFAULT_THRESHOLD`` — the
boundary the trigger is most worth testing at.
"""


def _mask(foreground_pixels: int, value: int = 255) -> Frame:
    """A mask with exactly ``foreground_pixels`` of its 100 pixels set."""
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask.reshape(-1)[:foreground_pixels] = value
    return mask


def test_default_threshold_is_one_percent():
    """The packaged default sits between a speck and a small object."""
    assert DEFAULT_THRESHOLD == 0.01
    assert MotionTrigger().threshold == DEFAULT_THRESHOLD


def test_threshold_is_kept_as_given():
    """A caller-supplied threshold is the one used."""
    assert MotionTrigger(threshold=0.25).threshold == 0.25


def test_an_integer_threshold_becomes_a_float():
    """``1`` and ``1.0`` mean the same thing, and are stored the same way."""
    threshold = MotionTrigger(threshold=1).threshold

    assert threshold == 1.0
    assert isinstance(threshold, float)


@pytest.mark.parametrize("threshold", [None, "0.5", [0.5], b"0.5", True, False])
def test_non_numeric_threshold_is_rejected(threshold):
    """A threshold that is not a number fails at construction."""
    with pytest.raises(ValueError, match="threshold must be a number"):
        MotionTrigger(threshold=threshold)


@pytest.mark.parametrize("threshold", [0, 0.0, -0.01, -1, 1.01, 2])
def test_threshold_outside_the_range_is_rejected(threshold):
    """Zero triggers on nothing and above one triggers on nothing."""
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        MotionTrigger(threshold=threshold)


def test_a_threshold_of_one_is_allowed():
    """Demanding the whole frame is extreme but meaningful, so it is allowed."""
    trigger = MotionTrigger(threshold=1.0)

    assert trigger.update(_mask(99)) is False
    assert trigger.update(_mask(100)) is True


def test_starts_idle():
    """A fresh trigger has seen nothing, so it reports nothing."""
    trigger = MotionTrigger()

    assert trigger.is_active is False
    assert trigger.last_fraction == 0.0


def test_motion_at_exactly_the_threshold_counts():
    """The comparison is ``>=``: landing on the threshold is motion."""
    trigger = MotionTrigger(threshold=0.01)

    assert trigger.update(_mask(1)) is True


def test_motion_below_the_threshold_does_not_count():
    """One pixel short of the threshold is still stillness."""
    trigger = MotionTrigger(threshold=0.02)

    assert trigger.update(_mask(1)) is False


def test_update_returns_what_is_active_reports():
    """The return value and the property are the same answer."""
    trigger = MotionTrigger()

    returned = trigger.update(_mask(40))

    assert returned is trigger.is_active is True


def test_update_records_the_measured_fraction():
    """``last_fraction`` is the measurement, whatever the verdict was."""
    trigger = MotionTrigger(threshold=0.5)

    trigger.update(_mask(12))

    assert trigger.is_active is False
    assert trigger.last_fraction == pytest.approx(0.12)


def test_the_very_first_mask_can_activate():
    """There is no warm-up here; that belongs to the background subtractor."""
    assert MotionTrigger().update(_mask(50)) is True


def test_any_non_zero_pixel_counts_as_foreground():
    """Foreground is counted, not thresholded: a mask of ones still counts.

    ``KNNBackgroundSubtractor`` emits a strictly binary 0/255 mask, but
    the trigger does not depend on that — a custom subtractor labelling
    foreground with 1 is measured the same way.
    """
    trigger = MotionTrigger(threshold=0.05)

    assert trigger.update(_mask(10, value=1)) is True
    assert trigger.last_fraction == pytest.approx(0.10)


def test_falls_back_to_idle_when_the_motion_stops():
    """Idle to active and back again, following the masks it is given."""
    trigger = MotionTrigger(threshold=0.05)

    assert trigger.update(_mask(20)) is True
    assert trigger.update(_mask(0)) is False
    assert trigger.is_active is False


def test_reset_returns_to_idle():
    """Reset discards the verdict as well as the measurement."""
    trigger = MotionTrigger(threshold=0.05)
    trigger.update(_mask(50))

    trigger.reset()

    assert trigger.is_active is False
    assert trigger.last_fraction == 0.0


def test_reset_returns_none():
    """Reset is a command, not a query."""
    assert MotionTrigger().reset() is None


def test_reset_keeps_the_threshold():
    """The dial is configuration, not accumulated state."""
    trigger = MotionTrigger(threshold=0.25)

    trigger.reset()

    assert trigger.threshold == 0.25


def test_the_trigger_is_usable_after_a_reset():
    """A reset trigger is a working trigger, not a spent one."""
    trigger = MotionTrigger(threshold=0.05)
    trigger.update(_mask(50))
    trigger.reset()

    assert trigger.update(_mask(50)) is True


@pytest.mark.parametrize("name", ["threshold", "is_active", "last_fraction"])
def test_the_properties_are_read_only(name):
    """Nothing outside the trigger may rewrite what it decided or was told."""
    trigger = MotionTrigger()

    with pytest.raises(AttributeError):
        setattr(trigger, name, 0.5)


def test_a_non_square_mask_is_measured_by_area():
    """The fraction is over every pixel, whatever shape they arrive in."""
    trigger = MotionTrigger(threshold=0.1)
    mask = np.zeros((4, 25), dtype=np.uint8)
    mask.reshape(-1)[:20] = 255

    assert trigger.update(mask) is True
    assert trigger.last_fraction == pytest.approx(0.2)
