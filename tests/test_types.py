import numpy as np
import pytest

from amprep import Frame, MotionImage


def test_types_are_importable():
    assert Frame is not None
    assert MotionImage is not None


def test_motion_image_post_init_accepts_uint8():
    MotionImage(
        data=np.array([[1, 2], [3, 4]], dtype=np.uint8), frame_indices=(0, 1)
    )


def test_motion_image_holds_data_and_indices():
    data = np.zeros((64, 64, 3), dtype=np.uint8)
    image = MotionImage(data=data, frame_indices=(0, 3, 6, 9))

    assert image.data.dtype == np.uint8
    assert image.frame_indices == (0, 3, 6, 9)


def test_motion_image_post_init_raises_type_error():
    with pytest.raises(TypeError):
        MotionImage(
            data=np.array([[1, 2], [3, 4]], dtype=np.int32), frame_indices=(0, 1)
        )
