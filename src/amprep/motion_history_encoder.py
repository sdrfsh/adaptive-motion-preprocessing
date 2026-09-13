import cv2
import numpy as np

from amprep.types import MotionImage, Window


class MotionHistoryEncoder:
    """Paints sampled masks into one gray image: old = dim, new = bright.

    The last stage. Four masks go in and one picture comes out, each mask
    painted at its own brightness so the result reads as a trail::

        mask 1 (oldest)  -> gray  64
        mask 2           -> gray 128
        mask 3           -> gray 191
        mask 4 (newest)  -> gray 255

    Where the masks overlap the brighter value wins, so the subject's
    current position is always on top and the dimmer trail shows where it
    came from.

    Size is the caller's business. Given no size the masks are painted at
    the size they arrive in, untouched — a 100x200 source yields a
    100x200 image, and a model built for that source needs nothing done
    to it. The shape stays fixed across windows because the source
    resolution does, not because anything here enforces it.

    Give a size and every mask is scaled to it instead, for a model whose
    input differs from the camera. Scaling uses ``INTER_AREA``, which
    averages the pixels it merges rather than picking one of them: a
    subject two pixels wide survives a large reduction as a faint value
    instead of disappearing at most positions.

    Args:
        width: Output width in pixels, or ``None`` to keep the source
            width. Must be given together with ``height``.
        height: Output height in pixels, or ``None`` to keep the source
            height. Must be given together with ``width``.
    """

    def __init__(self, *, width: int | None = None, height: int | None = None) -> None:
        if (width is None) != (height is None):
            raise ValueError(
                "width and height must be given together, or neither: "
                f"got width={width!r}, height={height!r}"
            )
        for name, value in (("width", width), ("height", height)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        self._width = width
        self._height = height

    @property
    def width(self) -> int | None:
        """Output width in pixels, or ``None`` to keep the source width."""
        return self._width

    @property
    def height(self) -> int | None:
        """Output height in pixels, or ``None`` to keep the source height."""
        return self._height

    def encode(self, window: Window, indices: tuple[int, ...]) -> MotionImage:
        """Paint the masks at ``indices`` into one motion image.

        Args:
            window: The window the masks come from. See ``Window``.
            indices: Which of its frames to paint, strictly increasing
                and within the window, oldest first. The sampler
                produces them in exactly that form.

        Returns:
            A ``MotionImage`` whose ``data`` is ``uint8`` and is either
            the configured size or the size of the first painted mask.

        Raises:
            ValueError: If ``indices`` is empty, not strictly increasing,
                or reaches outside the window. Brightness encodes time,
                so an out-of-order index would silently paint the past
                on top of the present.
        """
        if not indices:
            raise ValueError("indices must not be empty")
        if list(indices) != sorted(set(indices)):
            raise ValueError(f"indices must be strictly increasing, got {indices}")
        if indices[0] < 0 or indices[-1] >= len(window):
            raise ValueError(f"indices out of range for a window of {len(window)}")

        if self._height is None or self._width is None:
            # No size asked for, so the first mask sets it and nothing is
            # scaled. Later masks are only touched if the source somehow
            # changed resolution mid-window.
            shape = window.masks[indices[0]].shape[:2]
        else:
            shape = (self._height, self._width)

        image = np.zeros(shape, dtype=np.uint8)
        for position, index in enumerate(indices):
            brightness = np.uint8(round(255 * (position + 1) / len(indices)))
            mask = window.masks[index]
            if mask.shape[:2] != shape:
                # cv2 wants (width, height). numpy shapes are (height, width).
                mask = cv2.resize(
                    mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
                )
            painted = np.where(mask > 0, brightness, np.uint8(0))
            # Brighter wins, and brighter means newer, so the newest mask
            # ends up on top wherever the trail crosses itself.
            np.maximum(image, painted, out=image)

        return MotionImage(data=image, frame_indices=tuple(indices))
