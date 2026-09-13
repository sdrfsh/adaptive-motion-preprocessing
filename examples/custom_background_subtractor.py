"""Plug your own background subtractor into the pipeline.

Usage:
    python examples/custom_background_subtractor.py

Subclass ``BackgroundSubtractor``, implement ``_apply`` and ``reset``, and
pass an instance by keyword. Nothing else changes: the same motion trigger,
windowing, sampling and encoding run behind it, and the motion images come
out in the same shape. The script runs the default pipeline and the custom
one over the same synthetic clip so the two can be compared.
"""

import cv2
import numpy as np

from amprep import AdaptiveMotionPreprocessor, BackgroundSubtractor


class FrameDifferenceSubtractor(BackgroundSubtractor):
    """Foreground is whatever changed since the previous frame.

    Far cruder than the default KNN model, but it is useful from its very
    first frame, so it has no warm-up to declare.

    ``_apply`` receives a ``uint8`` ``(H, W, 3)`` BGR frame and must return a
    ``uint8`` ``(H, W)`` mask. ``apply``, which you do not override, checks
    both. ``reset`` forgets the previous frame when the scene changes.
    """

    def __init__(self, threshold: int = 25) -> None:
        self._threshold = threshold
        self._previous: np.ndarray | None = None

    def _apply(self, frame: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._previous is None:
            mask = np.zeros_like(grey)
        else:
            changed = cv2.absdiff(grey, self._previous) > self._threshold
            mask = np.where(changed, np.uint8(255), np.uint8(0))
        self._previous = grey
        return mask

    def reset(self) -> None:
        self._previous = None


def synthetic_clip() -> list[np.ndarray]:
    """Ten still frames, a square sliding right for thirty, ten still again."""
    frames = []
    for index in range(50):
        frame = np.full((120, 240, 3), 90, dtype=np.uint8)
        if 10 <= index < 40:
            left = 10 + (index - 10) * 6
            frame[40:80, left : left + 40] = 200
        frames.append(frame)
    return frames


def show(title: str, preprocessor: AdaptiveMotionPreprocessor) -> None:
    print(title)
    for image in preprocessor.process(synthetic_clip()):
        print(
            f"  shape {image.data.shape}, dtype {image.data.dtype}, "
            f"frames {image.frame_indices}"
        )


def main() -> None:
    show("Default background subtractor (KNN):", AdaptiveMotionPreprocessor())
    show(
        "Custom background subtractor (FrameDifferenceSubtractor):",
        AdaptiveMotionPreprocessor(background_subtractor=FrameDifferenceSubtractor()),
    )


if __name__ == "__main__":
    main()
