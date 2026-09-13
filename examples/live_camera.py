"""Watch the preprocessor work on your webcam, live.

Usage:
    python examples/live_camera.py
    python examples/live_camera.py --camera 1 --window-frames 15

One window shows the camera on the left and the latest motion image on
the right. Move in front of the camera: while you keep moving, a new
motion image appears every ``window_frames`` frames. Press q or Esc, or
close the window, to quit.

The first few frames only teach the background, so stay out of shot for
a moment after starting.
"""

import argparse
import sys

import cv2
import numpy as np

from amprep import AdaptiveMotionPreprocessor, MotionImage

WINDOW = "amprep live: camera | motion image"
QUIT_KEYS = {ord("q"), 27}  # q, Esc
MAX_DISPLAY_WIDTH = 1600
"""Wider views are shrunk for the screen. Only the display, never the input."""


def compose(frame: np.ndarray, image: MotionImage | None, count: int) -> np.ndarray:
    """Put the camera frame and the latest motion image side by side."""
    left = frame.copy()
    if image is None:
        right = np.zeros_like(frame)
        caption = "waiting for motion..."
    else:
        right = cv2.cvtColor(image.data, cv2.COLOR_GRAY2BGR)
        caption = f"motion image #{count}  frames {image.frame_indices}"
    _label(left, "camera")
    _label(right, caption)
    return np.hstack([left, right])


def _label(canvas: np.ndarray, text: str) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )


class LiveView:
    """The on-screen window, plus the latest motion image to draw in it."""

    def __init__(self) -> None:
        self.image: MotionImage | None = None
        self.count = 0

    def show(self, frame: np.ndarray) -> bool:
        """Draw one frame. Returns ``False`` once the user wants to quit."""
        view = compose(frame, self.image, self.count)
        if view.shape[1] > MAX_DISPLAY_WIDTH:
            scale = MAX_DISPLAY_WIDTH / view.shape[1]
            view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.imshow(WINDOW, view)

        if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
            return False
        # Closing the window with its X button makes it invisible.
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def camera_frames(index: int, view: LiveView):
    """Yield webcam frames, drawing each one after the pipeline has seen it.

    Drawing happens here rather than in the loop over ``process()``,
    because that loop only runs when a motion image comes out, while the
    live video has to update on every frame.
    """
    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open camera {index}")
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
            # By now the pipeline has processed this frame, and any motion
            # image it produced is already in the view.
            if not view.show(frame):
                return
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--window-frames", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()

    preprocessor = AdaptiveMotionPreprocessor(
        motion_threshold=args.threshold, window_frames=args.window_frames
    )
    view = LiveView()
    try:
        for image in preprocessor.process(camera_frames(args.camera, view)):
            view.image = image
            view.count += 1
    except OSError as error:
        sys.exit(str(error))
    finally:
        cv2.destroyAllWindows()

    print(f"{view.count} motion images")


if __name__ == "__main__":
    main()
