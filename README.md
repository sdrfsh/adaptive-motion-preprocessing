# Adaptive Motion Preprocessing

Turns video frames into motion images for a neural network. While something
moves, each window of frames becomes one `uint8` grayscale image. Older
frames are painted dimmer and newer ones brighter. Which frames get sampled
depends on how fast the motion is, and every image comes out in the same shape.

## Install

```sh
pip install adaptive-motion-preprocessing
```

The import name is `amprep`.

## Example

The package takes any iterable of `uint8` BGR frames and never opens a video
itself. Reading one is a few lines:

```python
import cv2

from amprep import AdaptiveMotionPreprocessor


def frames_from(path):
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise OSError(f"cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


for image in AdaptiveMotionPreprocessor().process(frames_from("clip.mp4")):
    print(image.data.shape)
```

While motion lasts, one image comes out every `window_frames` frames (default
10). That count is frames, not seconds: 10 frames is about 0.33 s at 30 fps
and 1 s at 10 fps. The package never reads the frame rate, so converting
between the two is up to you.

More: [examples/from_video_file.py](examples/from_video_file.py) ·
[examples/custom_background_subtractor.py](examples/custom_background_subtractor.py)
(plugging in your own stage)
