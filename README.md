# Adaptive Motion Preprocessing

Adaptive Motion Preprocessing converts video frames into encoded motion images for downstream neural-network inference. It adaptively selects frames while preserving a consistent output format for the model.

## Installation

```powershell
pip install adaptive-motion-preprocessing
```

## Usage

The input is an iterable of frames — a `uint8` `(H, W, 3)` BGR array each. A
list, a generator, a custom iterator and a loop around a live camera are all
accepted and all treated the same, because by the time a frame reaches the
pipeline there is nothing left to tell them apart.

Reading video is deliberately your job, not the package's. Nothing inside
`amprep` opens a camera or a file, so nothing inside it can leak a device
handle. Producing frames from a file takes a few lines:

```python
import cv2
from amprep import AdaptiveMotionPreprocessor


def frames_from(path):
    capture = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


for image in AdaptiveMotionPreprocessor().process(frames_from("clip.mp4")):
    ...
```

Keep the `try`/`finally`. Without it the capture handle leaks whenever the
loop is left early — a `break`, an exception, or simply not consuming the
generator to the end.

Frames must arrive in capture order. The pipeline accumulates state across
consecutive frames, so a shuffled stream describes motion that never happened.

`process` consumes the stream lazily, one frame at a time, and never
materialises it, so an unbounded source such as a camera is fine.

Status: in development.
