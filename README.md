# 🎞️ Adaptive Motion Preprocessing

Turn video into motion images your neural network can read. 

While something moves, each window of frames becomes one grayscale picture:
older frames dim, newer ones bright, so a single image shows where the
subject went and how fast. Every image has the same shape, so your model
never gets a surprise. 

## 📦 Install

```sh
pip install adaptive-motion-preprocessing
```

Then `import amprep`.

## 🚀 Quick start

You bring the frames (any iterable of `uint8` BGR arrays) and the package
does the rest:

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

## 🎥 Try it on your webcam

See your camera and the motion images side by side, live:

```sh
git clone https://github.com/sdrfsh/adaptive-motion-preprocessing
cd adaptive-motion-preprocessing
pip install -e .
python examples/live_camera.py
```

Stay out of shot for a second while it learns the background, then move.
Press `q` or `Esc` to quit. Add `--camera 1` for an external webcam, or
`--threshold 0.03` if it triggers when nothing is moving.

## ⚙️ Settings

All optional keyword arguments of `AdaptiveMotionPreprocessor(...)`:

| Setting | Default | What it does |
| --- | --- | --- |
| `motion_threshold` | `0.01` | Share of the frame that must be moving before frames are collected |
| `window_frames` | `10` | Frames per window, and one image per full window |
| `sample_frames` | `4` | Frames painted into each image (at most `window_frames`) |
| `width`, `height` | `None` | Output size; leave unset to keep the frame size, or set both |
| `noise_reducer` | median filter | Your own `NoiseReducer` subclass |
| `background_subtractor` | KNN | Your own `BackgroundSubtractor` subclass |

## 💡 Good to know

- ⏱️ **Frames, not seconds.** 10 frames is about 0.33 s at 30 fps and 1 s at
  10 fps. The package never reads the frame rate, so that math is yours.
- 🔁 **A steady stream.** While motion lasts you get one image every
  `window_frames` frames. A half-full window is dropped when motion stops.
- 🌱 **Warm-up.** The default subtractor spends its first 4 frames learning
  the background, so they never produce images. Change it with
  `KNNBackgroundSubtractor(warmup_frames=...)`.
- 🎬 **New scene?** Call `reset()`. It forgets the background, any half-built
  window and the frame size. Otherwise state carries over between
  `process()` calls.
- 📐 **One frame size per scene.** Frames that change size mid-stream raise a
  `ValueError`. Call `reset()` first if the change is on purpose.

## 📚 Examples

- [examples/live_camera.py](https://github.com/sdrfsh/adaptive-motion-preprocessing/blob/main/examples/live_camera.py): watch it live on your webcam, camera and motion image side by side
- [examples/from_video_file.py](https://github.com/sdrfsh/adaptive-motion-preprocessing/blob/main/examples/from_video_file.py): run it on a video file
- [examples/custom_background_subtractor.py](https://github.com/sdrfsh/adaptive-motion-preprocessing/blob/main/examples/custom_background_subtractor.py): plug in your own stage
