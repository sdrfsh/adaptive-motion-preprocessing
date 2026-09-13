"""The examples run, and the README's recipe is the one they ship.

Each example is run as its own process, the way a reader would run it, so
a missing import or a broken ``__main__`` fails here rather than for them.
"""

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from amprep import MotionImage

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def _load(script: str):
    """Import an example as a module, without running its ``__main__``."""
    spec = importlib.util.spec_from_file_location(Path(script).stem, EXAMPLES / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_readme_frames_from_is_the_example_verbatim():
    """Copied by hand, so this is what stops the two drifting apart."""
    snippet = inspect.getsource(_load("from_video_file.py").frames_from)

    assert snippet in (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_links_every_example():
    """Full GitHub URLs, because PyPI shows the README without the repo."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    base = "https://github.com/sdrfsh/adaptive-motion-preprocessing/blob/main"

    for script in sorted(EXAMPLES.glob("*.py")):
        assert f"({base}/examples/{script.name})" in readme


def test_from_video_file_prints_a_shape_per_motion_image(tmp_path):
    """A real file on disk, written with OpenCV's built-in MJPEG encoder."""
    path = tmp_path / "clip.avi"
    height, width = 120, 160
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height)
    )
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((height, width, 3), 90, dtype=np.uint8)
        if 10 <= index < 30:
            left = 5 + (index - 10) * 5
            frame[40:80, left : left + 40] = 200
        writer.write(frame)
    writer.release()

    result = _run("from_video_file.py", str(path))

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines
    assert all(line.endswith(f"shape {(height, width)}") for line in lines)


def test_from_video_file_fails_loudly_on_a_missing_file(tmp_path):
    """A bad path must not look like a video with no motion in it."""
    result = _run("from_video_file.py", str(tmp_path / "missing.mp4"))

    assert result.returncode != 0
    assert "cannot open video" in result.stderr


def test_custom_background_subtractor_runs_both_pipelines():
    result = _run("custom_background_subtractor.py")

    assert result.returncode == 0, result.stderr
    assert "FrameDifferenceSubtractor" in result.stdout
    assert "shape (120, 240), dtype uint8" in result.stdout


def test_live_camera_starts_without_a_camera():
    """No webcam in CI, so ``--help`` proves the imports and arguments."""
    result = _run("live_camera.py", "--help")

    assert result.returncode == 0, result.stderr
    assert "--window-frames" in result.stdout


def test_live_camera_shows_the_camera_and_motion_image_side_by_side():
    """The drawing half of the live example, checked without a window."""
    live = _load("live_camera.py")
    frame = np.full((48, 64, 3), 90, dtype=np.uint8)
    image = MotionImage(data=np.full((48, 64), 255, np.uint8), frame_indices=(6, 9))

    waiting = live.compose(frame, None, 0)
    showing = live.compose(frame, image, 1)

    assert waiting.shape == showing.shape == (48, 128, 3)
    # Below the caption bar, the right half is the motion image itself.
    assert (waiting[40:, 64:] == 0).all()
    assert (showing[40:, 64:] == 255).all()
    # The camera frame is copied, not drawn on.
    assert (frame == 90).all()
