import gc
from pathlib import Path

import cv2
import numpy as np
import pytest

from amprep import Frame, VideoFileReader

FRAME_VALUES = (10, 60, 110, 160, 210)
"""Grey levels written to the sample clip, one solid frame each.

Distinct values make frame order readable from the decoded pixels alone.
They are spaced far wider than MJPEG's error on a flat image, so a frame
stays identifiable even though the clip is lossy.
"""

CLIP_FPS = 10.0
CLIP_SIZE = (64, 48)  # (width, height), the order VideoWriter expects

_REAL_VIDEO_CAPTURE = cv2.VideoCapture
"""The genuine class, bound before any test swaps the name for a spy."""


@pytest.fixture(scope="module")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write a short clip of solid grey frames and return its path.

    MJPEG in an AVI container is used because it is the codec most
    consistently present across the platform wheels this package is
    tested on. It is lossy, so tests compare decoded frames by mean
    value rather than for exact equality.
    """
    path = tmp_path_factory.mktemp("clips") / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), CLIP_FPS, CLIP_SIZE
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available to build the sample clip")

    try:
        for value in FRAME_VALUES:
            writer.write(np.full((CLIP_SIZE[1], CLIP_SIZE[0], 3), value, np.uint8))
    finally:
        writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("MJPG encoder produced no output")
    return path


class CaptureSpy:
    """A ``cv2.VideoCapture`` that records whether it was released.

    Whether a capture leaks is not observable from the reader's output,
    so the tests below watch the handle itself: every spy built during a
    test is kept in ``instances`` and can be asked afterwards whether
    ``release`` ever ran.
    """

    instances: list["CaptureSpy"] = []
    fps_override: float | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._capture = _REAL_VIDEO_CAPTURE(*args, **kwargs)
        self.released = False
        CaptureSpy.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV name
        return self._capture.isOpened()

    def read(self) -> tuple[bool, Frame]:
        return self._capture.read()

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS and CaptureSpy.fps_override is not None:
            return CaptureSpy.fps_override
        return self._capture.get(prop)

    def release(self) -> None:
        self.released = True
        self._capture.release()


@pytest.fixture
def captures(monkeypatch: pytest.MonkeyPatch) -> type[CaptureSpy]:
    """Replace ``cv2.VideoCapture`` with a spy for the duration of a test."""
    CaptureSpy.instances = []
    CaptureSpy.fps_override = None
    monkeypatch.setattr(cv2, "VideoCapture", CaptureSpy)
    return CaptureSpy


def _means(frames: list[Frame]) -> list[float]:
    return [float(frame.mean()) for frame in frames]


def test_yields_frames_in_order(sample_clip: Path):
    """Frames come out in the order the file stores them."""
    frames = list(VideoFileReader(sample_clip))

    assert len(frames) == len(FRAME_VALUES)
    assert _means(frames) == pytest.approx(list(FRAME_VALUES), abs=2.0)


def test_yielded_frames_match_the_frame_contract(sample_clip: Path):
    """Every frame is a uint8 BGR array of the clip's dimensions."""
    width, height = CLIP_SIZE

    for frame in VideoFileReader(sample_clip):
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (height, width, 3)


def test_frames_are_independent_arrays(sample_clip: Path):
    """Retaining a frame is safe: a later read does not overwrite it."""
    frames = list(VideoFileReader(sample_clip))

    assert len({id(frame) for frame in frames}) == len(frames)
    assert _means(frames) == pytest.approx(list(FRAME_VALUES), abs=2.0)


def test_reader_can_be_iterated_more_than_once(sample_clip: Path):
    """Each pass re-opens the file and starts again at the first frame."""
    reader = VideoFileReader(sample_clip)

    assert _means(list(reader)) == pytest.approx(_means(list(reader)), abs=2.0)


def test_capture_is_released_after_exhaustion(sample_clip: Path, captures):
    """No capture handle survives a fully consumed iterator."""
    list(VideoFileReader(sample_clip))

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_capture_is_released_when_iteration_is_abandoned(sample_clip: Path, captures):
    """Breaking out of the loop still releases the capture."""
    for _ in VideoFileReader(sample_clip):
        break
    gc.collect()

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_capture_is_released_when_the_consumer_raises(sample_clip: Path, captures):
    """An exception thrown into the iterator unwinds through the release."""
    iterator = iter(VideoFileReader(sample_clip))
    next(iterator)

    with pytest.raises(RuntimeError, match="consumer failed"):
        iterator.throw(RuntimeError("consumer failed"))

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_capture_is_released_when_the_iterator_is_closed(sample_clip: Path, captures):
    """Closing a part-way iterator releases the capture immediately."""
    iterator = iter(VideoFileReader(sample_clip))
    next(iterator)
    iterator.close()

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_fps_reports_the_rate_the_file_declares(sample_clip: Path):
    """A file that stores a frame rate has it reported unchanged."""
    assert VideoFileReader(sample_clip).fps == pytest.approx(CLIP_FPS)


def test_fps_is_none_when_the_file_reports_no_rate(sample_clip: Path, captures):
    """A zero from OpenCV means the rate is unknown, reported as None."""
    captures.fps_override = 0.0

    assert VideoFileReader(sample_clip).fps is None


def test_fps_is_none_when_the_reported_rate_is_not_finite(sample_clip: Path, captures):
    """A non-finite rate is unusable and is reported as None too."""
    captures.fps_override = float("nan")

    assert VideoFileReader(sample_clip).fps is None


def test_fps_releases_the_capture_it_opens(sample_clip: Path, captures):
    """Asking for the frame rate leaves nothing open behind it."""
    assert VideoFileReader(sample_clip).fps is not None

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_fps_is_probed_once(sample_clip: Path, captures):
    """The frame rate is remembered rather than re-read on each access."""
    reader = VideoFileReader(sample_clip)

    assert reader.fps == pytest.approx(CLIP_FPS)
    assert reader.fps == pytest.approx(CLIP_FPS)
    assert len(captures.instances) == 1


def test_iterating_supplies_the_frame_rate_without_reopening(
    sample_clip: Path, captures
):
    """A pass over the frames answers a later fps query for free."""
    reader = VideoFileReader(sample_clip)
    list(reader)

    assert reader.fps == pytest.approx(CLIP_FPS)
    assert len(captures.instances) == 1


def test_missing_file_is_reported_at_construction(tmp_path: Path):
    """A path that names no file fails before any frame is requested."""
    with pytest.raises(FileNotFoundError, match="No such video file"):
        VideoFileReader(tmp_path / "absent.avi")


def test_directory_is_not_a_video_file(tmp_path: Path):
    """A directory is rejected the same way a missing file is."""
    with pytest.raises(FileNotFoundError):
        VideoFileReader(tmp_path)


@pytest.mark.parametrize("value", [None, 3, b"clip.avi", ["clip.avi"]])
def test_path_must_be_a_string_or_path_like(value):
    """Values that cannot name a file are rejected by type."""
    with pytest.raises(TypeError, match="path must be a string or path-like"):
        VideoFileReader(value)


def test_string_paths_are_accepted(sample_clip: Path):
    """A plain string names the file as well as a Path does."""
    assert list(VideoFileReader(str(sample_clip)))


def test_undecodable_file_raises_os_error(tmp_path: Path):
    """A file OpenCV cannot open is an error, not an empty iterator."""
    path = tmp_path / "not-a-video.avi"
    path.write_bytes(b"this is not video data")

    with pytest.raises(OSError, match="Could not open video file"):
        list(VideoFileReader(path))


def test_undecodable_file_leaves_no_capture_open(tmp_path: Path, captures):
    """The failed capture is released before the error propagates."""
    path = tmp_path / "not-a-video.avi"
    path.write_bytes(b"this is not video data")

    with pytest.raises(OSError):
        list(VideoFileReader(path))

    assert captures.instances
    assert all(capture.released for capture in captures.instances)


def test_path_property_exposes_the_source(sample_clip: Path):
    """The reader reports the file it was built for."""
    assert VideoFileReader(str(sample_clip)).path == sample_clip
