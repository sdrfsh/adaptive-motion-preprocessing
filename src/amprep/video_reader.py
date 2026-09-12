from collections.abc import Iterator
from math import isfinite
from os import PathLike
from pathlib import Path

import cv2

from amprep.types import Frame


class VideoFileReader:
    """Reads the frames of a video file, in order.

    The input side of the pipeline: a path goes in and ``Frame`` objects
    come out, one per decoded frame, in the order the file stores them.

    The reader holds no open capture between calls. A capture is opened
    when iteration begins and released in a ``finally``, so it closes on
    exhaustion, when the consumer raises, and when an iterator is
    abandoned part-way through. There is deliberately no ``close`` method
    and no context manager: a caller cannot forget to release something
    the reader never hands out.

    Two consequences worth knowing:

    Each pass opens its own capture and starts again at the first frame,
    so the reader is a re-readable handle on a file rather than a
    one-shot iterator. Iterating it twice reads the file twice.

    Every yielded frame is an independent array, freshly allocated by the
    decoder. A stage that collects frames into a window can hold on to
    them without copying; the next ``read`` will not overwrite one it
    already handed out.

    Args:
        path: Path to the video file. It must exist when the reader is
            constructed — a missing file is reported here rather than at
            the first frame — but it is opened only during iteration.

    Raises:
        TypeError: If ``path`` is not a string or path-like object.
        FileNotFoundError: If ``path`` does not name an existing file.
    """

    def __init__(self, path: str | PathLike[str]) -> None:
        if not isinstance(path, (str, PathLike)):
            raise TypeError(f"path must be a string or path-like, got {type(path)}")

        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(f"No such video file: {self._path}")

        self._fps: float | None = None
        self._fps_known = False

    @property
    def path(self) -> Path:
        """The video file this reader was constructed for."""
        return self._path

    @property
    def fps(self) -> float | None:
        """The frame rate the file reports, or ``None`` if it reports none.

        Not every container stores a usable frame rate, and OpenCV
        signals that by returning zero or a non-finite value rather than
        by failing. Those are reported here as ``None`` so a caller has
        to decide what to do about a missing rate, instead of quietly
        computing with a zero.

        Reading this before iterating opens the file briefly to ask; the
        answer is then remembered, and iterating first makes it free.
        """
        if not self._fps_known:
            capture = self._open()
            try:
                self._remember_fps(capture)
            finally:
                capture.release()
        return self._fps

    def __iter__(self) -> Iterator[Frame]:
        """Yield each frame of the file in order.

        Yields:
            Frames as ``uint8`` BGR arrays of shape ``(H, W, 3)``, as
            decoded by OpenCV. See ``Frame`` for the full contract.

        Raises:
            OSError: If the file cannot be opened for decoding. Because
                this method is a generator, that is raised when
                iteration starts rather than when the iterator is
                created — which is the same moment for a ``for`` loop.
        """
        capture = self._open()
        try:
            self._remember_fps(capture)
            while True:
                ok, frame = capture.read()
                # A false flag is how OpenCV reports both the end of the
                # file and a frame it could not decode; neither leaves
                # anything further worth reading.
                if not ok:
                    return
                yield frame
        finally:
            capture.release()

    def _open(self) -> cv2.VideoCapture:
        """Return an opened capture on the file, or raise.

        OpenCV reports an unusable file through ``isOpened`` rather than
        by raising, and the unopened capture still owns resources, so it
        is released before the error is raised.
        """
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Could not open video file: {self._path}")
        return capture

    def _remember_fps(self, capture: cv2.VideoCapture) -> None:
        """Cache the frame rate from an already-open capture."""
        if self._fps_known:
            return

        reported = capture.get(cv2.CAP_PROP_FPS)
        self._fps = float(reported) if isfinite(reported) and reported > 0 else None
        self._fps_known = True
