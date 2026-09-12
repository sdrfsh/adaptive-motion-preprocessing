from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import amprep
from amprep import AdaptiveMotionPreprocessor, Frame, NoiseReducer

FRAME_SIZE = (8, 8, 3)
"""Small enough to be cheap, large enough to satisfy the frame contract."""


def _frame(value: int = 0) -> Frame:
    """Build one valid frame: ``uint8``, ``(H, W, 3)``, BGR."""
    return np.full(FRAME_SIZE, value, np.uint8)


class FrameIterator:
    """A hand-written iterator, neither a list nor a generator.

    ``process`` promises to accept any iterable, and a list and a
    generator between them still leave the third case untested: an object
    that implements the iterator protocol itself. This is that case.
    """

    def __init__(self, count: int) -> None:
        self._remaining = list(range(count))

    def __iter__(self) -> "FrameIterator":
        return self

    def __next__(self) -> Frame:
        if not self._remaining:
            raise StopIteration
        return _frame(self._remaining.pop(0))


class CountingReducer(NoiseReducer):
    """A reducer that passes frames through and counts them.

    Nothing reaches the output of the pipeline yet, so the number of
    frames that reached the first stage is what the input-side tests
    measure instead.
    """

    def __init__(self) -> None:
        self.calls = 0

    def _apply(self, frame: Frame) -> Frame:
        self.calls += 1
        return frame


def _frames_reaching_the_pipeline(frames) -> int:
    """Run ``frames`` through a preprocessor and report how many arrived."""
    reducer = CountingReducer()
    list(AdaptiveMotionPreprocessor(noise_reducer=reducer).process(frames))
    return reducer.calls


def test_accepts_a_list():
    """A materialised list of frames is a valid input."""
    assert _frames_reaching_the_pipeline([_frame(1), _frame(2), _frame(3)]) == 3


def test_accepts_a_generator():
    """A generator is a valid input."""
    generator = (_frame(value) for value in (1, 2, 3))

    assert _frames_reaching_the_pipeline(generator) == 3


def test_accepts_a_custom_iterator():
    """An object implementing the iterator protocol is a valid input."""
    assert _frames_reaching_the_pipeline(FrameIterator(3)) == 3


def test_the_three_input_kinds_are_treated_identically():
    """How the frames were delivered makes no difference to the pipeline."""
    as_list = [_frame(value) for value in (1, 2, 3)]
    as_generator = (_frame(value) for value in (1, 2, 3))
    as_iterator = FrameIterator(3)

    counts = [
        _frames_reaching_the_pipeline(as_list),
        _frames_reaching_the_pipeline(as_generator),
        _frames_reaching_the_pipeline(as_iterator),
    ]

    assert counts == [3, 3, 3]


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param([], id="list"),
        pytest.param((_frame() for _ in ()), id="generator"),
        pytest.param(FrameIterator(0), id="iterator"),
        pytest.param(iter([]), id="exhausted-iterator"),
    ],
)
def test_an_empty_input_yields_nothing_and_raises_nothing(empty):
    """An input with no frames in it is not an error, just no output."""
    assert list(AdaptiveMotionPreprocessor().process(empty)) == []


def test_nothing_is_pulled_before_iteration_begins():
    """``process`` is a generator: building it touches the input not at all."""
    pulled: list[int] = []

    def recording_frames() -> Iterator[Frame]:
        for index in range(3):
            pulled.append(index)
            yield _frame(index)

    images = AdaptiveMotionPreprocessor().process(recording_frames())

    assert pulled == []

    list(images)

    assert pulled == [0, 1, 2]


def test_frames_are_consumed_one_at_a_time():
    """The stream is processed as it is pulled, never drained up front.

    A count of frames pulled cannot tell "read all three, then process
    all three" apart from "read one, process one, three times" — both end
    at three. The order the two interleave can, so the pulls and the
    stage calls are recorded into one log and the strict alternation is
    asserted.
    """
    events: list[str] = []

    def recording_frames() -> Iterator[Frame]:
        for index in range(3):
            events.append(f"pull {index}")
            yield _frame(index)

    class LoggingReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            events.append("process")
            return frame

    preprocessor = AdaptiveMotionPreprocessor(noise_reducer=LoggingReducer())
    list(preprocessor.process(recording_frames()))

    assert events == [
        "pull 0",
        "process",
        "pull 1",
        "process",
        "pull 2",
        "process",
    ]


def test_an_endless_stream_is_untouched_until_it_is_iterated():
    """An unbounded input neither hangs nor reads ahead of demand.

    The stronger property — consuming an endless stream part way and
    walking away — cannot be asserted yet: with no encoder behind it
    ``process`` yields nothing, so any attempt to take "the first few"
    outputs would consume the input forever. What is testable now is
    that handing over an endless source costs nothing until it is asked
    for, which is the half that makes a live camera safe to pass in.
    """
    pulled = 0

    def endless_frames() -> Iterator[Frame]:
        nonlocal pulled
        while True:
            pulled += 1
            yield _frame()

    reducer = CountingReducer()
    images = AdaptiveMotionPreprocessor(noise_reducer=reducer).process(endless_frames())
    images.close()

    # Closing before the first pull leaves the source untouched, which is
    # the property that makes handing over a live camera safe.
    assert pulled == 0
    assert reducer.calls == 0


def test_the_package_does_no_video_io():
    """No module under ``src/amprep/`` opens or writes video itself.

    Reading and writing video is the caller's job, so the package must
    not quietly grow a capture of its own. ``cv2`` stays available for
    the filters themselves — this looks only for the I/O entry points.
    """
    package = Path(amprep.__file__).parent

    found = [
        f"{path.name}: {name}"
        for path in sorted(package.rglob("*.py"))
        for name in ("VideoCapture", "VideoWriter")
        if name in path.read_text(encoding="utf-8")
    ]

    assert found == []
