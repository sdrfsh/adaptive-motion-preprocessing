from collections.abc import Iterable, Iterator

from amprep.background_subtractor import BackgroundSubtractor, KNNBackgroundSubtractor
from amprep.noise_reducer import MedianNoiseReducer, NoiseReducer
from amprep.types import Frame, MotionImage


class AdaptiveMotionPreprocessor:
    """Turns video frames into encoded motion images.

    The public entry point: one class holding the whole pipeline, with
    each stage swappable for your own implementation.

    Every stage argument defaults to ``None``, which is replaced here
    with the packaged implementation of that stage. Resolving the
    default inside ``__init__`` — rather than behind a factory or at the
    first frame — means an assembled preprocessor always holds real
    stage objects, so there is no second code path in which a stage is
    still missing.

    Args:
        noise_reducer: Removes sensor noise before subtraction. Defaults
            to ``MedianNoiseReducer``.
        background_subtractor: Separates moving foreground from the
            learned background. Defaults to ``KNNBackgroundSubtractor``.
    """

    def __init__(
        self,
        noise_reducer: NoiseReducer | None = None,
        background_subtractor: BackgroundSubtractor | None = None,
    ) -> None:
        if noise_reducer is None:
            noise_reducer = MedianNoiseReducer()
        elif not isinstance(noise_reducer, NoiseReducer):
            raise TypeError(
                f"noise_reducer must be a NoiseReducer, got {type(noise_reducer)}"
            )

        if background_subtractor is None:
            background_subtractor = KNNBackgroundSubtractor()
        elif not isinstance(background_subtractor, BackgroundSubtractor):
            raise TypeError(
                "background_subtractor must be a BackgroundSubtractor, "
                f"got {type(background_subtractor)}"
            )

        self._noise_reducer = noise_reducer
        self._background_subtractor = background_subtractor

    def process(self, frames: Iterable[Frame]) -> Iterator[MotionImage]:
        """Turn a stream of frames into encoded motion images.

        The input side of the pipeline, and the whole of it: the package
        does no video I/O. Any iterable of frames is accepted — a list, a
        generator, a custom iterator, a loop around a camera the caller
        opened — because by the time a frame arrives here it is a
        ``uint8`` BGR array, and where it came from is neither recoverable
        nor needed. Capture stays outside the package on purpose: nothing
        in here owns a device handle, so nothing in here can leak one.
        The README carries the few lines that read frames from a file.

        The stream is consumed lazily, one frame at a time, and is never
        materialised. An unbounded source is therefore fine — a live
        camera can be handed over and abandoned whenever the caller
        likes — and because this is a generator it does not touch the
        input at all until it is iterated.

        Frames must arrive in capture order. The stages behind this one
        accumulate state across consecutive frames, so a shuffled stream
        describes motion that never happened.

        Args:
            frames: The frames to process, in capture order. ``Frame``
                gives the contract each one must satisfy; it is enforced
                by ``NoiseReducer.apply`` on every frame, so it is not
                re-checked here.

        Yields:
            One ``MotionImage`` per completed window.
        """
        for frame in frames:
            denoised = self._noise_reducer.apply(frame)
            self._background_subtractor.apply(denoised)
            # Whatever this frame completed goes here, and so far that is
            # nothing: the stages that turn a mask into a window and a
            # window into a MotionImage are not built, so no window ever
            # completes. The empty ``yield from`` is not a placeholder for
            # its own sake — a function is a generator only if a ``yield``
            # appears in its body, and consuming the input lazily depends
            # on this being one.
            yield from ()
