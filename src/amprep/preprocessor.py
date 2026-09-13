from collections.abc import Iterable, Iterator

from amprep.adaptive_sampler import DEFAULT_SAMPLE_FRAMES, AdaptiveFrameSampler
from amprep.background_subtractor import BackgroundSubtractor, KNNBackgroundSubtractor
from amprep.frame_window import DEFAULT_WINDOW_FRAMES, FrameWindowCollector
from amprep.motion_history_encoder import MotionHistoryEncoder
from amprep.motion_trigger import DEFAULT_THRESHOLD, MotionTrigger
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

    While motion lasts, one ``MotionImage`` comes out every
    ``window_frames`` frames. A subject that lingers produces a
    continuous stream, so a caller wiring this to a network should expect
    that.

    State carries across ``process()`` calls: splitting one stream into
    chunks gives the same result as one call. Call ``reset()`` when the
    scene changes. Frames must keep one size until then.

    Args:
        noise_reducer: Removes sensor noise before subtraction. Defaults
            to ``MedianNoiseReducer``.
        background_subtractor: Separates moving foreground from the
            learned background. Defaults to ``KNNBackgroundSubtractor``.
        motion_threshold: Share of the frame that must move. Default 1%.
        window_frames: Frames per window. Default 10.
        sample_frames: Frames kept per window. Default 4.
        full_speed: Change rate at which sampling is tightest. ``None``,
            the default, learns it from the scene; pass a number to fix
            it.
        width: Output width, or ``None`` to keep the frame width.
        height: Output height, or ``None`` to keep the frame height.
    """

    def __init__(
        self,
        noise_reducer: NoiseReducer | None = None,
        background_subtractor: BackgroundSubtractor | None = None,
        *,
        motion_threshold: float = DEFAULT_THRESHOLD,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        sample_frames: int = DEFAULT_SAMPLE_FRAMES,
        full_speed: float | None = None,
        width: int | None = None,
        height: int | None = None,
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

        # Each stage checks its own setting.
        self._trigger = MotionTrigger(threshold=motion_threshold)
        self._collector = FrameWindowCollector(window_frames=window_frames)
        self._sampler = AdaptiveFrameSampler(
            sample_frames=sample_frames, full_speed=full_speed
        )
        self._encoder = MotionHistoryEncoder(width=width, height=height)

        # The one check no single stage can do.
        if self._sampler.sample_frames > self._collector.window_frames:
            raise ValueError(
                f"sample_frames ({sample_frames}) cannot exceed "
                f"window_frames ({window_frames})"
            )

        self._frames_seen = 0
        self._frame_shape: tuple[int, ...] | None = None

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

        Raises:
            ValueError: If the frame size changes mid-stream. Call
                ``reset()`` first when it is meant to.
        """
        for frame in frames:
            denoised = self._noise_reducer.apply(frame)  # also validates the frame
            self._check_frame_shape(denoised)
            mask = self._background_subtractor.apply(denoised)

            # The model still learns from these frames; only its verdict
            # is ignored.
            self._frames_seen += 1
            if self._frames_seen <= self._background_subtractor.warmup_frames:
                continue

            active = self._trigger.update(mask)
            window = self._collector.update(frame, mask, active)
            if window is not None:
                yield self._encoder.encode(window, self._sampler.select(window))

    def reset(self) -> None:
        """Start over for a new scene.

        Forgets the background, motion, any half-full window, the learned
        ``full_speed``, the warm-up count and the frame size. Never called
        automatically: a fixed camera fed in chunks should keep what it
        has learned.
        """
        self._background_subtractor.reset()
        self._trigger.reset()
        self._collector.reset()
        self._sampler.reset()
        self._frames_seen = 0
        self._frame_shape = None

    def _check_frame_shape(self, frame: Frame) -> None:
        """Raise if the frame size changed since the stream began."""
        if self._frame_shape is None:
            self._frame_shape = frame.shape
        elif frame.shape != self._frame_shape:
            raise ValueError(
                f"frame size changed from {self._frame_shape} to {frame.shape}; "
                "call reset() before processing frames of a different size"
            )
