from amprep.background_subtractor import BackgroundSubtractor, KNNBackgroundSubtractor
from amprep.noise_reducer import MedianNoiseReducer, NoiseReducer


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
