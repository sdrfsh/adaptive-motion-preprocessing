import numpy as np
import pytest

from amprep import (
    BackgroundSubtractor,
    Frame,
    KNNBackgroundSubtractor,
    MedianNoiseReducer,
    NoiseReducer,
)
from amprep.preprocessor import AdaptiveMotionPreprocessor


def _frame(height: int = 8, width: int = 8) -> Frame:
    """Return a valid uint8 BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _mask(frame: Frame) -> Frame:
    """Return a valid all-background mask for ``frame``."""
    return np.zeros(frame.shape[:2], dtype=np.uint8)


@pytest.mark.parametrize(
    ("abc", "hooks"),
    [
        (NoiseReducer, {"_apply"}),
        (BackgroundSubtractor, {"_apply", "reset"}),
    ],
)
def test_abstract_hooks_are_exactly_the_documented_ones(abc, hooks):
    """The hooks a subclass owes are the ones the class docstring names.

    Pinning the set catches a hook quietly gaining or losing its
    ``@abstractmethod``, which would change what third-party subclasses
    are required to provide without any test failing on its own.
    """
    assert abc.__abstractmethods__ == frozenset(hooks)


@pytest.mark.parametrize("abc", [NoiseReducer, BackgroundSubtractor])
def test_abc_cannot_be_instantiated_directly(abc):
    """Neither ABC is usable as a stage on its own."""
    with pytest.raises(TypeError):
        abc()


def test_overriding_public_apply_does_not_satisfy_noise_reducer():
    """Replacing ``apply`` is not implementing the hook.

    The inviting mistake: both ABCs tell subclasses not to override
    ``apply``, so a subclass that does it anyway has skipped the
    validation and left ``_apply`` abstract. It must not instantiate.
    """

    class SneakyReducer(NoiseReducer):
        def apply(self, frame: Frame) -> Frame:
            return frame

    with pytest.raises(TypeError, match="_apply"):
        SneakyReducer()


def test_overriding_public_apply_does_not_satisfy_background_subtractor():
    """The same mistake is caught on the subtractor side."""

    class SneakySubtractor(BackgroundSubtractor):
        def apply(self, frame: Frame) -> Frame:
            return _mask(frame)

        def reset(self) -> None:
            pass

    with pytest.raises(TypeError, match="_apply"):
        SneakySubtractor()


def test_noise_reducer_subclass_completes_in_one_step():
    """Implementing the single hook is all a reducer owes."""

    class WholeReducer(NoiseReducer):
        def _apply(self, frame: Frame) -> Frame:
            return frame

    assert WholeReducer().apply(_frame()).shape == (8, 8, 3)


def test_partly_implemented_base_stays_abstract_until_a_child_finishes_it():
    """A half-done intermediate class is rejected; its completed child is not.

    Subclassing in two steps — a base that fixes ``reset`` and children
    that vary ``_apply`` — is a reasonable thing for a user to do, so the
    enforcement has to survive a hierarchy rather than only a direct
    subclass.
    """

    class ResetOnlyBase(BackgroundSubtractor):
        def reset(self) -> None:
            pass

    with pytest.raises(TypeError, match="_apply"):
        ResetOnlyBase()

    class CompletedSubtractor(ResetOnlyBase):
        def _apply(self, frame: Frame) -> Frame:
            return _mask(frame)

    assert CompletedSubtractor().apply(_frame()).shape == (8, 8)


@pytest.mark.parametrize(
    ("default", "abc"),
    [
        (MedianNoiseReducer, NoiseReducer),
        (KNNBackgroundSubtractor, BackgroundSubtractor),
    ],
)
def test_packaged_defaults_are_concrete_implementations_of_their_abc(default, abc):
    """Each shipped default leaves no hook unimplemented.

    The defaults are held to the contract they ask of everyone else: if
    one ever went abstract, it would fail at construction rather than
    when a caller first hands it a frame.
    """
    instance = default()

    assert isinstance(instance, abc)
    assert not type(instance).__abstractmethods__


class RecordingReducer(NoiseReducer):
    """A minimal user-supplied reducer that records the frames it saw."""

    def __init__(self) -> None:
        self.seen: list[Frame] = []

    def _apply(self, frame: Frame) -> Frame:
        self.seen.append(frame)
        return frame


class RecordingSubtractor(BackgroundSubtractor):
    """A minimal user-supplied subtractor that records the frames it saw."""

    def __init__(self) -> None:
        self.seen: list[Frame] = []
        self.resets = 0

    def _apply(self, frame: Frame) -> Frame:
        self.seen.append(frame)
        return _mask(frame)

    def reset(self) -> None:
        self.resets += 1


def test_omitting_both_kwargs_installs_the_packaged_defaults():
    """An unconfigured preprocessor is assembled from the shipped stages."""
    preprocessor = AdaptiveMotionPreprocessor()

    assert isinstance(preprocessor._noise_reducer, MedianNoiseReducer)
    assert isinstance(preprocessor._background_subtractor, KNNBackgroundSubtractor)


def test_each_preprocessor_gets_its_own_default_stages():
    """Defaults are built per instance, not shared between preprocessors.

    The subtractor carries a background model learned from whatever
    frames it has seen, so two preprocessors sharing one instance would
    silently contaminate each other's output.
    """
    first = AdaptiveMotionPreprocessor()
    second = AdaptiveMotionPreprocessor()

    assert first._noise_reducer is not second._noise_reducer
    assert first._background_subtractor is not second._background_subtractor


def test_custom_noise_reducer_replaces_the_default():
    """The caller's reducer instance is the one installed."""
    reducer = RecordingReducer()

    preprocessor = AdaptiveMotionPreprocessor(noise_reducer=reducer)

    assert preprocessor._noise_reducer is reducer
    assert not isinstance(preprocessor._noise_reducer, MedianNoiseReducer)


def test_custom_background_subtractor_replaces_the_default():
    """The caller's subtractor instance is the one installed."""
    subtractor = RecordingSubtractor()

    preprocessor = AdaptiveMotionPreprocessor(background_subtractor=subtractor)

    assert preprocessor._background_subtractor is subtractor
    assert not isinstance(preprocessor._background_subtractor, KNNBackgroundSubtractor)


def test_overriding_one_stage_leaves_the_other_on_its_default():
    """Swapping a stage swaps that stage and nothing else."""
    reducer = RecordingReducer()

    preprocessor = AdaptiveMotionPreprocessor(noise_reducer=reducer)

    assert preprocessor._noise_reducer is reducer
    assert isinstance(preprocessor._background_subtractor, KNNBackgroundSubtractor)


def test_both_stages_can_be_overridden_at_once():
    """Neither default survives when the caller supplies both stages."""
    reducer = RecordingReducer()
    subtractor = RecordingSubtractor()

    preprocessor = AdaptiveMotionPreprocessor(
        noise_reducer=reducer, background_subtractor=subtractor
    )

    assert preprocessor._noise_reducer is reducer
    assert preprocessor._background_subtractor is subtractor


def test_a_packaged_default_may_be_passed_explicitly_with_custom_tuning():
    """Supplying a retuned default is an override like any other."""
    reducer = MedianNoiseReducer(ksize=3)

    preprocessor = AdaptiveMotionPreprocessor(noise_reducer=reducer)

    assert preprocessor._noise_reducer is reducer
    assert preprocessor._noise_reducer._ksize == 3


def test_installed_custom_stages_are_the_objects_that_run():
    """Frames reach the caller's class, and its own ``_apply`` handles them.

    The end-to-end proof arrives with ``process``; what is checked here
    is that the instance the preprocessor holds is a working stage whose
    hook is the one the base class delegates to.
    """
    reducer = RecordingReducer()
    subtractor = RecordingSubtractor()
    preprocessor = AdaptiveMotionPreprocessor(
        noise_reducer=reducer, background_subtractor=subtractor
    )
    frame = _frame()

    preprocessor._noise_reducer.apply(frame)
    preprocessor._background_subtractor.apply(frame)

    assert [f is frame for f in reducer.seen] == [True]
    assert [f is frame for f in subtractor.seen] == [True]


@pytest.mark.parametrize("value", ["a reducer", object(), MedianNoiseReducer])
def test_a_noise_reducer_that_is_not_one_is_rejected(value):
    """A stage that does not implement the ABC fails at construction.

    Including the class itself rather than an instance: forgetting the
    parentheses would otherwise be found much later, by an attribute
    error deep in a frame loop.
    """
    with pytest.raises(TypeError, match="noise_reducer must be a NoiseReducer"):
        AdaptiveMotionPreprocessor(noise_reducer=value)


@pytest.mark.parametrize("value", ["a subtractor", object(), KNNBackgroundSubtractor])
def test_a_background_subtractor_that_is_not_one_is_rejected(value):
    """The same guard covers the subtractor slot."""
    with pytest.raises(
        TypeError, match="background_subtractor must be a BackgroundSubtractor"
    ):
        AdaptiveMotionPreprocessor(background_subtractor=value)
