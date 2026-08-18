import logging
import random
from abc import ABC

from toktagger.api.schemas.annotations import Annotation
from toktagger.api.schemas.projects import QueryStrategyType
from toktagger.api.schemas.samples import Sample

logger = logging.getLogger(__name__)


class QueryStrategy(ABC):
    """Base class for query strategies"""

    def __init__(
        self,
        samples: list[Sample],
        annotations: list[Annotation] | None = None,
    ):
        self.samples = samples
        self.annotations = annotations if annotations is not None else []

    def _get_matching_sample(self, current_sample_id: str) -> int:
        index = next(
            (
                i
                for i, sample in enumerate(self.samples)
                if sample.id == current_sample_id
            ),
            None,
        )
        if index is None:
            raise RuntimeError("Current sample ID not found in the list of samples.")
        return index

    def get_next_sample(self, visited_sample_ids: list[str]) -> Sample:
        """Get the next sample"""
        next_sample = next(
            (sample for sample in self.samples if sample.id not in visited_sample_ids),
            None,
        )
        if not next_sample:
            raise RuntimeError("No more samples available!")
        return next_sample


class SequentialQueryStrategy(QueryStrategy):
    """Sequential query strategy

    Chooses the next sample from the ordered list of samples
    """

    def get_next_sample(self, visited_sample_ids: list[str]) -> Sample:
        """Get the next sample, by finding the index of the last visited sample and doing +1"""
        if not visited_sample_ids:
            # Pressed Jump to Shot, should recommend next non-validated sample
            next_sample = next(
                (sample for sample in self.samples if not sample.validated_annotations),
                None,
            )
            # If all are validated, just return first sample in table
            if not next_sample:
                return self.samples[0]
            return next_sample

        # Find index of last visited sample in samples list
        index = self._get_matching_sample(visited_sample_ids[-1])
        next_index = index + 1
        if next_index == len(self.samples):
            raise RuntimeError("No more samples available!")

        return self.samples[next_index]


class RandomQueryStrategy(QueryStrategy):
    """Random query strategy

    Randomly chooses a sample as the next one to show to the user
    """

    def _random_shuffle_samples(self, samples: list[Sample], seed: int):
        rng = random.Random(seed)
        rng.shuffle(samples)
        return samples

    def __init__(
        self,
        samples: list[Sample],
        annotations: list[Annotation] | None = None,
        seed: int = 42,
    ):
        # Get list of non-validated samples and shuffle
        samples_nonvalidated = [
            sample for sample in samples if not sample.validated_annotations
        ]
        self.samples = self._random_shuffle_samples(samples_nonvalidated, seed)

        # Then get validated samples and shuffle
        samples_validated = [
            sample for sample in samples if sample.validated_annotations
        ]
        self.samples += self._random_shuffle_samples(samples_validated, seed)


class UncertaintyQueryStrategy(RandomQueryStrategy):
    """Uncertainty-based query strategy

    Chooses the next sample based on uncertainty scores from existing annotations.
    If no annotations exist, falls back to random sampling.
    """

    def __init__(
        self,
        samples: list[Sample],
        annotations: list[Annotation] | None = None,
        seed: int = 42,
    ):
        self.annotations = [ann for ann in annotations if not ann.validated]
        samples = self._random_shuffle_samples(samples=samples, seed=seed)

        self.annotations = sorted(
            self.annotations, key=lambda ann: ann.uncertainty, reverse=True
        )
        sample_ids = [annotation.sample_id for annotation in self.annotations]
        sample_ids = list(dict.fromkeys(sample_ids))
        # List of samples which have unvalidated annotations
        samples_nonvalidated = [
            sample for sample in samples if not sample.validated_annotations
        ]
        # Sort in order of annotation uncertainty, samples with no annotations go last (randomised)
        self.samples = sorted(
            samples_nonvalidated,
            key=lambda sample: (
                sample_ids.index(sample.id) if sample.id in sample_ids else float("inf")
            ),
        )
        # Then add in samples which *are* validated at the end
        self.samples += [sample for sample in samples if sample.validated_annotations]


QUERY_STRATEGIES = {
    QueryStrategyType.RANDOM: RandomQueryStrategy,
    QueryStrategyType.SEQUENTIAL: SequentialQueryStrategy,
    QueryStrategyType.UNCERTAINTY: UncertaintyQueryStrategy,
}
