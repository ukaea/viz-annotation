import pytest

from tests import db_definitions
from toktagger.api.core import query_strategy
from toktagger.api.schemas.annotations import TimePointOut
from toktagger.api.schemas.samples import Sample


@pytest.fixture
def samples():
    # Samples, sorted by shot_id, ascending
    sample_ins = [
        db_definitions.SAMPLE_1,
        db_definitions.SAMPLE_2,
        db_definitions.SAMPLE_3,
        db_definitions.SAMPLE_4,
    ]
    samples = [
        Sample(**sample_in.model_dump(), project_id="test", _id="test")
        for sample_in in sample_ins
    ]
    # Attach IDs to these - this would normally be done by the database
    for i in range(len(samples)):
        samples[i].id = f"sample_{i + 1}"

    return samples


@pytest.fixture
def annotations():
    # Annotations, sorted by uncertainty, descending. Only non-validated.
    annotation_ins = [
        db_definitions.ANNOTATION_5,
        db_definitions.ANNOTATION_4,
        db_definitions.ANNOTATION_3,
    ]
    annotations = [
        TimePointOut(
            **annotation_in.model_dump(),
            _id="test",
        )
        for annotation_in in annotation_ins
    ]
    # Attach sample IDs to these which would normally be done by the database
    annotations[0].sample_id = "sample_4"
    annotations[1].sample_id = "sample_1"
    annotations[2].sample_id = "sample_2"
    return annotations


def test_sequential_strategy(samples, annotations):
    strategy = query_strategy.SequentialQueryStrategy(
        samples.copy(), annotations.copy()
    )

    next_sample_id = None
    visited_sample_ids = []

    for i in range(len(samples)):
        next_sample = strategy.get_next_sample(visited_sample_ids)
        next_sample_id = next_sample.id
        assert next_sample == samples[i]
        visited_sample_ids.append(next_sample_id)

    # Should say there are no more samples
    with pytest.raises(RuntimeError, match="No more samples available!"):
        next_sample = strategy.get_next_sample(visited_sample_ids)


def test_random_strategy(samples, annotations):
    strategy = query_strategy.RandomQueryStrategy(
        samples.copy(), annotations.copy(), seed=42
    )

    visited_sample_ids = []
    for i in range(len(samples)):
        next_sample = strategy.get_next_sample(visited_sample_ids)
        visited_sample_ids.append(next_sample.id)

    assert len(visited_sample_ids) == len(samples)
    assert visited_sample_ids != [sample.id for sample in samples]

    with pytest.raises(RuntimeError, match="No more samples available!"):
        next_sample = strategy.get_next_sample(visited_sample_ids)


def test_uncertainty_strategy(samples, annotations):
    strategy = query_strategy.UncertaintyQueryStrategy(
        samples.copy(), annotations.copy()
    )
    expected_order = ["sample_4", "sample_2", "sample_1", "sample_3"]
    visited_sample_ids = []
    for i in range(len(samples)):
        next_sample = strategy.get_next_sample(visited_sample_ids)
        next_sample_id = next_sample.id
        assert next_sample_id == expected_order[i]
        visited_sample_ids.append(next_sample_id)

    # Should say there are no more samples
    with pytest.raises(RuntimeError, match="No more samples available!"):
        next_sample = strategy.get_next_sample(visited_sample_ids)
