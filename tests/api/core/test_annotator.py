import toktagger.api.core.annotators as annotators
import numpy
from scipy.datasets import electrocardiogram
from toktagger.api.schemas.annotations import SpectrogramMask
from toktagger.api.schemas.data import TimeSeriesData, MultiVariateTimeSeriesData
import numpy as np
from toktagger.api.schemas.annotators import (
    PeakDetectionParams,
    OutlierDetectionParams,
    JumpDetectionParams,
    SpectrogramThresholdParams,
    ChangePointDetectionParams,
)


def test_find_peaks():
    data = electrocardiogram()[2000:4000]
    ts_data = TimeSeriesData(time=np.arange(len(data)), values=data)
    mv_data = MultiVariateTimeSeriesData(values={"Ip": ts_data})

    params = PeakDetectionParams(signal_name="Ip", prominence=1, distance=1)
    annotator = annotators.PeakDetectionAnnotator(params)
    time_regions = annotator.predict(mv_data)

    assert len(time_regions) == 12
    for region in time_regions:
        assert region.time_min < region.time_max
        assert region.time_min >= 0
        assert region.time_max <= len(data)


def test_find_peaks_params():
    data = electrocardiogram()[2000:4000]
    ts_data = TimeSeriesData(time=numpy.arange(len(data)), values=data)
    mv_data = MultiVariateTimeSeriesData(values={"Ip": ts_data})

    small_params = PeakDetectionParams(signal_name="Ip", prominence=1, distance=1)
    # Create an annotator with the above params - this should detect all 7 peaks
    full_annotator = annotators.PeakDetectionAnnotator(small_params)
    time_regions = full_annotator.predict(mv_data)
    assert len(time_regions) == 12

    # Then create an annotator with more limiting params
    params = PeakDetectionParams(
        signal_name="Ip", prominence=1, distance=100, time_min=10, time_max=500
    )
    annotator = annotators.PeakDetectionAnnotator(params)
    time_regions = annotator.predict(mv_data)

    assert len(time_regions) == 3

    peak = time_regions[0]
    assert numpy.isclose(peak.time_min, 51, rtol=1e-1)
    assert numpy.isclose(peak.time_max, 85, rtol=1e-1)

    peak = time_regions[1]
    assert numpy.isclose(peak.time_min, 228, rtol=1e-1)
    assert numpy.isclose(peak.time_max, 273, rtol=1e-1)


def make_mv_data(values, signal_name="Ip"):
    ts_data = TimeSeriesData(time=np.arange(len(values)), values=values)
    return MultiVariateTimeSeriesData(values={signal_name: ts_data})


def test_outlier_detection_annotator_mad():
    # Create a signal with a clear outlier
    values = np.ones(100)
    values[50] = 100  # outlier
    mv_data = make_mv_data(values)

    params = OutlierDetectionParams(signal_name="Ip", threshold=5, method="mad")
    annotator = annotators.OutlierDetectionAnnotator(params)
    regions = annotator.predict(mv_data)

    assert len(regions) == 1
    assert regions[0].time_min == 50
    assert regions[0].time_max == 51


def test_outlier_detection_annotator_isoforest():
    # Create a signal with a clear outlier
    values = np.ones(100)
    values[50] = 100  # outlier
    mv_data = make_mv_data(values)

    params = OutlierDetectionParams(
        signal_name="Ip", contamination=0.1, method="isoforest"
    )
    annotator = annotators.OutlierDetectionAnnotator(params)
    regions = annotator.predict(mv_data)

    assert len(regions) == 1
    assert regions[0].time_min == 50
    assert regions[0].time_max == 51


def test_outlier_detection_annotator_no_outlier():
    values = np.ones(100)
    params = OutlierDetectionParams(signal_name="Ip", threshold=5, method="mad")
    mv_data = make_mv_data(values)
    annotator = annotators.OutlierDetectionAnnotator(params)
    regions = annotator.predict(mv_data)
    assert len(regions) == 0


def test_jump_annotator_detects_jump():
    # Create a signal with a jump at index 50
    values = np.concatenate([np.ones(50), np.ones(50) * 10])
    params = JumpDetectionParams(
        signal_name="Ip", threshold=1, min_distance=1, smoothing=0, num_points=100
    )
    mv_data = make_mv_data(values)
    annotator = annotators.JumpDetectionAnnotator(params)
    regions = annotator.predict(mv_data)
    assert len(regions) == 1
    assert 49 <= regions[0].time_min <= 51


def test_jump_annotator_no_jump():
    values = np.ones(100)
    params = JumpDetectionParams(
        signal_name="Ip", threshold=1, min_distance=1, smoothing=0, num_points=100
    )
    mv_data = make_mv_data(values)
    annotator = annotators.JumpDetectionAnnotator(params)
    regions = annotator.predict(mv_data)
    assert len(regions) == 0


def test_changepoint_annotator_detects_change():
    # Create a signal with a change in mean at index 30 and 70
    values = np.concatenate([np.ones(30), np.ones(40) * 5, np.ones(30) * 2])
    params = ChangePointDetectionParams(
        signal_name="Ip", threshold=1, method="pelt", num_points=100, penalty=10
    )
    mv_data = make_mv_data(values)
    annotator = annotators.ChangePointDetectionAnnotator(params)
    regions = annotator.predict(mv_data)
    # Should detect two changepoints
    assert len(regions) == 3
    assert any(0 <= r.time_min <= 2 for r in regions)
    assert any(28 <= r.time_min <= 32 for r in regions)
    assert any(68 <= r.time_min <= 72 for r in regions)


def test_changepoint_annotator_detects_change_hmm():
    # Create a signal with a change in mean at index 30 and 70
    values = np.concatenate([np.ones(30), np.ones(40) * 5, np.ones(30) * 2])
    params = ChangePointDetectionParams(
        signal_name="Ip", threshold=1, method="hmm", num_points=100, num_components=3
    )
    mv_data = make_mv_data(values)
    annotator = annotators.ChangePointDetectionAnnotator(params)
    regions = annotator.predict(mv_data)
    # Should detect two changepoints
    assert len(regions) == 3
    assert any(0 <= r.time_min <= 2 for r in regions)
    assert any(28 <= r.time_min <= 32 for r in regions)
    assert any(68 <= r.time_min <= 72 for r in regions)


def test_changepoint_annotator_no_change():
    values = np.ones(100)
    params = ChangePointDetectionParams(
        signal_name="Ip", threshold=1, method="pelt", num_points=100, penalty=10
    )
    mv_data = make_mv_data(values)
    annotator = annotators.ChangePointDetectionAnnotator(params)
    regions = annotator.predict(mv_data)

    assert len(regions) == 1
    assert np.isclose(regions[0].time_min, 0)
    assert np.isclose(regions[0].time_max, 99)


def test_spectrogram_threshold():
    data = electrocardiogram()[2000:4000]
    ts_data = TimeSeriesData(time=numpy.arange(len(data)), values=data)
    mv_data = MultiVariateTimeSeriesData(values={"Ip": ts_data})

    params = SpectrogramThresholdParams(signal_name="Ip", percentile=95)
    annotator = annotators.SpectrogramThresholdAnnotator(params)
    result = annotator.predict(mv_data)

    assert isinstance(result, SpectrogramMask)
    mask = numpy.array(result.values)
    assert mask.shape == (129, 17)
    assert mask.min() == 0
    assert mask.max() == 1
