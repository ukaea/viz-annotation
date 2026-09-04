import toktagger.api.core.data_loaders as data_loaders
import pytest
from toktagger.api.schemas.projects import Task
from typing import Type
from toktagger.api.schemas.samples import (
    Sample,
    TimeSeriesFileData,
    ShotData,
    ImageFileData,
    ImageArrayFileData,
)
from toktagger.api.schemas.data import (
    TimeSeriesData,
    MultiVariateTimeSeriesData,
    ImageData,
    ImageParams,
    DataParams,
)
import pathlib
import numpy
import xarray
from PIL import Image
import base64
import io


def test_image_file_loader_jpeg():
    img_file = ImageFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("mast_images")),
        type="jpeg",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=img_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.ImageDataLoader()
    image_data = data_loader.get_sample(
        sample, params=ImageParams(name="image", frame=1)
    )
    assert isinstance(image_data, ImageData)
    # Check we got back base64 encoded string
    assert isinstance(image_data.values, str)
    # Convert back to numpy array
    base64_decoded = base64.b64decode(image_data.values)
    image = Image.open(io.BytesIO(base64_decoded))
    assert numpy.array(image).shape == (1079, 881, 3)


def test_image_file_loader_jpeg_raw():
    img_file = ImageFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("mast_images")),
        type="jpeg",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=img_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.ImageDataLoader()
    image_data = data_loader.get_sample(
        sample,
        params=ImageParams(
            name="image",
            frame=1,
            return_raw=True,
        ),
    )
    assert isinstance(image_data, ImageData)
    # Check we got back a list
    assert isinstance(image_data.values, list)

    image_array = numpy.asarray(image_data.values)
    assert image_array.shape == (1079, 881, 3)


def test_image_file_loader_png():
    img_file = ImageFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("mast_images")),
        type="png",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=img_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.ImageDataLoader()
    image_data = data_loader.get_sample(
        sample, params=ImageParams(name="image", frame=1)
    )
    assert isinstance(image_data, ImageData)
    # Check we got back base64 encoded string
    assert isinstance(image_data.values, str)
    # Convert back to numpy array
    base64_decoded = base64.b64decode(image_data.values)
    image = Image.open(io.BytesIO(base64_decoded))
    assert numpy.array(image).shape == (1079, 881, 3)


def test_image_file_loader_png_raw():
    img_file = ImageFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("mast_images")),
        type="png",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=img_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.ImageDataLoader()
    image_data = data_loader.get_sample(
        sample,
        params=ImageParams(
            name="image",
            frame=1,
            return_raw=True,
        ),
    )
    assert isinstance(image_data, ImageData)
    # Check we got back a list
    assert isinstance(image_data.values, list)

    image_array = numpy.asarray(image_data.values)
    assert image_array.shape == (1079, 881, 3)


def test_image_file_loader_missing_frame():
    img_file = ImageFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("mast_images")),
        type="jpeg",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=img_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    with pytest.raises(data_loaders.FrameNotFoundError):
        data_loaders.ImageDataLoader().get_sample(
            sample, params=ImageParams(name="image", frame=999999)
        )


def test_parquet_file_loader():
    parquet_file = TimeSeriesFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("10000.parquet")),
        type="parquet",
        protocol="file",
        signal_names=["Ip", "dalpha"],
    )
    sample = Sample(
        shot_id=10000,
        data=parquet_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.TabularDataLoader()
    data = data_loader.get_sample(sample, params=DataParams(name="identity"))
    assert isinstance(data, MultiVariateTimeSeriesData)

    # Check both columns requested are present
    assert data.values.get("Ip")
    assert data.values.get("dalpha")

    # Check values - dalpha should be between 0 and 50
    dalpha_values = numpy.array(data.values.get("dalpha").values)
    ip_values = numpy.array(data.values.get("Ip").values)
    assert numpy.all((dalpha_values >= 0) & (dalpha_values <= 50))

    # Ip should have 4 values of 1000
    assert len(numpy.where(numpy.isclose(ip_values, 1000))[0]) == 4


def test_uda_loader(uda_test):
    try:
        import pyuda

        pyuda.Client().get("help::help()")
    except Exception:
        pytest.skip("Could not contact UDA server")

    uda_shot = ShotData(protocol="uda", signal_names=["ip", "ANE_DENSITY"])
    sample = Sample(
        shot_id=14892,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.UDADataLoader()
    data = data_loader.get_sample(sample, params=DataParams(name="identity"))
    assert isinstance(data, MultiVariateTimeSeriesData)

    # Check both columns requested are present
    assert data.values.get("ip")
    assert data.values.get("ANE_DENSITY")

    # Check it contains values and times
    ip_values = numpy.array(data.values.get("ip").values)
    times = numpy.array(data.values.get("ip").time)
    assert numpy.all((ip_values >= -50) & (ip_values <= 1000))
    assert numpy.max(times) < 1.5


def test_uda_camera_loader(uda_env_vars):
    try:
        import pyuda

        pyuda.Client().get("help::help()")
    except Exception:
        pytest.skip("Could not contact UDA server")

    camera_name = "rba"
    uda_shot = ShotData(protocol="uda", signal_names=[camera_name])
    sample = Sample(
        shot_id=30421,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.UDACameraDataLoader()
    data = data_loader.get_sample(sample, params=ImageParams(name="image", frame=0))
    assert isinstance(data, ImageData)
    # Check we got back base64 encoded string
    assert isinstance(data.values, str)
    # Convert back to numpy array
    base64_decoded = base64.b64decode(data.values)
    image = Image.open(io.BytesIO(base64_decoded))
    assert numpy.array(image).shape == (912, 768)


def test_uda_camera_loader_bit_depth_scaling(monkeypatch):
    # UDA can return a uint16 array even though the camera's declared bit
    # depth is smaller, e.g. RCO reports depth=8 for shot 54339. Conversion
    # to uint8 should use that declared depth rather than each frame's own
    # min/max, so scaling is consistent across frames.
    camera_name = "rco"
    uda_shot = ShotData(protocol="uda", signal_names=[camera_name])
    sample = Sample(
        shot_id=54339,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    raw_values = numpy.array([[0, 128], [200, 255]], dtype=numpy.uint16)
    fake_dataset = xarray.Dataset(
        {"data": xarray.DataArray(raw_values, dims=["x", "y"], attrs={"depth": 8})}
    )
    monkeypatch.setattr(
        data_loaders.xr, "open_dataset", lambda *args, **kwargs: fake_dataset
    )

    data_loader = data_loaders.UDACameraDataLoader()
    image_data = data_loader.get_sample(
        sample,
        params=ImageParams(name="image", frame=0, return_raw=True),
    )
    assert isinstance(image_data.values, list)
    frame_arr = numpy.asarray(image_data.values)

    # depth=8 means the declared max value is 255, so scaling is a no-op
    # and every value maps to itself.
    assert numpy.array_equal(frame_arr, raw_values.astype(numpy.uint8))


def test_uda_camera_loader_constant_frame_above_bit_depth_range(monkeypatch):
    # Regression test: a frame with a single constant value above 255 used
    # to be rescaled to all zeros (completely black), since the old code
    # computed range = max - min = 0 for a constant frame. Scaling by the
    # camera's declared bit depth avoids this.
    camera_name = "rco"
    uda_shot = ShotData(protocol="uda", signal_names=[camera_name])
    sample = Sample(
        shot_id=54339,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    raw_values = numpy.full((4, 4), 300, dtype=numpy.uint16)
    fake_dataset = xarray.Dataset(
        {"data": xarray.DataArray(raw_values, dims=["x", "y"], attrs={"depth": 8})}
    )
    monkeypatch.setattr(
        data_loaders.xr, "open_dataset", lambda *args, **kwargs: fake_dataset
    )

    data_loader = data_loaders.UDACameraDataLoader()
    image_data = data_loader.get_sample(
        sample, params=ImageParams(name="image", frame=0)
    )
    base64_decoded = base64.b64decode(image_data.values)
    frame_arr = numpy.array(Image.open(io.BytesIO(base64_decoded)))

    # Value is clipped to the top of the 8-bit range (255), not zeroed out.
    assert numpy.all(frame_arr == 255)


def test_uda_camera_loader_no_bit_depth_falls_back_to_clip(monkeypatch):
    # When the bit depth attribute isn't present, fall back to a plain
    # clip to the uint8 range rather than a per-frame min/max rescale.
    camera_name = "rco"
    uda_shot = ShotData(protocol="uda", signal_names=[camera_name])
    sample = Sample(
        shot_id=54339,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    raw_values = numpy.array([[0, 100], [300, 500]], dtype=numpy.uint16)
    fake_dataset = xarray.Dataset(
        {"data": xarray.DataArray(raw_values, dims=["x", "y"])}
    )
    monkeypatch.setattr(
        data_loaders.xr, "open_dataset", lambda *args, **kwargs: fake_dataset
    )

    data_loader = data_loaders.UDACameraDataLoader()
    image_data = data_loader.get_sample(
        sample, params=ImageParams(name="image", frame=0)
    )
    base64_decoded = base64.b64decode(image_data.values)
    frame_arr = numpy.array(Image.open(io.BytesIO(base64_decoded)))

    assert numpy.array_equal(frame_arr, numpy.array([[0, 100], [255, 255]]))


def test_uda_camera_loader_rco_rgb_uint16(monkeypatch):
    # Regression test for https://github.com/ukaea/toktagger/issues/320: RCO
    # returns a leading time dimension plus three-channel uint16 pixel data,
    # e.g. (1, height, width, 3), with a declared depth of 8. The loader should
    # squeeze the time dimension and convert to a (height, width, 3) uint8 PNG.
    camera_name = "rco"
    uda_shot = ShotData(protocol="uda", signal_names=[camera_name])
    sample = Sample(
        shot_id=54339,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    raw_values = numpy.array(
        [
            [
                [[0, 64, 128], [255, 200, 100]],
                [[10, 20, 30], [40, 50, 60]],
            ]
        ],
        dtype=numpy.uint16,
    )
    fake_dataset = xarray.Dataset(
        {
            "data": xarray.DataArray(
                raw_values,
                dims=["time", "height", "width", "channel"],
                attrs={"depth": 8},
            )
        }
    )
    monkeypatch.setattr(
        data_loaders.xr, "open_dataset", lambda *args, **kwargs: fake_dataset
    )

    data_loader = data_loaders.UDACameraDataLoader()
    image_data = data_loader.get_sample(
        sample, params=ImageParams(name="image", frame=0)
    )
    base64_decoded = base64.b64decode(image_data.values)
    frame_arr = numpy.array(Image.open(io.BytesIO(base64_decoded)))

    # depth=8 means the declared max value is 255, so scaling is a no-op and
    # every channel value maps to itself.
    assert frame_arr.shape == (2, 2, 3)
    assert frame_arr.dtype == numpy.uint8
    assert numpy.array_equal(frame_arr, numpy.squeeze(raw_values).astype(numpy.uint8))


def test_uda_camera_loader_missing_frame(monkeypatch):
    uda_shot = ShotData(protocol="uda", signal_names=["rba"])
    sample = Sample(
        shot_id=30421,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    fake_dataset = xarray.Dataset(
        {
            "data": xarray.DataArray(
                numpy.zeros((2, 2), dtype=numpy.uint8),
                dims=["height", "width"],
                attrs={"n_frames": 450},
            )
        }
    )

    def raise_missing_frame(*args, **kwargs):
        if kwargs["frame_number"] == 450:
            raise RuntimeError("Could not open UDA dataset")
        return fake_dataset

    monkeypatch.setattr(data_loaders.xr, "open_dataset", raise_missing_frame)

    with pytest.raises(data_loaders.FrameNotFoundError):
        data_loaders.UDACameraDataLoader().get_sample(
            sample, params=ImageParams(name="image", frame=450)
        )


def test_uda_camera_loader_non_frame_failure_remains_data_loader_error(monkeypatch):
    uda_shot = ShotData(protocol="uda", signal_names=["rba"])
    sample = Sample(
        shot_id=30421,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    fake_dataset = xarray.Dataset(
        {
            "data": xarray.DataArray(
                numpy.zeros((2, 2), dtype=numpy.uint8),
                dims=["height", "width"],
                attrs={"n_frames": 450},
            )
        }
    )

    def raise_uda_failure(*args, **kwargs):
        if kwargs["frame_number"] == 0:
            return fake_dataset
        raise RuntimeError("UDA server unavailable")

    monkeypatch.setattr(data_loaders.xr, "open_dataset", raise_uda_failure)

    with pytest.raises(data_loaders.DataLoaderError) as exc_info:
        data_loaders.UDACameraDataLoader().get_sample(
            sample, params=ImageParams(name="image", frame=1)
        )
    assert not isinstance(exc_info.value, data_loaders.FrameNotFoundError)


def test_image_array_file_loader_raw():
    arr_file = ImageArrayFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("single_arr.npy")),
        type="npy",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=arr_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    image_data = data_loaders.ArrayDataLoader().get_sample(
        sample, params=ImageParams(name="image", frame=0, return_raw=True)
    )

    assert isinstance(image_data.values, list)
    image_array = numpy.asarray(image_data.values)
    assert image_array.shape == (10, 10)


def test_image_array_file_loader_upper_out_of_range_frame():
    arr_file = ImageArrayFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath("single_arr.npy")),
        type="npy",
        protocol="file",
    )
    sample = Sample(
        shot_id=10000,
        data=arr_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )

    with pytest.raises(data_loaders.FrameNotFoundError):
        data_loaders.ArrayDataLoader().get_sample(
            sample, params=ImageParams(name="image", frame=2)
        )


def test_uda_loader_data_doesnt_exist(uda_env_vars):
    try:
        import pyuda

        pyuda.Client().get("help::help()")
    except Exception:
        pytest.skip("Could not contact UDA server")

    uda_shot = ShotData(protocol="uda", signal_names=["doesnt_exist"])
    sample = Sample(
        shot_id=10000,
        data=uda_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.UDADataLoader()

    try:
        data_loader.get_sample(sample, params=DataParams(name="identity"))
        pytest.fail("Expected DataLoaderError not raised")
    except data_loaders.DataLoaderError:
        pass


def test_sal_loader():
    try:
        from sal.client import SALClient

        client = SALClient("https://sal.jetdata.eu")
        client.prompt_for_password = False
        client.authenticate()
    except Exception:
        pytest.skip("Could not contact SAL server")

    sal_shot = ShotData(protocol="sal", signal_names=["ppf/signal/jetppf/magn/ipla"])
    sample = Sample(
        shot_id=87737,
        data=sal_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.SALDataLoader()
    data = data_loader.get_sample(sample, DataParams(name="identity"))

    ip_values = numpy.array(data.values.get("ppf/signal/jetppf/magn/ipla").values)
    assert numpy.min(ip_values) == -1837277.75
    assert numpy.max(ip_values) == 4664.5283203125


def test_fair_mast_dataloader():
    fair_mast_shot = data_loaders.ShotData(
        protocol="fair_mast",
        signal_names=["magnetics/ip"],
    )

    sample = Sample(
        shot_id=30421,
        data=fair_mast_shot,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.FAIRMASTDataLoader()
    # FAIR MAST server seems to be a bit flaky at the moment, causing intermittent CI failures
    # Will catch TimeoutError here, and skip the test if encountered
    try:
        data = data_loader.get_sample(sample, params=DataParams(name="identity"))
    except TimeoutError:
        pytest.skip(
            "Timeout error when connecting to FAIR MAST server - skipping test..."
        )
    assert isinstance(data, MultiVariateTimeSeriesData)

    # Check both columns requested are present
    assert data.values.get("magnetics/ip")

    # Check it contains values and times
    ip_values = numpy.array(data.values.get("magnetics/ip").values)
    assert numpy.min(ip_values) == -40806.55078125
    assert numpy.max(ip_values) == 649008.875


@pytest.mark.parametrize(
    "file_name",
    [
        "multiple_arrs.npz",
        "single_arr.npz",
        "single_arr.npy",
        "rgb_arr.npz",
        "floats_arr.npy",
    ],
)
@pytest.mark.parametrize("frame", [None, 1])
def test_image_array_file_loader(file_name: str, frame: int | None):
    # Data in numpy arrays has shape (2, 10, 10) for (num_frames, x, y)
    arr_file = ImageArrayFileData(
        file_name=str(pathlib.Path(__file__).parents[2].joinpath(file_name)),
        type=file_name.split(".")[1],
        protocol="file",
        signal_name="y" if file_name == "multiple_arrs.npz" else None,
    )
    sample = Sample(
        shot_id=10000,
        data=arr_file,
        _id="test",
        project_id="test",
        validated_annotations=False,
    )
    data_loader = data_loaders.ArrayDataLoader()
    data = data_loader.get_sample(sample, params=ImageParams(name="image", frame=frame))
    assert isinstance(data, ImageData)

    # Decode base64 back to raw PNG bytes
    png_bytes = base64.b64decode(data.values)

    # Load image from bytes
    im = Image.open(io.BytesIO(png_bytes))

    # Convert to NumPy array
    frame_arr = numpy.array(im)

    # Check it is 10x10
    if file_name == "rgb_arr.npz":
        # With RGB channels
        assert frame_arr.shape == (10, 10, 3)
    else:
        # Greyscale
        assert frame_arr.shape == (10, 10)

    # Check it has correct values
    # If frame not specified, first frame
    # Data is constructed from a range reshaped, so...
    if file_name == "rgb_arr.npz":
        # Check second and third channels are ones
        assert numpy.allclose(frame_arr[..., 1].flatten(), numpy.ones(100))
        assert numpy.allclose(frame_arr[..., 2].flatten(), numpy.ones(100))
        frame_arr = frame_arr[..., 0]
    if frame:
        assert numpy.allclose(frame_arr.flatten(), numpy.arange(100, 200))
    else:
        assert numpy.allclose(frame_arr.flatten(), numpy.arange(0, 100))


@pytest.mark.asyncio
async def test_custom_data_loader(api_client):
    # Check that you cannot create a project with 'test' data loader
    in_project = {
        "name": "test_project",
        "task": Task.VIDEO,
        "query_strategy": "random",
        "data_loader": "test",  # <--- invalid
    }
    response = await api_client.post("/projects", json=in_project)
    assert response.status_code == 422
    assert "Invalid data loader 'test'" in response.json()["detail"][0]["msg"]

    # Create a custom data loader
    @data_loaders.LoaderRegistry.register("test")
    class CustomLoader(data_loaders.DataLoader):
        @classmethod
        def sample_data_type(self) -> Type[ShotData]:
            return ShotData

        def get_sample(self, sample: Sample, params: DataParams, **kwargs):
            shot_id = sample.shot_id
            # Return some data, use something from sample to check it is passed in correctly
            return MultiVariateTimeSeriesData(
                values={
                    "test_vals": TimeSeriesData(
                        time=[0, 1], values=[shot_id, shot_id + 1]
                    )
                }
            )

    # Try again to create project with test dataloader, should be valid now
    response = await api_client.post("/projects", json=in_project)
    assert response.status_code == 200
    _project_id = response.json()["_id"]

    # Now create a sample, contents dont matter
    shot_id = 10
    in_sample = [
        {
            "shot_id": shot_id,
            "data": {
                "protocol": "uda",
                "signal_names": ["Ip", "dalpha"],
            },
        },
    ]
    response = await api_client.post(f"/projects/{_project_id}/samples", json=in_sample)
    assert response.status_code == 200
    _sample_id = response.json()[0]

    # And get data from that sample, should use new data loader
    response = await api_client.post(
        f"/projects/{_project_id}/samples/{_sample_id}/data"
    )
    assert response.status_code == 200
    assert response.json()["values"]["test_vals"]["time"] == [0, 1]
    assert response.json()["values"]["test_vals"]["values"] == [shot_id, shot_id + 1]


@pytest.mark.parametrize(
    "name,data_loader,sample_data_model",
    [
        ("image", data_loaders.ImageDataLoader, ImageFileData),
        ("tabular", data_loaders.TabularDataLoader, TimeSeriesFileData),
        ("uda", data_loaders.UDADataLoader, ShotData),
        ("sal", data_loaders.SALDataLoader, ShotData),
        ("fair_mast", data_loaders.FAIRMASTDataLoader, ShotData),
    ],
)
def test_loader_registry(name, data_loader, sample_data_model):
    # Check the registry returns the correct class
    assert data_loaders.LoaderRegistry.get(name) == data_loader

    # Check the registry returns the correct sample data schema
    assert (
        data_loaders.LoaderRegistry.get_data_schema(name)
        == sample_data_model.model_json_schema()
    )
