import base64
import inspect
import io
import os
import pathlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Type

import numpy as np
import pandas as pd
import pydantic
import xarray as xr
from PIL import Image

from toktagger.api.schemas.data import (
    Data,
    DataParamTypes,
    ImageParams,
    DataParams,
    ImageData,
    MultiVariateTimeSeriesData,
    TimeSeriesData,
)
from toktagger.api.schemas.samples import (
    FileData,
    Sample,
    ShotData,
    TimeSeriesFileData,
    ImageFileData,
    ImageArrayFileData,
    DataTypes,
)
import toktagger.api.config as config

# Set up UDA environment variables with defaults if not already set. This is required for
# the pyuda client to work correctly outside of Freia.
os.environ["UDA_HOST"] = os.environ.get("UDA_HOST", config.settings.uda.host)
os.environ["UDA_META_PLUGINNAME"] = os.environ.get(
    "UDA_META_PLUGINNAME", config.settings.uda.meta_pluginname
)
os.environ["UDA_METANEW_PLUGINNAME"] = os.environ.get(
    "UDA_METANEW_PLUGINNAME", config.settings.uda.metanew_pluginname
)

# Setup SAL environment variables with defaults if not already set. This is required for
# the SAL client to work correctly.
os.environ["SAL_HOST"] = os.environ.get("SAL_HOST", config.settings.sal.host)


class DataLoaderError(Exception):
    """Custom exception for data loader errors."""


class DataLoader(ABC):
    @classmethod
    @abstractmethod
    def sample_data_type(
        cls,
    ) -> Type[DataTypes]:
        # Return whatever type the data loader expects to be passed in as sample_data when getting the sample
        pass

    @abstractmethod
    def get_sample(
        self,
        sample: Sample,
        params: DataParamTypes = DataParams(),
        **kwargs,
    ) -> Data:
        pass


class LoaderRegistry:
    _registry: dict[str, DataLoader] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(loader_class: DataLoader):
            if not issubclass(loader_class, DataLoader):
                raise ValueError(
                    f"Loader '{name}' does not inherit from DataLoader base class."
                )
            if (sample_data_type := loader_class.sample_data_type()) not in (
                ShotData,
                FileData,
                ImageFileData,
                ImageArrayFileData,
                TimeSeriesFileData,
            ):
                raise ValueError(
                    f"Loader '{name}' must expect a supported data type as an input, but got '{sample_data_type}'."
                )
            cls._registry[name] = loader_class
            return loader_class

        return decorator

    @classmethod
    def get(cls, name: str):
        loader_class: DataLoader | None = cls._registry.get(name)
        if not loader_class:
            raise ValueError(f"No DataLoader class called '{name}' found in registry!")
        return loader_class

    @classmethod
    def names(cls):
        return list(cls._registry.keys())

    @classmethod
    def get_data_schema(cls, name: str):
        loader_class: DataLoader | None = cls._registry.get(name)
        if not loader_class:
            raise ValueError(f"No DataLoader class called '{name}' found in registry!")
        return loader_class.sample_data_type().model_json_schema()


@LoaderRegistry.register("image")
class ImageDataLoader(DataLoader):
    """DataLoader for retrieving data using a folder of image files"""

    @classmethod
    def sample_data_type(cls) -> Type[ImageFileData]:
        return ImageFileData

    @pydantic.validate_call
    def get_sample(self, sample: Sample, params: ImageParams, **kwargs) -> ImageData:
        if not isinstance(sample.data, ImageFileData):
            raise TypeError(
                f"Expected sample data of type 'ImageFileData' but got '{type(sample.data)}'"
            )

        sample_data: ImageFileData = sample.data

        # Find directory of images
        dir_path = pathlib.Path(sample_data.file_name)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(
                f"Could not find directory at '{dir_path}', relative to {pathlib.Path().cwd()} - {list(pathlib.Path().cwd().iterdir())}"
            )
        # Open image which represents frame selected
        if params.name != "image":
            raise ValueError("Must provide image data parameters!")
        elif params.frame is None:
            files = sorted(dir_path.iterdir())
            if len(files) == 0:
                raise FileNotFoundError("No files exist in specified directory!")
            file_path = files[0]
        else:
            file_path = dir_path.joinpath(f"{params.frame}.{sample_data.type}")
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find image file at '{file_path}', relative to {pathlib.Path().cwd()}"
            )
        im = Image.open(file_path)
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageData(
            frame=file_path.name.split(".")[0],
            values=base64.b64encode(buffer.getvalue()).decode(),
        )


@LoaderRegistry.register("image-array")
class ArrayDataLoader(DataLoader):
    """DataLoader for retrieving data using Numpy array files."""

    @classmethod
    def sample_data_type(cls) -> Type[ImageArrayFileData]:
        return ImageArrayFileData

    @pydantic.validate_call
    def get_sample(self, sample: Sample, params: ImageParams, **kwargs) -> ImageData:
        if not isinstance(sample.data, ImageArrayFileData):
            raise TypeError(
                f"Expected sample data of type 'ImageArrayFileData' but got '{type(sample.data)}'"
            )

        sample_data: ImageArrayFileData = sample.data

        # Find file
        file_path = pathlib.Path(sample_data.file_name)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find directory at '{file_path}', relative to {pathlib.Path().cwd()} - {list(pathlib.Path().cwd().iterdir())}"
            )
        # Load array
        if sample_data.type == "npz":
            with np.load(file_path, allow_pickle=False) as img_data:
                arr_names = img_data.files
                if sample_data.signal_name is not None:
                    if sample_data.signal_name in arr_names:
                        arr: np.ndarray = img_data[sample_data.signal_name]
                    else:
                        raise DataLoaderError(
                            f"Signal name {sample_data.signal_name} not found in array file! Available keys are: {arr_names}"
                        )
                else:
                    if len(arr_names) == 1:
                        arr: np.ndarray = img_data[arr_names[0]]
                    else:
                        raise DataLoaderError(
                            f"Signal name not provided, but multiple options found in array file! Available keys are: {arr_names}"
                        )
        else:
            arr: np.ndarray = np.load(file_path, allow_pickle=False)

        # Check array is 3D, frame x height x width, or 4D, frame x height x width x rgb
        if len(arr.shape) != 3 and not (len(arr.shape) == 4 and arr.shape[-1] == 3):
            raise DataLoaderError(
                f"""Expected array to have three dimensions representing (frame, height, width),
                or 4 dimensions representing (frame, height, width, RGB),
                but found {len(arr.shape)} dimensions!"""
            )

        # If any values > 255, scale arr to be 1-255
        if np.any(arr > 255):
            val_range = arr.max() - arr.min()
            arr = arr - arr.min()
            # Avoid divide by zero in case where image is uniform
            if val_range:
                arr = arr / val_range
            arr *= 255
        arr = arr.astype(np.uint8)

        if params.name != "image":
            raise DataLoaderError("Must provide image data parameters!")

        frame = params.frame if params.frame is not None else 0

        if frame < 0 or frame >= arr.shape[0]:
            raise DataLoaderError(
                f"Frame {frame} unavailable! Available frame range is 0 to {arr.shape[0] - 1}."
            )

        frame_arr = arr[frame, ...]

        im = Image.fromarray(frame_arr)
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
        buffer.seek(0)

        return ImageData(
            frame=frame,
            values=base64.b64encode(buffer.getvalue()).decode(),
        )


@LoaderRegistry.register("tabular")
class TabularDataLoader(DataLoader):
    """DataLoader for retrieving data from a tabular file format (e.g., CSV, Parquet)"""

    @classmethod
    def sample_data_type(cls) -> Type[TimeSeriesFileData]:
        return TimeSeriesFileData

    @pydantic.validate_call
    def get_sample(
        self,
        sample: Sample,
        params: DataParams,
        time_min: Optional[float] = None,
        time_max: Optional[float] = None,
        min_time_step: Optional[float] = None,
        **kwargs,
    ) -> MultiVariateTimeSeriesData:
        if not isinstance(sample.data, TimeSeriesFileData):
            raise TypeError(
                f"Expected sample data of type 'TimeSeriesFileData' but got '{type(sample.data)}'"
            )

        if not pathlib.Path(sample.data.file_name).exists():
            raise FileNotFoundError(
                f"Could not find file at '{sample.data.file_name}', relative to {pathlib.Path().cwd()}"
            )

        item: TimeSeriesFileData = sample.data

        if item.file_name.endswith(".csv"):
            df = pd.read_csv(item.file_name, usecols=item.signal_names)
        elif item.file_name.endswith(".tsv"):
            df = pd.read_csv(item.file_name, sep="\t", usecols=item.signal_names)
        elif item.file_name.endswith(".parquet"):
            df = pd.read_parquet(item.file_name, columns=item.signal_names)
        elif item.file_name.endswith(".json"):
            df = pd.read_json(item.file_name)
            df = df[item.signal_names]
        elif item.file_name.endswith(".xlsx"):
            df = pd.read_excel(item.file_name, usecols=item.signal_names)
        elif item.file_name.endswith(".feather"):
            df = pd.read_feather(item.file_name, columns=item.signal_names)
        else:
            raise ValueError(
                "Unsupported file format {}".format(Path(item.file_name).suffix)
            )

        df = df.fillna(0)

        df.index = pd.to_timedelta(df.index, unit="s")
        mean_diff = df.index.to_series().diff().dropna().mean().total_seconds()

        if min_time_step is not None and mean_diff < min_time_step:
            df = df.resample(rule=f"{min_time_step}s").interpolate(method="linear")

        if time_min is not None:
            df = df[df.index >= pd.to_timedelta(time_min, unit="s")]

        if time_max is not None:
            df = df[df.index <= pd.to_timedelta(time_max, unit="s")]

        data = df.to_dict(orient="list")
        time = df.index.total_seconds().to_list()

        results = {}
        for key, value in data.items():
            results[key] = TimeSeriesData(time=time, values=value)

        return MultiVariateTimeSeriesData(values=results)


@LoaderRegistry.register("uda")
class UDADataLoader(DataLoader):
    """DataLoader for retrieving data using the UDA access layer"""

    @classmethod
    def sample_data_type(self) -> Type[ShotData]:
        return ShotData

    def get_sample(
        self,
        sample: Sample,
        params: DataParams,
        time_min: Optional[float] = None,
        time_max: Optional[float] = None,
        min_time_step: Optional[float] = None,
        **kwargs,
    ) -> MultiVariateTimeSeriesData:
        if not isinstance(sample.data, ShotData):
            raise TypeError(
                f"Expected sample data of type 'ShotData' but got '{type(sample.data)}'"
            )
        if not params.name == "identity":
            raise TypeError(f"Expected blank parameters, but got '{params.name}'")

        sample_data: ShotData = sample.data

        results = {}
        for name in sample_data.signal_names:
            try:
                signal = xr.open_dataset(f"uda://{name}:{sample.shot_id}", engine="uda")
                data = signal["data"].values
                time = signal["time"].values

                if time_min is not None:
                    mask = time >= time_min
                    time = time[mask]
                    data = data[mask]

                if time_max is not None:
                    mask = time <= time_max
                    time = time[mask]
                    data = data[mask]

                if (
                    min_time_step is not None
                    and len(time) > 1
                    and np.diff(time).mean() < min_time_step
                ):
                    time_base = np.arange(time[0], time[-1], min_time_step)
                    data = np.interp(time_base, time, data)
                    time = time_base

                item = TimeSeriesData(time=time, values=data)
                results[name] = item
            except Exception:
                results[name] = None

        if all(values is None for values in results.values()):
            raise DataLoaderError(
                f"Could not load any signals for shot ID '{sample.shot_id}'. Check UDA connectivity and signal names."
            )

        return MultiVariateTimeSeriesData(values=results)


@LoaderRegistry.register("uda_camera")
class UDACameraDataLoader(DataLoader):
    """DataLoader for retrieving camera image data using the UDA access layer"""

    @classmethod
    def sample_data_type(self) -> Type[ShotData]:
        return ShotData

    def get_sample(
        self,
        sample: Sample,
        params: ImageParams,
        **kwargs,
    ) -> ImageData:
        if not isinstance(sample.data, ShotData):
            raise TypeError(
                f"Expected sample data of type 'ShotData' but got '{type(sample.data)}'"
            )

        sample_data: ShotData = sample.data

        if len(sample_data.signal_names) != 1:
            raise ValueError("UDA Camera DataLoader expects exactly one signal name.")

        signal_name = sample_data.signal_names[0]
        try:
            if params.frame is None:
                params.frame = 0  # Default to first frame if not specified

            print(
                f"Loading image signal '{signal_name}' for shot ID '{sample.shot_id}' at frame {params.frame}..."
            )
            signal = xr.open_dataset(
                f"uda://{signal_name}:{sample.shot_id}",
                engine="uda",
                frame_number=params.frame,
            )

            image_array = signal["data"].values
            image_array = np.squeeze(image_array)

            # Ensure at least 2D for Pillow (squeeze can reduce to 0D or 1D for tiny images)
            if image_array.ndim == 0:
                image_array = image_array.reshape(1, 1)
            elif image_array.ndim == 1:
                if image_array.shape[0] in (3, 4):
                    image_array = image_array.reshape(1, 1, -1)  # single pixel RGB/RGBA
                else:
                    image_array = image_array.reshape(-1, 1)  # 1D grayscale strip

            # Convert uint16 to uint8 for Pillow compatibility (Pillow doesn't support u2)
            if image_array.dtype == np.uint16:
                if np.any(image_array > 255):
                    val_range = image_array.max() - image_array.min()
                    image_array = image_array - image_array.min()
                    if val_range:
                        image_array = image_array / val_range
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)

            im = Image.fromarray(image_array)
            buffer = io.BytesIO()
            im.save(buffer, format="PNG")
            buffer.seek(0)

            return ImageData(
                frame=str(params.frame),
                values=base64.b64encode(buffer.getvalue()).decode(),
            )
        except Exception as e:
            raise DataLoaderError(
                f"Could not load image signal '{signal_name}' for shot ID '{sample.shot_id}': {e}"
            )


@LoaderRegistry.register("sal")
class SALDataLoader(DataLoader):
    """DataLoader for retrieving data using the SAL access layer"""

    @classmethod
    def sample_data_type(self) -> Type[ShotData]:
        return ShotData

    def get_sample(
        self,
        sample: Sample,
        params: DataParams,
        time_min: Optional[float] = None,
        time_max: Optional[float] = None,
        min_time_step: Optional[float] = None,
        **kwargs,
    ) -> MultiVariateTimeSeriesData:
        assert isinstance(sample.data, ShotData), "Sample data must be of type ShotData"
        sample_data: ShotData = sample.data

        if not params.name == "identity":
            raise TypeError(f"Expected blank parameters, but got '{params.name}'")

        has_user_credentials = Path("~/.sal/credentials").expanduser().exists()
        if not has_user_credentials:
            raise DataLoaderError(
                "SAL authentication credentials not found. Please set up SAL credentials at '~/.sal/credentials' to use the SAL data loader."
            )

        results = {}
        for name in sample_data.signal_names:
            full_name = f"pulse/{sample.shot_id}/{name}"
            try:
                signal = xr.open_dataset(f"sal://{full_name}", engine="sal")
                data = signal["data"].values
                time = signal["time"].values

                if time_min is not None:
                    mask = time >= time_min
                    time = time[mask]
                    data = data[mask]

                if time_max is not None:
                    mask = time <= time_max
                    time = time[mask]
                    data = data[mask]

                if (
                    min_time_step is not None
                    and len(time) > 1
                    and np.diff(time).mean() < min_time_step
                ):
                    time_base = np.arange(time[0], time[-1], min_time_step)
                    data = np.interp(time_base, time, data)
                    time = time_base

                print(time_min, time_max, min_time_step)

                item = TimeSeriesData(time=time, values=data)
                results[name] = item
            except Exception:
                results[name] = None

        if all(values is None for values in results.values()):
            raise DataLoaderError(
                f"Could not load any signals for shot ID '{sample.shot_id}'. Check SAL connectivity and signal names."
            )

        return MultiVariateTimeSeriesData(values=results)


@LoaderRegistry.register("fair_mast")
class FAIRMASTDataLoader(DataLoader):
    @classmethod
    def sample_data_type(self) -> Type[ShotData]:
        return ShotData

    def get_sample(
        self,
        sample: Sample,
        params: DataParams,
        time_min: Optional[float] = None,
        time_max: Optional[float] = None,
        min_time_step: Optional[float] = None,
        **kwargs,
    ) -> MultiVariateTimeSeriesData:
        assert isinstance(sample.data, ShotData), "Sample data must be of type ShotData"
        sample_data: ShotData = sample.data

        if not params.name == "identity":
            raise TypeError(f"Expected blank parameters, but got '{params.name}'")

        endpoint = "https://s3.echo.stfc.ac.uk/mast/level2/shots"
        file_path = f"{endpoint}/{sample.shot_id}.zarr"

        kwargs = {"chunks": None}

        # check if xarray version supports create_default_indexes argument, and if so, set it to False
        # to avoid unnecessary index creation which can cause performance issues with large datasets
        sig = inspect.signature(xr.open_dataset)
        if "create_default_indexes" in sig.parameters:
            kwargs["create_default_indexes"] = False

        data_tree = xr.open_datatree(file_path, **kwargs)

        results = {}
        for name in sample_data.signal_names:
            try:
                ds = data_tree[name]
                data = ds.values
                time = ds.time.values

                if time_min is not None:
                    mask = time >= time_min
                    time = time[mask]
                    data = data[mask]

                if time_max is not None:
                    mask = time <= time_max
                    time = time[mask]
                    data = data[mask]

                if (
                    min_time_step is not None
                    and len(time) > 1
                    and np.diff(time).mean() < min_time_step
                ):
                    time_base = np.arange(time[0], time[-1], min_time_step)
                    data = np.interp(time_base, time, data)
                    time = time_base

                item = TimeSeriesData(time=time, values=data)
                results[name] = item
            except Exception:
                results[name] = None

        return MultiVariateTimeSeriesData(values=results)
