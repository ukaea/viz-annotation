from enum import Enum

from toktagger.api.schemas import ConfiguredModel


class ViewType(str, Enum):
    IDENTITY = "identity"
    SPECTROGRAM = "spectrogram"


class ViewParams(ConfiguredModel):
    name: ViewType = ViewType.IDENTITY


class SpectrogramViewParams(ViewParams):
    nperseg: int | None = 256
    time_min: float | None = None
    time_max: float | None = None
    frequency_min: float | None = None
    frequency_max: float | None = None
    amplitude_min: float | None = None
    amplitude_max: float | None = None
    threshold_value: float | None = None


ViewParamTypes = ViewParams | SpectrogramViewParams
