from pydantic import BaseModel


class Sample(BaseModel):
    # Does this take anything by default..?
    pass


class FileSample(Sample):
    file_name: str


class ShotSample(Sample):
    shot_id: int


class UFOSample(Sample):
    camera: str
    frame: int
