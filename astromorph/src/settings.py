from typing import Union
from pydantic import BaseModel, Field


class TrainingSettings(BaseModel):
    datafile: str
    maskfile: Union[str, None] = Field(default=None)
    epochs: int = Field(default=10)
    network_name: str
    network_settings: dict
