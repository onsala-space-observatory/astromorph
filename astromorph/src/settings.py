from typing import Union
from pydantic import BaseModel, Field


class TrainingSettings(BaseModel):
    datafile: str
    maskfile: Union[str, None] = Field(default=None)
    epochs: int = Field(default=10)
    last_layer: str = Field(default="layer4")
