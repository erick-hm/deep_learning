from typing import Annotated

from pydantic import Field

PositiveFloat = Annotated[float, Field(gt=0)]
