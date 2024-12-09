from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class List(BaseModel, Generic[T]):
    items: list[T]
