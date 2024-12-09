from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    type: str
    description: str
