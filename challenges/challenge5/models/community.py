from models.entity import Entity
from models.relationship import Relationship
from pydantic import BaseModel


class Community(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
