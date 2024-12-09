from pydantic import BaseModel


class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_description: str
    relationship_strength: float
