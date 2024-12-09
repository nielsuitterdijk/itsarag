import json
import os
from typing import TypeVar

from chat import Chat
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from models.community import Community
from pydantic import BaseModel

B = TypeVar("B", bound=BaseModel)


class GraphRAGIndex:
    config = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }
    output_dir = "/output"
    input_dir = "/input"
    prompt_dir = "/prompts"

    def __init__(self, rag_dir: str = "./ragtest") -> None:
        self.dir = rag_dir
        self.chat = Chat()
        self.documents: list[Document] = []

    def build_context(self, documents: list[Document]):
        # TODO: Chunk documents
        community = self.extract_communities(documents)
        return community

    def extract_communities(self, documents: list[Document]) -> Community:
        all_communities_path = f"{self.dir}{self.output_dir}/all_communities.json"
        if os.path.exists(all_communities_path):
            return Community(**json.load(open(all_communities_path, "r")))

        prompt = open(f"{self.dir}{self.prompt_dir}/entity_extraction.txt", "r").read()

        communities: list[Community] = []
        for i, document in enumerate(self.chunk_documents(documents)):
            community_file_path = f"{self.dir}{self.output_dir}/community_{i}.json"
            if os.path.exists(community_file_path):
                print(f"Loading community for chunk {i}")
                communities.append(
                    Community(**json.load(open(community_file_path, "r")))
                )
                continue

            print(f"Extracting community from chunk {i}")
            response = self.chat.structured_complete(
                Community, prompt.replace("{input_text}", document.page_content)
            )
            communities.append(response)
            json.dump(response.model_dump(), open(community_file_path, "w"))

        all_communities = self.merge_communities(communities)
        json.dump(all_communities.model_dump(), open(all_communities_path, "w"))

        return all_communities

    def merge_communities(self, communities: list[Community]) -> Community:
        entities = {}
        relationships = {}

        for community in communities:
            for entity in community.entities:
                entities[entity.name] = entity
            for relationship in community.relationships:
                relationships[
                    relationship.source_entity + relationship.target_entity
                ] = relationship

        return Community(
            entities=list(entities.values()), relationships=list(relationships.values())
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        client = CharacterTextSplitter(
            chunk_size=5_000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        return client.split_documents(documents)
