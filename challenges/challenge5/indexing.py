import json
import os
from collections import defaultdict
from typing import TypeVar

from chat import Chat
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from models.community import Community
from models.entity import Entity
from models.relationship import Relationship
from pydantic import BaseModel
from tqdm import tqdm

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
        # https://microsoft.github.io/graphrag/index/default_dataflow/#entity-relationship-extraction
        community = self.extract_communities(
            documents
        )  # ? Do we need to keep link to document?
        # TODO: claim extraciton -- later
        # TODO: community detection ?  -- Hierarchical Leiden Clustering,
        # TODO: graph embedding node2vec ?
        # TODO: Community summary & embedding
        # TODO: Create document embedding - average of chunk embeddings
        # TODO: search algos

        return community

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        client = CharacterTextSplitter(
            chunk_size=5_000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        return client.split_documents(documents)

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
        """
        This function is consumed by extract_communities. That function generates a community for each chunk.
        Therefore, it's prone to duplicate entities & relations.
        This function aims to consolidate duplicates by summarizing
        """
        entity_dict: defaultdict[str, list[Entity]] = defaultdict(list)
        relationship_dict: defaultdict[str, list[Relationship]] = defaultdict(list)

        for community in communities:
            for entity in community.entities:
                # TODO: Duplicates may not have same formatted name
                entity_dict[entity.name].append(entity)
            for relationship in community.relationships:
                relationship_dict[
                    f"{relationship.source_entity}_{relationship.target_entity}"
                ].append(relationship)

        final_community = Community()
        entity_summary_prompt = open(
            f"{self.dir}{self.prompt_dir}/entity_summary.txt"
        ).read()
        print("Merging entities")
        for duplicate_entities in tqdm(entity_dict.values()):
            final_entity = self.chat.structured_complete(
                Entity,
                entity_summary_prompt.replace(
                    "{entities}",
                    json.dumps([e.model_dump() for e in duplicate_entities]),
                ),
            )
            final_community.entities.append(final_entity)
        relation_summary_prompt = open(
            f"{self.dir}{self.prompt_dir}/relationship_summary.txt"
        ).read()

        print("Merging relations")
        for duplicate_relations in tqdm(relationship_dict.values()):
            final_relationship = self.chat.structured_complete(
                Relationship,
                relation_summary_prompt.replace(
                    "{relationships}",
                    json.dumps([d.model_dump() for d in duplicate_relations]),
                ),
            )
            final_community.relationships.append(final_relationship)

        return final_community

    def detect_communities(self, community: Community) -> None:
        pass
