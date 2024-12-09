import os

import pandas as pd
import tiktoken
from graphrag.model.community import Community
from graphrag.model.community_report import CommunityReport
from graphrag.model.entity import Entity
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_reports,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import (
    GlobalSearch,
    GlobalSearchResult,
)


class Search:
    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }
    context_builder_params = {}

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            model=os.environ["AZURE_OPENAI_MODEL"],
            api_type=OpenaiApiType.AzureOpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
            max_retries=20,
        )
        self.encoder = tiktoken.encoding_for_model(os.environ["AZURE_OPENAI_MODEL"])
        self.search_engine: GlobalSearch | None = None
        self.context: GlobalCommunityContext | None = None

    def build_context(self):
        return GlobalCommunityContext(
            community_reports=self.load_reports(),
            communities=self.load_communities(),
            entities=self.load_entities(),
            token_encoder=self.encoder,
        )

    def load_reports(self) -> list[CommunityReport]:
        return []

    def load_communities(self) -> list[Community]:
        return []

    def load_entities(self) -> list[Entity]:
        return []

    def load_search_engine(self) -> GlobalSearch:
        if self.search_engine is not None:
            return self.search_engine
        if self.context is None:
            self.context = self.build_context()

        self.search_engine = GlobalSearch(
            llm=self.llm,
            context_builder=self.context,
            token_encoder=self.encoder,
            max_data_tokens=12_000,
            map_llm_params=self.map_llm_params,
            reduce_llm_params=self.reduce_llm_params,
            allow_general_knowledge=True,
            json_mode=True,
            context_builder_params=self.context_builder_params,
            concurrent_coroutines=32,
            response_type="multiple paragraphs",
        )
        return self.search_engine

    def search_global(self, query: str) -> GlobalSearchResult:
        if self.search_engine is None:
            self.search_engine = self.load_search_engine()

        return self.search_engine.search(query=query)

    def search_local(self):
        pass
