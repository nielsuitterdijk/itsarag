from langchain_community.vectorstores.azuresearch import AzureSearch
from embedder import get_embedder
from document_loader import DocumentLoader
from chunker import Chunker
from langchain_core.documents import Document
import os


class VectorStore:

    def __init__(
        self, index_name: str | None = None, fields: list | None = None
    ) -> None:
        self.store = AzureSearch(
            azure_search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            azure_search_key=os.environ["AZURE_SEARCH_API_KEY"],
            index_name=index_name if index_name else os.environ["AZURE_SEARCH_INDEX"],
            embedding_function=get_embedder().embed_query,
            additional_search_client_options={"retry_total": 4},
            fields=fields,
        )

    def populate(self):
        print("Loading documents")
        documents = DocumentLoader().load()

        print("Chunking")
        chunks = Chunker().chunk(documents)

        print("Adding chunks to index")
        self.store.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadata=[
                {
                    "title": chunk.metadata.get("source", "").split("/")[-1],
                    "source": chunk.metadata.get("source"),
                }
                for chunk in chunks
            ],
        )

    def search(self, query: str) -> list[Document]:
        return self.store.search(query, k=3, search_type="similarity")
