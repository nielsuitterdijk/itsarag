from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain_core.documents import Document
import os


class DocumentLoader:
    def __init__(self) -> None:
        self.loader = AzureBlobStorageContainerLoader(
            conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            container=os.environ["AZURE_STORAGE_CONTAINER"],
        )

    def load(self) -> list[Document]:
        return self.loader.load()
