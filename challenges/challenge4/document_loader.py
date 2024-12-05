from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document
import os


# https://python.langchain.com/docs/integrations/document_loaders/azure_document_intelligence/


class DocumentLoader:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    def load(self) -> list[Document]:
        results = []

        for file_path in os.listdir(self.folder_path)[:2]:
            print(f"Loading {file_path}")
            results.extend(
                AzureAIDocumentIntelligenceLoader(
                    api_endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
                    api_key=os.environ["AZURE_DOCUMENT_INTELLIGENCE_API_KEY"],
                    file_path=os.path.join(self.folder_path, file_path),
                    api_model="prebuilt-layout",
                ).load()
            )

        return results
