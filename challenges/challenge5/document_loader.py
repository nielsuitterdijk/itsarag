import json
import os

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document

# https://python.langchain.com/docs/integrations/document_loaders/azure_document_intelligence/


class DocumentLoader:
    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path

    def load_pdfs(self) -> list[Document]:
        results = self.load_txt()
        if results:
            print("Loaded parsed documents from disk")
            return results

        print("Parsing PDFs with Azure Document Intelligence")
        for file_path in os.listdir(self.input_path)[:2]:
            print(f"Loading {file_path}")
            results.extend(
                AzureAIDocumentIntelligenceLoader(
                    api_endpoint=os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"],
                    api_key=os.environ["AZURE_DOCUMENT_INTELLIGENCE_API_KEY"],
                    file_path=os.path.join(self.input_path, file_path),
                    api_model="prebuilt-layout",
                ).load()
            )

        self.store_results(results)
        return results

    def load_txt(self) -> list[Document]:
        if os.path.exists(self.output_path):
            data = json.load(open(self.output_path, "r"))
            return [Document(**obj) for obj in data]
        return []

    def store_results(self, results: list[Document]) -> None:
        json_results = [res.model_dump() for res in results]
        json.dump(json_results, open(self.output_path, "w"))
