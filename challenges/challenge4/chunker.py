from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document


class Chunker:
    def __init__(self) -> None:
        self.client = CharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        return self.client.split_documents(documents)
