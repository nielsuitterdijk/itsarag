import os
from langchain_openai import AzureChatOpenAI
from vector_store import VectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


class Chat:
    def __init__(self) -> None:
        for key in [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_API_VERSION",
        ]:
            print(f"{key}: {os.environ[key]}")
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # type: ignore
        )
        self.store = VectorStore()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context"
                    " to answer the question. If you don't know the answer, just say that you don't know. Use three "
                    "sentences maximum and keep the answer concise. Question: {question} Context: {context} Answer:",
                ),
            ]
        )

    def chat(self, message: str) -> str:
        references = self.retrieve(message)
        return self.generate(message, references)

    def retrieve(self, message: str) -> list[Document]:
        return self.store.search(message)

    def generate(self, message: str, documents: list[Document]) -> str:
        docs_content = "\n\n".join(doc.page_content for doc in documents)
        messages = self.prompt.invoke({"question": message, "context": docs_content})
        response = self.llm.invoke(messages)
        return str(response.content)
