import os
from typing import TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel

B = TypeVar("B", bound=BaseModel)


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

    def complete(self, system_prompt: str, message: dict) -> BaseMessage:
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        messages = prompt.invoke(message)
        return self.llm.invoke(messages)

    def structured_complete(self, pydantic_model: type[B], query: str) -> B:
        result = self.llm.with_structured_output(pydantic_model).invoke(query)
        return pydantic_model.model_validate(result)
