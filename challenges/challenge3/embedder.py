import os

from langchain_openai import AzureOpenAIEmbeddings


def get_embedder():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["EMBEDDING_DEPLOYMENT"],
        api_key=os.environ["EMBEDDING_API_KEY"],  # type: ignore
        azure_endpoint=os.environ["EMBEDDING_ENDPOINT"],
    )
