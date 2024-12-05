from langchain_openai import AzureOpenAIEmbeddings
import os


def get_embedder():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["EMBEDDING_DEPLOYMENT"],
        api_key=os.environ["EMBEDDING_API_KEY"],  # ignore
        azure_endpoint=os.environ["EMBEDDING_ENDPOINT"],
    )
