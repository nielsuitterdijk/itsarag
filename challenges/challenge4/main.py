# from vector_store import VectorStore
# from custom_index import get_fields
from dotenv import load_dotenv
from chat import Chat

load_dotenv(dotenv_path="../.env")


# https://python.langchain.com/docs/integrations/vectorstores/azuresearch/#perform-a-hybrid-search


# store = VectorStore(index_name="fsicustom", fields=get_fields())
# store.populate()
# results = store.search("Google revenue")
# print(results)

print(Chat().chat("What's googles revenue in 2000?"))
