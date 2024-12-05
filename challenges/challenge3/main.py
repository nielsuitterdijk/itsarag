from vector_store import VectorStore
from custom_index import get_fields
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")


# https://python.langchain.com/docs/integrations/vectorstores/azuresearch/#perform-a-hybrid-search


print("Initializing vector store")
store = VectorStore(index_name="fsicustom", fields=get_fields())

print("Populating")
store.populate()

print("Querying")
results = store.search("Google revenue")
print(results)
