from document_loader import DocumentLoader
from dotenv import load_dotenv
from indexing import GraphRAGIndex

load_dotenv(dotenv_path="../../.env")

documents = DocumentLoader(
    input_path="../../data/fsi/pdf/", output_path="ragtest/input/documents.json"
).load_pdfs()

index = GraphRAGIndex()
index.build_context(documents)
