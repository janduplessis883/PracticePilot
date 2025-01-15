from llama_index.core import SimpleDirectoryReader
from ragatouille import RAGPretrainedModel

reader = SimpleDirectoryReader(input_files=["data/attention.pdf"])

docs = reader.load_data()

full_document = ""

for doc in docs:
  full_document += doc.text


rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

rag.index(
    collection=[full_document],
    index_name="attention",
    max_document_length=512,
    split_documents=True,
)
