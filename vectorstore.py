from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# local imports
from models.docs import Docs

class VectorStore:
    def __init__(self, docs: Docs):
        self.docs = docs
        self.vector = None

    def set_vector(self, embeddings):
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(self.docs.get_docs())
        vector = FAISS.from_documents(documents, embeddings)
        self.vector = vector

    def get_vector(self):
        return self.vector
