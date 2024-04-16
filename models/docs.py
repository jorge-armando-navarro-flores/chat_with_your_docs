from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

class Docs:
    def __init__(self):
        self.docs = []

    def set_pdf_docs(self, url):
        loader = PyPDFLoader(url)
        docs = loader.load()
        self.docs = docs

    def set_youtube_docs(self, url: str):
        save_dir = "docs/youtube/"
        loader = GenericLoader(
            YoutubeAudioLoader([url], save_dir),
            OpenAIWhisperParser()
        )
        docs = loader.load()
        self.docs = docs

        # pip install yt_dlp

    def set_web_docs(self, url: str):
        loader = WebBaseLoader(url)
        docs = loader.load()
        self.docs = docs

    def get_docs(self):
        return self.docs