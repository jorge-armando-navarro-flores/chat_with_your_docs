import os
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

output_parser = StrOutputParser()


class LLM(ABC):
    def __init__(self, name: str):
        self.name = name
        self.llm = None
        self.embeddings = None

    def get_name(self):
        return self.name

    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings


class OllamaModel(LLM):
    def __init__(self, name: str):
        super().__init__(name)
        self.llm = Ollama(model=self.name)
        self.embeddings = OllamaEmbeddings()


class OpenAIModel(LLM):
    def __init__(self, name: str, api_key: str = None):
        super().__init__(name)
        os.environ["OPENAI_API_KEY"] = api_key
        self.llm = ChatOpenAI(model=self.name, openai_api_key=api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)


class HFModel(LLM):
    def __init__(self, name, token):
        super().__init__(name)
        self.llm = ChatHuggingFace(llm=HuggingFaceEndpoint(
            huggingfacehub_api_token=f"{token}",
            repo_id=self.name,
            task="text-generation",
            max_new_tokens=512,
            top_k=30,
            temperature=0.1,
            repetition_penalty=1.03,

        ))
        self.embeddings = HuggingFaceEmbeddings()


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


class Chain:
    def __init__(self, model: LLM):
        self.model = model
        self.chain = self.get_simple_chain()

    def get_simple_chain(self):
        return self.model.llm | output_parser

    def set_conversational_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, answer",
                ),
            ]
        )

        self.chain = prompt | self.model.llm | output_parser

    def set_retrieval_chain(self, vector: VectorStore):
        retriever = vector.get_vector().as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
                ),
            ]
        )
        retriever_chain = create_history_aware_retriever(self.model.llm, retriever, prompt)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the user's questions based on the below context:\n\n{context}",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
            ]
        )
        document_chain = create_stuff_documents_chain(self.model.llm, prompt)

        self.chain = create_retrieval_chain(retriever_chain, document_chain)

    def get_chain(self):
        return self.chain


class ChatBot:
    def __init__(self, chain: Chain):
        self.chat_history = []
        self.chain = chain

    def get_simple_answer(self, message):
        answer = self.chain.get_chain().invoke(message)
        self.add_history(message, answer)
        return answer

    def get_conversational_answer(self, message):
        answer = self.chain.get_chain().invoke(
            {"chat_history": self.chat_history, "input": message}
        )
        self.add_history(message, answer)
        return answer

    def get_retrieval_answer(self, message):
        answer = self.chain.get_chain().invoke(
            {"chat_history": self.chat_history, "input": message}
        )["answer"]
        self.add_history(message, answer)
        return answer

    def get_test_answer(self, message):
        return message

    def add_history(self, message, answer):
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=answer))


class ChatBotController:
    def __init__(self, model: LLM):
        self.model = model
        self.chain = Chain(self.model)
        self.chatbot = ChatBot(self.chain)
        self.model_types = {
            "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
            "Ollama": ["llama2:latest", "mistral:latest", "gemma:latest"],
            "HuggingFace": ["HuggingFaceH4/zephyr-7b-beta", "mistralai/Mistral-7B-v0.1"]
        }
        self.docs = Docs()
        self.vectorstore = None

    def set_model(self, model_type, model_ref, api_key=None):

        if model_ref not in self.model_types[model_type]:
            model_ref = self.model_types[model_type][0]

        try:
            if model_type.lower() == "openai":
                self.model = OpenAIModel(model_ref, api_key)
            elif model_type.lower() == "huggingface":
                self.model = HFModel(model_ref, api_key)
            elif model_type.lower() == "ollama":
                self.model = OllamaModel(model_ref)
            else:
                print("Nombre de modelo no válido.")

            self.chain = Chain(self.model)
            self.chatbot = ChatBot(self.chain)
        except Exception as e:
            return "Please Introduce a Valid API key"

        return "Model Ready"

    def set_retrieval(self, doc_type, doc_url=None, file_path=None):
        if doc_type == "PDF":
            self.docs.set_pdf_docs(file_path)
        elif doc_type == "YouTube":
            self.docs.set_youtube_docs(doc_url, )
        elif doc_type == "WEB":
            self.docs.set_web_docs(doc_url)

        self.vectorstore = VectorStore(self.docs)
        self.vectorstore.set_vector(self.model.get_embeddings())
        self.chain.set_retrieval_chain(self.vectorstore)
        self.chatbot = ChatBot(self.chain)
        return "Docs Ready"

    def predict(self, message, history):
        return self.chatbot.get_simple_answer(
            message) if not self.vectorstore else self.chatbot.get_retrieval_answer(message)


undo_button = gr.Button("↩️ Undo")
clear_button = gr.Button("🗑️  Clear")

chatbot = ChatBotController(OllamaModel("orca-mini"))


def filter_model_types(model_type):
    api_key_field = model_type != "Ollama"
    return gr.Text(placeholder="Input your API key", visible=api_key_field), gr.Dropdown(value=chatbot.model_types[model_type][0],
                                                                               choices=chatbot.model_types[model_type])


def filter_doc_types(doc_type):
    url_field = False
    file_field = False

    if doc_type == "PDF":
        file_field = True
    else:
        url_field = True
    return gr.Textbox("URL", visible=url_field), gr.File(visible=file_field)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            model_label = gr.Label("Set your Model")
            model_type = gr.Radio(label="Model Source", value="Ollama", choices=["Ollama", "OpenAI", "HuggingFace"])
            selected_model = gr.Dropdown(label="Model Selection", value="orca-mini", choices=["orca-mini", "llama2:latest"])
            api_key = gr.Textbox(label="API key", type="password", visible=False)
            model_type.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            selected_model.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            api_key.input(chatbot.set_model, inputs=[model_type, selected_model, api_key], outputs=[model_label])
            model_type.change(filter_model_types, [model_type], outputs=[api_key, selected_model])
            doc_label = gr.Label("Set your Docs")

            doc_type = gr.Radio(label="Model Type", value="PDF", choices=["PDF", "WEB", "YouTube"])
            url = gr.Textbox(label="Document Source", placeholder="URL", visible=False)
            file = gr.File()
            doc_type.change(filter_doc_types, inputs=[doc_type], outputs=[url, file])

            process_button = gr.Button("Process")
            process_button.click(chatbot.set_retrieval, inputs=[doc_type, url, file], outputs=[doc_label])

        with gr.Column(scale=3):
            gr.ChatInterface(chatbot.predict, retry_btn="🔄  Retry", undo_btn=undo_button, clear_btn=clear_button)

if __name__ == "__main__":
    demo.launch()
