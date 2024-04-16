# local imports
from models.llms import LLM, OllamaModel, OpenAIModel, HFModel
from models.chain import Chain
from models.chatbot import ChatBot
from models.docs import Docs
from vectorstore import VectorStore

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