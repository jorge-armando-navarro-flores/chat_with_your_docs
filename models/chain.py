from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

# local imports
from models.llms import LLM
from vectorstore import VectorStore

output_parser = StrOutputParser()



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