from langchain_core.messages import HumanMessage, AIMessage
# local imports
from models.chain import Chain

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