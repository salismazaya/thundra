import langchain
from typing import Dict
from dataclasses import dataclass
from enum import Enum
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from ..config import config_toml

@dataclass
class UserMemory:
    system_message: SystemMessage
    k: int
    memory: ConversationBufferWindowMemory


    @classmethod
    def create_ai_instance(cls, k: int):
        system_message = SystemMessage(content=config_toml['openai']['agent']['system_message'])
        return cls(
            system_message=system_message,
            k=k,
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=k,
                return_messages=True,
                chat_memory=ChatMessageHistory(
                    messages=[system_message]
                ),
            ),
        )

    def clear_history(self):
        self.memory.chat_memory.messages.clear()
        self.memory.chat_memory.messages.append(
            self.system_message
        )

    def get_memory(self):
        if self.memory.chat_memory.messages.__len__() > self.k * 2 - 1:
            self.memory.chat_memory.messages = [
                self.system_message,
                *self.memory.chat_memory.messages,
            ]  # unoptimized
        return self.memory


class AIMemory:
    def __init__(self, k: int = 13):
        self.k = k
        self.memory: Dict[str, UserMemory] = {}

    def get_memory(self, id: str):
        memoize = self.memory.get(id)
        if not memoize:
            memoize = UserMemory.create_ai_instance(
                config_toml['openai']['agent']['memory_size']
            )
            self.memory[id] = memoize
        return memoize.memory

    def delete_memory(self, id: str):
        if self.memory.get(id):
            del self.memory[id]

    def clear_history(self, id: str):
        if self.memory.get(id):
            self.memory[id].clear_history()


memory = AIMemory()
