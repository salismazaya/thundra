from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub
from neonize.client import NewClient
from neonize.proto.Neonize_pb2 import Message
from .utils import get_message_type
from .agents import agent
from .core.memory import build_system_message
from .core.llm import llm
from langchain_core.messages import SystemMessage

# class AgentExecutor(AgentExecutor):
#     def __init__(self, memory, *kwargs):
#         self.mem
#     def invoke(self, input: Dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any) -> Dict[str, Any]:
        


def execute_agent(memory, client: NewClient, message: Message):
    tools = [
        tool.agent(client, message)
        for tool in agent.filter_tools(get_message_type(message.Message).__class__)
    ]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    prompt.messages.insert(0, SystemMessage(content = build_system_message()))
    # return initialize_agent(
    #     agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,
    #     max_iterations=3,
    #     early_stopping_method="generate",
    #     memory=memory,
    #     # https://github.com/langchain-ai/langchain/issues/6334
    #     agent_kwargs = {
    #         'system_message': SystemMessage(content = build_system_message())
    #     }
    # )
    agent_ = create_openai_tools_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )
    return AgentExecutor(agent=agent_, tools=tools, verbose=True, memory=memory)
