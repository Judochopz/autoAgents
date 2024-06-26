import os
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent
from qdrant_client import QdrantClient
from langchain_community.tools.tavily_search import TavilySearchResults
from ARGO import ArgoWrapper
from CustomLLMAutogen import ARGO_LLM
os.environ["USER"] = 'tandoc'

from autogen.oai.client import OpenAIClient, OpenAIWrapper

def main():
    os.environ["USER"] = 'tandoc'
    os.environ['OPENAI_API_KEY'] = 'NA'
    argo_wrapper_instance = ArgoWrapper()
    argo_client_instance = ARGO_LLM(argo=argo_wrapper_instance,model_type='gpt4', temperature = 0.3)
    argo_client_instance
    argo_client = OpenAIClient(argo_client_instance)
    autogenClient = OpenAIWrapper()
    autogenClient._clients = [argo_client]
    autogenClient._clients

    assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    )
    # Converts to Argo client
    assistant.client = autogenClient

    assistant2 = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    )
    # Converts to Argo client
    assistant2.client = autogenClient

    # userProxy = UserProxyAgent(
    #     name="user_proxy",
    #     system_message="You are a user.",
    #     human_input_mode='NEVER',
    #     code_execution_config=False,
    # )

    assistant2.initiate_chat(assistant, message='Tell me a joke', max_turns =2)

if __name__ == "__main__":
    main()