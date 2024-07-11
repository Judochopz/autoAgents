from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import requests
import json
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from llama_cpp.llama_grammar import LlamaGrammar
from types import SimpleNamespace

# The ARGO_LLM class. Uses the _invoke_model helper function.
# It implements the _call function.

# Autogen changes can be found between _call and _identifying_params methods.
# Added new imports

class ARGO_LLM:

    argo: ArgoWrapper
    __name__ = 'ARGO_LLM'

    def __init__(self, argo, model_type='gpt4', temperature = 0.8):
        # May just pass an argo instance directly to chat.completions
        self.argo = argo(model=model_type, temperature=temperature)
        self.chat = chat(self.argo)

    base_url = ArgoWrapper.default_url # AutoGen Required

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

class chat:
    def __init__(self, argo: ArgoWrapper):
        self.completions = completions(argo)

class completions: 
    def __init__(self, argo: ArgoWrapper):
        def create(messages: List[str],
                stream: bool,
        ):
            # Enables the call function of autogen's OpenAIWrapper class
            #ISSUE: Can only respond to latest message, does not consider entire thread
            length = len(messages)
            prompt = 'These are the previous messages for context:\n'
            for message in messages[:length - 1]:
                prompt += message['content'] + '\n'
            prompt += 'The current prompt is:\n' + messages[-1]['content']
            response = self.argo.invoke(prompt)
            
            # SimpleNamespace creates necessary attributes
            message = SimpleNamespace(
                #ISSUE: currently lacks function_call and tool_calls
                    function_call = None,
                    tool_calls = None,
                    content = response['response'],
            )
            choice = SimpleNamespace(
                text = response['response'],
                message = message
            )
            result = SimpleNamespace(
                model = 'argo',
                usage = None,
                choices = [choice],
            )
            return result
        self.argo = argo
        self.create = create
    

class ARGO_EMBEDDING:
    argo: ArgoEmbeddingWrapper
    def __init__(self, argo_wrapper: ArgoEmbeddingWrapper):
        self.argo = argo_wrapper
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call( self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any, ) -> str:
        if stop is not None:
            print(f"STOP={stop}")
        response = self.argo.invoke(prompt)
        #print(f"ARGO Response: {response['embedding']}\nEND ARGO RESPONSE")
        return response['embedding']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

    def embed_documents(self, texts):
        return self._call(texts)

    def embed_query(self, query: str):
        # Handle embedding of a single query string
        # Assuming 'query' is a single string
        return self._call(query)[0]
