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

class ARGO_LLM(LLM):

    argo: ArgoWrapper
    grammar: Optional[LlamaGrammar] = None

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            print(f"STOP={stop}")
            # raise ValueError("stop kwargs are not permitted.")
        print(kwargs)
        response = self.argo.invoke(prompt)       
        #print(f"ARGO Response: {response['response']}\nEND ARGO RESPONSE")
        return response['response']

    #ISSUE: need to pass the same ArgoWrapper instance instead of instantiating it with each call.
    def create(messages: List[str],
               stream: bool
    ):
        # Enables the call function of autogen's OpenAIWrapper class
        argo = ArgoWrapper()
        #ISSUE: Can only respond to latest message, does not consider entire thread
        response = argo.invoke(messages[-1]['content'])
        
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

    base_url = ArgoWrapper.default_url # AutoGen Required
    chat = SimpleNamespace() # provides the chat.completions attribute
    chat.completions = SimpleNamespace() # provides the create method
    chat.completions.create = create

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

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
