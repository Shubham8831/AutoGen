from config.settings import GROQ_API_KEY, MODEL
from autogen_ext.models.openai import OpenAIChatCompletionClient


model_client =  OpenAIChatCompletionClient(
        base_url="https://api.groq.com/openai/v1",
        model=MODEL,
        api_key = GROQ_API_KEY,
        model_info={
            "family":'llama',
            "vision" :False,
            "function_calling":True,
            "json_output": True
        }
    )

