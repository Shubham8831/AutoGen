# BASIC HITL  AGENT -> USER (round robin)             #USEER PROXY AGENT

# FEEDBACK DURING A RUN 

# input() # We face this issue because in VScode python input function does not work as expected. and requires some extension.


import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
import os

load_dotenv()
key = os.getenv("Groq_api_key")



model_client =  OpenAIChatCompletionClient(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    api_key = key,
    model_info={
        "family":'llama',
        "vision" :False,
        "function_calling":True,
        "json_output": True
    }
)



# agent 1
assistant = AssistantAgent(
    name='Assistant',
    model_client=model_client,
    system_message="You are a helpful assistant.",
)

# user proxy agent - to take user feedback 
user_proxy_agent = UserProxyAgent(
    name='UserProxy',
    description="A proxy agent that represents the user.",
    input_func=input
)

# condition to terminate the team agent
termination = TextMentionTermination('APPROVE')


# Create a team with the assistant and user proxy agent
team = RoundRobinGroupChat(
    participants=[assistant, user_proxy_agent],
    termination_condition=termination
)


stream = team.run_stream(task = 'Write a 4 line poem about the ocean')

async def main():
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())