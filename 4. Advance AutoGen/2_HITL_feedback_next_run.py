# in previous HITL the state of team was unstable and process is hold up untile user provide a feeback or say error out 
# as previous one is blocking approch its recommended to use previous one for shorter interactions

# NO USER PROXY AGENT ( we are providing feedback after task is done)

# 2 ways to implement this flow / approach:

        # - use maximum number of turns 
        # - use termination condition



# 1st Approach - # Provide feedback to the next run : Using Max Turns
    # (allow us to pause the team for user input/ feedback)

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



# Three agents for Storytelling
narrator = AssistantAgent(
    name='Narrator',
    model_client=model_client)
hero = AssistantAgent(
    name='Hero',
    model_client=model_client)
guide = AssistantAgent(
    name='Guide',
    model_client=model_client)

# Team with max_turns of 1

team = RoundRobinGroupChat(
    participants=[narrator, hero, guide],
    max_turns=1)


async def main():
    task = ' Write a 3 part story about a mysterious forest less than 20 words.'
    
    while True:
        stream = team.run_stream(task=task)
        await Console(stream)

        # Here is when we take the feedback from the user
        feedback = input('Please provide your feedback(type "exit" to stop): ')

        if feedback.lower().strip() == 'exit':
            break

        task = feedback # Next task is the feedback if no exit in feedback

if __name__ == '__main__':
    # Run the main function
    # asyncio.run is used to run the main function in an event loop
    # This is necessary because the main function is asynchronous
    # and needs to be run in an event loop to work properly
    asyncio.run(main())