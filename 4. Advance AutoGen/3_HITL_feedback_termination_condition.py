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
api_key = os.getenv('OPENAI_API_KEY')
model_client = OpenAIChatCompletionClient(model='gpt-4o')

lazy_agent = AssistantAgent(
    name='Lazy_Agent',
    model_client=model_client,
    system_message='Give the answer of question asked with the information you have or are provided. If you cannot complete the task, just say "TERMINATE"'
)

text_termination = TextMentionTermination('TERMINATE')


lazy_agent_team = RoundRobinGroupChat(participants=[lazy_agent], termination_condition= text_termination, max_turns=2)


async def main():
    task = 'Give me the current weather of New York.'
    
    while True:
        stream = lazy_agent_team.run_stream(task=task)
        await Console(stream)

        # Here is when we take the feedback from the user
        feedback = input('Please provide your feedback(type "exit" to stop): ')

        if feedback.lower().strip() == 'exit':
            break

        task = feedback # Next task is the feedback

if __name__ == '__main__':
    # Run the main function
    # asyncio.run is used to run the main function in an event loop
    # This is necessary because the main function is asynchronous
    # and needs to be run in an event loop to work properly
    asyncio.run(main())