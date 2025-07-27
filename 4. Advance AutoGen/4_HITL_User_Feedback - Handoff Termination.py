# HANDOFF TERMINATION : useful when you want u want to pause the run and allow application or user to provide input when as agent hands off to them.














from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
model_client = OpenAIChatCompletionClient(model='gpt-4o')

# agent 
lazy_agent = AssistantAgent(
    name='Lazy_Agent',
    model_client=model_client,
    handoffs=[Handoff(target='user',message='Transfering to user')],  # this will always handoff to user ?
    system_message='if you cannot complete the task, just transfer it to user. When done, say "TERMINATE"'
)


# termination condition
handoff_termination = HandoffTermination('user')
text_termination = TextMentionTermination('TERMINATE')

# mix of both termination condion
termination_condition = handoff_termination | text_termination

# team
lazy_agent_team = RoundRobinGroupChat(
    participants=[lazy_agent],
    termination_condition=termination_condition)




async def main():
    task = 'Give me the current weather of New York.'
    await Console(lazy_agent_team.run_stream(task=task),output_stats=True)
    feedback = "The weather is sunny"
    # Here is when we take the feedback from the user
    await Console(lazy_agent_team.run_stream(task=feedback))

if __name__ == '__main__':
    # Run the main function
    # asyncio.run is used to run the main function in an event loop
    # This is necessary because the main function is asynchronous
    # and needs to be run in an event loop to work properly
    asyncio.run(main())