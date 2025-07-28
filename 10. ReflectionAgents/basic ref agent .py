#  REFLECTION AGENTS

import asyncio
from autogen_agentchat.agents import AssistantAgent 
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from dotenv  import load_dotenv
load_dotenv()
import os 

key = os.getenv("GROQ_API_KEY")

model_client = OpenAIChatCompletionClient(
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


# generator agent
generator = AssistantAgent(
    name = "generator",
    model_client=model_client,
    system_message="you are a funny short story writer "
)

# reflector agent
reflector = AssistantAgent(
    name = "reflector",
    model_client=model_client,
    system_message="""You are a critical reflection agent tasked with reviewing stories written by another AI agent.

Your job is to:

Identify any inconsistencies, plot holes, or unclear parts in the story.

Suggest improvements to pacing, tone, structure, or character development.

Ensure the story aligns with the original prompt or theme.

Comment on creativity, engagement, and originality.

When giving feedback:

Be constructive and specific.

If the story is already good, still suggest at least one small refinement.

Use clear and concise language.

End your review with one of the following:

“Needs revision.” – if the story has significant issues.

“Minor improvements suggested.” – if it’s mostly good but could be improved.

“Story is ready.” – if the story is polished and no further changes are needed.

Note: just provide feedback only in 10 words not more then that."""
)


# termination condition
text_termination = TextMentionTermination('exit')



# making team

team = RoundRobinGroupChat(
    participants=[generator, reflector],
    termination_condition= text_termination,
    max_turns=2
)

async def main(user_input):
    task = user_input
    
    while True:
        stream = team.run_stream(task=task)
        await Console(stream)

        # Here is when we take the feedback from the user
        feedback = input('Please provide your feedback(type "exit" to stop): ')

        if feedback.lower().strip() == 'exit':
            break

        task = feedback # Next task is the feedback

if __name__ == '__main__':
    asyncio.run(main(input("Give topic for the story: ")))


