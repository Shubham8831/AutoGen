from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.base import TaskResult 
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
load_dotenv()
import os 
key = os.getenv("GROQ_API_KEY")

async def team_config(position="AI Engineer"):
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

# Define our agents

    # interviewer agent
    # interviewee agent
    # career coach

    interviewer = AssistantAgent(
        name="interviewer",
        model_client=model_client,
        description=f"an ai agent that conducts interview for a {position} position.",
        system_message= f'''You are a professional interviewer for a {position} position.
        Ask one clear question at a time and Wait for user to respond. 
        Your job. is to continue and ask questions, don't pay any attention to career coach response. 
        Make sure to ask question based on Candidate's answer and your expertise in the field.
        Ask 3 questions in total covering technical skills and experience, problem-solving abilities, and cultural fit.
        After asking 3 questions, say 'exit' at the end of the interview.
        Make question under 30 words.'''
    )

    #user
    candidate = UserProxyAgent(
        name = "candidate",
        description=f"an agent that simulates a candidate fir a {position} position.",
        input_func=input
    )

    career_coach = AssistantAgent(
        name="carrer_coach",
        model_client=model_client,
        description=f"ai agent that provides feedback and advice to a canditate for a {position} position",
        system_message=f"""
        You are a career coach specializing in preparing candidates for {position} interviews.
        Provide constructive feedback on the candidate's responses and suggest improvements.
        After the interview, summarize the candidate's performance and provide actionable advice.
        Make it under 50 words.
        note = 'output in less then 50 words'
        """
    )

    team = RoundRobinGroupChat(
        participants=[interviewer, candidate, career_coach],
        termination_condition=TextMentionTermination("exit"),
        max_turns=20
    )
    return team


async def interview(team):
    async for item in team.run_stream(task = 'start the interview with the first question'):
         # stream of TextMessage (and other message types) followed by a TaskResult
        if isinstance(item, TextMessage):
            # this is one of the streamed messages
            yield item.content

        elif isinstance(item, TaskResult):
            # end of stream; if you want you can break here
            break


async def main():
    position = "AI and Machine learning Engineer"
    team = await team_config(position)

    async for message in interview(team):
        print("-"*70)
        print(message)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())