import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the OpenAI model client
openai_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

# Define a custom function to reverse a string
def reverse_string(text: str) -> str:
    """Reverse the given text.
        input : str


        output : str 
        The reversed String is returned.
    """
    return text[::-1]


# Register the custom function as a tool
reverse_tool = FunctionTool(reverse_string,description='A tool to reverse a string')

# Create an agent with the custom tool
agent = AssistantAgent(
    name="ReverseAgent",
    model_client=openai_client,
    system_message="You are a helpful assistant that can reverse text using the reverse_string tool. Give the result with the summary",
    tools=[reverse_tool],
    reflect_on_tool_use=True, 
    # if false it will give only the output from tool 
    # if True then it will reflect on the output given by the tool
)

# Define a task
task = "Reverse the text 'Hello, how are you Doing?'"

# Run the agent
async def main():
    result = await agent.run(task=task)

    # print(result)    
    print(f"Agent Response: {result.messages[-1].content}")

    print(reverse_string('Hello, how are you Doing?'))

if __name__ == "__main__":
    asyncio.run(main())