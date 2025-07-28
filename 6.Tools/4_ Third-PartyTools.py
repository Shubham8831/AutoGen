import asyncio
from dotenv import load_dotenv
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from langchain_community.utilities import GoogleSerperAPIWrapper #tool from langchian

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set up Serper API key
#os.environ["SERPER_API_KEY"] = "8c8e65ec131852cd74158b38f7e31c921da8d5e9"  # Replace with your key

# Initialize the search tool wrapper
search_tool_wrapper = GoogleSerperAPIWrapper(type='news')

# Create a simple function to use as a tool
def search_web(query: str) -> str:
    """Search the web using Serper API"""
    try:
        return search_tool_wrapper.run(query)
    except Exception as e:
        return f"Search failed: {str(e)}"

# Initialize OpenAI client
openai_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key)

# Create an agent with the search tool
search_agent = AssistantAgent(
    name='SearchAgent',
    model_client=openai_client,
    system_message="""You are a helpful assistant that can search the web to find current information. 
    When asked a question, use the search_web tool to find relevant information and provide a comprehensive answer based on the search results.""",
    description='Searches the internet and provides detailed answers based on search results',
    tools=[search_web],
    reflect_on_tool_use=True
)

# Test function to demonstrate the tool
async def demonstrate_search():
    """Demonstrate the search functionality"""
    print("=== AutoGen Third-Party Tools Demonstration ===\n")
    
    # Test queries for demonstration
    test_queries = [
        "Who won the last IPL tournament in cricket in 2025?",
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        
        try:
            # Run the agent with the query
            result = await search_agent.run(task=query)
            print(f"Response: {result.messages[-1].content}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "="*70 + "\n")

# Main execution
async def main():
    await demonstrate_search()

if __name__ == "__main__":
    asyncio.run(main())