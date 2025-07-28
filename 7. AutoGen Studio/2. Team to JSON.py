from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import  TextMentionTermination

agent = AssistantAgent(
        name="weather_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key="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"
        ),
    )

agent_team = RoundRobinGroupChat([agent], termination_condition=TextMentionTermination("TERMINATE"))
config = agent_team.dump_component()
print(config.model_dump_json())

agent_team.load_component()
