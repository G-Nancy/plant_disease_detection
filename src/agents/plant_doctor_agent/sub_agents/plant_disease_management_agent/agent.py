# author: Nancy Goyal


"""Agent  to handle the specific task of finding management recommendations."""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from .tools.recommendation_tool import get_management_recommendations
from . import prompt

plant_disease_management_agent = LlmAgent(
    name="plant_disease_management_agent",
    # model="gemini-1.5-pro",
    model=LiteLlm(model="openai/gpt-4o-mini"),

    description=(
        "Specialist agent that provides disease management to cure recommendations."
    ),
    instruction=prompt.MANAGEMENT_AGENT_PROMPT,
    tools=[get_management_recommendations],
)