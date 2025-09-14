# author: Nancy Goyal


"""Agent for the image analysis specialist."""

from google.adk.agents import LlmAgent

from . import prompt
from .tools.image_prediction_tool import predict_disease_from_image

image_analysis_agent = LlmAgent(
    name="image_analysis_agent",
    model="gemini-2.5-flash",
    description=(
        "Specialist agent that analyzes an image of a plant leaf "
        "to predict a specific disease using a custom model."
    ),
    instruction=prompt.IMAGE_ANALYSIS_PROMPT,
    tools=[predict_disease_from_image],
)