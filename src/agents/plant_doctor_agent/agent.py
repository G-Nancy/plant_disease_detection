# author: Nancy Goyal

"""Root agent for the plant disease prediction application."""

import google.auth
from google.adk.agents import  LlmAgent
from google.adk.models.lite_llm import LiteLlm 

from . import prompt
from .sub_agents.image_analysis_agent import image_analysis_agent
from .sub_agents.symptom_to_disease_agent import symptom_to_disease_agent
from .sub_agents.plant_disease_management_agent import plant_disease_management_agent

# from .sub_agents.rag_agent import rag_agent

import os

# from .sub_agents.rag_agent import rag_agent

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


plant_doctor = LlmAgent(
    name="plant_doctor_agent",
    # model="gemini-2.5-flash",
    model=LiteLlm(model="openai/gpt-4o-mini"),

    description=(
        "I am a plant doctor that diagnoses diseases when asked by text using symptom_to_disease_agent,"
        "when asked with image then using image_analysis_agent  and provides treatment using plant_disease_management_agent."
    ),
    instruction=prompt.ROOT_PROMPT,
    sub_agents=[symptom_to_disease_agent, plant_disease_management_agent, image_analysis_agent,],
)

root_agent = plant_doctor