# author: Nancy Goyal

"""Agent for the RAG-based information retrieval specialist."""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from .tools.openai_doc_qa_tool import ask_document_question

from . import prompt

symptom_to_disease_agent = LlmAgent(
    name="symptom_to_disease_agent",
    # model="gemini-2.5-flash",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description=(
        "Specialist agent that performs Q&A on a document by retrieving "
        "relevant information from tool."
    ),
    # instruction="Use tool to provide answer only",
    instruction=prompt.SYMPTOM_ANALYSIS_PROMPT,
    tools=[ask_document_question],
)