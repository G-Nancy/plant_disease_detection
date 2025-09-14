<<<<<<< HEAD
# Plant Doctor AI 👩‍⚕️🌱

## Overview
A specialized, multi-agent system designed to diagnose plant diseases and provide actionable management recommendations. This system uses a combination of generative AI models and custom tools to analyze user input and provide comprehensive solutions.

All disease diagnosis and management information is based on a pre-processed document, the "Handbook of Plant Disease Identification and Management."

For details on GCP deployment, please read and follow the instructions in the `deploy/README.md` file.

## Agent Details
The Plant Doctor AI is built as a multi-agent system orchestrated by a **Root Agent**. The Root Agent delegates tasks to specialized sub-agents, each equipped with its own tools and knowledge base.

### Key Features
* **Multi-Agent Architecture**: A hierarchical system where a central agent delegates tasks to specialized sub-agents for diagnosis and management.
* **Symptom-Based Diagnosis**: A specialist **RAG agent** that analyzes a text description of plant symptoms to provide a preliminary disease diagnosis using a knowledge base.
* **Image-Based Diagnosis**: A **custom model** trained using **transfer learning on 70,000 images** with the **MobileNetV2** architecture to provide a definitive diagnosis from a plant leaf image.
* **RAG-Powered Recommendations**: Uses a Retrieval-Augmented Generation (RAG) tool to search a local ChromaDB for disease management and cure recommendations, ensuring answers are grounded in the provided document.
* **Language Model Flexibility**: The system is designed to be model-agnostic, capable of using both Google Gemini models for orchestration and OpenAI models for specific tasks via LiteLLM.

### Agent Workflow
1.  The **Root Agent** receives a user's request (either a text description of symptoms or an image).
2.  Based on the input type, it delegates to either the **`symptom_to_disease_agent`** or the **`image_analysis_agent`**.
3.  The chosen diagnosis agent returns a disease name or "Diagnosis Unclear."
4.  The **Root Agent** then delegates the disease name to the **`management_agent`**.
5.  The **`management_agent`** uses its RAG tool to find and synthesize recommendations from the document.
6.  The **Root Agent** combines the diagnosis and recommendations into a single, comprehensive response for the user.

### Architecture
![Provider Search Agent Architecture](docs/provider-search-agent-arch.png)

## Setup and Installation

### Prerequisites

- **Google Credentials:** You need a GCP project _or_ Gemini API key for local testing. You need a GCP project for deployment to Cloud Run.
- **UV:** Ensure that you have uv installed. If you don't already, please follow the installation instructions at [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

### Project Setup

1. **Install dependencies in a virtual environment:** `make install`

1. **Run static code analysis:** `make check`

1. **Set up Environment Variables:** Create a file named `.env` and update values as needed.

    ```bash
    # If using API key: ML Dev backend config.
    GOOGLE_API_KEY=YOUR_VALUE_HERE
    OPENAI_API_KEY=YOUR_VALUE_HERE
    GOOGLE_GENAI_USE_VERTEXAI=false

    # If using Vertex on GCP: Vertex backend config
    GOOGLE_CLOUD_PROJECT=YOUR_VALUE_HERE
    GOOGLE_CLOUD_LOCATION=YOUR_VALUE_HERE
    GOOGLE_GENAI_USE_VERTEXAI=true
    ```

1. **If you're using a GCP project, authenticate with GCP and enable VertexAI:**

    ```bash
    gcloud auth login --update-adc
    gcloud auth application-default login
    gcloud config set project PROJECT_ID
    gcloud services enable aiplatform.googleapis.com
    ```

You are now ready to start development on your project!

## Running the Agent

Run the agent(s) API server with the command: `make api_server`

Run the agent with the ADK Web UI with the command: `make web`

## Running Tests

Tests assess the overall executability of the agents. All tests are located under the `tests/` directory.
Run tests with the command `make test`

=======

### Additional - custom trained vision model
dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data
Additional data sources for future work:
https://www.nature.com/articles/sdata201566
https://www.chc.ucsb.edu/data/chirps?utm_source=chatgpt.com
for Time and weather data: https://power.larc.nasa.gov/docs/services/api/temporal/daily/?utm_source=chatgpt.com#__tabbed_1_3


#### Research work:
https://www.sciencedirect.com/science/article/pii/S2772375524000133
https://www.sciencedirect.com/science/article/pii/S2666285X22000218?ref=pdf_download&fr=RR-2&rr=97f277261b5c9295
https://www.mdpi.com/2571-8800/8/1/4
https://arxiv.org/pdf/2409.04038
https://arxiv.org/pdf/2504.20419
