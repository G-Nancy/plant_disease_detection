import openai
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .. import prompt   # Import the prompt from the prompt file


def get_management_recommendations(disease_name: str) -> str:
    """
    Retrieves and synthesizes management and cure recommendations for a given plant disease.

    Args:
        disease_name: The name of the plant disease to find recommendations for.

    Returns:
        A string containing a list of management recommendations.
    """
    with open("/Users/gnancy/work/5h1va/DSML/plant_disease/code/openai.txt","r") as file:
        key = file.readline().strip()

    # Initialize OpenAI Embeddings
    model_name = "text-embedding-3-large"
    embeddings = OpenAIEmbeddings(model=model_name, api_key=key)
    client = openai.OpenAI(api_key=key)
    
    # Load the existing ChromaDB from the persistence directory
    # Note: No documents are added here; it just connects to the existing collection.
    db = Chroma(persist_directory="/Users/gnancy/work/5h1va/DSML/plant_disease/code/chroma_db_oa", embedding_function=embeddings)
    print("Connected to pre-existing ChromaDB.")


    try:
        # Step 1: Retrieve documents relevant to the disease name and "management"
        query_text = f"{disease_name} management recommendations"
        retrieved_docs = db.similarity_search(query_text, k=5)

        # Step 2: Prepare the context for the LLM
        formatted_context = "\n\n".join([f"Content:\n{doc.page_content}" for doc in retrieved_docs])

        # Step 3: Construct the prompt using the imported template
        prompt = prompt.TOOL_PROMPT.format(
            formatted_context=formatted_context,
            disease_name=disease_name
        )

        # Step 4: Call the LLM to generate the answer
        completion = client.chat.completions.create(
            messages = [
            {'role': 'user', 'content': prompt}
            ],
            model = 'gpt-4o-mini',
            temperature = 0.01,
            max_tokens = 5000)

        # Extract the answer
        answer = completion.choices[0].message.content
        return answer

    except Exception as e:
        return f"An error occurred during recommendation generation: {e}"