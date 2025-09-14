from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage

def ask_document_question(question: str) -> str:
    """
    Answers a question by retrieving information from the local ChromaDB
    and generating a response using the Gemini LLM.

    Args:
        question: The user's question about the document.

    Returns:
        A generated answer based on the document content.
    """
    print(f"Tool received question: '{question}'")

    # --- Initialize components using Google Gemini ---
    # The API key will be read from the GOOGLE_API_KEY environment variable by default.
    # You can also set it explicitly: genai.configure(api_key="YOUR_API_KEY")
    
    # Initialize Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Initialize the Gemini Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.01)
    
    # Load the existing ChromaDB from the persistence directory
    # Note: No documents are added here; it just connects to the existing collection.
    db = Chroma(persist_directory="./chroma_db_oa", embedding_function=embeddings)
    print("Connected to pre-existing ChromaDB.")

    # Step 4: Retrieve relevant documents from ChromaDB
    retrieved_docs = db.similarity_search(question, k=10)

    # Prepare the context for the LLM
    formatted_context = "\n\n".join([f"Content:\n{doc.page_content}" for doc in retrieved_docs])

    # Step 5: Construct the query prompt
    prompt_text = f"""
    ## INTRODUCTION
    You are a Chatbot designed to help answer technical questions about plant diseases based on the symptoms or specific plant asks.

    ## ROLE
    You are a symptom analysis specialist. Your primary role is to diagnose plant diseases and recommend management strategies based on a user's description of symptoms. You will use the provided document context to find relevant information.

    ## CONTEXT
    Technical Documentation:
    '''
    {formatted_context}
    '''

    ## USER QUERY
    The user has described the symptoms or asked this question:{question}

    ## TASK
    Based *only* on the provided context:
    1.  Identify the potential plant disease(s) that match the described symptoms.
    2.  If the disease is identified, list its key characteristics and symptoms as described in the text.
    3.  If available, provide management or control recommendations.
    4.  If the context does not contain relevant information, state that the information is not available in the provided documentation.
    5. answer in Markdown format.

    ## RESTRICTIONS
    * Do not use any outside knowledge. Your entire response must be derived from the provided 'Technical Documentation'.
    * Present your findings in a structured, easy-to-read format.
    Refer to the plant-specific diseases by their names.
    * Be clear, transparent, and factual: only state what is in the context without providing opinions or subjectivity.
    * Answer the question based solely on the context above; if you do not know the answer, be clear with the user that you do not know.

    ## RESPONSE STRUCTURE
    '''
    ### [Disease Name or Diagnosis Title]

    **Symptoms:**
    - [Symptom 1]
    - [Symptom 2]
    ...

    **Management/Recommendations:**
    - [Recommendation 1]
    - [Recommendation 2]
    ...

    Source:
    • From The Handbook of Plant Disease Identification and Management
    '''
    """

    # Step 6: Call the LLM to generate the answer
    try:
        # Pass the formatted prompt to the Gemini LLM
        messages = [HumanMessage(content=prompt_text)]
        completion = llm.invoke(messages)
        answer = completion.content
        return answer
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"