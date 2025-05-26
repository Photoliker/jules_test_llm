"""
main.py
FastAPI backend for a Chat application with RAG (Retrieval Augmented Generation) capabilities.

Features:
- Web interface for chat.
- Streaming responses from LLM.
- Chat history management.
- RAG using Qdrant vector database and Sentence Transformers for embeddings.
- Document upload endpoint for populating the RAG database.
- Stop generation button functionality.
"""

import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import aiohttp # For asynchronous HTTP requests to the LLM API
import json
import uuid # For generating unique IDs for documents in Qdrant

# RAG specific imports
from qdrant_client import QdrantClient, models # Qdrant client and data models
from sentence_transformers import SentenceTransformer # For generating text embeddings

# --- Application Constants ---

# LLM Configuration
LLM_MODEL_NAME = "gemma-3-27b-it"
"""Name of the Large Language Model to be used (must match a model available in the LLM server)."""
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
"""API endpoint for the LLM server (e.g., LM Studio OpenAI-compatible server)."""

# Qdrant (Vector Database) Configuration for RAG
QDRANT_HOST = "localhost" 
"""Hostname for the Qdrant instance."""
QDRANT_PORT = 6333
"""Port number for the Qdrant instance."""
QDRANT_COLLECTION_NAME = "my_documents"
"""Name of the collection in Qdrant to store document vectors."""
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
"""Name of the Sentence Transformer model used for generating text embeddings.
This model produces 384-dimensional vectors."""
VECTOR_SIZE = 384 
"""Dimensionality of the vectors produced by the EMBEDDING_MODEL_NAME."""

# Initialize FastAPI app
app = FastAPI(
    title="ChatLLM with RAG",
    description="A FastAPI application for chatting with an LLM, augmented by a Qdrant-based RAG system.",
    version="1.0.0"
)

# --- Initialize RAG Components (Qdrant Client and Embedding Model) ---

qdrant_client: QdrantClient | None = None
"""Global Qdrant client instance. Initialized on startup."""
embedding_model: SentenceTransformer | None = None
"""Global Sentence Transformer model instance. Initialized on startup."""

try:
    # Attempt to initialize Qdrant client
    # This might not immediately fail if Qdrant server is down, but will fail on first operation.
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    print(f"Qdrant client initialized successfully for host {QDRANT_HOST}:{QDRANT_PORT}.")
except Exception as e:
    print(f"CRITICAL: Error initializing Qdrant client: {e}. RAG features will be UNAVAILABLE.")
    # qdrant_client remains None, subsequent checks will prevent RAG operations.

try:
    # Attempt to initialize sentence transformer model
    # This may download the model from Hugging Face Hub if not cached locally.
    # Requires internet access and sufficient disk space on first run.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"SentenceTransformer model ('{EMBEDDING_MODEL_NAME}') loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Error initializing SentenceTransformer model ('{EMBEDDING_MODEL_NAME}'): {e}. RAG features will be UNAVAILABLE.")
    # embedding_model remains None.

# --- FastAPI Startup Event: Setup Qdrant Collection ---
@app.on_event("startup")
async def startup_event():
    """
    FastAPI startup event handler.
    Ensures the Qdrant collection for RAG is created if it doesn't exist.
    This is crucial for the RAG functionality.
    """
    global qdrant_client, embedding_model # Access global instances

    # Check if RAG components are initialized. If not, RAG cannot function.
    if not qdrant_client:
        print("Startup: Qdrant client not initialized. Skipping Qdrant collection setup.")
        return
    if not embedding_model:
        # VECTOR_SIZE is derived from the embedding model, so it's critical.
        print("Startup: Embedding model not initialized. Skipping Qdrant collection setup as VECTOR_SIZE might be incorrect.")
        return
        
    try:
        # Attempt to get collection details. If it doesn't exist, an exception is raised.
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Startup: Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")
    except Exception: # Catches exceptions typically meaning the collection doesn't exist
        print(f"Startup: Qdrant collection '{QDRANT_COLLECTION_NAME}' not found. Attempting to create it.")
        try:
            # Create the collection with specified vector parameters.
            # Using COSINE distance for semantic similarity search.
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)
            )
            print(f"Startup: Qdrant collection '{QDRANT_COLLECTION_NAME}' created successfully with vector size {VECTOR_SIZE}.")
        except Exception as creation_error:
            # Log critical error if collection creation fails, as RAG will not work.
            print(f"CRITICAL: Failed to create Qdrant collection '{QDRANT_COLLECTION_NAME}': {creation_error}. RAG will be non-functional.")


# --- Static Files and Templates ---
# Mount the 'static' directory to serve static files (like CSS, JS, images) under '/static' path.
app.mount("/static", StaticFiles(directory="static"), name="static")
# Initialize Jinja2 templates to render HTML from the 'templates' directory.
templates = Jinja2Templates(directory="templates")

# --- Pydantic Models for Request/Response Validation ---
class ChatRequest(BaseModel):
    """
    Pydantic model for the request body of the /chat endpoint.
    Ensures type validation for incoming chat requests.
    """
    message: str # The new message from the user.
    history: list[dict] # The previous chat history, list of {'role': 'user'/'assistant', 'content': 'message'}.

class DocumentUploadRequest(BaseModel):
    """
    Pydantic model for the request body of the /upload_document endpoint.
    Used for uploading text content to be indexed by RAG.
    """
    text_content: str # The raw text content of the document to be uploaded.

# --- LLM Interaction Helper ---
async def llm_request_stream(payload: dict):
    """
    Asynchronously sends a request to the LLM API and streams the response.

    Args:
        payload (dict): The payload to send to the LLM API, including model name, messages, and stream flag.

    Yields:
        str: Chunks of content from the LLM's streamed response.

    Raises:
        HTTPException: If the LLM API request fails (e.g., non-200 status code).
    """
    try:
        async with aiohttp.ClientSession() as session:
            # Make an asynchronous POST request with streaming content.
            async with session.post(LLM_API_URL, json=payload, stream_content=True) as response:
                if response.status != 200:
                    # If LLM server returns an error, read the error detail and raise HTTPException.
                    error_detail = await response.text()
                    print(f"LLM API Error ({response.status}): {error_detail}") # Server-side logging
                    raise HTTPException(status_code=response.status, detail=f"LLM API request failed: {error_detail}")
                
                # Iterate over the response content chunks asynchronously.
                async for chunk in response.content.iter_any(): # iter_any() reads whatever is available
                    if chunk:
                        chunk_str = chunk.decode('utf-8').strip()
                        # LLM stream often sends data in "data: {...}" format or "[DONE]" sentinel.
                        if chunk_str.startswith("data: "):
                            chunk_str = chunk_str[len("data: "):].strip() # Remove "data: " prefix
                        
                        if chunk_str == "[DONE]": # End of stream signal from some LLM servers
                            continue
                        
                        if not chunk_str: # Skip empty chunks after stripping
                            continue

                        try:
                            # Each chunk is expected to be a JSON object containing delta content.
                            json_chunk = json.loads(chunk_str)
                            # Extract the actual text content from the chunk.
                            # Path may vary based on LLM server: (OpenAI-like) choices[0].delta.content
                            content = json_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content: # Only yield if there's actual content
                                # print(f"LLM Chunk Content: '{content}'") # Debugging: log received content
                                yield content
                        except json.JSONDecodeError:
                            # Log if a chunk is not valid JSON (might happen with some streams or errors).
                            print(f"Warning: Failed to decode JSON chunk: '{chunk_str}'")
                            continue # Skip malformed chunks
    except aiohttp.ClientConnectorError as e:
        # Handle network issues connecting to the LLM API (e.g., server not running).
        print(f"Critical: Cannot connect to LLM API at {LLM_API_URL}: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: Could not connect to LLM API. {e}")
    except Exception as e:
        # Catch any other unexpected errors during LLM communication.
        print(f"Critical: Unexpected error in llm_request_stream: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Unexpected issue during LLM communication. {e}")


# --- FastAPI HTTP Endpoints ---

@app.post("/upload_document", summary="Upload a document for RAG", status_code=200)
async def upload_document(doc_request: DocumentUploadRequest):
    """
    Endpoint to upload and index a text document for the RAG system.
    The document text is provided in the request body as JSON.

    Request Body:
        - `text_content` (str): The raw text of the document.

    Response:
        - JSON message confirming success and providing the document ID.
        - HTTP 503 if RAG components (Qdrant/Embedding Model) are not available.
        - HTTP 400 if document content is empty.
        - HTTP 500 if any internal error occurs during processing or storage.
    """
    # Check if RAG system components are functional.
    if not qdrant_client or not embedding_model:
        print("Error: /upload_document called but RAG components are not available.")
        raise HTTPException(status_code=503, detail="RAG system is currently unavailable. Please check server logs.")

    text_content = doc_request.text_content
    if not text_content.strip(): # Basic validation for empty content.
        raise HTTPException(status_code=400, detail="Document content cannot be empty.")

    try:
        # Generate embedding for the document content using the pre-loaded SentenceTransformer model.
        # Note: For very large documents, implementing a chunking strategy before embedding
        # is recommended to stay within token limits and improve retrieval relevance.
        # This example assumes documents are small enough to be processed whole.
        print(f"Generating embedding for document content (length: {len(text_content)} chars)...")
        doc_vector = embedding_model.encode(text_content).tolist()
        print("Embedding generated successfully.")

        # Create a unique ID for this document point in Qdrant.
        point_id = str(uuid.uuid4())

        # Upsert (insert or update) the document vector and payload into Qdrant.
        # The payload stores the original text, allowing it to be retrieved during search.
        print(f"Upserting document to Qdrant with ID: {point_id}...")
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=doc_vector,
                    payload={"text": text_content, "source_filename": "text_upload"} # Example metadata
                )
            ]
        )
        print("Document upserted to Qdrant successfully.")
        return JSONResponse(content={"message": "Document uploaded and indexed successfully.", "doc_id": point_id}, status_code=200)
    except Exception as e:
        # Log the full error for server-side debugging.
        print(f"Error during document upload and indexing: {e}")
        # Return a generic error message to the client.
        raise HTTPException(status_code=500, detail=f"Failed to process and store document due to an internal error: {str(e)}")


@app.post("/chat", summary="Handle chat messages with optional RAG context")
async def chat(chat_request: ChatRequest):
    """
    Main endpoint for handling chat messages.
    It takes the user's message and chat history, optionally retrieves relevant
    context from Qdrant (RAG), then streams the LLM's response.

    Request Body:
        - `message` (str): The user's current message.
        - `history` (list[dict]): A list of previous messages in the format `{"role": "user/assistant", "content": "..."}`.

    Response:
        - `StreamingResponse`: A stream of text chunks from the LLM.
        - HTTP 503 if LLM API is unavailable.
    """
    user_message = chat_request.message
    context_for_llm = "" # Initialize empty context string

    # --- RAG Context Retrieval ---
    # Only attempt RAG if Qdrant client and embedding model are available.
    if qdrant_client and embedding_model:
        print(f"RAG: Processing message '{user_message[:50]}...' for context retrieval.")
        try:
            # 1. Generate embedding for the user's current message.
            query_vector = embedding_model.encode(user_message).tolist()
            print("RAG: Query vector generated.")

            # 2. Search Qdrant for documents semantically similar to the user's message.
            # `limit=3` retrieves the top 3 most relevant documents.
            search_results = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=3 
            )
            print(f"RAG: Found {len(search_results)} relevant documents in Qdrant.")

            # 3. Construct the context string from the payloads of retrieved documents.
            if search_results:
                retrieved_texts = [
                    result.payload['text'] for result in search_results 
                    if result.payload and 'text' in result.payload
                ]
                if retrieved_texts:
                    context_for_llm = "Relevant information found in documents:\n" + "\n---\n".join(retrieved_texts)
                    print(f"RAG: Context compiled for LLM:\n{context_for_llm[:200]}...") # Log snippet of context
        except Exception as e:
            # Log RAG retrieval errors but don't halt the chat; proceed without context.
            print(f"Error during RAG retrieval: {e}. Proceeding without RAG context.")
            context_for_llm = "System Note: Error occurred while retrieving relevant documents. Results may be less accurate."
    else:
        # Log if RAG components are unavailable.
        print("RAG components (Qdrant client or Embedding model) not available. Proceeding without RAG context.")
        # Optionally, inform the user if RAG is down, or proceed silently.
        # context_for_llm = "System Note: RAG system is currently unavailable."

    # --- Prepare Messages for LLM ---
    # Start with the existing chat history.
    llm_messages = list(chat_request.history) # Ensure it's a mutable list

    # Prepend RAG context to the user's current message if context was found.
    # This helps the LLM answer based on the retrieved documents.
    if context_for_llm:
        # Option: Add context as a system message (less direct, depends on LLM's handling of system prompts)
        # llm_messages.append({"role": "system", "content": f"Use the following context to answer the user's question:\n{context_for_llm}"})
        # llm_messages.append({"role": "user", "content": user_message})
        
        # Option: Prepend context directly to the user's current message (more direct)
        enhanced_user_message = f"Based on the following information:\n{context_for_llm}\n\nUser question: {user_message}"
        llm_messages.append({"role": "user", "content": enhanced_user_message})
    else:
        # No RAG context, just add the plain user message.
        llm_messages.append({"role": "user", "content": user_message})
    
    # --- LLM API Request ---
    # Prepare the full payload for the LLM API.
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": llm_messages, # Includes history, (RAG-enhanced) user message
        "stream": True # Enable streaming response
    }
    print(f"Sending payload to LLM: {json.dumps(payload, indent=2)[:500]}...") # Log snippet of payload

    # Return a streaming response by calling the llm_request_stream generator.
    # media_type="text/event-stream" is common for Server-Sent Events (SSE).
    return StreamingResponse(llm_request_stream(payload), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse, summary="Serve the main chat HTML page")
async def read_root(request: Request):
    """
    Serves the main HTML page for the chat application.
    Uses Jinja2 to render `index.html` from the `templates` directory.

    Args:
        request (Request): The incoming FastAPI request object.

    Returns:
        HTMLResponse: The rendered HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

# --- Main Execution Guard ---
if __name__ == "__main__":
    # This block runs when the script is executed directly (e.g., `python main.py`).
    # Starts the Uvicorn ASGI server to serve the FastAPI application.
    # Host "0.0.0.0" makes the server accessible on the network.
    # Port 8000 is the standard development port.
    # reload=True enables auto-reloading on code changes (useful for development).
    print("Starting Uvicorn server for FastAPI application...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
