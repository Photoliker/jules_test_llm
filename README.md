# ChatLLM with RAG Integration

## Description

This project is a web-based chat application that allows users to interact with a Large Language Model (LLM). It features streaming responses, chat history, Markdown rendering for LLM messages, and a Retrieval Augmented Generation (RAG) system using Qdrant as the vector database. Users can upload documents to the Qdrant database, and the application will retrieve relevant information from these documents to provide more contextually accurate answers from the LLM.

## Features

*   **Interactive Chat Interface:** Clean and modern UI for chatting.
*   **Streaming LLM Responses:** Messages from the LLM are displayed token by token in real-time.
*   **Markdown Rendering:** LLM responses are rendered as Markdown, allowing for formatted text, code blocks, etc.
*   **Chat History:** The conversation history is maintained and sent to the LLM for context.
*   **Retrieval Augmented Generation (RAG):**
    *   Uses **Qdrant** as a vector database.
    *   Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) for generating text embeddings.
    *   Endpoint (`/upload_document`) to upload text documents for indexing.
    *   Retrieves relevant document snippets to augment LLM prompts for more informed answers.
*   **Stop Generation:** A "Stop" button allows users to interrupt the LLM's response stream.
*   **FastAPI Backend:** Robust and asynchronous Python backend.
*   **Vanilla JavaScript Frontend:** Simple and efficient frontend without heavy frameworks.

## Setup and Installation

Follow these steps to set up and run the application:

**1. Python Virtual Environment:**

It's highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment (e.g., named .venv)
python3 -m venv .venv

# Activate the virtual environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

**2. Install Dependencies:**

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```
Key dependencies include `fastapi`, `uvicorn`, `qdrant-client`, and `sentence-transformers`. The latter two are crucial for the RAG functionality.

**⚠️ Important Note on RAG Dependencies:**
During development in the provided environment, a "No space left on device" error was encountered while installing dependencies, particularly `torch` (a sub-dependency of `sentence-transformers`). This prevented a full, clean installation of all RAG-related packages.
**If you encounter similar issues, the RAG features (document upload and context retrieval) may not function correctly or at all.** Ensure your environment has sufficient disk space and can successfully install all packages in `requirements.txt`.

**3. Set Up Qdrant Vector Database:**

The RAG system requires a running Qdrant instance. The application is configured to connect to Qdrant at `localhost:6333`.

The easiest way to run Qdrant is using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
This command mounts a local directory (`qdrant_storage`) for persistent storage.

**4. LLM Server:**

This application requires a separate LLM server that is compatible with the OpenAI API format for chat completions.
*   The application is configured in `main.py` to connect to an LLM API at `LLM_API_URL = "http://localhost:1234/v1/chat/completions"`.
*   The default model is `LLM_MODEL_NAME = "gemma-3-27b-it"`.

You can use tools like **LM Studio** or **Ollama (with an OpenAI-compatible proxy/adapter)** to serve a local LLM. Ensure the model specified in `LLM_MODEL_NAME` is loaded and accessible through the server's OpenAI-compatible endpoint.

## Running the Application

Once the setup is complete:

1.  Ensure your Python virtual environment is activated.
2.  Ensure your Qdrant instance is running.
3.  Ensure your LLM server is running and configured.
4.  Run the FastAPI application using Uvicorn:

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reloading on code changes (for development).
    *   `--host 0.0.0.0`: Makes the server accessible from your local network.
    *   `--port 8000`: Specifies the port. You can change this if needed.

## How to Use

**1. Access the Web Interface:**

Open your web browser and navigate to `http://localhost:8000` (or the port you configured).

**2. Chatting:**

*   Type your message in the input box at the bottom of the chat interface.
*   Press Enter or click the "Send" button.
*   The LLM's response will be streamed to the chat display.

**3. Stop Generation:**

*   If the LLM is generating a long response, you can click the "Stop" button to interrupt it.

**4. Uploading Documents (for RAG):**

If the RAG dependencies (`qdrant-client`, `sentence-transformers`, and their sub-dependencies like `torch`) were installed successfully and Qdrant is running, you can upload documents to be used for context retrieval.

Use a tool like `curl` or Postman to send a POST request to the `/upload_document` endpoint.

**Example using `curl`:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text_content": "Artificial intelligence (AI) is rapidly transforming various industries. Machine learning, a subset of AI, involves training algorithms on data to make predictions or decisions."}' \
     http://localhost:8000/upload_document
```

Response:
```json
{
  "message": "Document uploaded and indexed successfully.",
  "doc_id": "your-unique-document-id"
}
```

Once documents are uploaded, the chat system will automatically try to find relevant snippets from these documents based on your message and provide them as context to the LLM.

## File Structure

*   `main.py`: The FastAPI backend application. Contains all API endpoints, RAG logic, LLM interaction, and Qdrant setup.
*   `templates/index.html`: The main HTML file for the chat interface. Includes CSS for styling and JavaScript for frontend logic.
*   `static/marked.min.js`: The `marked.js` library used for rendering Markdown in the chat display. (Note: This file is assumed to be provided or obtained separately).
*   `requirements.txt`: Lists the Python dependencies for the project.
*   `README.md`: This file, providing documentation for the project.
*   `qdrant_storage/` (Optional, created by Docker): If using the Docker command above, this directory stores Qdrant data.

---
This `README.md` provides a comprehensive guide for setting up, running, and using the application.
