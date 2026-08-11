# Multi-PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, and **Cohere's LLMs and Embeddings**. This project allows users to upload multiple PDF documents (such as CVs, books, or technical articles) and ask intelligent, context-aware questions based on the content of those documents.

## Features

- **Multi-Document Support:** Query a single PDF (like a CV), a large book, or multiple documents simultaneously.
- **Advanced Retrieval:** Uses semantic search combined with **Cohere Rerank** (`rerank-multilingual-v3.0`) for highly accurate, hybrid-like retrieval.
- **Conversational Memory:** Chatbot remembers the history of the conversation to handle follow-up questions effectively.
- **Source Attribution:** Displays the source PDF filename and page number for every generated answer.
- **Interactive UI:** Features a sleek **Streamlit** chat interface with a sidebar to select/filter which PDFs to search and to adjust the retrieval parameter (`k`).
- **Kaggle Compatible:** Includes a built-in automated workflow using **Cloudflare Tunnels** to run the Streamlit interface directly from a Kaggle Notebook without port-forwarding issues.

## Technologies Used

- **Python**
- **LangChain** (Core, Community, and Cohere integrations)
- **Cohere API** (`embed-v4.0` for embeddings, `command-a-03-2025` for generation, and `rerank-multilingual-v3.0` for reranking)
- **ChromaDB** (Vector Database)
- **Streamlit** (Frontend UI)
- **PyPDFLoader** & **RecursiveCharacterTextSplitter** (Document processing)

## Repository Structure

- `notebooks/`: Contains the separated Jupyter Notebooks for each task.
  - `01_Task_CV_RAG.ipynb`: Testing RAG on a personal CV.
  - `02_Task_Book_RAG.ipynb`: Testing RAG on a full book with chunking size experiments.
  - `03_Task_Multi_PDF_RAG.ipynb`: Multi-PDF RAG pipeline.
- `gui/`: Contains the interactive web application.
  - `app.py`: The complete Streamlit web application script.
- `rag-system.ipynb`: The complete (original) notebook combining all tasks.
- `requirements.txt`: List of required Python packages to run the project.
- `Multi_PDF_RAG_Lab.docx`: The original lab instructions and requirements.

## How to Run (Local)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -U streamlit langchain langchain-cohere langchain-chroma pypdf chromadb
   ```

3. **Set your Cohere API Key:**
   ```bash
   export COHERE_API_KEY="your-api-key-here"
   ```
   *(Alternatively, you can input the key directly into the Streamlit sidebar).*

4. **Run the Streamlit App:**
   ```bash
   python -m streamlit run gui/app.py
   ```

## How to Run (Kaggle)

If you are running this project inside a Kaggle Notebook:

1. Upload the `gui/app.py` script or use the `%%writefile app.py` magic command in a cell.
2. Ensure your API key is saved in **Kaggle Secrets** as `coherekey`.
3. Run the Cloudflare Tunnel cell provided at the end of the `rag-system.ipynb` notebook.
4. Click the generated `trycloudflare.com` link to access your live Streamlit dashboard!

## Chunking Experiment Notes

In this project, different chunk sizes were tested to optimize context retrieval:
- **Books:** A larger `chunk_size` (e.g., 2000) was used to maintain the context of long explanations and paragraphs.
- **Technical PDFs / CVs:** A smaller `chunk_size` (e.g., 500 or 800) was used with a suitable overlap to retrieve highly specific definitions and bullet points accurately.
