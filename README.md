# ArXiv Research Distiller 🎓

**Bridge the gap between complex academic research and practical understanding.**

The **ArXiv Research Distiller** is a Python-based tool designed for developers, data scientists, and researchers. It ingests technical papers from [arXiv.org](https://arxiv.org/), parses the content, and uses local Large Language Models (LLMs) to synthesize clear, structured "Micro-Lessons."

## ✨ Features

*   **📄 Seamless Ingestion**: Fetch papers directly via ArXiv ID (e.g., `2310.06825`) or URL.
*   **🤖 AI-Powered Distillation**: Automatically generates a structured study guide containing:
    *   **The One-Liner**: A single sentence summary of the problem and solution.
    *   **Core Innovation**: The specific delta over previous state-of-the-art.
    *   **Key Concepts**: Definitions of technical terms used in the paper.
    *   **Study Guide**: Generated questions to test your understanding.
*   **💬 Chat with Paper**: A Retrieval-Augmented Generation (RAG) interface to ask specific follow-up questions about the paper's methodologies, hyperparameters, or results.
*   **🔒 Local & Private**: Runs entirely locally using [Ollama](https://ollama.com/), keeping your data private and free of API costs.

## 🛠️ Technical Stack

*   **Framework**: [Streamlit](https://streamlit.io/)
*   **LLM Orchestration**: [LangChain](https://www.langchain.com/)
*   **LLM Backend**: [Ollama](https://ollama.com/) (running locally)
*   **PDF Parsing**: [PyMuPDF](https://pymupdf.readthedocs.io/)
*   **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
*   **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)

## 🚀 Getting Started

### Prerequisites

1.  **Python 3.8+** installed.
2.  **[Ollama](https://ollama.com/download)** installed and running.
    *   Pull the default model (or any other model you prefer):
        ```bash
        ollama pull llama3.2
        ```
    *   Ensure the Ollama server is running (usually `ollama serve`).

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/arxiv-research-distiller.git
    cd arxiv-research-distiller
    ```

2.  Install python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser to the URL provided (usually `http://localhost:8501`).

3.  **In the Sidebar**:
    *   Enter the name of your local Ollama model (default: `llama3.2`).
    *   Paste an ArXiv ID (e.g., `1706.03762`) or full URL.
    *   Click **Fetch & Distill**.

4.  **Explore**: Read the generated study guide and use the chat interface at the bottom to dive deeper.

## ⚠️ Notes

*   **Performance**: The speed of distillation depends on your local hardware (CPU/GPU) and the size of the model used.
*   **Chunking**: Long papers are automatically split into chunks. Extremely long papers may take longer to process via the Map-Reduce summarization strategy.
