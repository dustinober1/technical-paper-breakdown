import streamlit as st
import arxiv
import fitz  # pymupdf
import os
import re
import tempfile
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="ArXiv Research Distiller",
    page_icon="🎓",
    layout="wide"
)

# --- Caching ---
@st.cache_resource
def get_embeddings():
    """Load the Embedding model using local CPU."""
    # Explicitly using CPU for stability in local deployments
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def extract_arxiv_id(input_str):
    """Extract ArXiv ID from URL or return as is."""
    match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', input_str)
    if match:
        return match.group(1)
    return input_str

@st.cache_data(show_spinner=False)
def fetch_arxiv_data(arxiv_id):
    """Fetch ArXiv paper metadata and PDF using tempfile for safety."""
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            paper = next(search.results())
        except StopIteration:
            return None

        # Use a temporary file to avoid race conditions and filesystem clutter
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            paper.download_pdf(dirpath=os.path.dirname(temp_pdf.name), filename=os.path.basename(temp_pdf.name))
            temp_pdf_path = temp_pdf.name
        
        # Read into memory
        with open(temp_pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        # Clean up immediately
        os.remove(temp_pdf_path)

        return {
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "abstract": paper.summary,
            "pdf_bytes": pdf_bytes
        }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def process_pdf(pdf_bytes):
    """Extract text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_vectorstore(chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_distillation_prompt():
    template = """
    You are an expert technical researcher. Analyze the following research paper text and produce a structured study guide.
    
    The output MUST strictly follow this format:
    
    1. **The One-Liner**: A single sentence explaining what this paper solves.
    2. **Core Innovation**: Explain the specific delta/improvement over previous State-of-the-Art (SOTA).
    3. **Key Concepts**:
        - Concept 1: Definition
        - Concept 2: Definition
        - Concept 3: Definition
    4. **Study Guide Questions**:
        - Question 1
        - Question 2
        - Question 3
    
    Text:
    {text}
    """
    return PromptTemplate(template=template, input_variables=["text"])

def get_map_reduce_prompt():
    map_template = """
    The following is a section of a research paper:
    {text}
    
    Summarize this section, highlighting key innovations, definitions, and main points.
    """
    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

    reduce_template = """
    The following is a set of summaries from a research paper:
    {text}
    
    Synthesize these into a structured study guide following this format:
    
    1. **The One-Liner**: A single sentence explaining what this paper solves.
    2. **Core Innovation**: Explain the specific delta/improvement over previous State-of-the-Art (SOTA).
    3. **Key Concepts**:
        - Concept 1: Definition
        - Concept 2: Definition
    4. **Study Guide Questions**:
        - Question 1
        - Question 2
        - Question 3
    """
    reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["text"])
    
    return map_prompt, reduce_prompt

# --- Main App Logic ---

def main():
    # Sidebar
    st.sidebar.title("Configuration")
    model_name = st.sidebar.text_input("Ollama Model Name", value="llama3.2")
    arxiv_input = st.sidebar.text_input("ArXiv ID or URL", placeholder="e.g., 2310.06825")
    fetch_btn = st.sidebar.button("Fetch & Distill")
    
    st.sidebar.info("⚠️ Ensure Ollama is running locally (e.g., `ollama serve`).")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How it works")
    st.sidebar.markdown("1. Fetches paper from ArXiv.\n2. Extracts text via PyMuPDF.\n3. Uses local LLM to generate a study guide.\n4. Enables Chat Q&A via RAG.")

    # Main Area
    st.title("🎓 ArXiv Research Distiller")
    st.markdown("Bridge the gap between complex research and practical understanding.")

    # State Management
    if "paper_data" not in st.session_state:
        st.session_state.paper_data = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "distillation" not in st.session_state:
        st.session_state.distillation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if fetch_btn and arxiv_input:
        arxiv_id = extract_arxiv_id(arxiv_input)
        with st.spinner(f"Fetching paper {arxiv_id} from ArXiv..."):
            data = fetch_arxiv_data(arxiv_id)
            if data:
                st.session_state.paper_data = data
                st.session_state.messages = [] # Reset chat
                st.session_state.distillation = None # Reset previous summary
                st.success(f"Fetched: {data['title']}")
                
                # Process PDF
                with st.spinner("Extracting and processing text..."):
                    raw_text = process_pdf(data["pdf_bytes"])
                    
                    # Chunking strategy
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = text_splitter.create_documents([raw_text])
                    
                    # Create Vector Store
                    st.session_state.vectorstore = create_vectorstore(chunks)
                    
                    # Distillation
                    with st.spinner("Distilling content with LLM..."):
                        llm = Ollama(model=model_name)
                        
                        try:
                            if len(chunks) > 10:
                                map_prompt, reduce_prompt = get_map_reduce_prompt()
                                chain = load_summarize_chain(
                                    llm, 
                                    chain_type="map_reduce", 
                                    map_prompt=map_prompt, 
                                    combine_prompt=reduce_prompt,
                                    token_max=3000
                                )
                                distillation = chain.run(chunks)
                            else:
                                prompt = get_distillation_prompt()
                                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                                distillation = chain.run(chunks)
                            
                            st.session_state.distillation = distillation
                        except Exception as e:
                            st.error(f"Error during distillation: {e}")
            else:
                st.error("Could not fetch paper. Please check the ArXiv ID.")

    # Display Results
    if st.session_state.paper_data:
        st.markdown(f"## {st.session_state.paper_data['title']}")
        st.markdown(f"**Authors:** {', '.join(st.session_state.paper_data['authors'])}")
        
        with st.expander("Show Abstract"):
            st.write(st.session_state.paper_data['abstract'])
        
        if st.session_state.distillation:
            st.markdown("### 📝 Study Guide")
            st.markdown(st.session_state.distillation)
        
        st.markdown("---")
        st.markdown("### 💬 Chat with Paper")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question about the paper..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            if st.session_state.vectorstore:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Re-instantiating LLM is cheap, but we use the existing vectorstore
                            llm = Ollama(model=model_name)
                            qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=st.session_state.vectorstore.as_retriever(),
                            )
                            response = qa_chain.run(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
            else:
                st.warning("Please fetch a paper first.")

if __name__ == "__main__":
    main()
