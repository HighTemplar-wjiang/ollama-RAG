import os
# Set CUDA_VISIBLE_DEVICES BEFORE any torch imports
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.docstore.document import Document
import ollama
import torch 

# --- Global variables / Session State ---
# Initial greeting message
INITIAL_ASSISTANT_MESSAGE = "Hello! Upload your PDFs and I'll help you query them after checking model availability. You can also restart our chat using the button in the sidebar."

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE}]
if "ollama_model_available" not in st.session_state:
    st.session_state.ollama_model_available = False

# --- Constants ---
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 3000 # Adjust as needed
CHUNK_OVERLAP = 500 # Adjust as needed
RETRIEVER_K = 5 # Number of chunks to retrieve, adjust as needed
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL_NAME = "llama3.3:70b" # Default model name, can be any open-source model
LLM_TEMPERATURE = 0.1 # Temperature for LLM generation

# --- Caching ---
@st.cache_resource
def get_embedding_model():
    st.write(f"Attempting to load embedding model '{EMBEDDING_MODEL_NAME}' on designated GPU (CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}')")
    needs_trust_remote_code = "bge-" in EMBEDDING_MODEL_NAME.lower() or "instructor-" in EMBEDDING_MODEL_NAME.lower() 
    if needs_trust_remote_code:
        st.write(f"Using trust_remote_code=True for model loading (heuristic: model name contains 'bge-' or 'instructor-').")
    
    model_kwargs_dict = {'device': 'cpu'} 
    if torch.cuda.is_available():
        model_kwargs_dict['device'] = 'cuda'
        st.write(f"PyTorch CUDA is available. Current device (as seen by PyTorch): {torch.cuda.get_device_name(0)}")
    else:
        st.warning("PyTorch CUDA not available, embedding model will run on CPU.")
    
    if needs_trust_remote_code:
        model_kwargs_dict['trust_remote_code'] = True
        
    try:
        model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs_dict)
        st.write(f"Embedding model loaded. Target device: {model_kwargs_dict['device']}")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.exception(e)
        return None

# --- Ollama Model Check ---
def check_ollama_model_availability():
    if st.session_state.get("ollama_model_checked_this_session", False): return st.session_state.ollama_model_available
    st.sidebar.write(f"Checking availability of Ollama model: `{LLM_MODEL_NAME}`...")
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        list_response = client.list()
        available_model_tags = [model_obj.model for model_obj in list_response.models if hasattr(model_obj, 'model')] if hasattr(list_response, 'models') and isinstance(list_response.models, list) else []
        if LLM_MODEL_NAME in available_model_tags:
            st.sidebar.success(f"Ollama model '{LLM_MODEL_NAME}' is available.")
            st.session_state.ollama_model_available = True
        else:
            st.sidebar.error(f"Ollama model '{LLM_MODEL_NAME}' not found!")
            st.sidebar.info(f"Available models: {available_model_tags}")
            st.session_state.ollama_model_available = False
    except Exception as e:
        st.sidebar.error(f"Error connecting to Ollama or listing models: {e}")
        st.session_state.ollama_model_available = False
    st.session_state.ollama_model_checked_this_session = True
    return st.session_state.ollama_model_available

# --- PDF Processing Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    new_files_processed_this_run = []
    for pdf in pdf_docs:
        if pdf.name not in st.session_state.processed_files:
            try:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text: text += page_text
                new_files_processed_this_run.append(pdf.name)
                st.session_state.processed_files.add(pdf.name)
            except Exception as e: st.sidebar.error(f"Error reading {pdf.name}: {e}")
        else: st.sidebar.info(f"'{pdf.name}' already processed.")
    if new_files_processed_this_run: st.sidebar.write(f"Extracted text from: {', '.join(new_files_processed_this_run)}")
    return text

def get_text_chunks(text, chunk_size_val, chunk_overlap_val):
    if not text: return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val, length_function=len)
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def update_vector_store(text_chunks, embeddings_model_instance):
    if not text_chunks: return st.session_state.vector_store
    if embeddings_model_instance is None:
        st.error("Embedding model is not available. Cannot update vector store.")
        return st.session_state.vector_store
    try:
        with st.spinner(f"Embedding {len(text_chunks)} chunks and updating vector store..."):
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISS.from_documents(text_chunks, embedding=embeddings_model_instance)
                st.sidebar.success("New vector store created.")
            else:
                st.session_state.vector_store.add_documents(text_chunks)
                st.sidebar.success("New documents added to existing vector store.")
    except Exception as e:
        st.error(f"Error updating FAISS vector store: {e}")
        st.exception(e)
    return st.session_state.vector_store

# --- LLM Interaction Functions ---
def create_rag_prompt(query, context_chunks):
    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the following context.
If the information is not in the context, say "I don't have enough information in the provided documents to answer that."
Do not make up information or use external knowledge.

Context:
{context_str}

User Question: {query}

Answer:"""
    return prompt

def query_llm_stream(prompt_with_context, model_name=LLM_MODEL_NAME):
    if not st.session_state.ollama_model_available:
        yield "Ollama model is not available. Please check the sidebar for status."
        return
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        llm_options = {"temperature": st.session_state.get('llm_temperature', LLM_TEMPERATURE)}
        stream = client.chat(model=model_name, messages=[{'role': 'user', 'content': prompt_with_context}], stream=True, options=llm_options)
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
            if chunk.get('done', False) and chunk.get('error'):
                st.error(f"Error from Ollama stream: {chunk['error']}")
                break
    except ollama.ResponseError as e:
        error_message = f"Ollama API Error: {getattr(e, 'error', 'Unknown error')} (Status code: {getattr(e, 'status_code', 'N/A')})"
        st.error(error_message)
        yield f"[Ollama Error: {error_message}]"
    except Exception as e:
        st.error(f"Error connecting to Ollama or querying model: {e}")
        st.exception(e)
        yield f"[Connection/Query Error: {e}]"

# --- Streamlit App Layout ---
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("Chat with Your PDFs using Open-Source LLMs 🧠 (RAG Streaming)")

if not st.session_state.get("ollama_model_checked_this_session", False):
    check_ollama_model_availability()

with st.sidebar:
    st.header("Controls & Settings ⚙️") # Changed header for grouping

    # Chat Controls
    st.subheader("Chat Controls")
    if st.button("Restart Chat", key="restart_chat_button", help="Clears the current conversation history."):
        st.session_state.messages = [{"role": "assistant", "content": "Chat restarted. How can I help you with your documents?"}]
        st.rerun() # Rerun to refresh the chat display immediately

    if st.button("Clear Knowledge Base & Chat", key="clear_button", help="Clears documents and conversation."):
        st.session_state.vector_store = None
        st.session_state.processed_files = set()
        st.session_state.messages = [{"role": "assistant", "content": "Knowledge base and chat cleared. Upload new PDFs to begin."}]
        st.rerun()
    st.divider()

    # RAG Settings
    st.subheader("RAG Settings")
    if 'chunk_size' not in st.session_state: st.session_state.chunk_size = CHUNK_SIZE
    if 'chunk_overlap' not in st.session_state: st.session_state.chunk_overlap = CHUNK_OVERLAP
    if 'retriever_k' not in st.session_state: st.session_state.retriever_k = RETRIEVER_K
    
    st.session_state.chunk_size = st.number_input("Chunk Size (chars)", key="sb_chunk_size", min_value=100, max_value=5000, value=st.session_state.chunk_size, step=100)
    st.session_state.chunk_overlap = st.number_input("Chunk Overlap (chars)", key="sb_chunk_overlap", min_value=0, max_value=1000, value=st.session_state.chunk_overlap, step=50)
    st.session_state.retriever_k = st.number_input("Chunks to Retrieve (k_retriever)", key="sb_retriever_k", min_value=1, max_value=20, value=st.session_state.retriever_k, step=1)
    
    # LLM Settings
    st.subheader("LLM Settings")
    if 'llm_temperature' not in st.session_state: st.session_state.llm_temperature = LLM_TEMPERATURE
    if 'use_top_k_for_llm' not in st.session_state: st.session_state.use_top_k_for_llm = 1 
    if 'enable_llm_top_k_test' not in st.session_state: st.session_state.enable_llm_top_k_test = True 

    st.session_state.llm_temperature = st.slider("LLM Temperature", min_value=0.0, max_value=2.0, value=st.session_state.llm_temperature, step=0.1)
    st.session_state.enable_llm_top_k_test = st.checkbox("TEST: Use only N retrieved chunk(s) for LLM Prompt", value=st.session_state.enable_llm_top_k_test, key="cb_enable_llm_top_k")
    if st.session_state.enable_llm_top_k_test:
        st.session_state.use_top_k_for_llm = st.number_input("N (chunks for LLM prompt if test enabled)", min_value=1, max_value=st.session_state.retriever_k, value=st.session_state.use_top_k_for_llm, step=1, key="ni_use_top_k_for_llm")

    st.divider()
    st.header("File Upload 📤")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process Uploaded Files", key="process_button"):
            # Check if chunk settings changed, which might imply need for user to understand reprocessing context
            # This simple check doesn't force re-clearing, just informs for new docs.
            if st.session_state.chunk_size != CHUNK_SIZE or st.session_state.chunk_overlap != CHUNK_OVERLAP:
                 st.sidebar.caption("Note: Chunk size/overlap for new docs is based on current RAG settings.")
            
            embeddings_model_instance = get_embedding_model() 
            if embeddings_model_instance:
                raw_text_from_new_files = get_pdf_text(uploaded_files)
                if raw_text_from_new_files:
                    text_chunks = get_text_chunks(raw_text_from_new_files, st.session_state.chunk_size, st.session_state.chunk_overlap)
                    if text_chunks: update_vector_store(text_chunks, embeddings_model_instance)
                    else: st.warning("No text could be extracted or chunked from new files.")
                else: st.info("No new files to process or text found in new files.")
            else: st.error("Cannot process files: Embedding model failed to load.")
    
    # Display status of knowledge base
    if st.session_state.vector_store:
        st.sidebar.success("Knowledge base is ready!")
        st.sidebar.write(f"Total unique files processed: {len(st.session_state.processed_files)}")
    elif uploaded_files and not st.session_state.vector_store : # If tried to process but failed
        st.sidebar.warning("Knowledge base processing may have encountered issues or is empty.")
    else: # Default state
        st.sidebar.info("Upload PDF files to build the knowledge base.")


if not st.session_state.ollama_model_available:
    st.error(f"Configured Ollama model '{LLM_MODEL_NAME}' is not available. Chat functionality will be impaired.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        full_response_content = "" 
        if not st.session_state.ollama_model_available:
            full_response_content = f"Cannot respond: Ollama model '{LLM_MODEL_NAME}' is not available."
            st.markdown(full_response_content)
        elif st.session_state.vector_store is not None:
            embeddings_model_instance = get_embedding_model()
            if embeddings_model_instance is None:
                full_response_content = "Cannot process query: Embedding model is not loaded."
                st.markdown(full_response_content)
            else:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": st.session_state.retriever_k})
                try:
                    all_relevant_docs = retriever.invoke(user_prompt) 
                    final_context_chunks_for_llm = []
                    if all_relevant_docs:
                        if st.session_state.enable_llm_top_k_test:
                            num_chunks_for_llm = min(st.session_state.use_top_k_for_llm, len(all_relevant_docs))
                            final_context_chunks_for_llm = [doc.page_content for doc in all_relevant_docs[:num_chunks_for_llm]]
                            st.info(f"🧪 Using Top-{num_chunks_for_llm} retrieved chunk(s) for LLM prompt.")
                        else:
                            final_context_chunks_for_llm = [doc.page_content for doc in all_relevant_docs]
                            # st.info(f"Using all {len(all_relevant_docs)} retrieved chunk(s) for LLM prompt.") # Can be noisy
                    
                    with st.expander("🔍 View All Retrieved Context Chunks", expanded=False):
                        if all_relevant_docs:
                            st.info(f"Displaying all {len(all_relevant_docs)} retrieved chunks (k_retriever={st.session_state.retriever_k}):")
                            for i, doc_item in enumerate(all_relevant_docs):
                                st.text_area(f"Retrieved Chunk {i+1}", doc_item.page_content, height=150, key=f"retrieved_chunk_{i}_{user_prompt}")
                        else: st.write("No context chunks were retrieved by the retriever.")
                    
                    if final_context_chunks_for_llm:
                        prompt_with_context = create_rag_prompt(user_prompt, final_context_chunks_for_llm)
                        placeholder = st.empty()
                        for token in query_llm_stream(prompt_with_context):
                            full_response_content += token
                            placeholder.markdown(full_response_content + "▌")
                        placeholder.markdown(full_response_content)
                    else:
                        full_response_content = "I couldn't find relevant information in your documents for that query (no chunks passed to LLM)."
                        st.markdown(full_response_content)
                except Exception as e:
                    st.error(f"Error during RAG pipeline: {str(e)}")
                    st.exception(e)
                    full_response_content = f"An error occurred during RAG processing: {str(e)}"
        else:
            full_response_content = "The knowledge base is empty. Please upload and process PDF files first."
            st.markdown(full_response_content)
        
    if full_response_content:
         st.session_state.messages.append({"role": "assistant", "content": full_response_content})