"""
RAG Application with Qwen Vision-Language Model
Main Streamlit application entry point.

This application provides a ChatGPT-style interface for:
- Uploading documents to a knowledge base (PDF, DOCX, TXT, images)
- Selecting which documents to use as context
- Chatting with the AI using retrieved context
- Attaching files (PDF, images) to queries for multimodal understanding
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from ui.chat_ui import (
    init_session_state,
    render_chat_history,
    add_user_message,
    add_assistant_message,
    render_sidebar,
    render_query_file_upload,
    display_query_images,
    create_chat_css,
    render_streaming_response
)
from rag.pipeline import RAGPipeline


# Page configuration
st.set_page_config(
    page_title="RAG Chat with Qwen VLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
create_chat_css()


def get_data_directory():
    """
    Get the appropriate data directory based on environment.
    On Streamlit Cloud, use /tmp since the app directory is read-only.
    """
    import platform
    
    # Check if running on Streamlit Cloud (Linux with limited write access)
    # or check for common cloud environment indicators
    is_cloud = (
        platform.system() == "Linux" and 
        os.environ.get("STREAMLIT_SHARING_MODE") or
        os.path.exists("/mount/src")  # Streamlit Cloud mounts source here
    )
    
    if is_cloud:
        # Use /tmp on Streamlit Cloud (writable)
        data_dir = "/tmp/rag_data"
    else:
        # Use local data directory for development
        data_dir = os.path.join(PROJECT_ROOT, "data")
    
    return data_dir


@st.cache_resource
def get_pipeline():
    """
    Initialize and cache the RAG pipeline.
    Using st.cache_resource ensures the pipeline is only created once.
    """
    data_dir = get_data_directory()
    return RAGPipeline(
        data_dir=data_dir,
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=512,
        chunk_overlap=50,
        top_k=5
    )


def main():
    """Main application function."""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🤖 RAG Chat with Qwen VLM")
    st.caption("Upload documents, ask questions, get answers with citations")
    
    # Initialize pipeline
    try:
        pipeline = get_pipeline()
        st.session_state.pipeline_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.info("""
        Please ensure you have:
        1. Installed all requirements: `pip install -r requirements.txt`
        2. Sufficient memory/VRAM for the embedding model
        """)
        return
    
    # Render sidebar and get selected sources
    selected_sources, use_all = render_sidebar(pipeline)
    
    # Main chat area
    main_container = st.container()
    
    with main_container:
        # Render chat history
        render_chat_history()
    
    # Chat input area at the bottom
    st.divider()
    
    # Query file upload (collapsible) - initialize outside expander for proper scoping
    query_files = []
    with st.expander("📎 Attach files to your query", expanded=False):
        query_files = render_query_file_upload()
        if query_files:
            display_query_images(query_files)
    
    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Get image data for display - safely handle empty or None query_files
        query_image_data = []
        if query_files:
            query_image_data = [
                data for data, name in query_files
                if name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        
        # Add user message to history
        add_user_message(user_query, images=query_image_data)
        
        # Display user message
        with st.chat_message("user"):
            if query_image_data:
                cols = st.columns(min(len(query_image_data), 3))
                for i, img in enumerate(query_image_data):
                    with cols[i % 3]:
                        st.image(img, use_container_width=True)
            st.markdown(user_query)
        
        # Determine which sources to use
        if use_all:
            filter_sources = None  # Use all documents
        else:
            # Use only selected documents (can be empty list if none selected)
            filter_sources = selected_sources
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare query images for vision model
                    query_images = query_image_data if query_image_data else None
                    
                    # Check if streaming is supported
                    use_streaming = hasattr(pipeline.llm, 'generate_stream')
                    
                    if use_streaming:
                        # Streaming response
                        placeholder = st.empty()
                        response_gen = pipeline.query(
                            user_query=user_query,
                            filter_sources=filter_sources,
                            query_files=query_files if query_files else None,
                            query_images=query_images,
                            stream=True
                        )
                        result = render_streaming_response(response_gen, placeholder)
                    else:
                        # Non-streaming response
                        result = pipeline.query(
                            user_query=user_query,
                            filter_sources=filter_sources,
                            query_files=query_files if query_files else None,
                            query_images=query_images,
                            stream=False
                        )
                        st.markdown(result['answer'])
                    
                    # Show sources
                    if result.get('sources'):
                        with st.expander("📚 View Sources"):
                            for source in result['sources']:
                                st.caption(f"• {source}")
                    
                    # Add assistant message to history
                    add_assistant_message(
                        result['answer'],
                        sources=result.get('sources', [])
                    )
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    add_assistant_message(error_msg)


if __name__ == "__main__":
    main()
