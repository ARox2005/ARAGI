"""
Streamlit UI components for the RAG chat application.
Provides chat bubbles, sidebar, and file upload components.
"""

import streamlit as st
from typing import List, Optional, Tuple, Dict, Any
import base64
from pathlib import Path
import time


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'selected_sources' not in st.session_state:
        st.session_state.selected_sources = []
    
    if 'use_all_documents' not in st.session_state:
        st.session_state.use_all_documents = True
    
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def render_chat_message(message: Dict[str, Any], container=None):
    """
    Render a single chat message with proper styling.
    
    Args:
        message: Message dict with 'role', 'content', and optional 'images'
        container: Optional container to render in
    """
    target = container if container else st
    
    role = message.get('role', 'user')
    content = message.get('content', '')
    images = message.get('images', [])
    sources = message.get('sources', [])
    
    with target.chat_message(role):
        # Display images if present
        if images:
            cols = st.columns(min(len(images), 3))
            for i, img_data in enumerate(images):
                with cols[i % 3]:
                    if isinstance(img_data, bytes):
                        st.image(img_data, use_container_width=True)
                    else:
                        st.image(img_data, use_container_width=True)
        
        # Display text content
        st.markdown(content)
        
        # Display source citations if present
        if sources:
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.caption(f"• {source}")


def render_chat_history():
    """Render all messages in the chat history."""
    for message in st.session_state.messages:
        render_chat_message(message)


def add_user_message(content: str, images: Optional[List[bytes]] = None):
    """
    Add a user message to the chat history.
    
    Args:
        content: Message content
        images: Optional list of image bytes
    """
    message = {
        'role': 'user',
        'content': content,
        'images': images or []
    }
    st.session_state.messages.append(message)


def add_assistant_message(
    content: str,
    sources: Optional[List[str]] = None,
    images: Optional[List[bytes]] = None
):
    """
    Add an assistant message to the chat history.
    
    Args:
        content: Message content
        sources: Optional list of source filenames
        images: Optional list of images
    """
    message = {
        'role': 'assistant',
        'content': content,
        'sources': sources or [],
        'images': images or []
    }
    st.session_state.messages.append(message)


def render_sidebar(pipeline) -> Tuple[List[str], bool]:
    """
    Render the sidebar with document upload and selection.
    
    Args:
        pipeline: RAGPipeline instance
        
    Returns:
        Tuple of (selected_sources, use_all_documents)
    """
    with st.sidebar:
        st.header("📚 Knowledge Base")
        
        # Document upload section
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files to knowledge base",
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key='kb_uploader',
            help="Supported formats: PDF, DOCX, TXT, PNG, JPG, JPEG"
        )
        
        if uploaded_files:
            if st.button("📤 Add to Knowledge Base", type="primary"):
                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        # Save uploaded file
                        file_bytes = file.read()
                        save_path = Path(pipeline.knowledge_base_dir) / file.name
                        
                        with open(save_path, 'wb') as f:
                            f.write(file_bytes)
                        
                        # Ingest document
                        result = pipeline.ingest_document(str(save_path))
                        
                        if result['success']:
                            st.success(f"✅ {file.name}: {result['chunks_created']} chunks")
                        else:
                            st.error(f"❌ {file.name}: {result.get('error', 'Unknown error')}")
        
        st.divider()
        
        # Document selection section
        st.subheader("Select Context Documents")
        
        # Toggle for using all documents
        use_all = st.toggle(
            "Use all documents",
            value=st.session_state.use_all_documents,
            help="When enabled, all documents are used as context"
        )
        st.session_state.use_all_documents = use_all
        
        # Get available sources
        source_files = pipeline.get_source_filenames()
        
        if not source_files:
            st.info("No documents uploaded yet")
            selected_sources = []
        else:
            if not use_all:
                # Multi-select for document selection
                source_names = [name for _, name in source_files]
                selected_names = st.multiselect(
                    "Choose documents",
                    options=source_names,
                    default=st.session_state.selected_sources if st.session_state.selected_sources else source_names,
                    help="Select specific documents to use as context"
                )
                
                # Map selected names back to paths
                name_to_path = {name: path for path, name in source_files}
                selected_sources = [name_to_path[name] for name in selected_names]
                st.session_state.selected_sources = selected_names
            else:
                selected_sources = [path for path, _ in source_files]
                
                # Show document list
                with st.expander("📄 Available Documents", expanded=True):
                    for path, name in source_files:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.caption(name)
                        with col2:
                            if st.button("🗑️", key=f"del_{name}", help="Remove"):
                                pipeline.delete_source(path)
                                st.rerun()
        
        st.divider()
        
        # Pipeline stats
        st.subheader("📊 Stats")
        stats = pipeline.get_stats()
        st.caption(f"Documents: {stats['total_sources']}")
        st.caption(f"Chunks: {stats['total_chunks']}")
        
        st.divider()
        
        # Danger zone - Delete all
        st.subheader("⚠️ Danger Zone")
        
        # Use session state for confirmation
        if 'confirm_delete_all' not in st.session_state:
            st.session_state.confirm_delete_all = False
        
        if not st.session_state.confirm_delete_all:
            if st.button("🗑️ Delete All Documents", type="secondary", use_container_width=True):
                st.session_state.confirm_delete_all = True
                st.rerun()
        else:
            st.warning("⚠️ This will permanently delete ALL documents and indexes!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    with st.spinner("Deleting everything..."):
                        result = pipeline.delete_all()
                        if result['success']:
                            st.success(f"Deleted {result['files_deleted']} files and {result['chunks_deleted']} chunks")
                        else:
                            st.error(f"Errors: {', '.join(result['errors'])}")
                    st.session_state.confirm_delete_all = False
                    st.session_state.messages = []  # Also clear chat
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_delete_all = False
                    st.rerun()
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        return selected_sources, use_all


def render_query_file_upload() -> List[Tuple[bytes, str]]:
    """
    Render file upload near the chat input for query files.
    
    Returns:
        List of (file_bytes, filename) tuples
    """
    uploaded = st.file_uploader(
        "Attach files to query",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='query_uploader',
        label_visibility='collapsed',
        help="Upload PDF or image files to include in your query"
    )
    
    query_files = []
    if uploaded:
        for file in uploaded:
            file_bytes = file.read()
            query_files.append((file_bytes, file.name))
            # Reset file position for potential re-reads
            file.seek(0)
    
    return query_files


def display_query_images(query_files: List[Tuple[bytes, str]]):
    """
    Display uploaded query images inline.
    
    Args:
        query_files: List of (bytes, filename) tuples
    """
    image_files = [
        (data, name) for data, name in query_files
        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if image_files:
        st.caption("📎 Attached images:")
        cols = st.columns(min(len(image_files), 4))
        for i, (data, name) in enumerate(image_files):
            with cols[i % 4]:
                st.image(data, caption=name, use_container_width=True)


def create_chat_css():
    """Inject custom CSS for chat styling that works in both light and dark modes."""
    st.markdown("""
    <style>
    /* Chat container styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessageContent"] {
        padding: 0.5rem 1rem;
    }
    
    /* File uploader compactness */
    .stFileUploader {
        padding: 0.5rem 0;
    }
    
    /* Remove hardcoded background colors - let Streamlit theme handle it */
    /* This ensures proper appearance in both light and dark modes */
    
    </style>
    """, unsafe_allow_html=True)


def render_streaming_response(response_generator, placeholder):
    """
    Render streaming response tokens.
    
    Args:
        response_generator: Generator yielding response tokens
        placeholder: Streamlit placeholder to update
        
    Returns:
        Final response dict
    """
    full_response = ""
    final_sources = []
    
    for chunk in response_generator:
        token = chunk.get('token', '')
        full_response += token
        
        # Update placeholder with current response
        placeholder.markdown(full_response + "▌")
        
        # Small delay to ensure UI renders each token (creates streaming effect)
        time.sleep(0.01)
        
        if chunk.get('done'):
            final_sources = chunk.get('sources', [])
            break
    
    # Final update without cursor
    placeholder.markdown(full_response)
    
    return {
        'answer': full_response,
        'sources': final_sources
    }
