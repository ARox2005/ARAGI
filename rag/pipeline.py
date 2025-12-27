"""
RAG Pipeline module that orchestrates document retrieval and generation.
Combines embeddings, vector store, and Qwen VLM for end-to-end RAG.
"""

import os
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path

from ingestion.loaders import DocumentLoader
from ingestion.ocr import OCRProcessor
from ingestion.chunking import TextChunker
from embeddings.embedder import EmbeddingModel
from vectorstore.store import FAISSVectorStore
from models.qwen_vlm import QwenVLM, load_qwen_model
from utils.file_utils import (
    is_image_file, is_pdf_file, is_docx_file, is_text_file,
    get_file_extension
)


class RAGPipeline:
    """
    Complete RAG pipeline for document retrieval and generation.
    Handles document ingestion, embedding, retrieval, and response generation.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir: Directory to store data and vector index
            embedding_model_name: Sentence transformer model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            device: Device for models ('cuda', 'cpu', or None for auto)
        """
        self.data_dir = data_dir
        self.knowledge_base_dir = os.path.join(data_dir, "knowledge_base")
        self.vector_store_dir = os.path.join(data_dir, "vector_store")
        self.top_k = top_k
        
        # Ensure directories exist
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        # Initialize components
        print("Initializing RAG Pipeline components...")
        
        # Document processing
        self.document_loader = DocumentLoader()
        self.ocr_processor = OCRProcessor()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Embeddings
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name,
            device=device
        )
        
        # Vector store
        if FAISSVectorStore.exists(self.vector_store_dir):
            print("Loading existing vector store...")
            self.vector_store = FAISSVectorStore.load(self.vector_store_dir)
        else:
            print("Creating new vector store...")
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedding_model.get_dimension()
            )
        
        # Language model (lazy loading)
        self._llm = None
        self._llm_loaded = False
        
        print("RAG Pipeline initialized!")
    
    @property
    def llm(self):
        """Lazy load the LLM."""
        if not self._llm_loaded:
            print("Loading Qwen VLM (this may take a few minutes)...")
            self._llm = load_qwen_model(prefer_vision=True)
            self._llm_loaded = True
        return self._llm
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the knowledge base.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        # Normalize to absolute path for consistent storage and retrieval
        file_path = os.path.abspath(file_path)
        filename = Path(file_path).name
        extension = get_file_extension(filename)
        
        try:
            # Extract text based on file type
            if is_image_file(filename):
                # Use OCR for images
                result = self.ocr_processor.extract_with_metadata(file_path)
                text = result['text']
                doc_data = {
                    'text': text,
                    'source': file_path,
                    'filename': filename,
                    'type': 'image'
                }
            else:
                # Use document loader for text files
                doc_data = self.document_loader.load(file_path)
                # Override source with absolute path
                doc_data['source'] = file_path
                doc_data['type'] = extension.replace('.', '')
            
            if not doc_data['text'].strip():
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'No text content extracted'
                }
            
            # Chunk the document
            chunks = self.chunker.chunk_document(doc_data)
            
            if not chunks:
                return {
                    'success': False,
                    'filename': filename,
                    'error': 'No chunks created'
                }
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.embed_batch(texts)
            
            # Add to vector store
            doc_ids = self.vector_store.add_batch(embeddings, chunks)
            
            # Save vector store
            self.vector_store.save(self.vector_store_dir)
            
            return {
                'success': True,
                'filename': filename,
                'chunks_created': len(chunks),
                'doc_ids': doc_ids
            }
            
        except Exception as e:
            return {
                'success': False,
                'filename': filename,
                'error': str(e)
            }
    
    def ingest_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of ingestion results
        """
        results = []
        for path in file_paths:
            result = self.ingest_document(path)
            results.append(result)
        return results
    
    def get_available_sources(self) -> List[str]:
        """Get list of all available document sources."""
        return self.vector_store.get_all_sources()
    
    def get_source_filenames(self) -> List[Tuple[str, str]]:
        """Get list of (source_path, filename) tuples."""
        sources = self.get_available_sources()
        return [(source, Path(source).name) for source in sources]
    
    def retrieve(
        self,
        query: str,
        filter_sources: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        min_score: float = 0.3  # Minimum similarity score threshold
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            filter_sources: Optional list of source files to filter
            top_k: Number of results (uses default if None)
            min_score: Minimum similarity score threshold (0-1). Documents below this are filtered out.
            
        Returns:
            List of retrieved document chunks with scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_sources=filter_sources
        )
        
        # Format results and filter by minimum score
        retrieved = []
        for doc, score in results:
            # Filter out low-relevance documents
            if score < min_score:
                print(f"[RAG DEBUG] Filtered out low-score document: {doc.get('filename', 'unknown')} (score: {score:.3f} < {min_score})")
                continue
                
            retrieved.append({
                'text': doc.get('text', ''),
                'source': doc.get('source', ''),
                'filename': doc.get('filename', ''),
                'score': score,
                'chunk_index': doc.get('chunk_index', 0)
            })
        
        return retrieved
    
    def build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents with citations.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string with source citations
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.get('filename', 'Unknown')
            text = doc.get('text', '')
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def extract_query_file_content(
        self,
        file_bytes: bytes,
        filename: str
    ) -> str:
        """
        Extract text content from a query file (PDF or image).
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        import tempfile
        
        # Save to temp file
        suffix = get_file_extension(filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            if is_image_file(filename):
                # OCR for images
                text = self.ocr_processor.extract_text(tmp_path)
            elif is_pdf_file(filename):
                # Load PDF
                doc = self.document_loader.load(tmp_path)
                text = doc['text']
            else:
                text = ""
            
            return text
            
        finally:
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except:
                pass
    
    def query(
        self,
        user_query: str,
        filter_sources: Optional[List[str]] = None,
        query_files: Optional[List[Tuple[bytes, str]]] = None,
        query_images: Optional[List[bytes]] = None,
        stream: bool = False
    ):
        """
        Execute a RAG query.
        
        Args:
            user_query: User's question
            filter_sources: Optional source files to filter
            query_files: Optional list of (bytes, filename) for query files
            query_images: Optional list of image bytes for vision
            stream: Whether to stream the response
            
        Returns:
            Response dict with answer and sources, or generator if streaming
        """
        # Process query files to extract text
        query_file_texts = []
        if query_files:
            for file_bytes, filename in query_files:
                text = self.extract_query_file_content(file_bytes, filename)
                if text.strip():
                    query_file_texts.append(f"[Uploaded: {filename}]\n{text}")
        
        # Combine user query with file content
        full_query = user_query
        if query_file_texts:
            full_query = user_query + "\n\nAttached file content:\n" + "\n\n".join(query_file_texts)
        
        # Retrieve relevant documents
        print(f"[RAG DEBUG] Retrieving documents for query: {full_query[:100]}...")
        print(f"[RAG DEBUG] Vector store has {self.vector_store.count()} documents")
        print(f"[RAG DEBUG] Filter sources: {filter_sources}")
        
        retrieved = self.retrieve(full_query, filter_sources=filter_sources)
        
        print(f"[RAG DEBUG] Retrieved {len(retrieved)} documents")
        for i, doc in enumerate(retrieved):
            print(f"[RAG DEBUG] Doc {i+1}: {doc.get('filename', 'unknown')} (score: {doc.get('score', 0):.3f})")
        
        # Build context
        context = self.build_context(retrieved)
        
        print(f"[RAG DEBUG] Context length: {len(context)} characters")
        if context:
            print(f"[RAG DEBUG] Context preview: {context[:200]}...")
        else:
            print("[RAG DEBUG] WARNING: Context is empty!")
        
        # Prepare citation info
        sources = list(set(doc['filename'] for doc in retrieved if doc['filename']))
        
        # Generate response
        if stream:
            return self._stream_response(
                user_query, context, query_images, retrieved, sources
            )
        else:
            response = self.llm.generate(
                query=user_query,
                context=context,
                images=query_images
            )
            
            # Add source citations to response if not already present
            if sources and "source" not in response.lower():
                response += f"\n\n📚 **Sources:** {', '.join(sources)}"
            
            return {
                'answer': response,
                'sources': sources,
                'retrieved_docs': retrieved
            }
    
    def _stream_response(
        self,
        query: str,
        context: str,
        images: Optional[List[bytes]],
        retrieved: List[Dict],
        sources: List[str]
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream response tokens."""
        full_response = ""
        
        for token in self.llm.generate_stream(
            query=query,
            context=context,
            images=images
        ):
            full_response += token
            yield {
                'token': token,
                'done': False
            }
        
        # Add sources at the end
        if sources and "source" not in full_response.lower():
            source_text = f"\n\n📚 **Sources:** {', '.join(sources)}"
            yield {
                'token': source_text,
                'done': True,
                'sources': sources,
                'retrieved_docs': retrieved
            }
        else:
            yield {
                'token': '',
                'done': True,
                'sources': sources,
                'retrieved_docs': retrieved
            }
    
    def delete_source(self, source_path: str) -> bool:
        """
        Delete a source and its chunks from the vector store.
        
        Args:
            source_path: Path of source to delete
            
        Returns:
            True if deleted successfully
        """
        removed = self.vector_store.remove_by_source(source_path)
        if removed > 0:
            self.vector_store.save(self.vector_store_dir)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'total_chunks': self.vector_store.count(),
            'total_sources': len(self.get_available_sources()),
            'embedding_dim': self.embedding_model.get_dimension(),
            'llm_loaded': self._llm_loaded,
            'vision_available': self._llm.is_vision_available() if self._llm_loaded else None
        }
    
    def delete_all(self) -> Dict[str, Any]:
        """
        Delete ALL documents from knowledge base and vector store.
        This includes:
        - Clearing the vector store index
        - Deleting all files in the knowledge base folder
        - Removing the saved vector store files
        
        Returns:
            Dictionary with deletion results
        """
        import shutil
        
        results = {
            'files_deleted': 0,
            'chunks_deleted': 0,
            'success': True,
            'errors': []
        }
        
        try:
            # Count chunks before clearing
            results['chunks_deleted'] = self.vector_store.count()
            
            # Clear the vector store in memory
            self.vector_store.clear()
            
            # Delete all files in knowledge base folder
            if os.path.exists(self.knowledge_base_dir):
                for filename in os.listdir(self.knowledge_base_dir):
                    file_path = os.path.join(self.knowledge_base_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            results['files_deleted'] += 1
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                            results['files_deleted'] += 1
                    except Exception as e:
                        results['errors'].append(f"Failed to delete {filename}: {str(e)}")
            
            # Delete vector store files on disk
            if os.path.exists(self.vector_store_dir):
                for filename in os.listdir(self.vector_store_dir):
                    file_path = os.path.join(self.vector_store_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        results['errors'].append(f"Failed to delete vector file {filename}: {str(e)}")
            
            # Save empty vector store
            self.vector_store.save(self.vector_store_dir)
            
        except Exception as e:
            results['success'] = False
            results['errors'].append(str(e))
        
        return results
