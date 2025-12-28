"""
Vector store module using ChromaDB for persistent similarity search.
Provides document storage, retrieval, and automatic persistence.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import uuid

import chromadb
from chromadb.config import Settings


class ChromaVectorStore:
    """
    ChromaDB-based vector store for document embeddings.
    Supports adding, removing, searching, with automatic persistence.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 384,  # Default for all-MiniLM-L6-v2
        persist_directory: Optional[str] = None,
        collection_name: str = "rag_documents"
    ):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors (stored for compatibility)
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.embedding_dim = embedding_dim
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create ChromaDB client with persistent storage
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        return str(uuid.uuid4())
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> str:
        """
        Add a single document to the store.
        
        Args:
            embedding: Document embedding vector
            metadata: Document metadata (text, source, etc.)
            
        Returns:
            Document ID
        """
        doc_id = self._generate_id()
        
        # Extract text from metadata (ChromaDB stores documents separately)
        text = metadata.get('text', '')
        
        # Prepare metadata for ChromaDB (must be flat dict with str/int/float/bool values)
        chroma_metadata = {
            'source': str(metadata.get('source', '')),
            'filename': str(metadata.get('filename', '')),
            'chunk_index': int(metadata.get('chunk_index', 0)),
        }
        
        # Add to collection
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            documents=[text],
            metadatas=[chroma_metadata]
        )
        
        return doc_id
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple documents to the store.
        
        Args:
            embeddings: Array of embedding vectors, shape (n, embedding_dim)
            metadatas: List of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadata dicts")
        
        doc_ids = [self._generate_id() for _ in range(len(metadatas))]
        
        # Extract texts and prepare metadata
        texts = [m.get('text', '') for m in metadatas]
        chroma_metadatas = [
            {
                'source': str(m.get('source', '')),
                'filename': str(m.get('filename', '')),
                'chunk_index': int(m.get('chunk_index', 0)),
            }
            for m in metadatas
        ]
        
        # Add to collection
        self.collection.add(
            ids=doc_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=chroma_metadatas
        )
        
        return doc_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_sources: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_sources: Optional list of source files to filter by
            
        Returns:
            List of (document_metadata, similarity_score) tuples
        """
        if self.collection.count() == 0:
            return []
        
        # Handle empty filter list - no documents should match
        if filter_sources is not None and len(filter_sources) == 0:
            return []
        
        # Build where filter for sources
        where_filter = None
        if filter_sources is not None and len(filter_sources) > 0:
            # ChromaDB uses $in operator for multiple values
            # Normalize paths for matching
            normalized_sources = []
            for src in filter_sources:
                normalized_sources.append(src)
                normalized_sources.append(os.path.normpath(src))
                normalized_sources.append(os.path.abspath(src))
                normalized_sources.append(os.path.basename(src))
            
            # Remove duplicates
            normalized_sources = list(set(normalized_sources))
            
            where_filter = {
                "$or": [
                    {"source": {"$in": normalized_sources}},
                    {"filename": {"$in": normalized_sources}}
                ]
            }
        
        # Adjust top_k to not exceed collection size
        actual_top_k = min(top_k, self.collection.count())
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=actual_top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances (lower is better for cosine)
                # Convert to similarity score (higher is better)
                distance = results['distances'][0][i] if results['distances'] else 0
                # For cosine distance: similarity = 1 - distance
                similarity = 1 - distance
                
                doc = {
                    'id': doc_id,
                    'text': results['documents'][0][i] if results['documents'] else '',
                    'source': results['metadatas'][0][i].get('source', '') if results['metadatas'] else '',
                    'filename': results['metadatas'][0][i].get('filename', '') if results['metadatas'] else '',
                    'chunk_index': results['metadatas'][0][i].get('chunk_index', 0) if results['metadatas'] else 0,
                }
                
                formatted_results.append((doc, similarity))
        
        return formatted_results
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            if result and result['ids']:
                metadata = result['metadatas'][0] if result['metadatas'] else {}
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else '',
                    **metadata
                }
        except Exception:
            pass
        return None
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique source files in the store."""
        if self.collection.count() == 0:
            return []
        
        # Get all documents metadata
        results = self.collection.get(include=["metadatas"])
        
        sources = set()
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                source = metadata.get('source', '')
                if source:
                    sources.add(source)
        
        return sorted(list(sources))
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        results = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"]
        )
        
        documents = []
        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i] if results['metadatas'] else {}
                documents.append({
                    'id': doc_id,
                    'text': results['documents'][i] if results['documents'] else '',
                    **metadata
                })
        
        return documents
    
    def count(self) -> int:
        """Return number of documents in store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        # Delete the collection and recreate it
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def remove_by_source(self, source: str) -> int:
        """
        Remove all documents from a specific source.
        
        Args:
            source: Source file path to remove
            
        Returns:
            Number of documents removed
        """
        # Get count before deletion
        docs_before = self.collection.get(where={"source": source})
        count_before = len(docs_before['ids']) if docs_before and docs_before['ids'] else 0
        
        if count_before > 0:
            # Delete by source
            self.collection.delete(where={"source": source})
        
        return count_before
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        Note: ChromaDB with PersistentClient auto-saves, so this is a no-op.
        
        Args:
            directory: Directory to save to (ignored, uses persist_directory)
        """
        # ChromaDB PersistentClient automatically persists
        pass
    
    @classmethod
    def load(cls, directory: str) -> 'ChromaVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded ChromaVectorStore instance
        """
        # Create instance with the persist directory
        return cls(persist_directory=directory)
    
    @classmethod
    def exists(cls, directory: str) -> bool:
        """Check if a saved vector store exists at the given directory."""
        # Check if the chroma.sqlite3 file exists (ChromaDB's main storage file)
        chroma_db_path = os.path.join(directory, "chroma.sqlite3")
        return os.path.exists(chroma_db_path)


# Alias for backward compatibility
FAISSVectorStore = ChromaVectorStore
