"""
Vector store module using FAISS for efficient similarity search.
Provides document storage, retrieval, and persistence.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import faiss


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings.
    Supports adding, removing, searching, and persisting documents.
    """
    
    def __init__(self, embedding_dim: int, index_type: str = 'flat'):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat' for exact search, 'ivf' for approximate)
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == 'flat':
            # Exact search using L2 distance (converted to cosine via normalization)
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        else:
            # For larger datasets, could use IVF index
            self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Store document metadata
        self.documents: List[Dict[str, Any]] = []
        
        # Map from document ID to index position
        self.id_to_position: Dict[str, int] = {}
        
        # Counter for generating unique IDs
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique document ID."""
        doc_id = f"doc_{self._id_counter}"
        self._id_counter += 1
        return doc_id
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
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
        
        # Normalize and prepare embedding
        normalized = self._normalize_embedding(embedding).astype('float32')
        normalized = normalized.reshape(1, -1)
        
        # Add to FAISS index
        self.index.add(normalized)
        
        # Store metadata
        position = len(self.documents)
        self.documents.append({
            'id': doc_id,
            **metadata
        })
        self.id_to_position[doc_id] = position
        
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
        
        doc_ids = []
        
        # Normalize all embeddings
        normalized = np.array([
            self._normalize_embedding(emb) for emb in embeddings
        ]).astype('float32')
        
        # Add to FAISS index
        self.index.add(normalized)
        
        # Store metadata
        for metadata in metadatas:
            doc_id = self._generate_id()
            position = len(self.documents)
            
            self.documents.append({
                'id': doc_id,
                **metadata
            })
            self.id_to_position[doc_id] = position
            doc_ids.append(doc_id)
        
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
        if self.index.ntotal == 0:
            return []
        
        # Normalize query embedding
        query = self._normalize_embedding(query_embedding).astype('float32')
        query = query.reshape(1, -1)
        
        # Search (get more results if filtering)
        search_k = top_k * 3 if filter_sources else top_k
        search_k = min(search_k, self.index.ntotal)
        
        # Normalize filter sources for comparison (handle both absolute and relative paths)
        normalized_filter = None
        if filter_sources is not None:
            # If filter_sources is an empty list, no documents should match
            if len(filter_sources) == 0:
                return []  # No documents selected, return empty results
            
            normalized_filter = set()
            for src in filter_sources:
                # Add both the original and normalized versions for matching
                normalized_filter.add(src)
                normalized_filter.add(os.path.normpath(src))
                normalized_filter.add(os.path.abspath(src))
                # Also add just the filename for fallback matching
                normalized_filter.add(os.path.basename(src))
        
        # Perform search
        scores, indices = self.index.search(query, search_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # No more results
                break
            
            # Bounds check to handle corrupted/out-of-sync index
            if idx < 0 or idx >= len(self.documents):
                print(f"[VectorStore WARNING] Invalid index {idx}, skipping (documents length: {len(self.documents)})")
                continue
            
            doc = self.documents[idx]
            
            # Apply source filter if specified
            if normalized_filter:
                doc_source = doc.get('source', '')
                doc_filename = doc.get('filename', '')
                # Check if document matches any of the filter criteria
                match = (
                    doc_source in normalized_filter or
                    os.path.normpath(doc_source) in normalized_filter or
                    os.path.abspath(doc_source) in normalized_filter or
                    doc_filename in normalized_filter
                )
                if not match:
                    continue
            
            results.append((doc, float(score)))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        position = self.id_to_position.get(doc_id)
        if position is not None:
            return self.documents[position]
        return None
    
    def get_all_sources(self) -> List[str]:
        """Get list of all unique source files in the store."""
        sources = set()
        for doc in self.documents:
            source = doc.get('source', '')
            if source:
                sources.add(source)
        return sorted(list(sources))
    
    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        return [doc for doc in self.documents if doc.get('source') == source]
    
    def count(self) -> int:
        """Return number of documents in store."""
        return len(self.documents)
    
    def clear(self) -> None:
        """Clear all documents from the store."""
        self.index.reset()
        self.documents = []
        self.id_to_position = {}
        self._id_counter = 0
    
    def remove_by_source(self, source: str) -> int:
        """
        Remove all documents from a specific source.
        Note: This rebuilds the entire index.
        
        Args:
            source: Source file path to remove
            
        Returns:
            Number of documents removed
        """
        # Find documents to keep
        docs_to_keep = [
            doc for doc in self.documents
            if doc.get('source') != source
        ]
        
        removed_count = len(self.documents) - len(docs_to_keep)
        
        if removed_count == 0:
            return 0
        
        # We need to rebuild the index
        # First, get embeddings for docs to keep (we need to re-embed or store embeddings)
        # For simplicity, we'll just rebuild with stored metadata
        # In production, you'd want to store embeddings too
        
        self.documents = docs_to_keep
        self.id_to_position = {
            doc['id']: i for i, doc in enumerate(self.documents)
        }
        
        return removed_count
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save to
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_position': self.id_to_position,
                '_id_counter': self._id_counter,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'FAISSVectorStore':
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create instance
        store = cls(
            embedding_dim=data['embedding_dim'],
            index_type=data['index_type']
        )
        
        # Load FAISS index
        index_path = os.path.join(directory, 'faiss.index')
        store.index = faiss.read_index(index_path)
        
        # Restore metadata
        store.documents = data['documents']
        store.id_to_position = data['id_to_position']
        store._id_counter = data['_id_counter']
        
        return store
    
    @classmethod
    def exists(cls, directory: str) -> bool:
        """Check if a saved vector store exists at the given directory."""
        index_path = os.path.join(directory, 'faiss.index')
        metadata_path = os.path.join(directory, 'metadata.pkl')
        return os.path.exists(index_path) and os.path.exists(metadata_path)
