#!/usr/bin/env python3
"""
Multi-Document RAG (Retrieval Augmented Generation) System

A production-grade RAG system that supports:
- Multiple named databases with persistent storage
- Folder and individual document ingestion
- Vector embeddings with FAISS similarity search
- Query expansion and distance filtering
- Result pagination for deeper exploration
- LLM answer generation via OpenRouter

Usage:
  Set the OPENROUTER_API_KEY environment variable.

Commands:
  rag.py set <database>              Create/switch database
  rag.py add <path>                  Add document or folder
  rag.py query "question"            Query the active database
  rag.py q "question"                Alias for query
  rag.py more                        Get more results from last query
  rag.py list-dbs                    List all databases
  rag.py info                        Show database info

Examples:
  rag.py set research                          # Create/switch to 'research' database
  rag.py add document.txt                      # Add single document
  rag.py add ./notes --recursive               # Add folder recursively
  rag.py q "What is the main topic?" --k 5     # Query with parameters
  rag.py more --k 10                           # Get 10 more results

See https://github.com/yourusername/rag for full documentation.
"""

# /// script
# dependencies = [
#     "sentence-transformers",
#     "faiss-cpu",
#     "openai",
#     "nltk",
#     "requests",
# ]
# ///


#### Imports and setup (dependencies, environment variables, constants) ###
# Lazy imports for heavy dependencies - only load when needed
import os
import argparse
import sys
import hashlib
import pickle
import json
from pathlib import Path
from datetime import datetime
import shutil
import time
from typing import List, Dict, Optional, Tuple

# Set environment variable before any heavy imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


#### Configuration (file paths, model settings, API clients) ###
# OpenRouter API key is loaded from environment when needed


def call_llm_with_fallback(models: List[str], messages: List[Dict], max_tokens: int = 1000) -> str:
    """Call LLM with model fallback using OpenRouter's native fallback support"""
    import requests

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "models": models,
        "messages": messages,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        # Debug: print response for troubleshooting
        if response.status_code != 200:
            print(f"DEBUG: Status {response.status_code}, Response: {response.text[:500]}")

        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response
        try:
            error_detail = response.json()
            raise Exception(f"HTTP {response.status_code}: {error_detail}")
        except:
            raise Exception(f"HTTP {response.status_code}: {e}")
    except Exception as e:
        raise Exception(f"All model fallbacks failed: {e}")


# File paths
# test_file_path = '/Users/maxbourke/Code/RAG/Sample data/Single-text-file-sample/Ep5 Research Brief Theresa Paste.md'
test_file_path = '/Users/maxbourke/Code/RAG/Sample data/Single-text-file-sample/Ep5 Research Brief Theresa Paste TRUNCATED 01.md'
file_path = test_file_path

#### Function definitions (chunking, search, RAG functions) ###

def chunk_text_sentences(text, target_chunk_size=800, overlap=150):
    """Split text into chunks respecting sentence boundaries"""
    import nltk
    # Download required NLTK data (only runs once)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed target size, start new chunk
        if len(current_chunk) + len(sentence) > target_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap by keeping last few sentences
            sentences_in_chunk = nltk.sent_tokenize(current_chunk)
            if len(sentences_in_chunk) > 2:
                # Keep last 2 sentences for overlap
                overlap_text = " ".join(sentences_in_chunk[-2:])
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# Simple text chunking function
def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at a sentence or paragraph if possible
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:  # Only if we find a good break point
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
    return chunks

# Test retrieval function
def search_chunks(query, k=3):
    """Find the k most similar chunks to the query"""
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'chunk': chunks[idx],
            'distance': distance,
            'index': idx
        })
    
    return results

def generate_rag_query(question, k=3, cite=False):
    """Complete RAG: retrieve relevant chunks and format an answer"""
    # Retrieve relevant chunks
    results = search_chunks(question, k=k)
    

    # Create context from retrieved chunks
    context = "\n\n---\n\n".join([result['chunk'] for result in results])

    if cite:
        print(f"Citing {len(results)} chunks")
        print(f"Cited chunks: {results}")

    prompt = f"""Based on the retrieved context, please answer the question.

    <context>
    {context}
    </context>

    <question>
    {question}
    </question>

    Please provide a clear, factual answer based on the context provided. If the context doesn't fully answer the question, say so."""

    return prompt

def rag_with_llm(question, model="x-ai/grok-4-fast:free", expansion_model=None, k=3, use_expansion=True, expansion_k=5, cite=False, distance_threshold=1.0):
    """Complete RAG with LLM generation and optional query expansion"""
    
    if expansion_model is None:
        expansion_model = model  # Use generation model for expansion if not specified
    
    if use_expansion:
        # Get expanded queries
        queries = expand_query(question, client, expansion_model)
        if cite:
            print(f"Expanded queries: {queries}")
        
        # Search with each query and collect results
        all_results = []
        for i, query in enumerate(queries):
            results = search_chunks(query, k=k)
            if cite:
                print(f"Query {i+1} found chunks: {[r['index'] for r in results]}")
            all_results.extend(results)
        
        # Remove duplicates by chunk index and keep best distances
        seen_indices = {}
        for result in all_results:
            idx = result['index']
            if idx not in seen_indices or result['distance'] < seen_indices[idx]['distance']:
                seen_indices[idx] = result
        
        # Sort by distance and take top expansion_k (more than regular k)
        final_results = sorted(seen_indices.values(), key=lambda x: x['distance'])[:expansion_k]
        # Filter final_results to only include those with distance <= threshold
        chunks_before_filter = len(final_results)
        final_results = [result for result in final_results if result['distance'] <= distance_threshold]


        if cite:
            print(f"Total results before dedup: {len(all_results)}")
            print(f"Unique chunk indices found: {len(seen_indices)}")
            print(f"Found {chunks_before_filter} unique chunks, {len(final_results)} within distance {distance_threshold:.1f}")
            print(f"Using {len(final_results)} chunks from expanded search")
            for result in final_results:
                print("-"*30)
                print(f"Chunk {result['index']} (distance: {result['distance']}):\n\n{result['chunk']}\n\n")

            print('------------------------------')
    else:
        # Use original single query approach
        final_results = search_chunks(question, k=k)
        if cite:
            print(f"Using {len(final_results)} chunks from single query")

    # Generate prompt with the original question (not expanded queries)
    context = "\n\n".join([result['chunk'] for result in final_results])
    
    prompt = f"""Based on the retrieved context, please answer the question.

<context>
{context}
</context>

<question>
{question}
</question>

Please provide a clear, factual answer based on the context provided. If the context doesn't fully answer the question, say so."""

    # Get LLM response
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return completion.choices[0].message.content


def rag_with_llm_OLD2(question, model="x-ai/grok-4-fast:free", k=3, use_expansion=True):
    """Complete RAG with LLM generation and optional query expansion"""
    cite_flag = True

    if use_expansion:
        # Get expanded queries
        queries = expand_query(question, client)
        if cite_flag:
            print(f"Expanded queries: {queries}")
        
        # Search with each query and collect results
        all_results = []
        for i, query in enumerate(queries):
            results = search_chunks(query, k=k)
            if cite_flag:
                print(f"Query {i+1} found chunks: {[r['index'] for r in results]}")
            all_results.extend(results)
        
        if cite_flag:
            print(f"Total results before dedup: {len(all_results)}")
            print(f"Unique chunk indices found: {len(set(r['index'] for r in all_results))}")

        
        # Remove duplicates by chunk index and keep best distances
        seen_indices = {}
        for result in all_results:
            idx = result['index']
            if idx not in seen_indices or result['distance'] < seen_indices[idx]['distance']:
                seen_indices[idx] = result
        
        # Sort by distance and take top k
        final_results = sorted(seen_indices.values(), key=lambda x: x['distance'])[:k]
        
    else:
        # Use original single query approach
        final_results = search_chunks(question, k=k)
    
    # Generate prompt with the original question (not expanded queries)
    context = "\n\n".join([result['chunk'] for result in final_results])
    
    prompt = f"""Based on the retrieved context, please answer the question.

<context>
{context}
</context>

<question>
{question}
</question>

Please provide a clear, factual answer based on the context provided. If the context doesn't fully answer the question, say so."""

    if cite_flag:
        print(f"Using {len(final_results)} chunks from expanded search")

    # Get LLM response
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return completion.choices[0].message.content


def rag_with_llm_OLD(question, model="x-ai/grok-4-fast:free", k=3):
    """Complete RAG with LLM generation"""
    # Generate prompt
    prompt = generate_rag_query(question, k=k, cite=True)

    # Get LLM response
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    return completion.choices[0].message.content


def expand_query(original_query, models):
    """Use LLM to generate alternative phrasings of the query"""
    expansion_prompt = f"""Generate 3 alternative ways to phrase this search query to find more relevant information. Focus on synonyms, related terms, and different phrasings.

<original_query>
{original_query}
</original_query>

Return just the 3 alternative queries, one per line, without numbering or explanation."""

    try:
        messages = [{"role": "user", "content": expansion_prompt}]
        response = call_llm_with_fallback(models, messages, max_tokens=150)

        expanded_queries = response.strip().split('\n')
        expanded_queries = [q.strip() for q in expanded_queries if q.strip()]

        return [original_query] + expanded_queries[:3]  # Original + up to 3 expansions
    except Exception as e:
        print(f"Warning: Query expansion failed ({e}). Using original query only.")
        return [original_query]  # Fallback to original if expansion fails


def calculate_content_hash(content):
    """Calculate SHA256 hash of document content"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def get_cache_dir(file_path):
    """Get cache directory path for a given input file"""
    cache_base = Path.cwd() / ".rag_cache"
    cache_base.mkdir(exist_ok=True)
    
    # Use the input file's name (without extension) for cache folder
    file_stem = Path(file_path).stem
    cache_dir = cache_base / file_stem
    cache_dir.mkdir(exist_ok=True)
    
    return cache_dir


def save_processed_data(file_path, content_hash, chunks, embeddings, model_name):
    """Save processed data to cache"""
    import numpy as np
    import faiss

    cache_dir = get_cache_dir(file_path)
    
    # Save metadata
    metadata = {
        'content_hash': content_hash,
        'model_name': model_name,
        'chunk_count': len(chunks),
        'embedding_dim': len(embeddings[0]) if embeddings.size > 0 else 0,
        'created_at': str(Path(file_path).stat().st_mtime)
    }
    
    with open(cache_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save chunks as text file for easy inspection
    with open(cache_dir / 'chunks.txt', 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"=== CHUNK {i} ===\n{chunk}\n\n")
    
    # Save chunks as pickle for exact reconstruction
    with open(cache_dir / 'chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save embeddings
    np.save(cache_dir / 'embeddings.npy', embeddings)
    
    # Save FAISS index
    embeddings_np = embeddings.astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, str(cache_dir / 'faiss_index.bin'))
    
    return cache_dir


def load_cached_data(file_path, content_hash, model_name):
    """Load cached data if it exists and matches the current content"""
    import numpy as np
    import faiss

    cache_dir = get_cache_dir(file_path)
    metadata_path = cache_dir / 'metadata.json'
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if hash and model match
        if (metadata['content_hash'] != content_hash or 
            metadata['model_name'] != model_name):
            return None
        
        # Load cached data
        with open(cache_dir / 'chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
        
        embeddings = np.load(cache_dir / 'embeddings.npy')
        
        # Load FAISS index
        index = faiss.read_index(str(cache_dir / 'faiss_index.bin'))
        
        return {
            'chunks': chunks,
            'embeddings': embeddings,
            'index': index,
            'metadata': metadata
        }
    
    except (FileNotFoundError, json.JSONDecodeError, pickle.PickleError) as e:
        # If any file is missing or corrupted, return None to reprocess
        return None


def get_config_file_path():
    """Get path to the configuration file"""
    return Path.cwd() / ".rag_config.json"


def load_config():
    """Load configuration from file, return defaults if not found"""
    config_path = get_config_file_path()

    default_config = {
        "generation_models": [
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free"
        ],
        "query_expansion_models": [
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free"
        ],
        "embedding_model": "all-MiniLM-L6-v2",
        "active_database": "default"
    }

    if not config_path.exists():
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Ensure all required keys exist with defaults
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
        return config
    except (json.JSONDecodeError, FileNotFoundError):
        return default_config


def save_config(config):
    """Save configuration to file"""
    config_path = get_config_file_path()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def set_default_models(generation_model=None, query_expansion_model=None, embedding_model=None):
    """Set default models and save to config"""
    config = load_config()

    if generation_model:
        config["generation_model"] = generation_model
    if query_expansion_model:
        config["query_expansion_model"] = query_expansion_model
    if embedding_model:
        config["embedding_model"] = embedding_model

    save_config(config)
    return config


def get_active_database():
    """Get the name of the currently active database"""
    config = load_config()
    return config.get("active_database", "default")


def set_active_database(name: str):
    """Set the active database and save to config"""
    config = load_config()
    config["active_database"] = name
    save_config(config)


#### RAGDatabase Class ###

class RAGDatabase:
    """Multi-document RAG database with persistent storage"""

    def __init__(self, name: str, base_path: str = ".rag_databases", embedding_model: str = "all-MiniLM-L6-v2"):
        self.name = name
        self.base_path = Path(base_path)
        self.db_path = self.base_path / name
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.chunks = []  # All chunks from all documents
        self.chunk_map = {}  # Maps global chunk index to (doc_hash, local_chunk_idx)
        self.document_metadata = {}  # Maps doc_hash to metadata
        self.index = None  # FAISS index

    def _ensure_directories(self):
        """Create database directory structure"""
        self.db_path.mkdir(parents=True, exist_ok=True)
        (self.db_path / "documents").mkdir(exist_ok=True)

    def _get_document_dir(self, doc_hash: str) -> Path:
        """Get directory path for a specific document"""
        return self.db_path / "documents" / doc_hash[:16]  # Use first 16 chars of hash

    def _load_embedding_model(self, verbose: bool = False):
        """Load the embedding model if not already loaded"""
        if self.embedding_model is None:
            if verbose:
                print(f"Loading embedding model: {self.embedding_model_name}...", end='', flush=True)
            start_time = time.time()
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            elapsed = time.time() - start_time
            if verbose:
                print(f" ({elapsed:.1f}s)")

    def exists(self) -> bool:
        """Check if database exists on disk"""
        return self.db_path.exists() and (self.db_path / "documents").exists()

    def load(self) -> bool:
        """Load existing database from disk"""
        import faiss

        if not self.exists():
            return False

        try:
            # Load database config
            config_path = self.db_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    db_config = json.load(f)
                    self.embedding_model_name = db_config.get("embedding_model", self.embedding_model_name)

            # Load chunk map
            chunk_map_path = self.db_path / "chunk_map.json"
            if chunk_map_path.exists():
                with open(chunk_map_path, 'r') as f:
                    # Chunk map is stored as {str(idx): doc_hash}
                    self.chunk_map = {int(k): v for k, v in json.load(f).items()}

            # Load all document metadata
            docs_dir = self.db_path / "documents"
            if docs_dir.exists():
                for doc_dir in docs_dir.iterdir():
                    if doc_dir.is_dir():
                        metadata_path = doc_dir / "metadata.json"
                        if metadata_path.exists():
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                self.document_metadata[metadata['doc_id']] = metadata

            # Load master FAISS index
            index_path = self.db_path / "master_index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))

            # Load all chunks (we'll need them for search results)
            self._load_all_chunks()

            return True

        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def _load_all_chunks(self):
        """Load all chunks from all documents in order"""
        self.chunks = []
        # Sort by global chunk index
        for chunk_idx in sorted(self.chunk_map.keys()):
            doc_hash = self.chunk_map[chunk_idx]
            doc_dir = self._get_document_dir(doc_hash)
            chunks_path = doc_dir / "chunks.pkl"

            if chunks_path.exists():
                with open(chunks_path, 'rb') as f:
                    doc_chunks = pickle.load(f)
                    # Calculate local chunk index within this document
                    # Find how many chunks from this doc we've already added
                    local_idx = sum(1 for idx in range(chunk_idx) if self.chunk_map.get(idx) == doc_hash)
                    if local_idx < len(doc_chunks):
                        self.chunks.append(doc_chunks[local_idx])

    def save_database_config(self):
        """Save database-level configuration"""
        config = {
            "embedding_model": self.embedding_model_name,
            "created_at": datetime.now().isoformat(),
            "document_count": len(self.document_metadata)
        }
        with open(self.db_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def get_info(self) -> Dict:
        """Get database statistics"""
        total_chunks = len(self.chunks) if self.chunks else 0
        if total_chunks == 0 and self.index:
            total_chunks = self.index.ntotal

        return {
            "name": self.name,
            "document_count": len(self.document_metadata),
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_model_name,
            "exists": self.exists()
        }

    def add_document(self, file_path: str, verbose: bool = False) -> Tuple[str, int]:
        """
        Add a document to the database
        Returns: (status, chunk_count) where status is 'added', 'updated', or 'skipped'
        """
        import numpy as np

        self._ensure_directories()
        self._load_embedding_model(verbose=verbose)

        # Read and hash document
        try:
            content = read_file(file_path)
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {e}")

        content_hash = calculate_content_hash(content)
        doc_hash = content_hash[:16]
        doc_dir = self._get_document_dir(doc_hash)

        # Check if document already exists
        if doc_hash in self.document_metadata:
            existing_meta = self.document_metadata[doc_hash]
            if existing_meta['content_hash'] == content_hash:
                if verbose:
                    print(f"✓ Document already in database (skipped): {Path(file_path).name}")
                return ('skipped', existing_meta['chunk_count'])

        # Process document
        if verbose:
            print(f"Processing: {Path(file_path).name}")

        chunks = chunk_text_sentences(content)

        if verbose:
            print(f"  - Created {len(chunks)} chunks")

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=verbose)

        # Save document data
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "doc_id": doc_hash,
            "file_path": str(Path(file_path).absolute()),
            "filename": Path(file_path).name,
            "content_hash": content_hash,
            "embedding_model": self.embedding_model_name,
            "chunk_count": len(chunks),
            "embedding_dim": len(embeddings[0]) if len(embeddings) > 0 else 0,
            "added_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        with open(doc_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save chunks
        with open(doc_dir / "chunks.pkl", 'wb') as f:
            pickle.dump(chunks, f)

        with open(doc_dir / "chunks.txt", 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== CHUNK {i} ===\n{chunk}\n\n")

        # Save embeddings
        np.save(doc_dir / "embeddings.npy", embeddings)

        # Update document metadata
        self.document_metadata[doc_hash] = metadata

        # Rebuild master index
        self._rebuild_master_index(verbose=verbose)

        status = 'added' if doc_hash not in self.document_metadata else 'updated'
        if verbose:
            print(f"✓ {status.capitalize()}: {metadata['filename']} ({len(chunks)} chunks)")

        return (status, len(chunks))

    def _rebuild_master_index(self, verbose: bool = False):
        """Rebuild the master FAISS index from all documents"""
        import numpy as np
        import faiss

        if verbose:
            print("Rebuilding master index...")

        all_embeddings = []
        new_chunk_map = {}
        new_chunks = []
        global_idx = 0

        # Iterate through all documents
        for doc_hash in sorted(self.document_metadata.keys()):
            doc_dir = self._get_document_dir(doc_hash)

            # Load embeddings
            embeddings_path = doc_dir / "embeddings.npy"
            if not embeddings_path.exists():
                continue

            embeddings = np.load(embeddings_path)
            all_embeddings.append(embeddings)

            # Load chunks
            chunks_path = doc_dir / "chunks.pkl"
            if chunks_path.exists():
                with open(chunks_path, 'rb') as f:
                    doc_chunks = pickle.load(f)
                    new_chunks.extend(doc_chunks)

                    # Update chunk map
                    for local_idx in range(len(doc_chunks)):
                        new_chunk_map[global_idx] = doc_hash
                        global_idx += 1

        if len(all_embeddings) == 0:
            if verbose:
                print("No embeddings to index")
            return

        # Concatenate all embeddings
        all_embeddings_np = np.vstack(all_embeddings).astype('float32')

        # Create FAISS index
        dimension = all_embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(all_embeddings_np)

        # Update chunk map and chunks
        self.chunk_map = new_chunk_map
        self.chunks = new_chunks

        # Save master index
        faiss.write_index(self.index, str(self.db_path / "master_index.faiss"))

        # Save chunk map
        with open(self.db_path / "chunk_map.json", 'w') as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in self.chunk_map.items()}, f, indent=2)

        # Save database config
        self.save_database_config()

        if verbose:
            print(f"✓ Master index rebuilt: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 3, verbose: bool = False) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None:
            raise Exception("No index loaded. Add documents first.")

        self._load_embedding_model(verbose=verbose)

        if verbose:
            print(f"Encoding query...", end='', flush=True)
        start_time = time.time()
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        elapsed = time.time() - start_time
        if verbose:
            print(f" ({elapsed:.2f}s)")

        if verbose:
            print(f"Searching index for top {k} matches...", end='', flush=True)
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        elapsed = time.time() - start_time
        if verbose:
            print(f" ({elapsed:.3f}s)")

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                doc_hash = self.chunk_map.get(idx, "unknown")
                metadata = self.document_metadata.get(doc_hash, {})

                results.append({
                    'chunk': self.chunks[idx],
                    'distance': float(distance),
                    'index': int(idx),
                    'doc_hash': doc_hash,
                    'source_file': metadata.get('file_path', 'unknown'),
                    'filename': metadata.get('filename', 'unknown')
                })

        return results


def parse_arguments():
    """Parse command line arguments with subcommands"""
    parser = argparse.ArgumentParser(
        description="Multi-Document RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s set research                              # Create/switch to database
  %(prog)s add document.txt                           # Add single document
  %(prog)s add ./notes --recursive                    # Add folder recursively
  %(prog)s q "What is the main topic?" --k 5          # Query with custom k
  %(prog)s more --k 10                                # Get 10 more results
  %(prog)s list-dbs                                   # List all databases
        """)

    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # SET command
    set_parser = subparsers.add_parser('set', help='Create or switch to a database')
    set_parser.add_argument('database', help='Database name')

    # ADD command
    add_parser = subparsers.add_parser('add', help='Add document(s) to the active database')
    add_parser.add_argument('path', help='File or folder path')
    add_parser.add_argument('--recursive', '-r', action='store_true',
                           help='Recursively add files from folders')
    add_parser.add_argument('--include', help='File patterns to include (e.g., "*.md,*.txt")')
    add_parser.add_argument('--exclude', help='File patterns to exclude (e.g., "draft_*")')

    # QUERY command (and alias Q)
    query_parser = subparsers.add_parser('query', aliases=['q'], help='Query the active database')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--k', type=int, default=3, help='Number of base chunks to retrieve')
    query_parser.add_argument('--expansion-k', type=int, default=7,
                             help='Number of chunks for expanded queries')
    query_parser.add_argument('--no-expansion', action='store_true',
                             help='Disable query expansion')
    query_parser.add_argument('--distance-threshold', type=float, default=10.0,
                             help='Maximum distance for chunk inclusion')
    query_parser.add_argument('--gen-model', help='Override generation model')
    query_parser.add_argument('--exp-model', help='Override expansion model')
    query_parser.add_argument('--cite', action='store_true', help='Show source chunks')
    query_parser.add_argument('--json', action='store_true', help='Output as JSON')
    query_parser.add_argument('--db', help='Use specific database (temp override)')

    # MORE command
    more_parser = subparsers.add_parser('more', help='Get more results from last query')
    more_parser.add_argument('--k', type=int, default=5, help='Number of additional results')
    more_parser.add_argument('--offset', type=int, help='Start from specific offset')
    more_parser.add_argument('--regenerate', action='store_true',
                            help='Re-generate answer with more context')

    # LIST-DBS command
    list_parser = subparsers.add_parser('list-dbs', aliases=['list'], help='List all databases')

    # INFO command
    info_parser = subparsers.add_parser('info', aliases=['status'], help='Show active database info')

    args = parser.parse_args()
    return args

        
#### Command Handlers ###

def cmd_set(args):
    """Handle SET command"""
    db_name = args.database
    config = load_config()

    # Create or load database
    db = RAGDatabase(db_name, embedding_model=config['embedding_model'])

    if db.exists():
        db.load()
        status = "switched to"
    else:
        db._ensure_directories()
        db.save_database_config()
        status = "created"

    # Set as active database
    set_active_database(db_name)

    # Show info
    info = db.get_info()
    print(f"✓ Active database: {db_name} ({status})")
    if info['document_count'] > 0:
        print(f"  Documents: {info['document_count']}")
        print(f"  Total chunks: {info['total_chunks']}")
        print(f"  Embedding model: {info['embedding_model']}")


def cmd_add(args):
    """Handle ADD command"""
    config = load_config()
    db_name = get_active_database()

    # Load database
    db = RAGDatabase(db_name, embedding_model=config['embedding_model'])
    if not db.exists():
        print(f"Error: Database '{db_name}' does not exist. Create it with: rag.py set {db_name}")
        sys.exit(1)

    db.load()

    # Check if path is file or directory
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)

    if path.is_file():
        # Add single file
        try:
            status, chunk_count = db.add_document(str(path), verbose=args.verbose)
            if status != 'skipped':
                print(f"✓ Added {path.name} ({chunk_count} chunks)")
        except Exception as e:
            print(f"Error adding {path.name}: {e}")
            sys.exit(1)
    else:
        # Folder ingestion
        files = _collect_files(path, args.recursive, args.include, args.exclude)

        if len(files) == 0:
            print(f"No matching files found in {path}")
            return

        print(f"Found {len(files)} file(s) to process")

        added = 0
        skipped = 0
        failed = 0
        errors = []

        for file in files:
            try:
                status, chunk_count = db.add_document(str(file), verbose=args.verbose)
                if status == 'skipped':
                    skipped += 1
                else:
                    added += 1
                    if not args.verbose:
                        print(f"✓ {file.name}")
            except Exception as e:
                failed += 1
                errors.append((file.name, str(e)))
                if args.verbose:
                    print(f"✗ Error: {file.name}: {e}")

        print(f"\n{'='*50}")
        print(f"Summary: ✓ {added} added, ↻ {skipped} skipped, ✗ {failed} failed")

        if failed > 0 and not args.verbose:
            print("\nErrors:")
            for filename, error in errors:
                print(f"  {filename}: {error}")


def _collect_files(path: Path, recursive: bool, include: Optional[str], exclude: Optional[str]) -> List[Path]:
    """Collect files from a directory based on filters"""
    # Default patterns
    if include is None:
        include_patterns = ["*.txt", "*.md"]
    else:
        include_patterns = [p.strip() for p in include.split(',')]

    if exclude is None:
        exclude_patterns = []
    else:
        exclude_patterns = [p.strip() for p in exclude.split(',')]

    files = []

    if recursive:
        # Use rglob for recursive search
        for pattern in include_patterns:
            files.extend(path.rglob(pattern))
    else:
        # Use glob for non-recursive search
        for pattern in include_patterns:
            files.extend(path.glob(pattern))

    # Filter out excluded patterns
    if exclude_patterns:
        filtered_files = []
        for file in files:
            exclude_match = False
            for pattern in exclude_patterns:
                if file.match(pattern):
                    exclude_match = True
                    break
            if not exclude_match:
                filtered_files.append(file)
        files = filtered_files

    # Only return files (not directories)
    files = [f for f in files if f.is_file()]

    return sorted(set(files))  # Remove duplicates and sort


def cmd_query(args):
    """Handle QUERY command"""
    config = load_config()

    # Determine which database to use
    if args.db:
        db_name = args.db
    else:
        db_name = get_active_database()

    # Load database
    db = RAGDatabase(db_name, embedding_model=config['embedding_model'])
    if not db.exists():
        print(f"Error: Database '{db_name}' does not exist.")
        sys.exit(1)

    if args.verbose:
        print(f"Loading database: {db_name}...")

    db.load()

    if args.verbose:
        print(f"✓ Loaded {len(db.document_metadata)} documents, {len(db.chunks)} chunks")

    if db.index is None or len(db.chunks) == 0:
        print(f"Error: Database '{db_name}' is empty. Add documents first.")
        sys.exit(1)

    # Get models from config/args
    # Support both single model override and model arrays
    if args.gen_model:
        generation_models = [args.gen_model]
    else:
        generation_models = config.get('generation_models', config.get('generation_model', []))
        if isinstance(generation_models, str):
            generation_models = [generation_models]

    if args.exp_model:
        expansion_models = [args.exp_model]
    else:
        expansion_models = config.get('query_expansion_models', config.get('query_expansion_model', []))
        if isinstance(expansion_models, str):
            expansion_models = [expansion_models]

    if args.verbose:
        print(f"Generation models: {generation_models[0]} (+{len(generation_models)-1} fallbacks)")
        print(f"Expansion models: {expansion_models[0]} (+{len(expansion_models)-1} fallbacks)")

    # Execute query
    use_expansion = not args.no_expansion

    if use_expansion:
        # Get expanded queries
        if not args.verbose:
            print("Expanding query...", end='', flush=True)
        elif args.verbose:
            print("Expanding query with LLM...", end='', flush=True)
        start_time = time.time()
        queries = expand_query(args.question, expansion_models)
        elapsed = time.time() - start_time
        print(f" ({elapsed:.1f}s)")
        if args.cite or args.verbose:
            print(f"Expanded queries: {queries}")

        # Search with each query
        if not args.verbose:
            print(f"Searching index...", end='', flush=True)
            start_time = time.time()
        all_results = []
        for i, query in enumerate(queries):
            if args.verbose:
                print(f"Searching with query variant {i+1}/{len(queries)}...")
            results = db.search(query, k=args.k, verbose=args.verbose)
            if args.cite or args.verbose:
                print(f"Query {i+1} found chunks: {[r['index'] for r in results]}")
            all_results.extend(results)

        # Deduplicate by chunk index, keep best distance
        seen_indices = {}
        for result in all_results:
            idx = result['index']
            if idx not in seen_indices or result['distance'] < seen_indices[idx]['distance']:
                seen_indices[idx] = result

        # Sort by distance and take top expansion_k
        final_results = sorted(seen_indices.values(), key=lambda x: x['distance'])[:args.expansion_k]

        # Apply distance threshold - keep chunks below absolute threshold
        chunks_before_filter = len(final_results)
        final_results = [result for result in final_results if result['distance'] <= args.distance_threshold]

        if not args.verbose:
            elapsed = time.time() - start_time
            print(f" ({elapsed:.2f}s)")

        if args.cite or args.verbose:
            print(f"Found {chunks_before_filter} unique chunks, {len(final_results)} within distance {args.distance_threshold:.1f} (use --distance-threshold to adjust)")
    else:
        # Single query
        if not args.verbose:
            print("Searching index...", end='', flush=True)
            start_time = time.time()
        elif args.verbose:
            print("Performing single query search...")
        final_results = db.search(args.question, k=args.k, verbose=args.verbose)
        if not args.verbose:
            elapsed = time.time() - start_time
            print(f" ({elapsed:.2f}s)")
        if args.cite or args.verbose:
            print(f"Using {len(final_results)} chunks from single query")

    # Show citations if requested
    if args.cite:
        print("\n" + "="*60)
        print("Source Chunks:")
        print("="*60)
        for result in final_results:
            print(f"\n[Distance: {result['distance']:.2f}] {result['filename']}")
            print(f"{result['chunk'][:200]}...")

    # Generate answer with LLM
    context = "\n\n".join([result['chunk'] for result in final_results])

    prompt = f"""Based on the retrieved context, please answer the question.

<context>
{context}
</context>

<question>
{args.question}
</question>

Please provide a clear, factual answer based on the context provided. If the context doesn't fully answer the question, say so."""

    if not args.verbose:
        print("Generating answer...", end='', flush=True)
    elif args.verbose:
        print(f"Sending query and context to LLM ({generation_models[0]})...", end='', flush=True)

    start_time = time.time()
    try:
        messages = [{"role": "user", "content": prompt}]
        answer = call_llm_with_fallback(generation_models, messages, max_tokens=1000)
        elapsed = time.time() - start_time
        print(f" ({elapsed:.1f}s)")
        if args.verbose:
            print("✓ Received answer from LLM")
    except Exception as e:
        print(f"Error generating answer: {e}")
        print("Returning search results only.")
        answer = f"[Error: Could not generate answer. Found {len(final_results)} relevant chunks.]"

    # Output
    print("\n" + "="*60)
    print(f"Q: {args.question}")
    print("="*60)
    print(answer)

    if args.json:
        # Also output JSON
        output = {
            "query": args.question,
            "answer": answer,
            "chunks": [{
                "text": r['chunk'],
                "distance": r['distance'],
                "source_file": r['source_file'],
                "filename": r['filename']
            } for r in final_results],
            "metadata": {
                "database": db_name,
                "generation_models": generation_models,
                "expansion_models": expansion_models,
                "k": args.k,
                "expansion_k": args.expansion_k,
                "use_expansion": use_expansion
            }
        }
        print("\n" + "="*60)
        print("JSON Output:")
        print("="*60)
        print(json.dumps(output, indent=2))


def cmd_list_dbs(args):
    """Handle LIST-DBS command"""
    base_path = Path(".rag_databases")
    if not base_path.exists():
        print("No databases found.")
        return

    active_db = get_active_database()
    print("Available databases:")

    for db_dir in sorted(base_path.iterdir()):
        if db_dir.is_dir():
            db = RAGDatabase(db_dir.name)
            if db.exists():
                db.load()
                info = db.get_info()
                marker = "*" if db_dir.name == active_db else " "
                print(f"  {marker} {db_dir.name} ({info['document_count']} docs, {info['total_chunks']} chunks)")

    print("\n(* = active)")


def cmd_info(args):
    """Handle INFO command"""
    config = load_config()
    db_name = get_active_database()

    db = RAGDatabase(db_name)
    if not db.exists():
        print(f"Error: Active database '{db_name}' does not exist.")
        sys.exit(1)

    db.load()
    info = db.get_info()

    print(f"Active database: {db_name}")
    print(f"Documents: {info['document_count']}")
    print(f"Total chunks: {info['total_chunks']}")
    print(f"Embedding model: {info['embedding_model']}")


#### Main execution ###

def main():
    args = parse_arguments()

    # Ensure default database exists on first run
    if not Path(".rag_config.json").exists():
        print("First run detected. Creating default database...")
        db = RAGDatabase("default")
        db._ensure_directories()
        db.save_database_config()
        set_active_database("default")
        save_config(load_config())  # Create config file

    # Handle commands
    if not args.command:
        print("Error: No command specified. Use --help for usage.")
        sys.exit(1)

    if args.command == 'set':
        cmd_set(args)
    elif args.command == 'add':
        cmd_add(args)
    elif args.command in ['list-dbs', 'list']:
        cmd_list_dbs(args)
    elif args.command in ['info', 'status']:
        cmd_info(args)
    elif args.command in ['query', 'q']:
        cmd_query(args)
    elif args.command == 'more':
        print("More command coming in Phase 5...")
        sys.exit(1)
    else:
        print(f"Error: Unknown command '{args.command}'")
        sys.exit(1)

if __name__ == "__main__":
    main() 

    
