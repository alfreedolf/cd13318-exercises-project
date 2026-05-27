import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    backend_dirs = []
    for p in current_dir.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        name = p.name.lower()
        # name-based heuristic
        if "chroma" in name or name.startswith("db_") or "chromadb" in name:
            backend_dirs.append(p)
            continue
        # marker-file heuristic
        markers = ["chroma.db", "persist", "persistence", ".chromadb"]
        if any((p / m).exists() for m in markers):
            backend_dirs.append(p)
    # Optionally sort and dedupe
    backend_dirs = sorted(set(backend_dirs), key=lambda x: str(x))

    # Loop through each discovered directory
    for backend_dir in backend_dirs:
        try:
            client = chromadb.PersistentClient(path=str(backend_dir), settings=Settings())
            collections = []
            if hasattr(client, "list_collections"):
                collections = client.list_collections()

            for collection in collections:
                collection_name = None
                if isinstance(collection, dict):
                    collection_name = collection.get("name")
                else:
                    collection_name = getattr(collection, "name", None)

                if not collection_name:
                    continue

                unique_key = f"{backend_dir.name}:{collection_name}"
                display_name = f"{backend_dir.name} / {collection_name}"
                document_count = "unknown"

                try:
                    coll = client.get_collection(name=collection_name)
                    count_attr = getattr(coll, "count", None)
                    if callable(count_attr):
                        document_count = count_attr()
                    elif isinstance(count_attr, int):
                        document_count = count_attr
                except Exception:
                    document_count = "unknown"

                backends[unique_key] = {
                    "path": str(backend_dir),
                    "collection": collection_name,
                    "display_name": display_name,
                    "document_count": document_count,
                }
        except Exception as e:
            error_message = str(e) or "unable to open backend"
            if len(error_message) > 80:
                error_message = error_message[:77] + "..."
            fallback_key = f"{backend_dir.name}:unavailable"
            backends[fallback_key] = {
                "path": str(backend_dir),
                "collection": None,
                "display_name": f"{backend_dir.name} (error: {error_message})",
                "document_count": "unavailable",
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # TODO: Create a chomadb persistentclient
    client = chromadb.PersistentClient(path=chroma_dir)
    # TODO: Return the collection with the collection_name
    return client.get_collection(name=collection_name)

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # TODO: Initialize filter variable to None (represents no filtering)
    filter = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter:
        mission_filter = mission_filter.strip().lower()
        if mission_filter not in ["all", "none", "any"]:
            filter = {"mission": mission_filter}
    # TODO: If filter conditions are met, create filter dictionary with appropriate field-value pairs [DONE]
    # TODO: Execute database query with the following parameters: [DONE]
        results = collection.query(
                                    query_texts=[query],
                                    n_results=n_results,
                                    where=filter
                            )
        # TODO: Pass search query in the required format [DONE]
        # TODO: Set maximum number of results to return [DONE]
        # TODO: Apply conditional filter (None for no filtering, dictionary for specific filtering) [DONE]

    # TODO: Return query results to caller [DONE]
    else:
        results = collection.query(
                                    query_texts=[query],
                                    n_results=n_results,
                            )
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # TODO: Initialize list with header text for context section
    context_parts = ["Relevant Information from Database:"]

    # TODO: Loop through paired documents and their metadata using enumeration
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # TODO: Extract mission information from metadata with fallback value
        mission_info = meta.get("mission", "Unknown Mission")
        # TODO: Clean up mission name formatting (replace underscores, capitalize)
        mission_info = mission_info.replace("_", " ").title()
        # TODO: Extract source information from metadata with fallback value  
        source_info = meta.get("source", "Unknown Source")
        # TODO: Extract category information from metadata with fallback value
        category_info = meta.get("category", "Unknown Category")
        # TODO: Clean up category name formatting (replace underscores, capitalize)
        category_info = category_info.replace("_", " ").title()
        # TODO: Create formatted source header with index number and extracted information
        source_header = f"{idx}. [{mission_info}] [{category_info}] {source_info}"
        # TODO: Add source header to context parts list
        context_parts.append(source_header)
        # TODO: Check document length and truncate if necessary
        max_length = 500
        if len(doc) > max_length:
            doc = doc[:max_length] + "..."
        # TODO: Add truncated or full document content to context parts list
        context_parts.append(doc)

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)