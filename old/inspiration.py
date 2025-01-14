#gpto1-generated code. Might return to this later, but most of it is probably not usable.

import os
import hashlib
import json
import time
import datetime
from typing import List, Dict, Any

# For PDF reading (example: PyPDF2)
import PyPDF2

# For PPTX reading (example: python-pptx)
from pptx import Presentation

# For text embeddings (placeholder)
import numpy as np

# Chroma imports
# Make sure to install Chroma: pip install chromadb
import chromadb
from chromadb.config import Settings

###############################################################################
#                             Utility Functions                               #
###############################################################################

def calculate_folder_hash(folder_path: str) -> str:
    """
    Calculates a hash value based on the file contents of a folder.
    If none of the files have changed, the hash remains the same.
    This helps us skip unnecessary vector store rebuilding.
    """
    md5 = hashlib.md5()
    
    # Walk through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            
            # We only consider certain file types
            # (In practice, you might consider only pdf, pptx, txt, etc.)
            if file_name.lower().endswith(('.pdf', '.txt', '.pptx')):
                with open(file_path, 'rb') as f:
                    # Update the hash
                    data = f.read()
                    md5.update(data)
    
    return md5.hexdigest()

def dummy_embedding_function(text: str) -> List[float]:
    """
    A placeholder embedding function that converts a string into a numeric vector.
    Replace with a real embedding model (e.g., OpenAI embeddings or SentenceTransformers).
    """
    # Toy example: just convert character ord values into a vector (not meaningful)
    vec = [ord(c) % 256 for c in text[:64]]  # only up to 64 chars
    # Pad or truncate to a fixed length, for instance 64
    vec = vec + [0]*(64-len(vec)) if len(vec) < 64 else vec[:64]
    # Convert to float
    vec = list(map(float, vec))
    return vec






###############################################################################
#                            Vector Store Creation                            #
###############################################################################




###############################################################################
#                            Multi-Agent RAG Setup                            #
###############################################################################

def create_mrag(path_to_sources_dir: str):
    """
    Scans each subfolder in `path_to_sources_dir`, builds (or loads) the corresponding
    vector store (Chroma), and sets up the multi-agent RAG system.
    
    Returns a dictionary where:
    - key: agent name (subfolder name + "_rag")
    - value: the Chroma collection
    """
    multi_agent_system = {}
    for item in os.listdir(path_to_sources_dir):
        subfolder_path = os.path.join(path_to_sources_dir, item)
        if os.path.isdir(subfolder_path):
            # For each subfolder, we create a separate vector store collection
            agent_name = f"{item}_rag"
            collection_name = agent_name  # or any unique naming convention

            collection = build_or_load_chroma_vectorstore(
                subfolder_path=subfolder_path,
                emb_function=dummy_embedding_function,
                collection_name=collection_name
            )
            multi_agent_system[agent_name] = {
                "collection": collection,
                "subfolder_path": subfolder_path
            }
    return multi_agent_system


###############################################################################
#                              Query & Routing                                #
###############################################################################

def load_agent_instructions(subfolder_path: str) -> str:
    """
    Loads optional query rewriting or special instructions from a small JSON or TXT file.
    If none is found, returns an empty string.
    """
    # e.g., "lectures_rag.json" or "lectures_rag.txt"
    # For simplicity, look for any .json or .txt in the subfolder matching "*_rag*"
    # In practice, you might define a more direct naming scheme.
    instructions = ""
    for file_name in os.listdir(subfolder_path):
        if file_name.endswith('.json') or file_name.endswith('.txt'):
            file_path = os.path.join(subfolder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    instructions = f.read()
            except:
                pass
    return instructions

def route_query_to_agents(query: str, multi_agent_system: Dict[str, Any],
                          advanced_routing: bool = False) -> List[str]:
    """
    Decides which agents to query based on the input query.
    If advanced_routing is False, returns all agents.
    If advanced_routing is True, you can implement logic or mini-LLM filtering
    to select a subset.
    """
    if not advanced_routing:
        return list(multi_agent_system.keys())

    # Example "advanced" approach (very naive):
    # If the query contains certain keywords, route to specific agent(s).
    # This can be replaced by a more sophisticated approach (LLM-based).
    selected_agents = []
    lower_query = query.lower()
    for agent_name in multi_agent_system.keys():
        if "lecture" in lower_query and "lecture" in agent_name:
            selected_agents.append(agent_name)
        elif "article" in lower_query and "article" in agent_name:
            selected_agents.append(agent_name)
        else:
            # Or default to including it
            selected_agents.append(agent_name)
    return selected_agents


def query_agent(agent_name: str, query: str, 
                multi_agent_system: Dict[str, Any],
                top_k: int = 3) -> Dict[str, Any]:
    """
    Queries the given agent's vector store and returns partial results.
    """
    collection = multi_agent_system[agent_name]["collection"]
    subfolder_path = multi_agent_system[agent_name]["subfolder_path"]

    # Load agent instructions for optional query rewriting
    instructions = load_agent_instructions(subfolder_path)
    if instructions:
        # Very naive rewriting: just prepend instructions. 
        # Real logic might parse JSON instructions or apply an LLM to transform the query.
        modified_query = instructions + " " + query
    else:
        modified_query = query

    # Convert the query into an embedding
    query_embedding = dummy_embedding_function(modified_query)

    # Perform the retrieval in Chroma
    # See Chroma's docs for advanced usage
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Build partial "response" (in real RAG, you'd pass retrieved docs + query to an LLM)
    # For demonstration, we'll just join the retrieved documents as the agent's response.
    retrieved_docs = results['documents'][0] if results['documents'] else []
    retrieved_metas = results['metadatas'][0] if results['metadatas'] else []
    retrieved_ids = results['ids'][0] if results['ids'] else []
    retrieved_scores = results['distances'][0] if results['distances'] else []

    # Let's create a naive "agent response" from the chunk text
    # In a real system, you'd feed these chunks + query into an LLM to get the summary.
    agent_response_text = " ".join(retrieved_docs[:2])  # Just taking first 2 doc chunks

    # Format partial results
    partial_result = {
        "text_response": agent_response_text,
        "context": [],
        "metadata": {
            "retrieval_time": datetime.datetime.now().isoformat(),
            "num_retrieved_documents": len(retrieved_docs),
            "relevance_scores": [float(s) for s in retrieved_scores]
        }
    }

    # Add more structured context info
    for doc, meta in zip(retrieved_docs, retrieved_metas):
        context_info = {
            "file_name": meta["source_file"],
            "chunk_index": meta["chunk_index"],
            "content": doc
        }
        partial_result["context"].append(context_info)

    return partial_result


###############################################################################
#                           Answer Aggregation                                #
###############################################################################

def detect_conflicts(agent_responses: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    A naive function to check if there are contradictory statements among agents.
    In practice, you'd use an LLM to do more sophisticated conflict detection.
    This function checks if any agent's text_response is significantly different from another's.
    """
    conflicts = []
    agent_names = list(agent_responses.keys())
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a1 = agent_names[i]
            a2 = agent_names[j]
            resp1 = agent_responses[a1]["text_response"]
            resp2 = agent_responses[a2]["text_response"]
            # A trivial check: if they differ by more than some arbitrary length or substring mismatch
            if len(resp1) > 30 and len(resp2) > 30 and resp1[:30] != resp2[:30]:
                conflicts.append(f"{a1} disagrees with {a2} (text differs).")
    return conflicts

def aggregate_responses(agent_responses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates partial answers from each agent into a final JSON output.
    - Detects conflicts
    - Constructs a consolidated 'text_response'
    - Gathers metadata
    """
    # Conflict detection (placeholder)
    conflicts_found = detect_conflicts(agent_responses)

    # Build consolidated text_response
    # Real approach: feed agent_responses into an LLM for summary and conflict mention.
    # Here we simply combine them.
    consolidated = []
    for agent_name, resp_data in agent_responses.items():
        snippet = f"[{agent_name}] says: {resp_data['text_response']}"
        consolidated.append(snippet)
    consolidated_text = "\n".join(consolidated)

    # If conflicts, mention them in the final text response
    if conflicts_found:
        conflict_msg = "Conflicts detected: " + "; ".join(conflicts_found)
        final_text_response = f"{consolidated_text}\n\nNote: {conflict_msg}"
    else:
        final_text_response = consolidated_text

    # Prepare final JSON
    final_json = {
        "text_response": final_text_response,
        "agents": {},
        "metadata": {}
    }

    # Add each agent's partial response under 'agents'
    for agent_name, resp_data in agent_responses.items():
        final_json["agents"][agent_name] = {
            "text_response": resp_data["text_response"],
            "context": resp_data["context"]
        }

    # Add some retrieval metadata
    retrieval_details = {}
    for agent_name, resp_data in agent_responses.items():
        retrieval_details[agent_name] = {
            "num_retrieved_documents": resp_data["metadata"]["num_retrieved_documents"],
            "relevance_scores": resp_data["metadata"]["relevance_scores"],
            "query_time": resp_data["metadata"]["retrieval_time"]
        }

    final_json["metadata"]["retrieval_details"] = retrieval_details

    return final_json


###############################################################################
#                                 Main Scripts                                #
###############################################################################

def setup_script(path_to_sources_dir: str):
    """
    Example "setup script" usage:
      python main.py setup --path /path/to/sources
    This builds or loads the multi-agent RAG system.
    """
    multi_rag = create_mrag(path_to_sources_dir)
    print(f"Multi-Agent RAG system setup complete with agents: {list(multi_rag.keys())}")
    return multi_rag

def query_script(query: str, multi_rag: Dict[str, Any],
                 advanced_routing: bool = False) -> str:
    """
    Example "query script" usage:
      python main.py query --user_query "some question"
    This takes a user query, routes to relevant agents, retrieves docs,
    aggregates partial answers, and returns final JSON.
    """
    # Decide which agents to query
    relevant_agents = route_query_to_agents(query, multi_rag, advanced_routing=advanced_routing)

    # Query each agent
    agent_responses = {}
    for agent_name in relevant_agents:
        agent_responses[agent_name] = query_agent(agent_name, query, multi_rag)

    # Aggregate answers
    final_output = aggregate_responses(agent_responses)

    # Return (or print) the JSON string
    final_json_str = json.dumps(final_output, indent=2)
    return final_json_str


###############################################################################
#                                  Entry Point                                #
###############################################################################

if __name__ == "__main__":
    """
    You could run this script in two modes:
      1) setup: Build or load all subfolder vector stores
      2) query: Pass a user query to the system
    
    Example:
      python main.py setup --path sources
      python main.py query --path sources --user_query "What does the lecture say about quantum entanglement?"
    """

    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent RAG System")
    parser.add_argument("command", choices=["setup", "query"], help="Which command to run")
    parser.add_argument("--path", type=str, default="sources", help="Path to sources directory")
    parser.add_argument("--user_query", type=str, default=None, help="User query for RAG")
    parser.add_argument("--advanced_routing", action='store_true', help="Use advanced routing")

    args = parser.parse_args()

    if args.command == "setup":
        # Build/load the multi-RAG system
        multi_rag = setup_script(args.path)
        # For demonstration, we might save it to a global variable or pickle.
        # For actual usage, you'd keep it in memory or re-initialize as needed.
    elif args.command == "query":
        # Typically, you'd call `setup_script` once at the beginning (or load from saved).
        # We'll do it here for demonstration, but you may optimize by skipping repeated setup.
        multi_rag = create_mrag(args.path)
        if not args.user_query:
            print("Please provide --user_query for the query command.")
        else:
            output = query_script(args.user_query, multi_rag, advanced_routing=args.advanced_routing)
            print(output)

