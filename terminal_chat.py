"""
=====================================================================
Loom AI - Document Retriever Service | Terminal Chat Interface
=====================================================================
Interactive terminal-based chat for testing the document retriever.
This allows you to search documents directly from the command line
without needing the full agent framework running.

Usage:
    python terminal_chat.py

You'll be prompted to enter:
  1. Tenant name (which chatbot's knowledge base to search)
  2. Search configuration (top_k, min_confidence)
  3. Your search queries (type 'quit' to exit)

This is useful for:
  - Testing if documents were indexed correctly after CSV upload
  - Debugging search quality and relevance
  - Verifying embeddings and chunk sizes are producing good results
=====================================================================
"""

import sys
import os

# ---------------------------------------------------------------------------
# Add the project root to the Python path so we can import app modules.
# This is needed when running the script directly (not via FastAPI).
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.services import weaviate_client, embedding
from app.services.document import search


def print_banner():
    """Display the Loom AI terminal chat banner."""
    print("\n" + "=" * 60)
    print("  Loom AI - Document Retriever | Terminal Chat")
    print(f"  Environment: {settings.environment}")
    print(f"  Weaviate: {settings.weaviate_url}")
    print(f"  Embedding model: {settings.embedding_model}")
    print("=" * 60)


def print_results(results: dict):
    """
    Pretty-print search results in the terminal.
    Shows each result with its confidence score, doc_id, tags,
    and a truncated preview of the context.
    """
    if results["total"] == 0:
        print("\n  No results found. Try a different query or check your documents.\n")
        return

    print(f"\n  Found {results['total']} result(s):\n")
    print("-" * 60)

    for i, r in enumerate(results["results"], 1):
        # Truncate context to 200 chars for terminal readability
        context_preview = r["context"][:200]
        if len(r["context"]) > 200:
            context_preview += "..."

        print(f"  [{i}] Confidence: {r['confidence']:.4f}")
        print(f"      Doc ID: {r['doc_id']} | Chunk: {r['chunk_index']}")
        print(f"      Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
        print(f"      Context: {context_preview}")
        print("-" * 60)
    print()


def main():
    """
    Main loop for the terminal chat interface.
    Connects to Weaviate, prompts for tenant and config,
    then enters an interactive search loop.
    """
    print_banner()

    # --- Step 1: Connect to Weaviate ---
    try:
        weaviate_client.connect()
        print("\n  [OK] Connected to Weaviate")
    except Exception as e:
        print(f"\n  [ERROR] Could not connect to Weaviate: {e}")
        print("  Make sure Weaviate is running (docker compose up -d)")
        sys.exit(1)

    # --- Step 2: Get tenant name from user ---
    tenant_name = input("\n  Enter tenant name: ").strip()
    if not tenant_name:
        print("  Tenant name cannot be empty.")
        weaviate_client.disconnect()
        sys.exit(1)

    # Check if the tenant collection exists
    if not weaviate_client.collection_exists(tenant_name):
        print(f"  [WARNING] No collection found for tenant '{tenant_name}'.")
        print("  Upload documents first via the API: POST /api/v1/documents/upload/{tenant_name}")
        weaviate_client.disconnect()
        sys.exit(1)

    print(f"  [OK] Using tenant: {tenant_name}")

    # --- Step 3: Configure search parameters ---
    try:
        top_k_input = input("  Number of results to retrieve (default: 5): ").strip()
        top_k = int(top_k_input) if top_k_input else 5

        confidence_input = input("  Minimum confidence threshold 0.0-1.0 (default: 0.0): ").strip()
        min_confidence = float(confidence_input) if confidence_input else 0.0

        tags_input = input("  Filter by tags (comma-separated, or press Enter for none): ").strip()
        tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else None
    except ValueError:
        print("  Invalid input. Using defaults.")
        top_k = 5
        min_confidence = 0.0
        tags = None

    print(f"\n  Config: top_k={top_k}, min_confidence={min_confidence}, tags={tags}")
    print("  Type your search query below. Type 'quit' or 'exit' to stop.\n")

    # --- Step 4: Interactive search loop ---
    while True:
        try:
            query = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D gracefully
            print("\n\n  Goodbye!")
            break

        # Exit commands
        if query.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!")
            break

        # Skip empty queries
        if not query:
            print("  Please enter a search query.\n")
            continue

        # Perform the search
        try:
            results = search(
                tenant_name=tenant_name,
                query=query,
                top_k=top_k,
                min_confidence=min_confidence,
                tags=tags,
            )
            print_results(results)
        except Exception as e:
            print(f"\n  [ERROR] Search failed: {e}\n")

    # --- Cleanup: disconnect from Weaviate ---
    weaviate_client.disconnect()
    print("  [OK] Disconnected from Weaviate\n")


# ---------------------------------------------------------------------------
# Entry point - run this file directly to start the terminal chat.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
