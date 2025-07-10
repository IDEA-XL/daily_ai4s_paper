import json
import logging
from pathlib import Path
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_FILE = Path("processed_papers_cache.json")

def load_processed_ids() -> Set[str]:
    """
    Loads the set of processed paper IDs from the JSON cache file.
    Returns an empty set if the file doesn't exist.
    """
    if not CACHE_FILE.exists():
        return set()
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Could not read cache file at {CACHE_FILE}: {e}. Starting with an empty cache.")
        return set()

def save_processed_ids(new_ids: Set[str]):
    """
    Adds new processed paper IDs to the cache file.
    It loads existing IDs, adds the new ones, and saves the updated set.
    """
    existing_ids = load_processed_ids()
    updated_ids = existing_ids.union(new_ids)
    
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"processed_ids": list(updated_ids)}, f, indent=4)
        logging.info(f"Successfully saved {len(new_ids)} new IDs to the cache. Total cached IDs: {len(updated_ids)}.")
    except IOError as e:
        logging.error(f"Failed to save cache file at {CACHE_FILE}: {e}")

if __name__ == '__main__':
    # Example usage and testing
    print("--- Testing Cache System ---")
    
    # Clear cache for a clean test
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    
    # 1. Load from non-existent file
    ids_1 = load_processed_ids()
    print(f"Initial load (no file): {ids_1}")
    assert ids_1 == set()

    # 2. Save some IDs
    new_ids_to_save = {"id1", "id2"}
    save_processed_ids(new_ids_to_save)
    print(f"Saved IDs: {new_ids_to_save}")

    # 3. Load them back
    ids_2 = load_processed_ids()
    print(f"Loaded IDs: {ids_2}")
    assert ids_2 == {"id1", "id2"}

    # 4. Save more IDs
    more_ids_to_save = {"id2", "id3", "id4"}
    save_processed_ids(more_ids_to_save)
    print(f"Saved more IDs: {more_ids_to_save}")

    # 5. Load again to check union
    ids_3 = load_processed_ids()
    print(f"Loaded final set: {ids_3}")
    assert ids_3 == {"id1", "id2", "id3", "id4"}

    print("\n--- Cache System Test Passed ---")
    
    # Clean up the test file
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
