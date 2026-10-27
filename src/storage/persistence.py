import json
import shutil
from pathlib import Path

from src.retrieval.vector_store import VectorStore


STORAGE_DIR = Path("data/vectorstore")
STATE_FILE = STORAGE_DIR / "state.json"


def save_session(store, documents, chunks, indexed_files):
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    store.save(STORAGE_DIR)

    state = {
        "documents": documents,
        "chunks": chunks,
        "indexed_files": indexed_files,
    }

    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_session():
    if not STATE_FILE.exists():
        return None

    store = VectorStore.load(STORAGE_DIR)

    state = json.loads(
        STATE_FILE.read_text(encoding="utf-8")
    )

    if len(state["chunks"]) != store.index.ntotal:
        raise ValueError(
            "Saved chunks do not match the FAISS index"
        )

    return {
        "vector_store": store,
        "documents": state["documents"],
        "chunks": state["chunks"],
        "indexed_files": state["indexed_files"],
    }


def clear_session():
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
