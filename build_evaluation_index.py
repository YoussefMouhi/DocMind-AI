from pathlib import Path
import json

from src.embeddings.embedder import DocumentEmbedder
from src.retrieval.vector_store import VectorStore

SOURCE = Path("data/evaluation/ai_knowledge_base.txt")
OUTPUT = Path("data/evaluation/vectorstore")
CHUNKS_FILE = Path("data/evaluation/indexed_chunks.json")

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing evaluation document: {SOURCE}")

text = SOURCE.read_text(encoding="utf-8")

# Separate sections using the knowledge base's uppercase headings.
sections = []
heading = None
content = []

def add_section():
    if heading and content:
        sections.append({
            "source": SOURCE.name,
            "section": heading,
            "text": " ".join(content).strip()
        })

for line in text.splitlines():
    line = line.strip()
    if not line:
        continue

    if line.isupper() and len(line) < 100:
        add_section()
        heading = line
        content = []
    else:
        content.append(line)

add_section()

if not sections:
    raise RuntimeError("No document sections were detected.")

chunks = [
    {
        "chunk_id": i,
        "source": section["source"],
        "section": section["section"],
        "text": section["text"],
    }
    for i, section in enumerate(sections)
]

print(f"Detected {len(chunks)} knowledge-base sections.")

embedder = DocumentEmbedder()
embeddings = embedder.encode([chunk["text"] for chunk in chunks])

store = VectorStore(embeddings.shape[1])
store.add_documents(chunks, embeddings)
store.save(OUTPUT)

CHUNKS_FILE.write_text(
    json.dumps(chunks, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Evaluation FAISS index saved to: {OUTPUT}")
print(f"Chunk metadata saved to: {CHUNKS_FILE}")
print(f"Indexed chunks: {len(chunks)}")
