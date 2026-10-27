from src.storage.persistence import load_session

saved = load_session()

if saved is None:
    raise SystemExit(
        "No saved index found. Process a document first."
    )

chunks = saved["chunks"]

print(f"\nTotal chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    print("\n" + "=" * 70)
    print(f"Position: {index}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Source: {chunk['source']}")
    print(f"Text:\n{chunk['text']}")
