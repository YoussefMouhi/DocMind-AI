import re


def clean_text(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(documents, chunk_size=800, overlap=150):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    for document in documents:
        text = clean_text(document["text"])

        if not text:
            continue

        words = text.split()
        current_words = []
        current_length = 0

        def save_chunk():
            if current_words:
                chunks.append({
                    "chunk_id": len(chunks),
                    "source": document["source"],
                    "page": document["page"],
                    "text": " ".join(current_words),
                })

        for word in words:
            additional_length = len(word) + (1 if current_words else 0)

            if current_words and current_length + additional_length > chunk_size:
                save_chunk()

                previous_words = current_words[:]
                current_words = []
                current_length = 0

                for previous_word in reversed(previous_words):
                    extra = len(previous_word) + (1 if current_words else 0)

                    if current_length + extra > overlap:
                        break

                    current_words.insert(0, previous_word)
                    current_length += extra

                # Avoid carrying the entire previous chunk forward.
                if current_words == previous_words:
                    current_words = []
                    current_length = 0

            current_words.append(word)
            current_length += len(word) + (1 if len(current_words) > 1 else 0)

        save_chunk()

    return chunks
