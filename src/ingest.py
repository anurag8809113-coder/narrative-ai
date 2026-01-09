import pathway as pw

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, size=700, overlap=120):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def get_chunks(story_path):
    return chunk_text(read_text(story_path))


