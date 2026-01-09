import pathway as pw
from ingest import build_table

def main():
    story_path = "data/stories/1_story.txt"
    table = build_table("1", story_path)

    # Simple in-memory table server (no xpack, no docs, no embeddings libs)
    print("🚀 Simple Pathway server ready (no xpack)")
    pw.run()

if __name__ == "__main__":
    main()


