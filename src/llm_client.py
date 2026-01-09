import subprocess

def ask_llm(prompt: str) -> str:
    """
    Sends prompt to local Ollama (llama3) and returns response text.
    """
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()



