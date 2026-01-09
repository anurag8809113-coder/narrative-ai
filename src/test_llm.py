from src.llm_client import client
r = client.responses.create(model="deepseek-chat", input="Say hello in one word")
print(r.output[0].content[0].text)

