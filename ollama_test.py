from ollama import Client

client = Client(host="http://localhost:11434")

output = client.generate(
    model="llama3.2",
    prompt=f"what is 12!",
    stream=False,
)

print(output["response"])
