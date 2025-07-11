import ollama
import chromadb

client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

response = ollama.generate(
    model='llama3.2-vision',
    prompt= 'You are a robot assistant. Please analyze the object in the image. If you see a robot arm in the picture, ignore it and focus on the object. Be concise.',
    images= ['Images/53.jpg']
)

print(response.get("response", ""))