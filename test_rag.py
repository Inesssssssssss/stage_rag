import ollama
import chromadb

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embed(model="mxbai-embed-large", input=d)
  embeddings = response["embeddings"]
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[d]
  )

# an example input
input = "What animals are llamas related to?"
# generate an embedding for the input and retrieve the most relevant doc
response = ollama.embeddings(
      prompt=input,
      model="mxbai-embed-large"
    )
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=10
  #where={"distances": {"$lte":350} }
)
data = results['documents'][0]
print(f" data : {results}")

threshold = 0.2  # Ajuste selon besoin
relevant_docs = []
"""
for doc, emb in zip(results["documents"][0], results["embeddings"][0]):
    similarity = cosine_similarity([query_embedding], [emb])[0][0]
    if similarity >= threshold:
        relevant_docs.append(doc)
        relevant_embeddings.append(emb)

print(f"Relevant documents: {relevant_docs}")
"""
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    if dist <= threshold:
        relevant_docs.append(doc)

print(f"Relevant documents: {relevant_docs}")
output_no_rag = ollama.generate(
  model="llama3.1:8B",
  prompt=f"Respond to this prompt: {input}"
)

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="llama3.1:8B",
  prompt=f"Using this data : {data}. Respond to this prompt: {input}"
)

print(f"RAG answer : {output['response']}\n")

print(f"without RAG :{output_no_rag['response']}")