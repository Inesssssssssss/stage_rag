from llm_planner_vlm_llm.task_planner import TaskPlanner
import time
import chromadb
import ollama

def get_useful_doc(scollection,task):
    """
    Find the most useful information in the documents
    """
    response = ollama.embeddings(
        prompt=task,
        model="mxbai-embed-large"
        )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=10
    )
    # Generate a threshold to filter relevant documents ( thresold can be adjusted)
    threshold = 0.9
    relevant_docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist <= threshold:
            relevant_docs.append(doc)
    return relevant_docs


# Set of example documents to be used for rag
documents=['The black gloves belong to bob',
           'The computer fan is broken',]

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

# Generate a vision response for the image
response = ollama.generate(
    model='llama3.2-vision',
    prompt= 'You are a robot assistant. Please analyze the object in the image. If you see a robot arm in the picture, ignore it and focus on the object. Be concise.',
    images= ['../Images/55.jpg']
)
image_desc = response.get("response", "")
# Initialize the task planner
planner = TaskPlanner(
    model_name="qwen3:4b",  # Specify the model you have in Ollama
)
task = f"Put this object ({image_desc}) somewhere"

docs = get_useful_doc(collection, task)
print(f"Useful documents for the task : {docs}")

# Generate a plan for a task
start_time = time.time()
# We add collection to choose the best information in high_level_prompt
result = planner.plan(task, docs)
end_time = time.time()
print(f"Planning time: {end_time - start_time:.4f} seconds")
