from llm_planner_vlm.task_planner import TaskPlanner
import time
import chromadb
import ollama

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

# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3.2-vision",  # Specify the model you have in Ollama
)

# Generate a plan for a task
start_time = time.time()
# We add collection to choose the best information in high_level_prompt
result = planner.plan("Put the object somewhere", collection, "../Images/50.jpg")
end_time = time.time()
print(f"Planning time: {end_time - start_time:.4f} seconds")
