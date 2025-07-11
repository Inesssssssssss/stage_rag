from llm_planner.task_planner import TaskPlanner
import time
import chromadb
import ollama

# Set of example documents to be used for rag
documents = ["The user is in front of you",
             "Water is put in a mug",
             "Banana is food",
             "When the user is hungry, he wants to eat food",
             "I need a screw to hang the canva",]

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
    model_name="qwen3:4b",  # Specify the model you have in Ollama
)

# Generate a plan for a task
start_time = time.time()
# We add collection to choose the best information in high_level_prompt
result = planner.plan("I need to hang a canva on the wall", collection)
end_time = time.time()
print(f"Planning time: {end_time - start_time:.4f} seconds")
