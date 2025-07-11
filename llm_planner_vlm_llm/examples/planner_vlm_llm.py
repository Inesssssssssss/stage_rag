from llm_planner_vlm_llm.task_planner import TaskPlanner
import time
import chromadb
import ollama
import re

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
           'The fan is broken',
           'We can still use the white cloth']

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
im_desc = response.get("response", "")

# Initialize the task planner
planner = TaskPlanner(
    model_name="qwen3:4b",  # Specify the model you have in Ollama
)
task = f"Put this object ({im_desc}) somewhere"

docs = get_useful_doc(collection, task)
print(f"Useful documents for the task : {docs}")

prompt = f"""You are a fixed robotic arm equipped with a gripper. You can place objects into three distinct boxes:

- Box 1: for personal objects  
- Box 2: for tools  
- Box 3: for trash

You can use the information provided in a Python-style list: {docs}  
Try to link related pieces of information together when they refer to similar or identical objects or concepts.

Your task is the following: {task}  
Explain **simply and concisely** what you would do to complete this task, step by step, at a high level.  
The goal is to produce a draft plan of meaningful, human-level actions (e.g., "Grasp screwdriver", "Put pen in Box 1", etc.).

Keep your explanation brief and focused. Do not include implementation details or low-level actions."""
start_time = time.time()
response = ollama.generate(
    model='qwen3:4b',
    prompt=prompt
)
response = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
print(f"Response from the model: {response}")

# Generate a plan for a task
start_time = time.time()
# We add collection to choose the best information in high_level_prompt
result = planner.plan(response, docs)
end_time = time.time()
print(f"Planning time: {end_time - start_time:.4f} seconds")
