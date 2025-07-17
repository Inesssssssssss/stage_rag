from llm_planner_vlm_llm.task_planner import TaskPlanner
from llm_planner_vlm_llm.utils.other import get_useful_doc, get_image, get_draft
import time
import chromadb
import ollama
import re

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Set of example documents to be used for rag
documents=["The blue block is owned by Bob"]

# Generate a vision response for the image
get_image()
response = ollama.generate(
    model='llama3.2-vision',
    prompt= 'You are a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms. Only describe the objects',
    images= ['Images/live.png']
    , options={
        "temperature": 0.0,
        "num_predict": 1024
    }
)
im_desc = response.get("response", "")
print(f"Image description: {im_desc}")

# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3.1:8b",  # Specify the model you have in Ollama
)

docs = []

prompt = get_draft(docs, im_desc)
start_time = time.time()
response = ollama.generate(
    model='qwen3:4b',
    prompt=prompt,
    options={
            "temperature": 0.0,
            "num_predict": 1024
        }
)
response = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
print(f"Response from the model: {response}")
print("Do you wanna add something ?")
rep = input("Type no if the plan is correct")
if rep.lower() != 'no':
    documents.append(rep)

    # store each document in a vector embedding database
    for i, d in enumerate(documents):
        emb = ollama.embed(model="mxbai-embed-large", input=d)
        embeddings = emb["embeddings"]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d]
        )

    docs = get_useful_doc(collection, im_desc)
    print(f"Useful documents for the task: {docs}")
    prompt = get_draft(docs, im_desc)

    response = ollama.generate(
        model='qwen3:4b',
        prompt=prompt,
        options={
                    "temperature": 0.5,
                    "num_predict": 1024
                }
    )
    response = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
    print(f"Response from the model after adding: {response}")
# Generate a plan for a task
start_time = time.time()
# We add collection to choose the best information in high_level_prompt
result = planner.plan(response, docs)
end_time = time.time()
print(f"Planning time: {end_time - start_time:.4f} seconds")
