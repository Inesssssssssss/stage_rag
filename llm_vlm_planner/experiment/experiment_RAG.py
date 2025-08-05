from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_image, get_draft
import time
import chromadb
import ollama
import re
import ast
import numpy as np

n = 10  # Number of iterations for the experiment
# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Set of example documents to be used for rag
documents=[]

# Generate a vision response for the image
prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
    Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
    """
#get_image()
response = ollama.generate(
    model='qwen2.5vl',
    prompt = prompt_vlm,
    images= ['Images/live.png']
    , options={
        "temperature": 0.0,
        "num_predict": 1024
    }
)
im_desc = response.get("response", "")

match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
if match:
    obj_list_str = match.group(0)
    obj_list = ast.literal_eval(obj_list_str)
    print(f"Extracted object list: {obj_list}")
else:
    print("List not found")

true_box = {'scissors': 1, 'red cup': 1, 'banana': 3, 'screwdriver': 2}
fail_box = {'scissors': 0, 'red cup': 0, 'banana': 0, 'screwdriver': 0}

# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3.1:8b",  # Specify the model you have in Ollama
)
id_coll = 0
for i in range(n):
    for obj in obj_list:
        docs = get_useful_doc(collection, obj)
        print(f"Documents for {obj}: {docs}")
        prompt = get_draft(docs, obj)
        start_time = time.time()
        response = ollama.generate(
            model='qwen3:4b',
            prompt=prompt,
            options={
                "temperature": 1.0
            }
        )
        response = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
        print(f"Response from the model: {response}")
        
        # Generate a plan for a task
        start_time = time.time()
        # We add collection to choose the best information in high_level_prompt
        result = planner.plan(response, docs)
        if not str(true_box[obj]) in result[1]:
            print(f"Error: {obj} should be in box {true_box[obj]} but found in {result[1]}")
            info = input("Add an information for the model to correct the plan: ")

            # store each document in a vector embedding database
            emb = ollama.embed(model="mxbai-embed-large", input=info)
            embeddings = emb["embeddings"]
            collection.add(
                ids=[str(id_coll)],
                embeddings=embeddings,
                documents=[info]
            )
            id_coll += 1
            fail_box[obj] += 1
        end_time = time.time()
        print(f"Planning time: {end_time - start_time:.4f} seconds")

for obj in fail_box:
    print(f"Object: {obj}, Failures: {fail_box[obj]}, Successes: {n - fail_box[obj]}, Percentage: {100 * (n - fail_box[obj]) / n:.2f}%")