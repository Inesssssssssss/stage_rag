from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_image, get_draft
import time
import chromadb
import ollama
import re
import ast
import numpy as np

# Initialise le client ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Préparation du fichier de résultats texte
result_file_path = "results/experiment_results.txt"
result_file = open(result_file_path, "a")

result_file.write("\n===== New Experiment =====\n\n")

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
print(f"Image description: {im_desc}")
match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
if match:
    obj_list_str = match.group(0)
    obj_list = ast.literal_eval(obj_list_str)
    print(f"Extracted object list: {obj_list}")
else:
    print("List not found")

true_box_prob = {
    'scissors': [0.4, 0.3, 0.3],
    'red cup': [0.5, 0.0, 0.5],
    'banana': [0.5, 0.0, 0.5],
    'screwdriver': [0.3, 0.4, 0.3]
}

planner = TaskPlanner(model_name="llama3.1:8b")
id_coll = 0

for obj in obj_list:
    true_box = np.random.choice([1, 2, 3], p=true_box_prob[obj])
    docs = get_useful_doc(collection, obj)
    prompt = get_draft(docs, obj)

    response = ollama.generate(
        model='qwen3:4b',
        prompt=prompt,
        options={"temperature": 1.0}
    )
    response_text = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
    print(f"Response from the model: {response_text}")

    result = planner.plan(response_text, docs)

    corrections = []
    while str(true_box) not in result[1]:
        print(f"Error: {obj} should be in box {true_box} but found in {result[1]}")
        info = input("Add an information for the model to correct the plan: ")
        corrections.append(info)

        emb = ollama.embed(model="mxbai-embed-large", input=info)
        embeddings = emb["embeddings"]
        collection.add(
            ids=[str(id_coll)],
            embeddings=embeddings,
            documents=[info]
        )
        id_coll += 1

        docs = get_useful_doc(collection, obj)
        prompt = get_draft(docs, obj)

        response = ollama.generate(
            model='qwen3:4b',
            prompt=prompt,
            options={"temperature": 1.0}
        )
        response_text = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
        print(f"Response from the model: {response_text}")
        result = planner.plan(response_text, docs)

    # Écriture du résultat final dans le fichier texte
    result_file.write(f"Object: {obj}\n")
    result_file.write(f"  - True box: {true_box}\n")
    result_file.write(f"  - Final plan: {result[1]}\n")
    if corrections:
        result_file.write(f"  - Corrections needed ({len(corrections)}):\n")
        for i, correction in enumerate(corrections, 1):
            result_file.write(f"      {i}. {correction}\n")
    result_file.write("\n")

result_file.close()
