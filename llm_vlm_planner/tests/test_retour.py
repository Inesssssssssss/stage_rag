from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_image, get_draft
import time
import chromadb
import ollama
import re
import ast

# Initialize ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

documents = []

# Initial vision prompt
prompt_vlm = """
You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
"""

response = ollama.generate(
    model='qwen2.5vl',
    prompt=prompt_vlm,
    images=['Images/device_live.png'],
    options={"temperature": 0.0, "num_predict": 1024}
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
    obj_list = []

planner = TaskPlanner(model_name="llama3.1:8b")

for obj in obj_list:
    docs = []
    prompt = get_draft(docs, obj)

    response_llm = ollama.generate(
        model='qwen3:4b',
        prompt=prompt,
        options={"temperature": 0.0}
    )
    response_text = re.sub(r'<think>.*?</think>\s*', '', response_llm.get('response', ''), flags=re.DOTALL)
    print(f"Initial LLM response: {response_text}")

    # Dialogue loop: LLM <-> VLM
    while "ask vlm" in response_text.lower():
        # Extraire la question à poser au VLM
        match = re.search(r"ask vlm\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
        if not match:
            print("Malformed VLM request. Skipping.")
            break
        question = match.group(1).strip()
        print(f"LLM is asking VLM: {question}")

        # Appel au VLM
        vlm_response = ollama.generate(
            model='qwen2.5vl',
            prompt=question,
            images=['Images/device_live.png'],
            options={"temperature": 0.0}
        )
        vlm_reply = vlm_response.get('response', '').strip()
        print(f"VLM response: {vlm_reply}")

        # Reposer la question au LLM avec la réponse VLM intégrée
        new_prompt = f"{response_text}\n\nVLM answered: {vlm_reply}\n\nPlease continue accordingly."
        response_llm = ollama.generate(
            model='qwen3:4b',
            prompt=new_prompt,
            options={"temperature": 0.0}
        )
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_llm.get('response', ''), flags=re.DOTALL)
        print(f"Updated LLM response: {response_text}")

    # Interaction avec l'utilisateur
    print("Do you wanna add something ? (Type no if the plan is correct)")
    rep = input()

    while rep.lower() != 'no':
        documents.append(rep)

        for i, d in enumerate(documents):
            emb = ollama.embed(model="mxbai-embed-large", input=d)
            embeddings = emb["embeddings"]
            collection.add(ids=[str(i)], embeddings=embeddings, documents=[d])

        docs = get_useful_doc(collection, obj, 0.5)
        print(f"Useful documents for the task: {docs}")
        prompt = get_draft(docs, obj)

        response_llm = ollama.generate(
            model='qwen3:4b',
            prompt=prompt,
            options={"temperature": 0.5}
        )
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_llm.get('response', ''), flags=re.DOTALL)
        print(f"LLM response after update: {response_text}")

        print("Do you wanna add something ? (Type no if the plan is correct)")
        rep = input()

    # Final planning step
    start_time = time.time()
    result = planner.plan(response_text, docs, verbose=True)
    end_time = time.time()
    print(f"Planning time: {end_time - start_time:.4f} seconds")
