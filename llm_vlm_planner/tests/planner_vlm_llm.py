from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_image, get_draft
from llm_vlm_planner.utils.audio import record_audio, getTranscript
import time
import chromadb
import ollama
import re
import faster_whisper
import ast

def describe_image(image_path, prompt_vlm):
    response = ollama.generate(
        model='qwen2.5vl',
        prompt=prompt_vlm,
        images=[image_path],
        options={
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
        return obj_list
    else:
        print("List not found")
        return []

def embed_and_store_documents(collection, documents):
    for i, d in enumerate(documents):
        emb = ollama.embed(model="mxbai-embed-large", input=d)
        embeddings = emb["embeddings"]
        collection.add(
            ids=[str(i)],
            embeddings=embeddings,
            documents=[d]
        )

def plan_for_object(obj, collection, planner):
    docs = []
    prompt = get_draft(docs, obj)
    response = ollama.generate(
        model='qwen3:4b',
        prompt=prompt,
        options={"temperature": 0.0}
    )
    response_text = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
    print(f"Response from the model: {response_text}")

    documents = []
    print("Do you wanna add something ?(Type no if the plan is correct)")
    #file = record_audio()
    #file = "audio/No.m4a"
    #rep = getTranscript(file, "large-v3")
    rep = input()
    while rep.lower() != 'no':
        documents.append(rep)
        embed_and_store_documents(collection, documents)
        docs = get_useful_doc(collection, obj, 0.5)
        print(f"Useful documents for the task: {docs}")
        prompt = get_draft(docs, obj)
        response = ollama.generate(
            model='qwen3:4b',
            prompt=prompt,
            options={"temperature": 0.5}
        )
        response_text = re.sub(r'<think>.*?</think>\s*', '', response.get('response', ''), flags=re.DOTALL)
        print(f"Response from the model after adding: {response_text}")
        print("Do you wanna add something ?(Type no if the plan is correct)")
        rep = input()
    # Generate a plan for a task
    start_time = time.time()
    result = planner.plan(response_text, docs, verbose=True)
    end_time = time.time()
    print(f"Planning time: {end_time - start_time:.4f} seconds")

def main():
    # Initialize the ChromaDB client
    client = chromadb.Client()
    collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

    prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
    Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
    """
    image_path = 'Images/live.png'
    obj_list = describe_image(image_path, prompt_vlm)

    planner = TaskPlanner(
        model_name="llama3.1:8b",  # Specify the model you have in Ollama
    )

    for obj in obj_list:
        plan_for_object(obj, collection, planner)

if __name__ == "__main__":
    main()
