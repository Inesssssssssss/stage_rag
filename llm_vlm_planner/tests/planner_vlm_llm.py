from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_image, get_draft
from llm_vlm_planner.utils.audio import record_audio, getTranscript
import time
import chromadb
import ollama
import re
import faster_whisper
import ast

# Initialize the ChromaDB client
client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

# Set of example documents to be used for rag
documents=[]

# Generate a vision response for the image
prompt_vlm = """
You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms. Only describe the objects.
Identify and list **all** visible objects **on the table**. Return the result as a valid Python list of strings.

Return only the list, in this format:
["mug", "silver ring", "blue small pen", ...]

"""
#get_image()
response = ollama.generate(
    model='llama3.2-vision',
    prompt = prompt_vlm,
    images= ['Images/mess_live.png']
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
    obj_list = ast.literal_eval(obj_list_str)  #
    print(f"Extracted object list: {obj_list}")
else:
    print("List not found")

# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3.1:8b",  # Specify the model you have in Ollama
)

for obj in obj_list:
    docs = []
    prompt = get_draft(docs, obj)
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
    print("Do you wanna add something ?(Type no if the plan is correct)")
    #file = record_audio()
    file = "audio/No.m4a"
    #rep = getTranscript(file, "large-v3")
    rep = input()
    #pritnt(f"User answer: {rep}")
    while rep.lower() != 'no':
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

        docs = get_useful_doc(collection, obj, 0.5)
        print(f"Useful documents for the task: {docs}")
        prompt = get_draft(docs, obj)

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
        print("Do you wanna add something ?(Type no if the plan is correct)")
        rep = input()
    # Generate a plan for a task
    start_time = time.time()
    # We add collection to choose the best information in high_level_prompt
    result = planner.plan(response, docs, verbose=True)
    end_time = time.time()
    print(f"Planning time: {end_time - start_time:.4f} seconds")
