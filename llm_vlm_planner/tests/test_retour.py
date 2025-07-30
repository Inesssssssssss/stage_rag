import time
import re
import ast
import chromadb
import ollama
from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_draft

client = chromadb.Client()
collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})

documents = []

prompt_vlm = """
You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
"""

vlm_response = ollama.generate(
    model='qwen2.5vl',
    prompt=prompt_vlm,
    images=['Images/device_live.png'],
    options={"temperature": 0.0}
)
im_desc = vlm_response.get("response", "")
print(f"Image description: {im_desc}")

match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
if match:
    obj_list = ast.literal_eval(match.group(0))
    print(f"Extracted object list: {obj_list}")
else:
    print("No valid object list found.")
    obj_list = []

planner = TaskPlanner(model_name="llama3.1:8b")

for obj in obj_list:
    docs = []
    prompt = get_draft(docs, obj, vlm=True)

    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant that helps plan how to interact with physical objects."},
        {"role": "user", "content": prompt}
    ]

    while True:
        llm_response = ollama.chat(
            model='qwen3',
            messages=llm_messages,
            options={"temperature": 0.0}
        )
        response_text = llm_response['message']['content']
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
        print(f"\nLLM: {response_text}")
        llm_messages.append({"role": "assistant", "content": response_text})

        if "ask vlm" in response_text.lower():
            match = re.search(r"ask vlm\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
            if match:
                vlm_question = match.group(1).strip()
                print(f"\nLLM is asking VLM: {vlm_question}")

                vlm_response = ollama.generate(
                    model='qwen2.5vl',
                    prompt=vlm_question,
                    images=['Images/device_live.png'],
                    options={"temperature": 0.0}
                )
                vlm_text = vlm_response.get('response', '').strip()
                print(f"\nVLM: {vlm_text}")

                llm_messages.append({"role": "user", "content": f"VLM answered: {vlm_text}"})
                continue
            else:
                print("No valid VLM question detected.")
                break
        else:
            break  


    print("\nDo you wanna add something ? (Type no if the plan is correct)")
    rep = input()

    while rep.lower() != "no":
        documents.append(rep)

        for i, d in enumerate(documents):
            emb = ollama.embed(model="mxbai-embed-large", input=d)
            collection.add(
                ids=[str(i)],
                embeddings=emb["embeddings"],
                documents=[d]
            )

        docs = get_useful_doc(collection, obj, 0.5)
        print(f"Useful documents: {docs}")
        prompt = get_draft(docs, obj)
        llm_messages.append({"role": "user", "content": prompt})

        llm_response = ollama.chat(
            model='qwen3:4b',
            messages=llm_messages,
            options={"temperature": 0.0}
        )
        response_text = llm_response['message']['content']
        response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
        print(f"\nLLM: {response_text}")
        llm_messages.append({"role": "assistant", "content": response_text})

        print("\nDo you wanna add something ? (Type no if the plan is correct)")
        rep = input()

    print("\nGenerating final plan...")
    start_time = time.time()
    final_response = llm_messages[-1]["content"]
    result = planner.plan(final_response, docs, verbose=True)
    end_time = time.time()
    print(f"Planning time: {end_time - start_time:.2f} seconds")
