

import time
import re
import ast
import chromadb
import ollama
import os
from llm_vlm_planner.task_planner import TaskPlanner
from llm_vlm_planner.utils.other import get_useful_doc, get_draft

def describe_image(image_path, prompt_vlm):
    abs_image_path = os.path.abspath(image_path)
    vlm_response = ollama.generate(
        model='qwen2.5vl',
        prompt=prompt_vlm,
        images=[abs_image_path],
        options={"temperature": 0.0}
    )
    im_desc = vlm_response.get("response", "")
    print(f"Image description: {im_desc}")
    match = re.search(r'\[\s*.*?\s*\]', im_desc, re.DOTALL)
    if match:
        obj_list = ast.literal_eval(match.group(0))
        print(f"Extracted object list: {obj_list}")
        return obj_list
    else:
        print("No valid object list found.")
        return []

def embed_and_store_document(collection, doc, doc_id):
    emb = ollama.embed(model="mxbai-embed-large", input=doc)
    collection.add(
        ids=[str(doc_id)],
        embeddings=emb["embeddings"],
        documents=[doc]
    )

def llm_vlm_loop(obj, docs, image_path):
    prompt = get_draft(docs, obj, vlm=True)
    llm_messages = [
        {"role": "system", "content": "You are a helpful assistant that helps plan how to interact with physical objects."},
        {"role": "user", "content": prompt}
    ]
    while True:
        llm_response = ollama.chat(
            model='qwen3:4b',
            messages=llm_messages
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
                    images=[image_path],
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
    return llm_messages

def user_feedback_loop(obj, docs, collection, image_path, llm_messages, doc_id):
    print("\nDo you wanna add something ? (Type no if the plan is correct)")
    rep = input()
    while rep.lower() != "no":
        embed_and_store_document(collection, rep, doc_id)
        doc_id += 1
        docs = get_useful_doc(collection, obj)
        print(f"Useful documents: {docs}")
        prompt = get_draft(docs, obj, vlm=True)
        llm_messages.append({
            "role": "user",
            "content": prompt
        })
        while True:
            llm_response = ollama.chat(
                model='qwen3:4b',
                messages=llm_messages
            )
            response_text = llm_response['message']['content']
            response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
            print(f"\nLLM (updated): {response_text}")
            llm_messages.append({"role": "assistant", "content": response_text})
            if "ask vlm" in response_text.lower():
                match = re.search(r"ask vlm\s*:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    vlm_question = match.group(1).strip()
                    print(f"\nLLM is asking VLM: {vlm_question}")
                    vlm_response = ollama.generate(
                        model='qwen2.5vl',
                        prompt=vlm_question,
                        images=[image_path],
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

    return llm_messages, docs, doc_id

def main():
    client = chromadb.Client()
    collection = client.create_collection(name="docs", metadata={"hnsw:space": "cosine"})
    doc_id = 0
    prompt_vlm = """
        You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
        Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
        """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '..', 'Images', 'lego_live.png')
    obj_list = describe_image(image_path, prompt_vlm)
    planner = TaskPlanner(model_name="llama3.1:8b")
    for obj in obj_list:
        docs = get_useful_doc(collection, obj)
        llm_messages = llm_vlm_loop(obj, docs, image_path)
        llm_messages, docs, doc_id = user_feedback_loop(obj, docs, collection, image_path, llm_messages, doc_id)
        print("\nGenerating final plan...")
        start_time = time.time()
        final_response = llm_messages[-1]["content"]
        result = planner.plan(final_response, docs, verbose=True)
        end_time = time.time()
        print(f"Planning time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
