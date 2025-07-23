from llm_planner_vlm_llm.task_planner import TaskPlanner
from llm_planner_vlm_llm.utils.other import get_useful_doc, get_image, get_draft, get_list_obj
from llm_planner_vlm_llm.utils.audio import record_audio, getTranscript
import time
import chromadb
import ollama
import re
import faster_whisper
import ast

# Generate a vision response for the image
dico_res = dict()
#obj_list = get_list_obj()
obj_list = ["banana", "red mug", "screwdriver", "scissors"]
for obj in obj_list:
    dico_res[obj] = {1: 0, 2: 0, 3: 0}
print("dictionary of objects: ", dico_res)
# Initialize the task planner
planner = TaskPlanner(
    model_name="llama3.1:8b",  # Specify the model you have in Ollama
)

docs = []
for i in range(10):
    for obj in obj_list:
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
        if "box 1" in response.lower():
            dico_res[obj][1] += 1
        elif "box 2" in response.lower():
            dico_res[obj][2] += 1
        elif "box 3" in response.lower():
            dico_res[obj][3] += 1
        # Generate a plan for a task
        start_time = time.time()
        # We add collection to choose the best information in high_level_prompt
        result = planner.plan(response, docs)
        end_time = time.time()
        print(f"Planning time: {end_time - start_time:.4f} seconds")
print("Final dictionary of objects: ", dico_res)
