import ollama

# Generate a vision response for the image
prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table. Ignore the table and any robot arms and any qr code board that you see. Do not ignore any tools or items placed near the qr code board.
    Return the result as a valid Python list of strings. 

    """

response = ollama.generate(
    model='qwen2.5vl',
    prompt = prompt_vlm,
    images= ['/home/ines/RAG/llm_vlm_planner/Images/live.png']
    , options={
        "temperature": 0.0,
        "num_predict": 1024
    }
)
im_desc = response.get("response", "")
print(f"Image description: {im_desc}")
