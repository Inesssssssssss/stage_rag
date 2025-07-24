import ollama

# Generate a vision response for the image
prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects near the transparent qr code board. Do not ignore any tools or items placed near the qr code board.
    Identify and list all visible objects **on the table**. Return the result as a valid Python list of strings.
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
"""
input = input("Enter your input: ")

rep = ollama.chat(
    model = 'qwen2.5vl',
    messages = {"role": "user", "content": input}
)
print(f"Response from the model: {rep['message']['content']}")
"""