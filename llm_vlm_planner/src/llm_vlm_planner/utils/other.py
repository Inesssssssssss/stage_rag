import ollama
import chromadb
import cv2
from pyorbbecsdk import *
from typing import Union, Any, Optional
import numpy as np
import os
import re
import ast

def get_list_obj(model="llama3.2-vision"):
    prompt_vlm = """
    You're a robot assistant. Please look at the image and describe each object on the table simply. Ignore the table and any robot arms and any qr code board that you see. Only describe the objects.
    Identify and list **all** visible objects **on the table**. Return the result as a valid Python list of strings.

    Return only the list, in this format:
    ["mug", "silver ring", "blue small pen", ...]
    """
    #get_image()
    match = None
    while not match:
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
    obj_list_str = match.group(0)
    obj_list = ast.literal_eval(obj_list_str)
    print(f"Extracted object list: {obj_list}")
    return obj_list

def get_useful_doc(collection,task,threshold=0.5):
    """
    Find the most useful information in the documents
    """
    response = ollama.embeddings(
        prompt=task,
        model="mxbai-embed-large"
        )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=10
    )
    # Generate a threshold to filter relevant documents ( thresold can be adjusted)
    relevant_docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist <= threshold:
            relevant_docs.append(doc)
    return relevant_docs

def get_image():
    """
    Get an image from the camera and save it
    """
    config = Config()
    ctx = Context()
    dev_list = ctx.query_devices()
    pipeline = Pipeline(dev_list[0])
    device = pipeline.get_device()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    
    while True:
        try:
            frames: FrameSet = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            # covert to RGB format
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print("failed to convert frame to image")
                continue
            print("color image shape: ", color_image.shape)
            cv2.imshow("Color Viewer", color_image)
            key = cv2.waitKey(1)
            rep = "Images"
            os.makedirs(rep, exist_ok=True)
            path = os.path.join(rep, "live.png")
            cv2.imwrite(path, color_image)
            break
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()
    pipeline.stop()

def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

def get_draft(rag : str, image) -> str:
    """
    Generate a draft plan using the RAG model.
    
    Args:
        rag: The RAG model to use for generating the draft
        task: The semantic task to plan for
        
    Returns:
        A string containing the draft plan
    """
    
    rep = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config"))
    os.makedirs(rep, exist_ok=True)
    draft_file = os.path.join(rep, "draft_plan.txt")
    
    with open(draft_file, 'r') as f:
        file = f.read()
    prompt = re.sub(r"RAG_PLACEHOLDER", str(rag), file)
    prompt = re.sub(r"IMAGE_PLACEHOLDER", image, prompt)
    return prompt