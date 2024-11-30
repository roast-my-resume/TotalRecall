import os
import re
import av
import torch
import numpy as np
from transformers import LlavaNextVideoForConditionalGeneration,LlavaNextVideoProcessor

def validate_video_path(video_path: str):
    """
    Validate the video path to ensure it exists and is a valid video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        bool: True if the path is valid, False otherwise.
    """
    # check for existence
    if not os.path.exists(video_path):
        print(f"Error: The file at path '{video_path}' does not exist.")
        return False

    # check extension
    valid_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]
    if not any(video_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Error: The file '{video_path}' is not a recognized video format.")
        return False

    # try pyav
    try:
        container = av.open(video_path)
        container.close()  # close after successfully loaded
    except av.AVError as e:
        print(f"Error: The file at path '{video_path}' could not be opened as a video. Details: {e}")
        return False

    # pass
    return True

def parse_output(output_text: str):
    """
    Parse the output into title, event type, and content.

    Args:
        output_text (str): The raw output from the model.

    Returns:
        dict: Parsed fields (title, type, description).
    """
    title_match = re.search(r"Title:\s*(.+)", output_text, re.IGNORECASE)
    event_type_match = re.search(r"Event Type:\s*(.+)", output_text, re.IGNORECASE)
    content_match = re.search(r"Content:\s*(.+)", output_text, re.IGNORECASE)
    return {
        "title": title_match.group(1).strip() if title_match else "Unknown Title",
        "type": event_type_match.group(1).strip() if event_type_match else "Unknown Event Type",
        "description": content_match.group(1).strip() if content_match else "Unknown Content",
    }


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# conversation template for content
conversation1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Describe the video content strictly using one of the following templates:\n"
                    "Content: The video shows {} {} while {}.\n"
                    "Content: A {} can be seen {} {}.\n"
                    "Content: This clip features {} {} in {}.\n"
                    "Content: The scene captures {} {} during {}.\n"
                    "Please ensure your response adheres to one of these templates."
                ),
            },
            {"type": "video"},
            ],
    }
]

# conversation template for title and type
conversation2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Analyze the text and generate a short title summarizing its main content and an event type "
                    "categorizing the content. Please format your response as follows:\n\n"
                    "Title: <short_title>\n"
                    "Event Type: <event_type>\n\n"
                    "Examples:\n"
                    "- A video of children playing soccer:\n"
                    "  Title: Children playing soccer match\n"
                    "  Event Type: Sports\n\n"
                    "- A video showing a dog running in a park:\n"
                    "  Title: Dog running in park\n"
                    "  Event Type: Recreation\n\n"
                    "Provide your response strictly in this format."
                ),
            },
            {"type": "video"},
        ],
    },
]

def load_model():
    """
    Load the model once and reuse it.
    """
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
    return model, processor

def process_video(video_path: str, model, processor):
    """
    Given a path to video clip, process it with PyAv and use LlavaNextVideo to create captions.

    Args:
        video_path (str): The full path to the video.
        model (`LlavaNextVideoForConditionalGeneration.from_pretrained`): LLaVA-NeXT-Video model.
        processor (`LlavaNextVideoProcessor.from_pretrained`): LLaVA-NeXT-Video model processor.
    Returns:
        dict: Parsed results (title, type, description).
    """

    if not validate_video_path(video_path):
        return parse_output("")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    # Load the video as an np.array, sampling uniformly 8 frames (can sample more for longer videos)
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    prompt1 = processor.apply_chat_template(conversation1, add_generation_prompt=False)
    prompt2 = processor.apply_chat_template(conversation2, add_generation_prompt=False)

    # process
    inputs1 = processor(text=prompt2, videos=video, return_tensors="pt", padding=True)
    inputs1.to(model.device)
    inputs2 = processor(text=prompt1, videos=video, return_tensors="pt", padding=True)
    inputs2.to(model.device)

    output1 = model.generate(**inputs1, max_new_tokens=100)
    output2 = model.generate(**inputs2, max_new_tokens=100)
    res1 = '\n'.join(processor.decode(output1[0][2:], skip_special_tokens=True).split("\n")[-2:])
    res2 = processor.decode(output2[0][2:], skip_special_tokens=True).split("\n")[-1]
    concat_res = '\n'.join([res1, res2])

    # print(concat_res)
    results = parse_output(concat_res)
    torch.cuda.empty_cache()
    return results